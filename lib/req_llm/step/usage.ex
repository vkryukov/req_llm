defmodule ReqLLM.Step.Usage do
  @moduledoc """
  Centralized Req step that extracts token usage from provider responses,
  normalizes usage values across providers, computes costs, and emits telemetry.

  This step:
  * Extracts token usage numbers from provider responses
  * Normalizes usage data across different provider formats
  * Calculates costs using ReqLLM.Model cost metadata
  * Stores usage data in `response.private[:req_llm][:usage]`
  * Emits telemetry events for monitoring

  ## Usage

      request
      |> ReqLLM.Step.Usage.attach(model)

  ## Telemetry Events

  Emits `[:req_llm, :token_usage]` events with:
  * Measurements: `%{tokens: %{input: 123, output: 456, reasoning: 64}, cost: 0.0123}`
  * Metadata: `%{model: %LLMDB.Model{}}`
  """

  @event [:req_llm, :token_usage]

  @doc """
  Attaches the Usage step to a Req request.

  ## Parameters

  - `req` - The Req.Request struct
  - `model` - Optional ReqLLM.Model struct for cost calculation

  ## Examples

      request
      |> ReqLLM.Step.Usage.attach(model)

  """
  @spec attach(Req.Request.t(), LLMDB.Model.t() | nil) :: Req.Request.t()
  def attach(%Req.Request{} = req, model \\ nil) do
    req
    |> Req.Request.append_response_steps(llm_usage: &__MODULE__.handle/1)
    |> then(fn r ->
      if model, do: Req.Request.put_private(r, :req_llm_model, model), else: r
    end)
  end

  @doc false
  @spec handle({Req.Request.t(), Req.Response.t()}) :: {Req.Request.t(), Req.Response.t()}
  def handle({req, resp}) do
    with {:ok, model} <- fetch_model(req),
         provider_module = provider_module_from_model(model),
         {:ok, usage} <- extract_usage(resp.body, provider_module, model),
         {:ok, cost_breakdown} <- compute_cost_breakdown(usage, model) do
      total_cost = cost_breakdown && cost_breakdown.total_cost
      meta = %{tokens: usage, cost: total_cost}

      meta =
        if cost_breakdown do
          Map.merge(meta, %{
            input_cost: cost_breakdown.input_cost,
            output_cost: cost_breakdown.output_cost,
            total_cost: cost_breakdown.total_cost
          })
        else
          meta
        end

      :telemetry.execute(@event, meta, %{model: model})

      req_llm_data = Map.get(resp.private, :req_llm, %{})
      updated_req_llm_data = Map.put(req_llm_data, :usage, meta)

      updated_resp =
        case resp.body do
          %ReqLLM.Response{usage: response_usage} when is_map(response_usage) ->
            cached_read_tokens = usage[:cached_input] || 0
            cache_creation_tokens = usage[:cache_creation] || 0

            augmented_usage =
              response_usage
              |> Map.put_new(:input_tokens, usage.input)
              |> Map.put_new(:output_tokens, usage.output)
              |> maybe_put_total_tokens(usage.input, usage.output)
              |> Map.put(:reasoning_tokens, usage.reasoning)
              |> Map.put(:cached_tokens, cached_read_tokens)
              |> Map.put(:cache_creation_tokens, cache_creation_tokens)
              |> Map.put(:tool_usage, usage.tool_usage)
              |> Map.put(:image_usage, usage.image_usage)
              |> maybe_merge_cost(cost_breakdown)

            updated_body = %{resp.body | usage: augmented_usage}
            %{resp | body: updated_body}

          _ ->
            resp
        end

      updated_resp = %{
        updated_resp
        | private: Map.put(updated_resp.private, :req_llm, updated_req_llm_data)
      }

      {req, updated_resp}
    else
      _ -> {req, resp}
    end
  end

  @spec provider_module_from_model(LLMDB.Model.t()) :: module() | nil
  defp provider_module_from_model(%LLMDB.Model{provider: provider_id}) do
    case ReqLLM.provider(provider_id) do
      {:ok, module} -> module
      _ -> nil
    end
  end

  @spec extract_usage(any, module() | nil, LLMDB.Model.t() | nil) :: {:ok, map()} | :error
  defp extract_usage(body, provider_module, model) do
    case provider_module do
      nil -> fallback_extract_usage(body)
      module -> provider_extract_usage(body, module, model) || fallback_extract_usage(body)
    end
  end

  defp provider_extract_usage(body, module, model) when is_atom(module) do
    if function_exported?(module, :extract_usage, 2) do
      case module.extract_usage(body, model) do
        {:ok, usage} -> {:ok, normalize_usage(usage)}
        _ -> nil
      end
    end
  end

  @spec fallback_extract_usage(any) :: {:ok, map()} | :error
  defp fallback_extract_usage(%{"usage" => usage}) when is_map(usage) do
    {:ok, normalize_usage(usage)}
  end

  defp fallback_extract_usage(%{"prompt_tokens" => input, "completion_tokens" => output}) do
    {:ok, %{input: input, output: output, reasoning: 0, cached_input: 0, cache_creation: 0}}
  end

  defp fallback_extract_usage(%{"input_tokens" => input, "output_tokens" => output}) do
    {:ok, %{input: input, output: output, reasoning: 0, cached_input: 0, cache_creation: 0}}
  end

  defp fallback_extract_usage(%ReqLLM.Response{usage: usage}) when is_map(usage) do
    {:ok, normalize_usage(usage)}
  end

  defp fallback_extract_usage(_), do: :error

  @spec normalize_usage(map()) :: map()
  defp normalize_usage(usage) when is_map(usage) do
    input_includes_cached = detect_input_includes_cached(usage)

    input =
      usage[:input] || usage["input"] || usage["prompt_tokens"] || usage[:prompt_tokens] ||
        usage["input_tokens"] || usage[:input_tokens] || 0

    output =
      usage[:output] || usage["output"] || usage["completion_tokens"] ||
        usage[:completion_tokens] || usage["output_tokens"] || usage[:output_tokens] || 0

    reasoning =
      usage[:reasoning] || usage["reasoning"] || usage[:reasoning_tokens] ||
        usage["reasoning_tokens"] || get_reasoning_tokens(usage) || 0

    cached_input = get_cached_input_tokens(usage, input, input_includes_cached)
    cache_creation = get_cache_creation_tokens(usage, input, input_includes_cached)

    total_tokens = total_tokens_from_usage(usage, input, output)

    %{
      input: input,
      output: output,
      reasoning: reasoning,
      cached_input: cached_input,
      cache_creation: cache_creation,
      input_includes_cached: input_includes_cached,
      add_reasoning_to_cost: get_add_reasoning_to_cost(usage),
      tool_usage:
        ReqLLM.Usage.Normalize.tool_usage(
          Map.get(usage, :tool_usage) || Map.get(usage, "tool_usage")
        ),
      image_usage:
        ReqLLM.Usage.Normalize.image_usage(
          Map.get(usage, :image_usage) || Map.get(usage, "image_usage")
        ),
      input_tokens: input,
      output_tokens: output,
      total_tokens: total_tokens,
      cached_tokens: cached_input,
      cache_creation_tokens: cache_creation,
      reasoning_tokens: reasoning
    }
  end

  defp safe_total_tokens(input, output) when is_number(input) and is_number(output) do
    input + output
  end

  defp safe_total_tokens(_input, _output), do: nil

  defp total_tokens_from_usage(usage, input, output) do
    total =
      usage[:total_tokens] || usage["total_tokens"] ||
        usage[:totalTokenCount] || usage["totalTokenCount"]

    case total do
      value when is_number(value) -> value
      _ -> safe_total_tokens(input, output)
    end
  end

  defp maybe_put_total_tokens(map, input, output) do
    case safe_total_tokens(input, output) do
      nil -> map
      total -> Map.put_new(map, :total_tokens, total)
    end
  end

  defp detect_input_includes_cached(usage) do
    has_openai_format =
      get_in(usage, ["prompt_tokens_details", "cached_tokens"]) != nil or
        get_in(usage, [:prompt_tokens_details, :cached_tokens]) != nil or
        get_in(usage, ["input_tokens_details", "cached_tokens"]) != nil or
        get_in(usage, [:input_tokens_details, :cached_tokens]) != nil

    has_anthropic_format =
      Map.has_key?(usage, "cache_read_input_tokens") or
        Map.has_key?(usage, :cache_read_input_tokens) or
        Map.has_key?(usage, "cache_creation_input_tokens") or
        Map.has_key?(usage, :cache_creation_input_tokens) or
        Map.has_key?(usage, "cacheReadInputTokens") or
        Map.has_key?(usage, :cacheReadInputTokens) or
        Map.has_key?(usage, "cacheWriteInputTokens") or
        Map.has_key?(usage, :cacheWriteInputTokens) or
        Map.has_key?(usage, "cacheReadInputTokenCount") or
        Map.has_key?(usage, :cacheReadInputTokenCount) or
        Map.has_key?(usage, "cacheWriteInputTokenCount") or
        Map.has_key?(usage, :cacheWriteInputTokenCount)

    cond do
      has_openai_format -> true
      has_anthropic_format -> false
      true -> true
    end
  end

  defp get_add_reasoning_to_cost(usage) do
    usage[:add_reasoning_to_cost] || usage["add_reasoning_to_cost"] ||
      is_google_gemini_format(usage)
  end

  defp is_google_gemini_format(usage) do
    Map.has_key?(usage, "thoughtsTokenCount") or
      Map.has_key?(usage, :thoughtsTokenCount)
  end

  defp get_reasoning_tokens(usage) do
    reasoning =
      get_in(usage, ["completion_tokens_details", "reasoning_tokens"]) ||
        get_in(usage, [:completion_tokens_details, :reasoning_tokens])

    case reasoning do
      n when is_integer(n) -> n
      _ -> 0
    end
  end

  defp get_cached_input_tokens(usage, input, input_includes_cached) do
    cached =
      usage[:cache_read_input_tokens] || usage["cache_read_input_tokens"] ||
        usage[:cacheReadInputTokens] || usage["cacheReadInputTokens"] ||
        usage[:cacheReadInputTokenCount] || usage["cacheReadInputTokenCount"] ||
        usage[:cached_input] || usage["cached_input"] ||
        usage[:cached_tokens] || usage["cached_tokens"] ||
        get_in(usage, ["prompt_tokens_details", "cached_tokens"]) ||
        get_in(usage, [:prompt_tokens_details, :cached_tokens])

    if input_includes_cached do
      clamp_tokens(cached, input)
    else
      safe_to_int(cached)
    end
  end

  defp get_cache_creation_tokens(usage, input, input_includes_cached) do
    creation =
      usage[:cache_creation_input_tokens] || usage["cache_creation_input_tokens"] ||
        usage[:cacheWriteInputTokens] || usage["cacheWriteInputTokens"] ||
        usage[:cacheWriteInputTokenCount] || usage["cacheWriteInputTokenCount"] ||
        usage[:cache_write_input_tokens] || usage["cache_write_input_tokens"]

    if input_includes_cached do
      clamp_tokens(creation, input)
    else
      safe_to_int(creation)
    end
  end

  defp safe_to_int(nil), do: 0
  defp safe_to_int(n) when is_integer(n), do: max(n, 0)
  defp safe_to_int(n) when is_float(n), do: max(trunc(n), 0)
  defp safe_to_int(_), do: 0

  @spec fetch_model(Req.Request.t()) :: {:ok, LLMDB.Model.t()} | :error
  defp fetch_model(%Req.Request{private: private, options: options}) do
    case private[:req_llm_model] || options[:model] do
      %LLMDB.Model{} = model -> {:ok, model}
      _ -> :error
    end
  end

  @spec compute_cost_breakdown(map(), LLMDB.Model.t()) ::
          {:ok,
           %{
             input_cost: float(),
             output_cost: float(),
             total_cost: float(),
             cost: map()
           }
           | nil}
  defp compute_cost_breakdown(usage, %LLMDB.Model{} = model) do
    case ReqLLM.Billing.calculate(usage, model) do
      {:ok, nil} ->
        {:ok, nil}

      {:ok, cost} ->
        {:ok,
         %{
           input_cost: cost.input_cost,
           output_cost: cost.output_cost,
           total_cost: cost.total,
           cost: cost
         }}
    end
  end

  defp maybe_merge_cost(usage, nil), do: usage

  defp maybe_merge_cost(usage, cost_breakdown) do
    usage
    |> Map.put(:cost, cost_breakdown.cost)
    |> Map.put(:input_cost, cost_breakdown.input_cost)
    |> Map.put(:output_cost, cost_breakdown.output_cost)
    |> Map.put(:total_cost, cost_breakdown.total_cost)
  end

  # Safely clamps a value to a valid token count within bounds.
  # Converts the value to a number and clamps it between 0 and the maximum allowed value.
  # Returns 0 if the value cannot be converted to a number.
  @spec clamp_tokens(any(), number()) :: integer()
  defp clamp_tokens(value, max_allowed) do
    case safe_to_number(value) do
      {:ok, int} ->
        int
        |> max(0)
        |> min(max(max_allowed, 0))

      _ ->
        0
    end
  end

  @spec safe_to_number(any()) :: {:ok, number()} | :error
  defp safe_to_number(value) when is_integer(value), do: {:ok, value}
  defp safe_to_number(value) when is_float(value), do: {:ok, trunc(value)}
  defp safe_to_number(_), do: :error
end
