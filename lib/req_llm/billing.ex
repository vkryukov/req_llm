defmodule ReqLLM.Billing do
  @moduledoc """
  Component-based billing calculator for token and tool usage.
  """

  @spec calculate(map(), LLMDB.Model.t() | nil) :: {:ok, map() | nil}
  def calculate(_usage, nil), do: {:ok, nil}

  def calculate(usage, %LLMDB.Model{} = model) when is_map(usage) do
    pricing = Map.get(model, :pricing)

    components =
      case pricing do
        %{components: comps} when is_list(comps) -> comps
        %{"components" => comps} when is_list(comps) -> comps
        _ -> []
      end

    if components == [] do
      token_cost_fallback(usage, Map.get(model, :cost))
    else
      {:ok, compute_costs(usage, pricing)}
    end
  end

  defp token_cost_fallback(_usage, nil), do: {:ok, nil}

  defp token_cost_fallback(usage, cost_map) when is_map(cost_map) do
    case ReqLLM.Cost.calculate(usage, cost_map) do
      {:ok, nil} ->
        {:ok, nil}

      {:ok, breakdown} ->
        {:ok,
         %{
           tokens: breakdown.total_cost,
           tools: 0.0,
           images: 0.0,
           storage: 0.0,
           total: breakdown.total_cost,
           line_items: [],
           input_cost: breakdown.input_cost,
           output_cost: breakdown.output_cost
         }}
    end
  end

  defp compute_costs(usage, pricing) do
    components = Map.get(pricing, :components) || Map.get(pricing, "components") || []
    usage = adjust_output_for_reasoning(usage, components)

    {line_items, totals} =
      Enum.reduce(components, {[], %{tokens: 0.0, tools: 0.0, images: 0.0, storage: 0.0}}, fn
        component, {items, totals_acc} ->
          case component_cost(component, usage) do
            {:ok, count, cost, kind} ->
              updated_totals = Map.update!(totals_acc, kind, &Float.round(&1 + cost, 6))

              item = %{
                id: component_id(component),
                count: count,
                cost: cost,
                kind: kind
              }

              {items ++ [item], updated_totals}

            :skip ->
              {items, totals_acc}
          end
      end)

    token_costs = token_cost_breakdown(line_items)

    total_cost =
      Float.round(
        totals.tokens + totals.tools + totals.images + totals.storage,
        6
      )

    %{
      tokens: totals.tokens,
      tools: totals.tools,
      images: totals.images,
      storage: totals.storage,
      total: total_cost,
      line_items: line_items,
      input_cost: token_costs.input_cost,
      output_cost: token_costs.output_cost
    }
  end

  defp adjust_output_for_reasoning(usage, components) do
    add_reasoning =
      Map.get(usage, :add_reasoning_to_cost) || Map.get(usage, "add_reasoning_to_cost")

    has_reasoning_component =
      Enum.any?(components, fn component ->
        id = component_id(component)
        is_binary(id) and String.starts_with?(id, "token.reasoning")
      end)

    if add_reasoning && not has_reasoning_component do
      output_tokens = usage_value(usage, "output_tokens")
      reasoning_tokens = usage_value(usage, "reasoning_tokens")
      Map.put(usage, :output_tokens, output_tokens + reasoning_tokens)
    else
      usage
    end
  end

  defp component_cost(component, usage) do
    per = Map.get(component, :per) || Map.get(component, "per")
    rate = Map.get(component, :rate) || Map.get(component, "rate")

    if is_number(per) and per > 0 and is_number(rate) do
      case component_kind(component) do
        nil ->
          :skip

        kind ->
          case component_count(component, usage) do
            count when is_number(count) and count > 0 ->
              cost = Float.round(count / per * rate, 6)
              {:ok, count, cost, kind}

            _ ->
              :skip
          end
      end
    else
      :skip
    end
  end

  defp component_count(component, usage) do
    kind = component_kind(component)
    meter = Map.get(component, :meter) || Map.get(component, "meter")

    cond do
      is_binary(meter) ->
        usage_value(usage, meter)

      kind == :tokens ->
        token_usage_count(component, usage)

      kind == :tools ->
        tool_usage_count(component, usage)

      kind == :images ->
        image_usage_count(component, usage)

      kind == :storage ->
        usage_value(usage, "storage")

      true ->
        usage_value(usage, component_id(component))
    end
  end

  defp token_usage_count(component, usage) do
    id = component_id(component)
    input_includes_cached = input_includes_cached?(usage)

    cached_read = usage_value(usage, "cached_tokens")
    cache_write = usage_value(usage, "cache_creation_tokens")

    case id do
      value when is_binary(value) ->
        cond do
          String.starts_with?(value, "token.input") ->
            input_tokens = usage_value(usage, "input_tokens")

            if input_includes_cached do
              max(input_tokens - cached_read - cache_write, 0)
            else
              input_tokens
            end

          String.starts_with?(value, "token.output") ->
            usage_value(usage, "output_tokens")

          String.starts_with?(value, "token.reasoning") ->
            usage_value(usage, "reasoning_tokens")

          String.starts_with?(value, "token.cache_read") ->
            cached_read

          String.starts_with?(value, "token.cache_write") ->
            cache_write

          String.starts_with?(value, "token.cache") ->
            cached_read

          true ->
            usage_value(usage, value)
        end

      _ ->
        usage_value(usage, id)
    end
  end

  defp component_kind(component) do
    kind = Map.get(component, :kind) || Map.get(component, "kind")

    case kind do
      :token -> :tokens
      "token" -> :tokens
      :tool -> :tools
      "tool" -> :tools
      :image -> :images
      "image" -> :images
      :storage -> :storage
      "storage" -> :storage
      _ -> nil
    end
  end

  defp component_id(component) do
    Map.get(component, :id) || Map.get(component, "id")
  end

  defp usage_value(usage, key) when is_binary(key) do
    Map.get(usage, key) ||
      case existing_atom(key) do
        nil -> 0
        atom -> Map.get(usage, atom) || 0
      end
  end

  defp usage_value(usage, key), do: Map.get(usage, key) || 0

  defp tool_usage_count(component, usage) do
    tool = Map.get(component, :tool) || Map.get(component, "tool")
    tool_key = normalize_tool_key(tool)
    tool_usage = Map.get(usage, :tool_usage) || Map.get(usage, "tool_usage") || %{}
    entry = tool_usage_entry(tool_usage, tool_key)
    count = Map.get(entry, :count) || Map.get(entry, "count") || 0
    component_unit = component_unit(component)
    usage_unit = usage_unit(entry)

    if unit_match?(component_unit, usage_unit) do
      count
    else
      0
    end
  end

  defp normalize_tool_key(tool) when is_atom(tool), do: tool
  defp normalize_tool_key(tool) when is_binary(tool), do: tool
  defp normalize_tool_key(_), do: :unknown

  defp tool_usage_entry(tool_usage, key) when is_atom(key) do
    Map.get(tool_usage, key) || Map.get(tool_usage, Atom.to_string(key)) || %{}
  end

  defp tool_usage_entry(tool_usage, key) when is_binary(key) do
    Map.get(tool_usage, key) ||
      case existing_atom(key) do
        nil -> %{}
        atom -> Map.get(tool_usage, atom) || %{}
      end
  end

  defp input_includes_cached?(usage) do
    case Map.fetch(usage, :input_includes_cached) do
      {:ok, value} when is_boolean(value) ->
        value

      _ ->
        case Map.fetch(usage, "input_includes_cached") do
          {:ok, value} when is_boolean(value) -> value
          _ -> true
        end
    end
  end

  defp component_unit(component) do
    normalize_unit(Map.get(component, :unit) || Map.get(component, "unit"))
  end

  defp usage_unit(entry) do
    normalize_unit(Map.get(entry, :unit) || Map.get(entry, "unit"))
  end

  defp unit_match?(component_unit, usage_unit) do
    component_unit == nil or usage_unit == nil or component_unit == usage_unit
  end

  defp normalize_unit(nil), do: nil
  defp normalize_unit(unit) when is_atom(unit), do: Atom.to_string(unit)
  defp normalize_unit(unit) when is_binary(unit), do: unit
  defp normalize_unit(_), do: nil

  defp existing_atom(value) when is_binary(value) do
    String.to_existing_atom(value)
  rescue
    ArgumentError -> nil
  end

  defp image_usage_count(component, usage) do
    usage_map = Map.get(usage, :image_usage) || Map.get(usage, "image_usage") || %{}
    generated = Map.get(usage_map, :generated) || Map.get(usage_map, "generated") || %{}
    size_class = Map.get(component, :size_class) || Map.get(component, "size_class")

    case size_class do
      nil ->
        Map.get(generated, :count) || Map.get(generated, "count") || 0

      _ ->
        usage_size = Map.get(generated, :size_class) || Map.get(generated, "size_class")

        if usage_size == size_class do
          Map.get(generated, :count) || Map.get(generated, "count") || 0
        else
          0
        end
    end
  end

  defp token_cost_breakdown(line_items) do
    input_cost =
      line_items
      |> Enum.filter(&token_input_item?/1)
      |> Enum.reduce(0.0, fn item, acc -> Float.round(acc + item.cost, 6) end)

    output_cost =
      line_items
      |> Enum.filter(&token_output_item?/1)
      |> Enum.reduce(0.0, fn item, acc -> Float.round(acc + item.cost, 6) end)

    %{input_cost: input_cost, output_cost: output_cost}
  end

  defp token_input_item?(%{id: id}) when is_binary(id) do
    String.starts_with?(id, "token.input") or String.starts_with?(id, "token.cache")
  end

  defp token_input_item?(_), do: false

  defp token_output_item?(%{id: id}) when is_binary(id) do
    String.starts_with?(id, "token.output") or String.starts_with?(id, "token.reasoning")
  end

  defp token_output_item?(_), do: false
end
