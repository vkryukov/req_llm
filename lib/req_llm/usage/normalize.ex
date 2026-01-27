defmodule ReqLLM.Usage.Normalize do
  @moduledoc false

  @spec tool_usage(any()) :: map()
  def tool_usage(nil), do: %{}
  def tool_usage(usage) when is_map(usage), do: usage
  def tool_usage(_), do: %{}

  @spec image_usage(any()) :: map()
  def image_usage(nil), do: %{}
  def image_usage(usage) when is_map(usage), do: usage
  def image_usage(_), do: %{}
end
