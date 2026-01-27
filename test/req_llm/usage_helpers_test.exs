defmodule ReqLLM.UsageHelpersTest do
  use ExUnit.Case, async: true

  describe "ReqLLM.Usage.Tool" do
    test "build defaults to call unit" do
      assert ReqLLM.Usage.Tool.build(:web_search, 2) == %{web_search: %{count: 2, unit: :call}}
    end

    test "count respects unit matching" do
      usage = %{tool_usage: %{web_search: %{count: 3, unit: "query"}}}

      assert ReqLLM.Usage.Tool.count(usage, :web_search, :query) == 3
      assert ReqLLM.Usage.Tool.count(usage, :web_search, :call) == 0
    end
  end

  describe "ReqLLM.Usage.Image" do
    test "counts inline image parts" do
      parts = [
        %{"inlineData" => %{"mimeType" => "image/png", "data" => "AAA"}},
        %{"inlineData" => %{"mimeType" => "text/plain", "data" => "BBB"}},
        %{"inline_data" => %{"mime_type" => "image/jpeg", "data" => "CCC"}}
      ]

      assert ReqLLM.Usage.Image.count_inline_parts(parts) == 2
    end
  end

  describe "ReqLLM.Pricing" do
    test "resolves tool unit from pricing components" do
      model = %LLMDB.Model{
        id: "m1",
        provider: :test,
        pricing: %{components: [%{kind: :tool, tool: "web_search", unit: "query"}]}
      }

      assert ReqLLM.Pricing.tool_unit(model, :web_search) == "query"
    end
  end
end
