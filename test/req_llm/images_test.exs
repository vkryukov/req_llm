defmodule ReqLLM.ImagesTest do
  use ExUnit.Case, async: true

  alias ReqLLM.{Context, Images}

  test "supported_models/0 includes known image models by heuristic" do
    models = Images.supported_models()
    assert "openai:gpt-image-1" in models
    assert "google:gemini-2.0-flash-exp-image-generation" in models
  end

  test "validate_model/1 rejects non-image models" do
    assert {:error, _} = Images.validate_model("openai:gpt-4o")
  end

  test "generate_image/3 errors when context has no user text" do
    context = Context.new([Context.system("You are helpful.")])
    assert {:error, _} = Images.generate_image("openai:gpt-image-1", context, fixture: "noop")
  end
end
