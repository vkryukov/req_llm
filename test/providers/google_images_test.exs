defmodule ReqLLM.Providers.GoogleImagesTest do
  use ExUnit.Case, async: true

  alias ReqLLM.Providers.Google
  alias ReqLLM.{Context, Response}

  test "encode_body/1 builds generateContent image request JSON" do
    # Context with a user message containing the prompt
    context = %Context{
      messages: [
        %ReqLLM.Message{role: :user, content: "A cat in space"}
      ]
    }

    request =
      Req.new(url: "/models/gemini-2.0-flash-exp-image-generation:generateContent")
      |> Req.Request.register_options([
        :operation,
        :model,
        :aspect_ratio,
        :output_format,
        :context
      ])
      |> Req.Request.merge_options(
        operation: :image,
        model: "gemini-2.0-flash-exp-image-generation",
        aspect_ratio: "1:1",
        output_format: :png,
        context: context
      )

    encoded = Google.encode_body(request)
    body = Jason.decode!(encoded.body)

    assert get_in(body, ["generationConfig", "responseModalities"]) == ["IMAGE"]
    assert get_in(body, ["generationConfig", "imageConfig", "aspectRatio"]) == "1:1"
    assert get_in(body, ["generationConfig", "imageConfig", "mimeType"]) == nil

    assert get_in(body, ["contents", Access.at(0), "parts", Access.at(0), "text"]) ==
             "A cat in space"
  end

  test "decode_response/1 converts inlineData to ContentPart.image" do
    req =
      Req.new(url: "/models/gemini-2.0-flash-exp-image-generation:generateContent")
      |> Req.Request.register_options([:operation, :model, :context])
      |> Req.Request.merge_options(
        operation: :image,
        model: "gemini-2.0-flash-exp-image-generation",
        context: %Context{messages: []}
      )

    resp = %Req.Response{
      status: 200,
      headers: [],
      body: %{
        "candidates" => [
          %{
            "content" => %{
              "parts" => [
                %{
                  "inlineData" => %{
                    "mimeType" => "image/png",
                    "data" => Base.encode64("xyz")
                  }
                }
              ]
            }
          }
        ],
        "usageMetadata" => %{
          "promptTokenCount" => 1,
          "candidatesTokenCount" => 1,
          "totalTokenCount" => 2
        }
      }
    }

    {_req, updated} = Google.decode_response({req, resp})

    assert %Response{} = updated.body
    assert Response.image_data(updated.body) == "xyz"
    assert Response.usage(updated.body)[:total_tokens] == 2
  end
end
