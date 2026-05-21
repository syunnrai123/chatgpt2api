from __future__ import annotations

import unittest
from unittest import mock

from services.protocol import openai_v1_chat_complete, openai_v1_image_edit, openai_v1_image_generations, openai_v1_response


class ImageModelNormalizationTests(unittest.TestCase):
    def test_images_generations_normalizes_auto_model(self) -> None:
        captured = {}

        def fake_stream(request):
            captured["model"] = request.model
            return iter(())

        with mock.patch.object(openai_v1_image_generations, "stream_image_outputs_with_pool", fake_stream):
            openai_v1_image_generations.handle({"model": "auto", "prompt": "draw a cat"})

        self.assertEqual(captured["model"], "gpt-image-2")

    def test_images_generations_accepts_resolution_params(self) -> None:
        captured = {}

        def fake_stream(request):
            captured["size"] = request.size
            captured["resolution"] = request.resolution
            return iter(())

        with mock.patch.object(openai_v1_image_generations, "stream_image_outputs_with_pool", fake_stream):
            openai_v1_image_generations.handle({
                "model": "gpt-image-2",
                "prompt": "draw a cat",
                "params": {"size": "4:3", "size_tier": "4K"},
            })

        self.assertEqual(captured["size"], "4:3")
        self.assertEqual(captured["resolution"], "4K")

    def test_images_edits_normalizes_auto_model(self) -> None:
        captured = {}

        def fake_stream(request):
            captured["model"] = request.model
            return iter(())

        with mock.patch.object(openai_v1_image_edit, "stream_image_outputs_with_pool", fake_stream):
            openai_v1_image_edit.handle({
                "model": "auto",
                "prompt": "edit this",
                "images": [(b"image-bytes", "image.png", "image/png")],
            })

        self.assertEqual(captured["model"], "gpt-image-2")

    def test_chat_completions_image_args_normalizes_auto_model(self) -> None:
        model, prompt, count, images = openai_v1_chat_complete.chat_image_args({
            "model": "auto",
            "modalities": ["image"],
            "messages": [{"role": "user", "content": "draw a cat"}],
        })

        self.assertEqual(model, "gpt-image-2")
        self.assertEqual(prompt, "draw a cat")
        self.assertEqual(count, 1)
        self.assertEqual(images, [])

    def test_responses_image_tool_normalizes_auto_model(self) -> None:
        captured = {}

        def fake_stream(request):
            captured["model"] = request.model
            return iter(())

        def fake_response(_outputs, _prompt, model):
            captured["response_model"] = model
            return iter((openai_v1_response.response_completed("resp_test", model, 1, []),))

        with (
            mock.patch.object(openai_v1_response, "stream_image_outputs_with_pool", fake_stream),
            mock.patch.object(openai_v1_response, "stream_image_response", fake_response),
        ):
            openai_v1_response.handle({
                "model": "auto",
                "input": "draw a cat",
                "tools": [{"type": "image_generation"}],
            })

        self.assertEqual(captured["model"], "gpt-image-2")
        self.assertEqual(captured["response_model"], "gpt-image-2")

    def test_responses_image_tool_accepts_resolution_options(self) -> None:
        captured = {}

        def fake_stream(request):
            captured["size"] = request.size
            captured["resolution"] = request.resolution
            return iter(())

        def fake_response(_outputs, _prompt, model):
            return iter((openai_v1_response.response_completed("resp_test", model, 1, []),))

        with (
            mock.patch.object(openai_v1_response, "stream_image_outputs_with_pool", fake_stream),
            mock.patch.object(openai_v1_response, "stream_image_response", fake_response),
        ):
            openai_v1_response.handle({
                "model": "gpt-image-2",
                "input": "draw a cat",
                "tools": [{"type": "image_generation", "params": {"aspect_ratio": "16:9", "resolution": "4k"}}],
            })

        self.assertEqual(captured["size"], "16:9")
        self.assertEqual(captured["resolution"], "4k")


if __name__ == "__main__":
    unittest.main()
