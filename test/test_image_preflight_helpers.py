from __future__ import annotations

import unittest
from unittest import mock

import utils.helper as helper


class ImagePreflightHelperTests(unittest.TestCase):
    def test_chat_image_detection_does_not_decode_base64(self) -> None:
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "edit this"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,not-valid-yet"}},
                    ],
                }
            ]
        }

        with mock.patch.object(helper.base64, "b64decode", side_effect=AssertionError("should not decode")):
            self.assertTrue(helper.has_chat_image(body))

    def test_response_image_detection_does_not_decode_base64(self) -> None:
        input_value = [
            {"type": "input_text", "text": "edit this"},
            {"type": "input_image", "image_url": "data:image/png;base64,not-valid-yet"},
        ]

        with mock.patch.object(helper.base64, "b64decode", side_effect=AssertionError("should not decode")):
            self.assertTrue(helper.has_response_input_image(input_value))

    def test_chat_image_detection_accepts_image_generation_tool(self) -> None:
        self.assertTrue(helper.is_image_chat_request({
            "model": "auto",
            "tools": [{"type": "image_generation"}],
            "messages": [{"role": "user", "content": "draw a cat"}],
        }))

    def test_normalize_image_model_defaults_auto_to_gpt_image_2(self) -> None:
        self.assertEqual(helper.normalize_image_model("auto"), "gpt-image-2")
        self.assertEqual(helper.normalize_image_model(None), "gpt-image-2")
        self.assertEqual(helper.normalize_image_model("gpt-image-2"), "gpt-image-2")
        self.assertEqual(helper.normalize_image_model("codex-gpt-image-2"), "codex-gpt-image-2")
        self.assertEqual(helper.normalize_image_model("unknown-image-model"), "unknown-image-model")


if __name__ == "__main__":
    unittest.main()
