from __future__ import annotations

import base64
from types import SimpleNamespace
import unittest
from unittest import mock

from services.protocol import conversation


class ImageResponseStorageTests(unittest.TestCase):
    def test_format_image_result_returns_b64_without_storage_when_image_save_disabled(self) -> None:
        encoded = base64.b64encode(b"fake-image").decode("ascii")

        with (
            mock.patch.object(conversation, "config", SimpleNamespace(image_save_enabled=False)),
            mock.patch.object(conversation.image_storage_service, "save") as save,
        ):
            result = conversation.format_image_result(
                [{"b64_json": encoded}],
                "cat",
                "b64_json",
                "http://app.test",
                created=1,
            )

        self.assertEqual(result["data"], [{"b64_json": encoded, "revised_prompt": "cat"}])
        save.assert_not_called()

    def test_format_image_result_rejects_url_when_image_save_disabled(self) -> None:
        encoded = base64.b64encode(b"fake-image").decode("ascii")

        with (
            mock.patch.object(conversation, "config", SimpleNamespace(image_save_enabled=False)),
            mock.patch.object(conversation.image_storage_service, "save") as save,
        ):
            with self.assertRaises(conversation.ImageGenerationError) as raised:
                conversation.format_image_result(
                    [{"b64_json": encoded}],
                    "cat",
                    "url",
                    "http://app.test",
                    created=1,
                )

        self.assertEqual(raised.exception.status_code, 400)
        save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
