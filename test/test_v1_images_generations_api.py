from __future__ import annotations

import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.ai as ai_module


AUTH_HEADERS = {"Authorization": "Bearer chatgpt2api"}


class ImagesGenerationsApiTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def fake_handle(payload):
            self.calls.append(payload)
            return {"created": 1, "data": [{"url": "http://example.test/image.png"}]}

        self.handle_patcher = mock.patch.object(ai_module.openai_v1_image_generations, "handle", fake_handle)
        self.filter_patcher = mock.patch.object(ai_module, "filter_or_log", mock.AsyncMock())
        self.handle_patcher.start()
        self.filter_patcher.start()
        self.addCleanup(self.handle_patcher.stop)
        self.addCleanup(self.filter_patcher.stop)

        app = FastAPI()
        app.include_router(ai_module.create_router())
        self.client = TestClient(app)

    def test_image_generation_accepts_n_at_task_limit(self):
        response = self.client.post(
            "/v1/images/generations",
            headers=AUTH_HEADERS,
            json={"prompt": "cat", "model": "gpt-image-2", "n": 20},
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(self.calls[0]["n"], 20)

    def test_image_generation_rejects_n_over_task_limit(self):
        response = self.client.post(
            "/v1/images/generations",
            headers=AUTH_HEADERS,
            json={"prompt": "cat", "model": "gpt-image-2", "n": 21},
        )

        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
