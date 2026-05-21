import unittest

from services.protocol.conversation import build_image_prompt, normalize_image_resolution


class ImageResolutionPromptTests(unittest.TestCase):
    def test_image_prompt_includes_size_and_resolution_hints(self):
        prompt = build_image_prompt("一只猫", "16:9", "4K")

        self.assertIn("16:9 横屏构图", prompt)
        self.assertIn("4K 清晰度", prompt)
        self.assertIn("4096", prompt)

    def test_image_resolution_normalization_rejects_unknown_values(self):
        self.assertEqual(normalize_image_resolution("2K"), "2k")
        self.assertEqual(normalize_image_resolution("8k"), "")


if __name__ == "__main__":
    unittest.main()
