import unittest
from io import BytesIO

from PIL import Image

from services.protocol.conversation import (
    ImageGenerationError,
    build_image_prompt,
    ensure_requested_image_resolution,
    image_request_options,
    image_output_size,
    normalize_image_resolution,
)


class ImageResolutionPromptTests(unittest.TestCase):
    def test_image_prompt_includes_size_and_resolution_hints(self):
        prompt = build_image_prompt("一只猫", "16:9", "4K")

        self.assertIn("16:9 横屏构图", prompt)
        self.assertIn("4K 清晰度", prompt)
        self.assertIn("3312x1872", prompt)

    def test_image_resolution_normalization_rejects_unknown_values(self):
        self.assertEqual(normalize_image_resolution("2K"), "2k")
        self.assertEqual(normalize_image_resolution("8k"), "")

    def test_image_output_size_maps_resolution_and_ratio(self):
        self.assertEqual(image_output_size("4:3", "4k"), "2880x2160")
        self.assertEqual(image_output_size("16:9", "4k"), "3312x1872")
        self.assertEqual(image_output_size("21:9", "4k"), "3808x1632")
        self.assertEqual(image_output_size("1:1", "2k"), "1248x1248")
        self.assertEqual(image_output_size(None, "4k"), "")
        self.assertEqual(image_output_size("2500x500", "4k"), "")

    def test_exact_pixel_size_inferrs_resolution_tier(self):
        self.assertEqual(image_request_options({"size": "2880x2160"}), ("2880x2160", "4k"))
        self.assertEqual(image_request_options({"params": {"size": "1440x1088"}}), ("1440x1088", "2k"))
        self.assertEqual(image_request_options({"size": "2500x500"}), ("2500x500", "1k"))

    def test_low_resolution_result_rejects_high_resolution_request(self):
        buffer = BytesIO()
        Image.new("RGB", (1448, 1086), color="white").save(buffer, format="PNG")

        with self.assertRaisesRegex(ImageGenerationError, "requested resolution 4K"):
            ensure_requested_image_resolution(buffer.getvalue(), "4:3", "4k")

    def test_near_target_result_accepts_minor_encoder_variance(self):
        buffer = BytesIO()
        Image.new("RGB", (1448, 1086), color="white").save(buffer, format="PNG")

        ensure_requested_image_resolution(buffer.getvalue(), "4:3", "2k")

    def test_unspecified_ratio_accepts_high_resolution_area(self):
        buffer = BytesIO()
        Image.new("RGB", (3312, 1872), color="white").save(buffer, format="PNG")

        ensure_requested_image_resolution(buffer.getvalue(), None, "4k")

    def test_unspecified_ratio_rejects_low_resolution_area(self):
        buffer = BytesIO()
        Image.new("RGB", (1448, 1086), color="white").save(buffer, format="PNG")

        with self.assertRaisesRegex(ImageGenerationError, "requested resolution 4K"):
            ensure_requested_image_resolution(buffer.getvalue(), None, "4k")

    def test_custom_pixel_size_uses_exact_dimensions_instead_of_tier_area(self):
        buffer = BytesIO()
        Image.new("RGB", (2500, 500), color="white").save(buffer, format="PNG")

        ensure_requested_image_resolution(buffer.getvalue(), "2500x500", "4k")

    def test_custom_pixel_size_rejects_when_exact_dimensions_not_met(self):
        buffer = BytesIO()
        Image.new("RGB", (1448, 500), color="white").save(buffer, format="PNG")

        with self.assertRaisesRegex(ImageGenerationError, "requested size is 2500x500"):
            ensure_requested_image_resolution(buffer.getvalue(), "2500x500", "4k")


if __name__ == "__main__":
    unittest.main()
