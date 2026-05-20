from __future__ import annotations

import unittest

from fastapi import HTTPException

from services.image_limits import MAX_IMAGES_PER_REQUEST, parse_image_count_limit


class ImageLimitsTests(unittest.TestCase):
    def test_default_count(self):
        self.assertEqual(parse_image_count_limit(None), 1)
        self.assertEqual(parse_image_count_limit(""), 1)

    def test_accepts_upper_bound(self):
        self.assertEqual(parse_image_count_limit(MAX_IMAGES_PER_REQUEST), MAX_IMAGES_PER_REQUEST)

    def test_rejects_zero(self):
        with self.assertRaises(HTTPException) as raised:
            parse_image_count_limit(0)

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.detail["error"], f"n must be between 1 and {MAX_IMAGES_PER_REQUEST}")

    def test_rejects_over_upper_bound(self):
        with self.assertRaises(HTTPException) as raised:
            parse_image_count_limit(MAX_IMAGES_PER_REQUEST + 1)

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.detail["error"], f"n must be between 1 and {MAX_IMAGES_PER_REQUEST}")

    def test_rejects_non_integer(self):
        with self.assertRaises(HTTPException) as raised:
            parse_image_count_limit("bad")

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.detail["error"], "n must be an integer")


if __name__ == "__main__":
    unittest.main()
