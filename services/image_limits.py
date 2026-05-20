from __future__ import annotations

from fastapi import HTTPException


MAX_IMAGES_PER_REQUEST = 20


def parse_image_count_limit(raw_value: object) -> int:
    try:
        value = 1 if raw_value is None or raw_value == "" else int(raw_value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"error": "n must be an integer"}) from exc
    if value < 1 or value > MAX_IMAGES_PER_REQUEST:
        raise HTTPException(status_code=400, detail={"error": f"n must be between 1 and {MAX_IMAGES_PER_REQUEST}"})
    return value
