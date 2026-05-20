from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Callable

from services.auth_service import ImageQuotaError, ImageQuotaMode, auth_service
from utils.log import logger


def reserve_for_image_request(
    identity: dict[str, object],
    *,
    mode: ImageQuotaMode,
    count: object = 1,
    reservation_id: str = "",
) -> dict[str, Any] | None:
    return auth_service.reserve_image_quota(identity, mode=mode, count=count, reservation_id=reservation_id)


def refund_image_reservation(reservation: dict[str, Any] | None) -> bool:
    if not reservation:
        return True
    try:
        auth_service.refund_image_quota(reservation)
        return True
    except Exception as exc:
        logger.error({"event": "image_quota_refund_failed", "reservation": reservation, "error": str(exc)})
        return False


def confirm_image_reservation(reservation: dict[str, Any] | None) -> bool:
    if not reservation:
        return True
    try:
        auth_service.confirm_image_quota(reservation)
        return True
    except Exception as exc:
        logger.error({"event": "image_quota_confirm_failed", "reservation": reservation, "error": str(exc)})
        refund_image_reservation(reservation)
        return False


def finalize_image_reservation(reservation: dict[str, Any] | None, *, charge: bool) -> str:
    if not reservation:
        return "not_required"
    if not charge:
        return "refunded" if refund_image_reservation(reservation) else "pending"
    try:
        auth_service.confirm_image_quota(reservation)
        return "charged"
    except Exception as exc:
        logger.error({"event": "image_quota_confirm_failed", "reservation": reservation, "error": str(exc)})
        return "refunded" if refund_image_reservation(reservation) else "pending"


QuotaSuccessPredicate = Callable[[dict[str, Any]], bool]


def _default_image_success(item: dict[str, Any]) -> bool:
    data = item.get("data")
    return isinstance(data, list) and bool(data)


def run_image_handler_with_quota(
    handler: Callable[[dict[str, Any]], dict[str, Any] | Iterator[dict[str, Any]]],
    payload: dict[str, Any],
    reservation: dict[str, Any] | None,
    *,
    success_predicate: QuotaSuccessPredicate | None = None,
) -> dict[str, Any] | Iterator[dict[str, Any]]:
    is_success = success_predicate or _default_image_success
    try:
        result = handler(payload)
    except Exception:
        refund_image_reservation(reservation)
        raise

    if isinstance(result, dict):
        if is_success(result):
            confirm_image_reservation(reservation)
        else:
            refund_image_reservation(reservation)
        return result

    return _stream_with_quota(result, reservation, is_success)


def _stream_with_quota(
    items: Iterator[dict[str, Any]],
    reservation: dict[str, Any] | None,
    success_predicate: QuotaSuccessPredicate,
) -> Iterator[dict[str, Any]]:
    finalized = False
    failed = False
    completed = False
    try:
        for item in items:
            if isinstance(item, dict) and success_predicate(item) and not finalized:
                finalized = finalize_image_reservation(reservation, charge=True) != "pending"
            yield item
        completed = True
    except Exception:
        failed = True
        if not finalized:
            refund_image_reservation(reservation)
        raise
    except GeneratorExit:
        failed = True
        if not finalized:
            refund_image_reservation(reservation)
        raise
    finally:
        if not failed and completed and not finalized:
            refund_image_reservation(reservation)


__all__ = [
    "ImageQuotaError",
    "confirm_image_reservation",
    "finalize_image_reservation",
    "refund_image_reservation",
    "reserve_for_image_request",
    "run_image_handler_with_quota",
]
