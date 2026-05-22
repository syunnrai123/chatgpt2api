from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from functools import lru_cache
from threading import Lock
from typing import Any, Literal

from services.config import config
from services.storage.base import StorageBackend

AuthRole = Literal["admin", "user"]
ImageQuotaMode = Literal["generate", "edit"]
QUOTA_SCALE = Decimal("0.000001")
SYNC_QUOTA_RESERVATION_TTL = timedelta(hours=24)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _quota_decimal(value: object, default: Decimal | str | int = Decimal("0")) -> Decimal:
    try:
        quota = Decimal(str(value if value is not None else default))
    except (InvalidOperation, ValueError):
        quota = Decimal(str(default))
    if quota < 0:
        quota = Decimal("0")
    return quota.quantize(QUOTA_SCALE, rounding=ROUND_HALF_UP)


def _quota_number(value: Decimal | object) -> float:
    return float(_quota_decimal(value))


def _reservation_id(value: object) -> str:
    return str(value or "").strip()


def _split_ip_rules(value: object) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[\s,]+", value) if item.strip()]
    if isinstance(value, list):
        return [str(item or "").strip() for item in value if str(item or "").strip()]
    return []


def _normalize_ip_rules(value: object, *, raise_on_invalid: bool = False) -> list[str]:
    rules: list[str] = []
    seen: set[str] = set()
    for raw in _split_ip_rules(value):
        try:
            normalized = str(ipaddress.ip_network(raw, strict=False)) if "/" in raw else str(ipaddress.ip_address(raw))
        except ValueError:
            if raise_on_invalid:
                raise ValueError(f"IP 规则无效：{raw}") from None
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        rules.append(normalized)
    return rules


@lru_cache(maxsize=2048)
def _compiled_ip_rules(rules: tuple[str, ...]) -> tuple[tuple[object, ...], tuple[object, ...]]:
    addresses: list[object] = []
    networks: list[object] = []
    for rule in rules:
        if "/" in rule:
            networks.append(ipaddress.ip_network(rule, strict=False))
        else:
            addresses.append(ipaddress.ip_address(rule))
    return tuple(addresses), tuple(networks)


def _ip_rule_cache_key(rules: object) -> tuple[str, ...]:
    if isinstance(rules, list):
        return tuple(str(rule or "").strip() for rule in rules if str(rule or "").strip())
    return tuple(_normalize_ip_rules(rules))


def _ip_matches_rules(client_ip: object, rules: object) -> bool:
    rule_key = _ip_rule_cache_key(rules)
    if not rule_key:
        return True
    try:
        ip = ipaddress.ip_address(str(client_ip or "").strip())
    except ValueError:
        return False
    try:
        addresses, networks = _compiled_ip_rules(rule_key)
    except ValueError:
        return False
    return any(ip == address for address in addresses) or any(ip in network for network in networks)


def _parse_iso_datetime(value: object) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _sync_reservation_expires_at() -> str:
    return (datetime.now(timezone.utc) + SYNC_QUOTA_RESERVATION_TTL).isoformat()


class ImageQuotaError(ValueError):
    def __init__(self, message: str, *, required: float = 0, available: float = 0):
        super().__init__(message)
        self.required = required
        self.available = available


class AuthService:
    def __init__(self, storage: StorageBackend):
        self.storage = storage
        self._lock = Lock()
        self._items = self._load()
        self._last_used_flush_at: dict[str, datetime] = {}

    @staticmethod
    def _clean(value: object) -> str:
        return str(value or "").strip()

    @staticmethod
    def _default_name(role: object) -> str:
        return "管理员密钥" if str(role or "").strip().lower() == "admin" else "普通用户"

    def _normalize_item(self, raw: object) -> dict[str, object] | None:
        if not isinstance(raw, dict):
            return None
        role = self._clean(raw.get("role")).lower()
        if role not in {"admin", "user"}:
            return None
        key_hash = self._clean(raw.get("key_hash"))
        if not key_hash:
            return None
        item_id = self._clean(raw.get("id")) or uuid.uuid4().hex[:12]
        name = self._clean(raw.get("name")) or self._default_name(role)
        created_at = self._clean(raw.get("created_at")) or _now_iso()
        last_used_at = self._clean(raw.get("last_used_at")) or None
        image_quota = _quota_decimal(raw.get("image_quota"))
        image_quota_reservations = self._normalize_quota_reservations(raw.get("image_quota_reservations"))
        image_quota_reserved = min(self._reserved_quota(image_quota_reservations), image_quota)
        return {
            "id": item_id,
            "name": name,
            "role": role,
            "key_hash": key_hash,
            "enabled": bool(raw.get("enabled", True)),
            "allowed_ips": _normalize_ip_rules(raw.get("allowed_ips")),
            "image_quota": _quota_number(image_quota),
            "image_quota_reserved": _quota_number(image_quota_reserved),
            "image_quota_reservations": image_quota_reservations,
            "created_at": created_at,
            "last_used_at": last_used_at,
        }

    @staticmethod
    def _normalize_quota_reservations(raw: object) -> dict[str, dict[str, object]]:
        source = raw if isinstance(raw, dict) else {}
        reservations: dict[str, dict[str, object]] = {}
        for raw_id, raw_item in source.items():
            reservation_id = _reservation_id(raw_id)
            item = raw_item if isinstance(raw_item, dict) else {}
            cost = _quota_decimal(item.get("cost"))
            if not reservation_id or cost <= 0:
                continue
            mode = "edit" if item.get("mode") == "edit" else "generate"
            reservations[reservation_id] = {
                "id": reservation_id,
                "mode": mode,
                "cost": _quota_number(cost),
                "created_at": str(item.get("created_at") or _now_iso()),
            }
            expires_at = str(item.get("expires_at") or "").strip()
            if expires_at:
                reservations[reservation_id]["expires_at"] = expires_at
        return reservations

    @staticmethod
    def _active_reservations(item: dict[str, object]) -> dict[str, dict[str, object]]:
        reservations = item.get("image_quota_reservations")
        return AuthService._normalize_quota_reservations(reservations)

    @staticmethod
    def _reserved_quota(reservations: dict[str, dict[str, object]]) -> Decimal:
        total = Decimal("0")
        for item in reservations.values():
            total += _quota_decimal(item.get("cost"))
        return total

    def _load(self) -> list[dict[str, object]]:
        try:
            items = self.storage.load_auth_keys()
        except Exception:
            return []
        if not isinstance(items, list):
            return []
        return [normalized for item in items if (normalized := self._normalize_item(item)) is not None]

    def _save(self) -> None:
        self.storage.save_auth_keys(self._items)

    def _reload_locked(self) -> None:
        self._items = self._load()
        if self._prune_expired_quota_reservations_locked():
            try:
                self._save()
            except Exception:
                pass

    @staticmethod
    def _is_expired_quota_reservation(reservation: dict[str, object], now: datetime) -> bool:
        expires_at = _parse_iso_datetime(reservation.get("expires_at"))
        return expires_at is not None and expires_at <= now

    def _prune_expired_quota_reservations_locked(self) -> bool:
        now = datetime.now(timezone.utc)
        changed = False
        for index, item in enumerate(self._items):
            reservations = self._active_reservations(item)
            if not reservations:
                continue
            active = {
                reservation_id: reservation
                for reservation_id, reservation in reservations.items()
                if not self._is_expired_quota_reservation(reservation, now)
            }
            if len(active) == len(reservations):
                continue
            next_item = dict(item)
            balance = _quota_decimal(next_item.get("image_quota"))
            next_item["image_quota_reservations"] = active
            next_item["image_quota_reserved"] = _quota_number(min(self._reserved_quota(active), balance))
            self._items[index] = next_item
            changed = True
        return changed

    @staticmethod
    def _public_item(item: dict[str, object]) -> dict[str, object]:
        image_quota = _quota_decimal(item.get("image_quota"))
        image_quota_reserved = min(AuthService._reserved_quota(AuthService._active_reservations(item)), image_quota)
        public = {
            "id": item.get("id"),
            "name": item.get("name"),
            "role": item.get("role"),
            "enabled": bool(item.get("enabled", True)),
            "allowed_ips": _normalize_ip_rules(item.get("allowed_ips")),
            "created_at": item.get("created_at"),
            "last_used_at": item.get("last_used_at"),
        }
        if item.get("role") == "user":
            public.update({
                "image_quota": _quota_number(image_quota),
                "image_quota_reserved": _quota_number(image_quota_reserved),
                "image_quota_available": _quota_number(max(Decimal("0"), image_quota - image_quota_reserved)),
            })
        return public

    def list_keys(self, role: AuthRole | None = None) -> list[dict[str, object]]:
        with self._lock:
            self._reload_locked()
            items = [item for item in self._items if role is None or item.get("role") == role]
            return [self._public_item(item) for item in items]

    def _has_key_hash_locked(self, key_hash: str, *, exclude_id: str = "") -> bool:
        for item in self._items:
            item_id = self._clean(item.get("id"))
            if exclude_id and item_id == exclude_id:
                continue
            stored_hash = self._clean(item.get("key_hash"))
            if stored_hash and hmac.compare_digest(stored_hash, key_hash):
                return True
        return False

    def _build_key_hash_locked(self, raw_key: str, *, exclude_id: str = "") -> str:
        candidate = self._clean(raw_key)
        if not candidate:
            raise ValueError("请输入新的专用密钥")
        admin_key = self._clean(config.auth_key)
        if admin_key and hmac.compare_digest(candidate, admin_key):
            raise ValueError("这个密钥和管理员密钥冲突了，请换一个新的密钥")
        key_hash = _hash_key(candidate)
        if self._has_key_hash_locked(key_hash, exclude_id=exclude_id):
            raise ValueError("这个专用密钥已经存在，请换一个新的密钥")
        return key_hash

    def _has_name_locked(self, name: str, *, role: AuthRole | None = None, exclude_id: str = "") -> bool:
        candidate = self._clean(name)
        if not candidate:
            return False
        for item in self._items:
            item_id = self._clean(item.get("id"))
            if exclude_id and item_id == exclude_id:
                continue
            if role is not None and item.get("role") != role:
                continue
            if self._clean(item.get("name")) == candidate:
                return True
        return False

    def _build_default_name_locked(self, role: AuthRole, *, exclude_id: str = "") -> str:
        base_name = self._default_name(role)
        if not self._has_name_locked(base_name, role=role, exclude_id=exclude_id):
            return base_name
        suffix = 2
        while True:
            candidate = f"{base_name} {suffix}"
            if not self._has_name_locked(candidate, role=role, exclude_id=exclude_id):
                return candidate
            suffix += 1

    def _build_name_locked(self, name: str, *, role: AuthRole, exclude_id: str = "") -> str:
        candidate = self._clean(name)
        if not candidate:
            return self._build_default_name_locked(role, exclude_id=exclude_id)
        if self._has_name_locked(candidate, role=role, exclude_id=exclude_id):
            raise ValueError("这个名称已经在使用中了，换一个更容易区分的名称吧")
        return candidate

    def create_key(
        self,
        *,
        role: AuthRole,
        name: str = "",
        image_quota: object = 0,
        allowed_ips: object = None,
    ) -> tuple[dict[str, object], str]:
        with self._lock:
            self._reload_locked()
            normalized_name = self._build_name_locked(name, role=role)
            normalized_allowed_ips = _normalize_ip_rules(allowed_ips, raise_on_invalid=True)
            while True:
                raw_key = f"sk-{secrets.token_urlsafe(24)}"
                try:
                    key_hash = self._build_key_hash_locked(raw_key)
                    break
                except ValueError:
                    continue
            item = {
                "id": uuid.uuid4().hex[:12],
                "name": normalized_name,
                "role": role,
                "key_hash": key_hash,
                "enabled": True,
                "allowed_ips": normalized_allowed_ips,
                "image_quota": _quota_number(_quota_decimal(image_quota)) if role == "user" else 0,
                "image_quota_reserved": 0,
                "image_quota_reservations": {},
                "created_at": _now_iso(),
                "last_used_at": None,
            }
            self._items.append(item)
            self._save()
            return self._public_item(item), raw_key

    def update_key(
        self,
        key_id: str,
        updates: dict[str, object],
        *,
        role: AuthRole | None = None,
    ) -> dict[str, object] | None:
        normalized_id = self._clean(key_id)
        if not normalized_id:
            return None
        with self._lock:
            self._reload_locked()
            for index, item in enumerate(self._items):
                if item.get("id") != normalized_id:
                    continue
                if role is not None and item.get("role") != role:
                    return None
                next_item = dict(item)
                next_role = "admin" if str(next_item.get("role") or "").strip().lower() == "admin" else "user"
                if "name" in updates and updates.get("name") is not None:
                    next_item["name"] = self._build_name_locked(
                        str(updates.get("name") or ""),
                        role=next_role,
                        exclude_id=normalized_id,
                    )
                if "enabled" in updates and updates.get("enabled") is not None:
                    next_item["enabled"] = bool(updates.get("enabled"))
                if "key" in updates and updates.get("key") is not None:
                    next_item["key_hash"] = self._build_key_hash_locked(str(updates.get("key") or ""), exclude_id=normalized_id)
                if "allowed_ips" in updates and updates.get("allowed_ips") is not None:
                    next_item["allowed_ips"] = _normalize_ip_rules(updates.get("allowed_ips"), raise_on_invalid=True)
                if "image_quota" in updates and updates.get("image_quota") is not None:
                    next_quota = _quota_decimal(updates.get("image_quota"))
                    reservations = self._active_reservations(next_item)
                    next_item["image_quota"] = _quota_number(next_quota)
                    next_item["image_quota_reserved"] = _quota_number(min(self._reserved_quota(reservations), next_quota))
                    next_item["image_quota_reservations"] = reservations
                self._items[index] = next_item
                self._save()
                return self._public_item(next_item)
        return None

    def recharge_key(self, key_id: str, amount: object, *, role: AuthRole | None = "user") -> dict[str, object] | None:
        normalized_id = self._clean(key_id)
        recharge_amount = _quota_decimal(amount)
        if not normalized_id:
            return None
        if recharge_amount <= 0:
            raise ValueError("充值额度必须大于 0")
        with self._lock:
            self._reload_locked()
            for index, item in enumerate(self._items):
                if item.get("id") != normalized_id:
                    continue
                if role is not None and item.get("role") != role:
                    return None
                next_item = dict(item)
                next_item["image_quota"] = _quota_number(_quota_decimal(next_item.get("image_quota")) + recharge_amount)
                self._items[index] = next_item
                self._save()
                return self._public_item(next_item)
        return None

    def _image_quota_multiplier(self, mode: ImageQuotaMode) -> Decimal:
        raw_settings = config.get_image_quota_settings()
        key = "edit_multiplier" if mode == "edit" else "generation_multiplier"
        return _quota_decimal(raw_settings.get(key), Decimal("1"))

    def image_quota_cost(self, *, mode: ImageQuotaMode, count: object = 1) -> float:
        try:
            image_count = int(count or 1)
        except (TypeError, ValueError):
            image_count = 1
        image_count = max(1, image_count)
        return _quota_number(self._image_quota_multiplier(mode) * Decimal(image_count))

    def reserve_image_quota(
        self,
        identity: dict[str, object],
        *,
        mode: ImageQuotaMode,
        count: object = 1,
        reservation_id: str = "",
    ) -> dict[str, Any] | None:
        if identity.get("role") != "user":
            return None
        key_id = self._clean(identity.get("id"))
        if not key_id:
            raise ImageQuotaError("密钥无效或已失效，请重新登录")
        cost = _quota_decimal(self.image_quota_cost(mode=mode, count=count))
        provided_reservation_id = _reservation_id(reservation_id)
        normalized_reservation_id = provided_reservation_id or uuid.uuid4().hex
        with self._lock:
            self._reload_locked()
            for index, item in enumerate(self._items):
                if item.get("id") != key_id or item.get("role") != "user":
                    continue
                if not bool(item.get("enabled", True)):
                    raise ImageQuotaError("密钥无效或已失效，请重新登录")
                reservations = self._active_reservations(item)
                existing = reservations.get(normalized_reservation_id)
                if existing is not None:
                    return dict(existing, key_id=key_id)
                balance = _quota_decimal(item.get("image_quota"))
                reserved = min(self._reserved_quota(reservations), balance)
                available = max(Decimal("0"), balance - reserved)
                if available < cost:
                    raise ImageQuotaError(
                        "图片额度不足，请联系管理员充值",
                        required=_quota_number(cost),
                        available=_quota_number(available),
                    )
                next_item = dict(item)
                reservation = {
                    "id": normalized_reservation_id,
                    "key_id": key_id,
                    "mode": mode,
                    "cost": _quota_number(cost),
                    "created_at": _now_iso(),
                }
                if not provided_reservation_id:
                    reservation["expires_at"] = _sync_reservation_expires_at()
                reservations[normalized_reservation_id] = {key: value for key, value in reservation.items() if key != "key_id"}
                next_item["image_quota_reservations"] = reservations
                next_item["image_quota_reserved"] = _quota_number(self._reserved_quota(reservations))
                self._items[index] = next_item
                self._save()
                return reservation
        raise ImageQuotaError("密钥无效或已失效，请重新登录")

    def confirm_image_quota(
        self,
        reservation: dict[str, Any] | None,
        *,
        strict: bool = False,
    ) -> dict[str, object] | None:
        return self._finish_image_quota_reservation(reservation, charge=True, strict=strict)

    def refund_image_quota(
        self,
        reservation: dict[str, Any] | None,
        *,
        strict: bool = False,
    ) -> dict[str, object] | None:
        return self._finish_image_quota_reservation(reservation, charge=False, strict=strict)

    def _finish_image_quota_reservation(
        self,
        reservation: dict[str, Any] | None,
        *,
        charge: bool,
        strict: bool,
    ) -> dict[str, object] | None:
        if not reservation:
            return None
        key_id = self._clean(reservation.get("key_id"))
        normalized_reservation_id = _reservation_id(reservation.get("id"))
        if not key_id or not normalized_reservation_id:
            return None
        with self._lock:
            self._reload_locked()
            for index, item in enumerate(self._items):
                if item.get("id") != key_id:
                    continue
                reservations = self._active_reservations(item)
                active = reservations.pop(normalized_reservation_id, None)
                if active is None:
                    if strict:
                        raise ImageQuotaError("图片额度预占记录不存在或已完成")
                    return self._public_item(item)
                cost = _quota_decimal(active.get("cost"))
                if cost <= 0:
                    next_item = dict(item)
                    next_balance = _quota_decimal(item.get("image_quota"))
                    next_item["image_quota_reserved"] = _quota_number(min(self._reserved_quota(reservations), next_balance))
                    next_item["image_quota_reservations"] = reservations
                    self._items[index] = next_item
                    self._save()
                    return self._public_item(next_item)
                balance = _quota_decimal(item.get("image_quota"))
                next_item = dict(item)
                next_balance = max(Decimal("0"), balance - cost) if charge else balance
                next_reserved = self._reserved_quota(reservations)
                next_item["image_quota"] = _quota_number(next_balance)
                next_item["image_quota_reserved"] = _quota_number(min(next_reserved, next_balance))
                next_item["image_quota_reservations"] = reservations
                self._items[index] = next_item
                self._save()
                return self._public_item(next_item)
        return None

    def delete_key(self, key_id: str, *, role: AuthRole | None = None) -> bool:
        normalized_id = self._clean(key_id)
        if not normalized_id:
            return False
        with self._lock:
            self._reload_locked()
            before = len(self._items)
            self._items = [
                item
                for item in self._items
                if not (item.get("id") == normalized_id and (role is None or item.get("role") == role))
            ]
            if len(self._items) == before:
                return False
            self._save()
            return True

    def authenticate(self, raw_key: str, client_ip: str | None = None) -> dict[str, object] | None:
        candidate = self._clean(raw_key)
        if not candidate:
            return None
        candidate_hash = _hash_key(candidate)
        with self._lock:
            for index, item in enumerate(self._items):
                if not bool(item.get("enabled", True)):
                    continue
                stored_hash = self._clean(item.get("key_hash"))
                if not stored_hash or not hmac.compare_digest(stored_hash, candidate_hash):
                    continue
                if not _ip_matches_rules(client_ip, item.get("allowed_ips")):
                    return None
                next_item = dict(item)
                now = datetime.now(timezone.utc)
                next_item["last_used_at"] = now.isoformat()
                self._items[index] = next_item
                item_id = self._clean(next_item.get("id"))
                last_flush_at = self._last_used_flush_at.get(item_id)
                if last_flush_at is None or (now - last_flush_at).total_seconds() >= 60:
                    try:
                        self._save()
                        self._last_used_flush_at[item_id] = now
                    except Exception:
                        pass
                return self._public_item(next_item)
        return None


auth_service = AuthService(config.get_storage_backend())
