from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import json
import os
import re
import sys
from pathlib import Path
import time
from urllib.parse import urlparse

from services.storage.base import StorageBackend

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CONFIG_FILE = BASE_DIR / "config.json"
VERSION_FILE = BASE_DIR / "VERSION"
BACKUP_STATE_FILE = DATA_DIR / "backup_state.json"

DEFAULT_BACKUP_INCLUDE = {
    "config": True,
    "register": True,
    "cpa": True,
    "sub2api": True,
    "logs": True,
    "image_tasks": True,
    "accounts_snapshot": True,
    "auth_keys_snapshot": True,
    "images": False,
}

DEFAULT_IMAGE_STORAGE = {
    "enabled": False,
    "mode": "local",
    "webdav_url": "",
    "webdav_username": "",
    "webdav_password": "",
    "webdav_root_path": "chatgpt2api/images",
    "public_base_url": "",
}
DEFAULT_IMAGE_SAVE_ENABLED = False
DEFAULT_IMAGE_RETENTION_MINUTES = 30 * 24 * 60
DEFAULT_TRUST_PROXY_HEADERS = False
DEFAULT_TRUSTED_PROXY_IPS: list[str] = []
DEFAULT_MAX_VOLATILE_IMAGE_RESULTS = 64
DEFAULT_MAX_VOLATILE_IMAGE_BYTES = 256 * 1024 * 1024

DEFAULT_IMAGE_QUOTA = {
    "generation_multiplier": 1.0,
    "edit_multiplier": 1.0,
}
MIN_IMAGE_QUOTA_MULTIPLIER = 0.000001

DEFAULT_APP_TEXT = {
    "brand_name": "chatgpt2api",
    "github_label": "GitHub",
    "github_url": "https://github.com/basketikun/chatgpt2api",
    "register_eyebrow": "Register",
    "register_title": "ChatGPT注册机",
}


def _normalize_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if value is None:
        return default
    return bool(value)


def _normalize_positive_int(value: object, default: int, minimum: int = 0) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        normalized = default
    return max(minimum, normalized)


def _normalize_positive_float(value: object, default: float, minimum: float = 0.0) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        normalized = default
    return max(minimum, normalized)


def _split_string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[\s,]+", value) if item.strip()]
    if isinstance(value, list):
        return [str(item or "").strip() for item in value if str(item or "").strip()]
    return []


def _normalize_ip_rules(value: object, *, raise_on_invalid: bool = False) -> list[str]:
    rules: list[str] = []
    seen: set[str] = set()
    for raw in _split_string_list(value):
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


def _validate_proxy_header_settings(data: dict[str, object]) -> None:
    if not _normalize_bool(data.get("trust_proxy_headers"), DEFAULT_TRUST_PROXY_HEADERS):
        return
    trusted_proxy_ips = _normalize_ip_rules(
        data.get("trusted_proxy_ips", DEFAULT_TRUSTED_PROXY_IPS),
        raise_on_invalid=True,
    )
    if not trusted_proxy_ips:
        raise ValueError("开启反向代理 IP 头信任时必须填写可信反代 IP 或 CIDR")


def _normalize_backup_include(value: object) -> dict[str, bool]:
    source = value if isinstance(value, dict) else {}
    normalized = dict(DEFAULT_BACKUP_INCLUDE)
    for key in normalized:
        normalized[key] = _normalize_bool(source.get(key), normalized[key])
    return normalized


def _normalize_backup_settings(value: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else {}
    return {
        "enabled": _normalize_bool(source.get("enabled"), False),
        "provider": "cloudflare_r2",
        "account_id": str(source.get("account_id") or "").strip(),
        "access_key_id": str(source.get("access_key_id") or "").strip(),
        "secret_access_key": str(source.get("secret_access_key") or "").strip(),
        "bucket": str(source.get("bucket") or "").strip(),
        "prefix": str(source.get("prefix") or "backups").strip().strip("/") or "backups",
        "interval_minutes": _normalize_positive_int(source.get("interval_minutes"), 360, 1),
        "rotation_keep": _normalize_positive_int(source.get("rotation_keep"), 10, 0),
        "encrypt": _normalize_bool(source.get("encrypt"), False),
        "passphrase": str(source.get("passphrase") or "").strip(),
        "include": _normalize_backup_include(source.get("include")),
    }


def _normalize_backup_state(value: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else {}
    return {
        "last_started_at": str(source.get("last_started_at") or "").strip() or None,
        "last_finished_at": str(source.get("last_finished_at") or "").strip() or None,
        "last_status": str(source.get("last_status") or "idle").strip() or "idle",
        "last_error": str(source.get("last_error") or "").strip() or None,
        "last_object_key": str(source.get("last_object_key") or "").strip() or None,
    }


def _normalize_image_storage_settings(value: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else {}
    mode = str(source.get("mode") or "local").strip().lower()
    if mode not in {"local", "webdav", "both"}:
        mode = "local"
    enabled = _normalize_bool(source.get("enabled"), False)
    if not enabled:
        mode = "local"
    root_path = str(source.get("webdav_root_path") or DEFAULT_IMAGE_STORAGE["webdav_root_path"]).strip().strip("/")
    return {
        "enabled": enabled,
        "mode": mode,
        "webdav_url": str(source.get("webdav_url") or "").strip().rstrip("/"),
        "webdav_username": str(source.get("webdav_username") or "").strip(),
        "webdav_password": str(source.get("webdav_password") or "").strip(),
        "webdav_root_path": root_path or str(DEFAULT_IMAGE_STORAGE["webdav_root_path"]),
        "public_base_url": str(source.get("public_base_url") or "").strip().rstrip("/"),
    }


def _normalize_image_quota_settings(value: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else {}
    return {
        "generation_multiplier": _normalize_positive_float(
            source.get("generation_multiplier"),
            float(DEFAULT_IMAGE_QUOTA["generation_multiplier"]),
            MIN_IMAGE_QUOTA_MULTIPLIER,
        ),
        "edit_multiplier": _normalize_positive_float(
            source.get("edit_multiplier"),
            float(DEFAULT_IMAGE_QUOTA["edit_multiplier"]),
            MIN_IMAGE_QUOTA_MULTIPLIER,
        ),
    }


def _normalize_app_text_settings(value: object) -> dict[str, object]:
    source = value if isinstance(value, dict) else {}
    normalized: dict[str, object] = {}
    for key, default_value in DEFAULT_APP_TEXT.items():
        raw_value = source.get(key, default_value)
        text = str(raw_value or "").strip()
        normalized[key] = text or default_value
    parsed_github_url = urlparse(str(normalized["github_url"]))
    if parsed_github_url.scheme not in {"http", "https"} or not parsed_github_url.netloc:
        normalized["github_url"] = DEFAULT_APP_TEXT["github_url"]
    return normalized


def _validate_image_storage_settings(settings: dict[str, object]) -> None:
    if not _normalize_bool(settings.get("enabled"), False):
        return
    if not str(settings.get("webdav_url") or "").strip():
        raise ValueError("启用 WebDAV 图片存储后必须填写 WebDAV URL")
    if not str(settings.get("webdav_password") or "").strip():
        raise ValueError("启用 WebDAV 图片存储后必须填写 WebDAV 密码")


@dataclass(frozen=True)
class LoadedSettings:
    auth_key: str
    refresh_account_interval_minute: int


def _normalize_auth_key(value: object) -> str:
    return str(value or "").strip()


def _is_invalid_auth_key(value: object) -> bool:
    return _normalize_auth_key(value) == ""


def _read_json_object(path: Path, *, name: str) -> dict[str, object]:
    if not path.exists():
        return {}
    if path.is_dir():
        print(
            f"Warning: {name} at '{path}' is a directory, ignoring it and falling back to other configuration sources.",
            file=sys.stderr,
        )
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _load_settings() -> LoadedSettings:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_config = _read_json_object(CONFIG_FILE, name="config.json")
    auth_key = _normalize_auth_key(os.getenv("CHATGPT2API_AUTH_KEY") or raw_config.get("auth-key"))
    if _is_invalid_auth_key(auth_key):
        raise ValueError(
            "❌ auth-key 未设置！\n"
            "请在环境变量 CHATGPT2API_AUTH_KEY 中设置，或者在 config.json 中填写 auth-key。"
        )

    try:
        refresh_interval = int(raw_config.get("refresh_account_interval_minute", 5))
    except (TypeError, ValueError):
        refresh_interval = 5

    return LoadedSettings(
        auth_key=auth_key,
        refresh_account_interval_minute=refresh_interval,
    )


class ConfigStore:
    def __init__(self, path: Path):
        self.path = path
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.data = self._load()
        self._storage_backend: StorageBackend | None = None
        if _is_invalid_auth_key(self.auth_key):
            raise ValueError(
                "❌ auth-key 未设置！\n"
                "请按以下任意一种方式解决：\n"
                "1. 在 Render 的 Environment 变量中添加：\n"
                "   CHATGPT2API_AUTH_KEY = your_real_auth_key\n"
                "2. 或者在 config.json 中填写：\n"
                '   "auth-key": "your_real_auth_key"'
            )

    def _load(self) -> dict[str, object]:
        return _read_json_object(self.path, name="config.json")

    def _save(self) -> None:
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @property
    def auth_key(self) -> str:
        return _normalize_auth_key(os.getenv("CHATGPT2API_AUTH_KEY") or self.data.get("auth-key"))

    @property
    def accounts_file(self) -> Path:
        return DATA_DIR / "accounts.json"

    @property
    def refresh_account_interval_minute(self) -> int:
        try:
            return int(self.data.get("refresh_account_interval_minute", 5))
        except (TypeError, ValueError):
            return 5

    @property
    def image_retention_days(self) -> int:
        return max(1, int((self.image_retention_minutes + 1439) // 1440))

    @property
    def image_retention_minutes(self) -> int:
        try:
            if self.data.get("image_retention_minutes") is not None:
                return max(1, int(self.data.get("image_retention_minutes", DEFAULT_IMAGE_RETENTION_MINUTES)))
            return max(1, int(self.data.get("image_retention_days", 30)) * 1440)
        except (TypeError, ValueError):
            return DEFAULT_IMAGE_RETENTION_MINUTES

    @property
    def image_save_enabled(self) -> bool:
        return _normalize_bool(self.data.get("image_save_enabled"), DEFAULT_IMAGE_SAVE_ENABLED)

    @property
    def trust_proxy_headers(self) -> bool:
        return _normalize_bool(self.data.get("trust_proxy_headers"), DEFAULT_TRUST_PROXY_HEADERS)

    @property
    def trusted_proxy_ips(self) -> list[str]:
        try:
            return _normalize_ip_rules(
                self.data.get("trusted_proxy_ips", DEFAULT_TRUSTED_PROXY_IPS),
                raise_on_invalid=True,
            )
        except ValueError:
            return []

    @property
    def max_volatile_image_results(self) -> int:
        return _normalize_positive_int(
            self.data.get("max_volatile_image_results"),
            DEFAULT_MAX_VOLATILE_IMAGE_RESULTS,
            1,
        )

    @property
    def max_volatile_image_bytes(self) -> int:
        return _normalize_positive_int(
            self.data.get("max_volatile_image_bytes"),
            DEFAULT_MAX_VOLATILE_IMAGE_BYTES,
            1,
        )

    @property
    def image_poll_timeout_secs(self) -> int:
        try:
            return max(1, int(self.data.get("image_poll_timeout_secs", 120)))
        except (TypeError, ValueError):
            return 120

    @property
    def image_poll_interval_secs(self) -> float:
        try:
            return max(0.5, float(self.data.get("image_poll_interval_secs", 10.0)))
        except (TypeError, ValueError):
            return 10.0

    @property
    def image_poll_initial_wait_secs(self) -> float:
        """Image generation upstream takes ~30s; polling immediately wastes requests
        and trips a transient 429. Default 10s gives the conversation document time
        to commit before the first poll."""
        try:
            return max(0.0, float(self.data.get("image_poll_initial_wait_secs", 10.0)))
        except (TypeError, ValueError):
            return 10.0

    @property
    def image_account_concurrency(self) -> int:
        try:
            return min(8, max(1, int(self.data.get("image_account_concurrency", 3))))
        except (TypeError, ValueError):
            return 3

    @property
    def auto_remove_invalid_accounts(self) -> bool:
        value = self.data.get("auto_remove_invalid_accounts", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @property
    def auto_remove_rate_limited_accounts(self) -> bool:
        value = self.data.get("auto_remove_rate_limited_accounts", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @property
    def log_levels(self) -> list[str]:
        levels = self.data.get("log_levels")
        if not isinstance(levels, list):
            return []
        allowed = {"debug", "info", "warning", "error"}
        return [level for item in levels if (level := str(item or "").strip().lower()) in allowed]

    @property
    def sensitive_words(self) -> list[str]:
        words = self.data.get("sensitive_words")
        return [word for item in words if (word := str(item or "").strip())] if isinstance(words, list) else []

    @property
    def ai_review(self) -> dict[str, object]:
        value = self.data.get("ai_review")
        return value if isinstance(value, dict) else {}

    @property
    def global_system_prompt(self) -> str:
        return str(self.data.get("global_system_prompt") or "").strip()

    @property
    def images_dir(self) -> Path:
        path = DATA_DIR / "images"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def image_thumbnails_dir(self) -> Path:
        path = DATA_DIR / "image_thumbnails"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cleanup_old_images(self) -> int:
        cutoff = time.time() - self.image_retention_minutes * 60
        removed = 0
        for path in self.images_dir.rglob("*"):
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        for path in sorted((p for p in self.images_dir.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            try:
                path.rmdir()
            except OSError:
                pass
        return removed

    @property
    def base_url(self) -> str:
        return str(
            os.getenv("CHATGPT2API_BASE_URL")
            or self.data.get("base_url")
            or ""
        ).strip().rstrip("/")

    @property
    def app_version(self) -> str:
        try:
            value = VERSION_FILE.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return "0.0.0"
        return value or "0.0.0"

    def get(self) -> dict[str, object]:
        data = dict(self.data)
        data["refresh_account_interval_minute"] = self.refresh_account_interval_minute
        data["image_save_enabled"] = self.image_save_enabled
        data["image_retention_minutes"] = self.image_retention_minutes
        data["image_retention_days"] = self.image_retention_days
        data["trust_proxy_headers"] = self.trust_proxy_headers
        data["trusted_proxy_ips"] = self.trusted_proxy_ips
        data["max_volatile_image_results"] = self.max_volatile_image_results
        data["max_volatile_image_bytes"] = self.max_volatile_image_bytes
        data["image_poll_timeout_secs"] = self.image_poll_timeout_secs
        data["image_poll_interval_secs"] = self.image_poll_interval_secs
        data["image_poll_initial_wait_secs"] = self.image_poll_initial_wait_secs
        data["image_account_concurrency"] = self.image_account_concurrency
        data["auto_remove_invalid_accounts"] = self.auto_remove_invalid_accounts
        data["auto_remove_rate_limited_accounts"] = self.auto_remove_rate_limited_accounts
        data["log_levels"] = self.log_levels
        data["sensitive_words"] = self.sensitive_words
        data["ai_review"] = self.ai_review
        data["global_system_prompt"] = self.global_system_prompt
        data["backup"] = self.get_backup_settings()
        data["image_storage"] = self.get_image_storage_settings()
        data["image_quota"] = self.get_image_quota_settings()
        data["app_text"] = self.get_app_text_settings()
        data.pop("auth-key", None)
        return data

    def get_proxy_settings(self) -> str:
        return str(self.data.get("proxy") or "").strip()

    def update(self, data: dict[str, object]) -> dict[str, object]:
        incoming = dict(data or {})
        next_data = dict(self.data)
        next_data.update(incoming)
        if "image_save_enabled" in next_data:
            next_data["image_save_enabled"] = _normalize_bool(next_data.get("image_save_enabled"), DEFAULT_IMAGE_SAVE_ENABLED)
        if "image_retention_minutes" in incoming or "image_retention_minutes" in next_data:
            next_data["image_retention_minutes"] = _normalize_positive_int(
                next_data.get("image_retention_minutes"),
                DEFAULT_IMAGE_RETENTION_MINUTES,
                1,
            )
        if "image_retention_days" in incoming and "image_retention_minutes" not in incoming:
            next_data["image_retention_minutes"] = _normalize_positive_int(next_data.get("image_retention_days"), 30, 1) * 1440
        elif "image_retention_minutes" not in next_data and "image_retention_days" in next_data:
            next_data["image_retention_minutes"] = _normalize_positive_int(next_data.get("image_retention_days"), 30, 1) * 1440
        proxy_settings_touched = "trust_proxy_headers" in incoming or "trusted_proxy_ips" in incoming
        if "trust_proxy_headers" in incoming:
            next_data["trust_proxy_headers"] = _normalize_bool(
                next_data.get("trust_proxy_headers"),
                DEFAULT_TRUST_PROXY_HEADERS,
            )
        if "trusted_proxy_ips" in incoming:
            next_data["trusted_proxy_ips"] = _normalize_ip_rules(next_data.get("trusted_proxy_ips"), raise_on_invalid=True)
        if proxy_settings_touched:
            _validate_proxy_header_settings(next_data)
        if "max_volatile_image_results" in next_data:
            next_data["max_volatile_image_results"] = _normalize_positive_int(
                next_data.get("max_volatile_image_results"),
                DEFAULT_MAX_VOLATILE_IMAGE_RESULTS,
                1,
            )
        if "max_volatile_image_bytes" in next_data:
            next_data["max_volatile_image_bytes"] = _normalize_positive_int(
                next_data.get("max_volatile_image_bytes"),
                DEFAULT_MAX_VOLATILE_IMAGE_BYTES,
                1,
            )
        if "backup" in next_data:
            next_data["backup"] = _normalize_backup_settings(next_data.get("backup"))
        if "image_storage" in next_data:
            next_data["image_storage"] = _normalize_image_storage_settings(next_data.get("image_storage"))
            _validate_image_storage_settings(next_data["image_storage"])
        if "image_quota" in next_data:
            next_data["image_quota"] = _normalize_image_quota_settings(next_data.get("image_quota"))
        if "app_text" in next_data:
            next_data["app_text"] = _normalize_app_text_settings(next_data.get("app_text"))
        next_data.pop("backup_state", None)
        self.data = next_data
        self._save()
        return self.get()

    def get_backup_settings(self) -> dict[str, object]:
        return _normalize_backup_settings(self.data.get("backup"))

    def get_image_storage_settings(self) -> dict[str, object]:
        return _normalize_image_storage_settings(self.data.get("image_storage"))

    def get_image_quota_settings(self) -> dict[str, object]:
        return _normalize_image_quota_settings(self.data.get("image_quota"))

    def get_app_text_settings(self) -> dict[str, object]:
        return _normalize_app_text_settings(self.data.get("app_text"))

    def get_storage_backend(self) -> StorageBackend:
        """获取存储后端实例（单例）"""
        if self._storage_backend is None:
            from services.storage.factory import create_storage_backend
            self._storage_backend = create_storage_backend(DATA_DIR)
        return self._storage_backend


def load_backup_state() -> dict[str, object]:
    return _normalize_backup_state(_read_json_object(BACKUP_STATE_FILE, name="backup_state.json"))


def save_backup_state(state: dict[str, object]) -> dict[str, object]:
    normalized = _normalize_backup_state(state)
    BACKUP_STATE_FILE.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return normalized


config = ConfigStore(CONFIG_FILE)
