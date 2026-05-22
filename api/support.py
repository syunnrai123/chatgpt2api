from __future__ import annotations

from contextvars import ContextVar, Token
import ipaddress
from pathlib import Path
from threading import Event, Thread

from fastapi import HTTPException, Request

from services.account_service import account_service
from services.auth_service import auth_service
from services.config import config

BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIST_DIR = BASE_DIR / "web_dist"
_CURRENT_REQUEST: ContextVar[Request | None] = ContextVar("chatgpt2api_current_request", default=None)


def extract_bearer_token(authorization: str | None) -> str:
    scheme, _, value = str(authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return ""
    return value.strip()


def _legacy_admin_identity(token: str) -> dict[str, object] | None:
    auth_key = str(config.auth_key or "").strip()
    if auth_key and token == auth_key:
        return {"id": "admin", "name": "管理员", "role": "admin"}
    return None


def bind_request_context(request: Request) -> Token[Request | None]:
    return _CURRENT_REQUEST.set(request)


def reset_request_context(token: Token[Request | None]) -> None:
    _CURRENT_REQUEST.reset(token)


def _clean_client_ip(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("[") and "]" in text:
        return text[1:text.index("]")].strip()
    if text.count(":") == 1:
        host, _, port = text.partition(":")
        if host and port.isdigit():
            return host.strip()
    return text


def _valid_ip(value: object) -> str:
    text = _clean_client_ip(value)
    if not text:
        return ""
    try:
        return str(ipaddress.ip_address(text))
    except ValueError:
        return ""


def _ip_matches_rules(client_ip: object, rules: object) -> bool:
    if not isinstance(rules, list) or not rules:
        return False
    try:
        ip = ipaddress.ip_address(_clean_client_ip(client_ip))
    except ValueError:
        return False
    for rule in rules:
        try:
            raw_rule = str(rule or "").strip()
            if "/" in raw_rule:
                if ip in ipaddress.ip_network(raw_rule, strict=False):
                    return True
            elif ip == ipaddress.ip_address(raw_rule):
                return True
        except ValueError:
            continue
    return False


def _forwarded_chain_client_ip(forwarded_ips: list[str], peer_ip: str, trusted_proxy_ips: list[str]) -> str:
    chain = [*forwarded_ips, peer_ip]
    for candidate in reversed(chain):
        if not _ip_matches_rules(candidate, trusted_proxy_ips):
            return candidate
    return ""


def request_client_ip(request: Request | None = None) -> str:
    active_request = request or _CURRENT_REQUEST.get()
    if active_request is None:
        return ""
    peer_ip = _valid_ip(active_request.client.host if active_request.client else "")
    trusted_proxy_ips = config.trusted_proxy_ips
    if not config.trust_proxy_headers or not trusted_proxy_ips or not _ip_matches_rules(peer_ip, trusted_proxy_ips):
        return peer_ip
    forwarded_ips = [_valid_ip(candidate) for candidate in str(active_request.headers.get("x-forwarded-for", "") or "").split(",")]
    forwarded_ips = [client_ip for client_ip in forwarded_ips if client_ip]
    client_ip = _forwarded_chain_client_ip(forwarded_ips, peer_ip, trusted_proxy_ips)
    if client_ip:
        return client_ip
    real_ip = _valid_ip(active_request.headers.get("x-real-ip", ""))
    if real_ip and not _ip_matches_rules(real_ip, trusted_proxy_ips):
        return real_ip
    return peer_ip


def require_identity(authorization: str | None) -> dict[str, object]:
    token = extract_bearer_token(authorization)
    identity = _legacy_admin_identity(token) or auth_service.authenticate(token, client_ip=request_client_ip())
    if identity is None:
        raise HTTPException(status_code=401, detail={"error": "密钥无效或已失效，请重新登录"})
    return identity


def require_auth_key(authorization: str | None) -> None:
    require_identity(authorization)


def require_admin(authorization: str | None) -> dict[str, object]:
    identity = require_identity(authorization)
    if identity.get("role") != "admin":
        raise HTTPException(status_code=403, detail={"error": "需要管理员权限才能执行这个操作"})
    return identity


def resolve_image_base_url(request: Request) -> str:
    return config.base_url or f"{request.url.scheme}://{request.headers.get('host', request.url.netloc)}"


def raise_image_quota_error(exc: Exception) -> None:
    message = str(exc)
    if "no available image quota" in message.lower():
        raise HTTPException(status_code=429, detail={"error": "no available image quota"}) from exc
    raise HTTPException(status_code=502, detail={"error": message}) from exc


def sanitize_cpa_pool(pool: dict | None) -> dict | None:
    if not isinstance(pool, dict):
        return None
    return {key: value for key, value in pool.items() if key != "secret_key"}


def sanitize_cpa_pools(pools: list[dict]) -> list[dict]:
    return [sanitized for pool in pools if (sanitized := sanitize_cpa_pool(pool)) is not None]


def sanitize_sub2api_server(server: dict | None) -> dict | None:
    if not isinstance(server, dict):
        return None
    sanitized = {key: value for key, value in server.items() if key not in {"password", "api_key"}}
    sanitized["has_api_key"] = bool(str(server.get("api_key") or "").strip())
    return sanitized


def sanitize_sub2api_servers(servers: list[dict]) -> list[dict]:
    return [sanitized for server in servers if (sanitized := sanitize_sub2api_server(server)) is not None]


def start_limited_account_watcher(stop_event: Event) -> Thread:
    interval_seconds = config.refresh_account_interval_minute * 60

    def worker() -> None:
        while not stop_event.is_set():
            try:
                limited_tokens = account_service.list_limited_tokens()
                if limited_tokens:
                    print(f"[account-limited-watcher] checking {len(limited_tokens)} limited accounts")
                    account_service.refresh_accounts(limited_tokens)
            except Exception as exc:
                print(f"[account-limited-watcher] fail {exc}")
            stop_event.wait(interval_seconds)

    thread = Thread(target=worker, name="limited-account-watcher", daemon=True)
    thread.start()
    return thread


def resolve_web_asset(requested_path: str) -> Path | None:
    if not WEB_DIST_DIR.exists():
        return None
    clean_path = requested_path.strip("/")
    base_dir = WEB_DIST_DIR.resolve()
    candidates = [base_dir / "index.html"] if not clean_path else [
        base_dir / Path(clean_path),
        base_dir / clean_path / "index.html",
        base_dir / f"{clean_path}.html",
    ]
    for candidate in candidates:
        try:
            candidate.resolve().relative_to(base_dir)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    return None
