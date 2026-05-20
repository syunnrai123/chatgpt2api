from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from services.config import DATA_DIR, config
from services.content_filter import request_text
from services.auth_service import auth_service
from services.log_service import LOG_TYPE_CALL, log_service
from services.protocol import openai_v1_image_edit, openai_v1_image_generations
from utils.log import logger

TASK_STATUS_QUEUED = "queued"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_SUCCESS = "success"
TASK_STATUS_ERROR = "error"
TERMINAL_STATUSES = {TASK_STATUS_SUCCESS, TASK_STATUS_ERROR}
UNFINISHED_STATUSES = {TASK_STATUS_QUEUED, TASK_STATUS_RUNNING}
DEFAULT_TASK_WORKERS = 6
DEFAULT_TASK_QUEUE_SIZE = 64

ImageTaskWorkItem = tuple[str, str, str, dict[str, Any], dict[str, object], str]


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _timestamp(value: object) -> float:
    if not isinstance(value, str) or not value.strip():
        return 0.0
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value[:26], fmt).timestamp()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _clean(value: object, default: str = "") -> str:
    return str(value or default).strip()


def _owner_id(identity: dict[str, object]) -> str:
    return _clean(identity.get("id")) or "anonymous"


def _task_key(owner_id: str, task_id: str) -> str:
    return f"{owner_id}:{task_id}"


def _collect_image_urls(data: list[Any]) -> list[str]:
    urls: list[str] = []
    for item in data:
        if isinstance(item, dict):
            url = item.get("url")
            if isinstance(url, str) and url:
                urls.append(url)
    return urls


def _task_worker_count(value: object = None) -> int:
    if value is not None:
        try:
            return min(16, max(1, int(value)))
        except (TypeError, ValueError):
            return DEFAULT_TASK_WORKERS
    try:
        configured = int(config.image_account_concurrency)
    except (TypeError, ValueError):
        configured = DEFAULT_TASK_WORKERS // 2
    return min(16, max(2, configured * 2))


def _task_queue_size(value: object = None, worker_count: int = DEFAULT_TASK_WORKERS) -> int:
    if value is not None:
        try:
            return min(512, max(1, int(value)))
        except (TypeError, ValueError):
            return DEFAULT_TASK_QUEUE_SIZE
    return min(512, max(DEFAULT_TASK_QUEUE_SIZE, worker_count * 8))


def _public_quota_status(status: str, *, pending: str = "charge_pending") -> str:
    return pending if status == "pending" else status


def _refund_task_reservation(reservation: dict[str, Any] | None) -> bool:
    if not reservation:
        return True
    try:
        auth_service.refund_image_quota(reservation)
        return True
    except Exception as exc:
        logger.error({"event": "image_task_quota_refund_failed", "reservation": reservation, "error": str(exc)})
        return False


def _finalize_task_reservation(reservation: dict[str, Any] | None, *, charge: bool) -> str:
    if not reservation:
        return "not_required"
    if not charge:
        return "refunded" if _refund_task_reservation(reservation) else "pending"
    try:
        auth_service.confirm_image_quota(reservation)
        return "charged"
    except Exception as exc:
        logger.error({"event": "image_task_quota_confirm_failed", "reservation": reservation, "error": str(exc)})
        return "refunded" if _refund_task_reservation(reservation) else "pending"


def _public_task(task: dict[str, Any]) -> dict[str, Any]:
    item = {
        "id": task.get("id"),
        "status": task.get("status"),
        "mode": task.get("mode"),
        "model": task.get("model"),
        "size": task.get("size"),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
    }
    if task.get("data") is not None:
        item["data"] = task.get("data")
    if task.get("error"):
        item["error"] = task.get("error")
    if task.get("quota_cost") is not None:
        item["quota_cost"] = task.get("quota_cost")
    if task.get("quota_status"):
        item["quota_status"] = task.get("quota_status")
    return item


class ImageTaskService:
    def __init__(
        self,
        path: Path,
        *,
        generation_handler: Callable[[dict[str, Any]], dict[str, Any]] = openai_v1_image_generations.handle,
        edit_handler: Callable[[dict[str, Any]], dict[str, Any]] = openai_v1_image_edit.handle,
        retention_days_getter: Callable[[], int] | None = None,
        task_workers: int | None = None,
        task_queue_size: int | None = None,
    ):
        self.path = path
        self.generation_handler = generation_handler
        self.edit_handler = edit_handler
        self.retention_days_getter = retention_days_getter or (lambda: config.image_retention_days)
        self._lock = threading.RLock()
        self._task_workers = _task_worker_count(task_workers)
        self._task_queue: queue.Queue[ImageTaskWorkItem] = queue.Queue(
            maxsize=_task_queue_size(task_queue_size, self._task_workers)
        )
        self._workers_started = False
        self._worker_lock = threading.Lock()
        self._tasks: dict[str, dict[str, Any]] = {}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._tasks = self._load_locked()
            changed = self._recover_unfinished_locked()
            changed = self._cleanup_locked() or changed
            if changed:
                self._save_locked()

    def submit_generation(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        prompt: str,
        model: str,
        size: str | None,
        base_url: str,
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "model": model,
            "n": 1,
            "size": size,
            "response_format": "url",
            "base_url": base_url,
        }
        return self._submit(identity, client_task_id=client_task_id, mode="generate", payload=payload)

    def submit_edit(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        prompt: str,
        model: str,
        size: str | None,
        base_url: str,
        images: list[tuple[bytes, str, str]],
        count: object = 1,
        quota_reservation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "images": images,
            "model": model,
            "n": count,
            "size": size,
            "response_format": "url",
            "base_url": base_url,
        }
        return self._submit(identity, client_task_id=client_task_id, mode="edit", payload=payload, quota_reservation=quota_reservation)

    def prepare_edit(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        prompt: str,
        model: str,
        size: str | None,
        count: object = 1,
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "model": model,
            "n": count,
            "size": size,
        }
        return self._submit(identity, client_task_id=client_task_id, mode="edit", payload=payload, start=False, mark_created=True)

    def start_prepared_edit(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        prompt: str,
        model: str,
        size: str | None,
        base_url: str,
        images: list[tuple[bytes, str, str]],
        count: object = 1,
    ) -> dict[str, Any]:
        task_id = _clean(client_task_id)
        if not task_id:
            raise ValueError("client_task_id is required")
        owner = _owner_id(identity)
        key = _task_key(owner, task_id)
        payload = {
            "prompt": prompt,
            "images": images,
            "model": model,
            "n": count,
            "size": size,
            "response_format": "url",
            "base_url": base_url,
        }
        with self._lock:
            task = self._tasks.get(key)
            if task is None:
                raise ValueError("image task was not prepared")
            if task.get("mode") != "edit":
                raise ValueError("image task mode mismatch")
            if task.get("status") != TASK_STATUS_QUEUED:
                return _public_task(task)
            task["error"] = ""
            task["updated_at"] = _now_iso()
            try:
                self._save_locked()
            except Exception:
                reservation = task.get("quota_reservation")
                quota_status = _finalize_task_reservation(
                    reservation if isinstance(reservation, dict) else None,
                    charge=False,
                )
                if quota_status != "pending":
                    task.pop("quota_reservation", None)
                task["status"] = TASK_STATUS_ERROR
                task["error"] = "图片任务启动失败"
                task["data"] = []
                task["quota_status"] = (
                    _public_quota_status(quota_status, pending="refund_pending")
                    if reservation
                    else task.get("quota_status", "not_required")
                )
                task["updated_at"] = _now_iso()
                try:
                    self._save_locked()
                except Exception:
                    pass
                raise
            public = _public_task(task)

        try:
            self._start_task_thread(key, task_id, "edit", payload, identity, model)
        except Exception:
            self.fail_task(identity, task_id, "图片任务启动失败")
            raise
        return public

    def list_tasks(self, identity: dict[str, object], task_ids: list[str]) -> dict[str, Any]:
        owner = _owner_id(identity)
        requested_ids = [_clean(task_id) for task_id in task_ids if _clean(task_id)]
        with self._lock:
            if self._cleanup_locked():
                self._save_locked()
            items = []
            missing_ids = []
            for task_id in requested_ids:
                task = self._tasks.get(_task_key(owner, task_id))
                if task is None:
                    missing_ids.append(task_id)
                else:
                    items.append(_public_task(task))
            if not requested_ids:
                items = [
                    _public_task(task)
                    for task in self._tasks.values()
                    if task.get("owner_id") == owner
                ]
                items.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
                missing_ids = []
            return {"items": items, "missing_ids": missing_ids}

    def get_task(self, identity: dict[str, object], task_id: str) -> dict[str, Any] | None:
        owner = _owner_id(identity)
        normalized_task_id = _clean(task_id)
        if not normalized_task_id:
            return None
        with self._lock:
            task = self._tasks.get(_task_key(owner, normalized_task_id))
            return _public_task(task) if task is not None else None

    def _submit(
        self,
        identity: dict[str, object],
        *,
        client_task_id: str,
        mode: str,
        payload: dict[str, Any],
        quota_reservation: dict[str, Any] | None = None,
        start: bool = True,
        mark_created: bool = False,
    ) -> dict[str, Any]:
        task_id = _clean(client_task_id)
        if not task_id:
            raise ValueError("client_task_id is required")
        owner = _owner_id(identity)
        key = _task_key(owner, task_id)
        now = _now_iso()
        should_start = False
        with self._lock:
            cleaned = self._cleanup_locked()
            task = self._tasks.get(key)
            if task is not None:
                if cleaned:
                    self._save_locked()
                public = _public_task(task)
                if mark_created:
                    public["_created"] = False
                return public
            task = {
                "id": task_id,
                "owner_id": owner,
                "status": TASK_STATUS_QUEUED,
                "mode": mode,
                "model": _clean(payload.get("model"), "gpt-image-2"),
                "size": _clean(payload.get("size")),
                "created_at": now,
                "updated_at": now,
            }
            reservation = quota_reservation
            try:
                if reservation is None:
                    reservation = auth_service.reserve_image_quota(
                        identity,
                        mode="edit" if mode == "edit" else "generate",
                        count=payload.get("n") or 1,
                        reservation_id=key,
                    )
                if reservation:
                    task["quota_reservation"] = reservation
                    task["quota_cost"] = reservation.get("cost")
                    task["quota_status"] = "reserved"
                else:
                    task["quota_status"] = "not_required"
                self._tasks[key] = task
                self._save_locked()
            except Exception:
                self._tasks.pop(key, None)
                _refund_task_reservation(reservation)
                raise
            should_start = start

        if should_start:
            try:
                self._start_task_thread(key, task_id, mode, payload, identity, _clean(payload.get("model"), "gpt-image-2"))
            except Exception:
                self.fail_task(identity, task_id, "图片任务启动失败")
                raise
        public = _public_task(task)
        if mark_created:
            public["_created"] = True
        return public

    def fail_task(self, identity: dict[str, object], task_id: str, error_message: str) -> dict[str, Any] | None:
        owner = _owner_id(identity)
        normalized_task_id = _clean(task_id)
        if not normalized_task_id:
            return None
        key = _task_key(owner, normalized_task_id)
        task = self._get_task(key)
        reservation = task.get("quota_reservation") if task else None
        quota_status = _finalize_task_reservation(reservation if isinstance(reservation, dict) else None, charge=False)
        with self._lock:
            current = self._tasks.get(key)
            if current is None:
                return None
            if quota_status != "pending":
                current.pop("quota_reservation", None)
            current["status"] = TASK_STATUS_ERROR
            current["error"] = error_message or "image task failed"
            current["data"] = []
            current["quota_status"] = (
                _public_quota_status(quota_status, pending="refund_pending")
                if reservation
                else current.get("quota_status", "not_required")
            )
            current["updated_at"] = _now_iso()
            self._save_locked()
            return _public_task(current)

    def _start_task_thread(
        self,
        key: str,
        task_id: str,
        mode: str,
        payload: dict[str, Any],
        identity: dict[str, object],
        model: str,
    ) -> None:
        self._ensure_workers_started()
        try:
            self._task_queue.put((key, task_id, mode, payload, dict(identity), model), timeout=1.0)
        except queue.Full as exc:
            raise RuntimeError("图片任务队列已满，请稍后重试") from exc

    def _ensure_workers_started(self) -> None:
        if self._workers_started:
            return
        with self._worker_lock:
            if self._workers_started:
                return
            for index in range(self._task_workers):
                thread = threading.Thread(
                    target=self._task_worker_loop,
                    name=f"image-task-worker-{index + 1}",
                    daemon=True,
                )
                thread.start()
            self._workers_started = True

    def _task_worker_loop(self) -> None:
        while True:
            key, _task_id, mode, payload, identity, model = self._task_queue.get()
            try:
                self._run_task(key, mode, payload, identity, model)
            finally:
                self._task_queue.task_done()

    def _run_task(
        self,
        key: str,
        mode: str,
        payload: dict[str, Any],
        identity: dict[str, object],
        model: str,
    ) -> None:
        started = time.time()
        try:
            self._update_task(key, status=TASK_STATUS_RUNNING, error="")
            handler = self.edit_handler if mode == "edit" else self.generation_handler
            result = handler(payload)
            if not isinstance(result, dict):
                raise RuntimeError("image task returned streaming result unexpectedly")
            data = result.get("data")
            if not isinstance(data, list) or not data:
                upstream = _clean(result.get("message"))
                if upstream:
                    message = upstream
                else:
                    message = "号池中没有可用账号或所有账号均被限流，请检查号池状态（账号额度、是否被封禁、是否到达生图上限）"
                raise RuntimeError(message)
            task = self._get_task(key)
            reservation = task.get("quota_reservation") if task else None
            self._update_task(key, status=TASK_STATUS_SUCCESS, data=data, error="", quota_status="charge_pending")
            quota_status = _finalize_task_reservation(reservation if isinstance(reservation, dict) else None, charge=True)
            try:
                self._update_task(
                    key,
                    quota_status=_public_quota_status(quota_status),
                    remove_quota_reservation=quota_status != "pending",
                )
            except Exception:
                pass
            self._log_call(
                identity,
                mode,
                model,
                started,
                "调用完成",
                request_preview=request_text(payload.get("prompt")),
                urls=_collect_image_urls(data),
            )
        except Exception as exc:
            error_message = str(exc) or "image task failed"
            try:
                self._fail_task_by_key(key, error_message)
            except Exception:
                pass
            self._log_call(
                identity,
                mode,
                model,
                started,
                "调用失败",
                request_preview=request_text(payload.get("prompt")),
                status="failed",
                error=error_message,
            )

    def _log_call(
        self,
        identity: dict[str, object],
        mode: str,
        model: str,
        started: float,
        suffix: str,
        *,
        request_preview: str = "",
        status: str = "success",
        error: str = "",
        urls: list[str] | None = None,
    ) -> None:
        endpoint = "/v1/images/edits" if mode == "edit" else "/v1/images/generations"
        summary_prefix = "图生图" if mode == "edit" else "文生图"
        detail = {
            "key_id": identity.get("id"),
            "key_name": identity.get("name"),
            "role": identity.get("role"),
            "endpoint": endpoint,
            "model": model,
            "started_at": datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M:%S"),
            "ended_at": _now_iso(),
            "duration_ms": int((time.time() - started) * 1000),
            "status": status,
        }
        if request_preview:
            detail["request_text"] = request_preview
        if error:
            detail["error"] = error
        if urls:
            detail["urls"] = list(dict.fromkeys(urls))
        try:
            log_service.add(LOG_TYPE_CALL, f"{summary_prefix}{suffix}", detail)
        except Exception:
            pass

    def _get_task(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(key)
            return dict(task) if task is not None else None

    def _clear_task_quota_reservation(self, key: str) -> None:
        with self._lock:
            task = self._tasks.get(key)
            if task is not None:
                task.pop("quota_reservation", None)

    def _fail_task_by_key(self, key: str, error_message: str) -> None:
        task = self._get_task(key)
        reservation = task.get("quota_reservation") if task else None
        quota_status = _finalize_task_reservation(reservation if isinstance(reservation, dict) else None, charge=False)
        self._update_task(
            key,
            status=TASK_STATUS_ERROR,
            error=error_message,
            data=[],
            quota_status=_public_quota_status(quota_status, pending="refund_pending") if reservation else "not_required",
            remove_quota_reservation=quota_status != "pending",
        )

    def _update_task(self, key: str, remove_quota_reservation: bool = False, **updates: Any) -> None:
        with self._lock:
            task = self._tasks.get(key)
            if task is None:
                return
            if remove_quota_reservation:
                task.pop("quota_reservation", None)
            task.update(updates)
            task["updated_at"] = _now_iso()
            self._save_locked()

    def _load_locked(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        raw_items = raw.get("tasks") if isinstance(raw, dict) else raw
        if not isinstance(raw_items, list):
            return {}
        tasks: dict[str, dict[str, Any]] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            task_id = _clean(item.get("id"))
            owner = _clean(item.get("owner_id"))
            if not task_id or not owner:
                continue
            status = _clean(item.get("status"))
            if status not in {TASK_STATUS_QUEUED, TASK_STATUS_RUNNING, TASK_STATUS_SUCCESS, TASK_STATUS_ERROR}:
                status = TASK_STATUS_ERROR
            task = {
                "id": task_id,
                "owner_id": owner,
                "status": status,
                "mode": "edit" if item.get("mode") == "edit" else "generate",
                "model": _clean(item.get("model"), "gpt-image-2"),
                "size": _clean(item.get("size")),
                "created_at": _clean(item.get("created_at"), _now_iso()),
                "updated_at": _clean(item.get("updated_at"), _clean(item.get("created_at"), _now_iso())),
            }
            data = item.get("data")
            if isinstance(data, list):
                task["data"] = data
            error = _clean(item.get("error"))
            if error:
                task["error"] = error
            reservation = item.get("quota_reservation")
            if isinstance(reservation, dict):
                task["quota_reservation"] = dict(reservation)
            if item.get("quota_cost") is not None:
                task["quota_cost"] = item.get("quota_cost")
            quota_status = _clean(item.get("quota_status"))
            if quota_status:
                task["quota_status"] = quota_status
            tasks[_task_key(owner, task_id)] = task
        return tasks

    def _save_locked(self) -> None:
        items = sorted(self._tasks.values(), key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps({"tasks": items}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(self.path)

    def _recover_unfinished_locked(self) -> bool:
        changed = False
        for task in self._tasks.values():
            if task.get("status") in UNFINISHED_STATUSES:
                reservation = task.get("quota_reservation")
                quota_status = _finalize_task_reservation(reservation if isinstance(reservation, dict) else None, charge=False)
                if quota_status != "pending":
                    task.pop("quota_reservation", None)
                task["status"] = TASK_STATUS_ERROR
                task["error"] = "服务已重启，未完成的图片任务已中断"
                task["quota_status"] = (
                    _public_quota_status(quota_status, pending="refund_pending")
                    if reservation
                    else "not_required"
                )
                task["updated_at"] = _now_iso()
                changed = True
                continue
            reservation = task.get("quota_reservation")
            if not isinstance(reservation, dict) or task.get("status") not in TERMINAL_STATUSES:
                continue
            if task.get("status") == TASK_STATUS_SUCCESS:
                quota_status = _finalize_task_reservation(reservation, charge=True)
                pending_status = "charge_pending"
            else:
                quota_status = _finalize_task_reservation(reservation, charge=False)
                pending_status = "refund_pending"
            task["quota_status"] = _public_quota_status(quota_status, pending=pending_status)
            if quota_status != "pending":
                task.pop("quota_reservation", None)
            else:
                logger.warning({"event": "image_task_quota_recovery_pending", "task_id": task.get("id")})
            task["updated_at"] = _now_iso()
            changed = True
        return changed

    def _cleanup_locked(self) -> bool:
        try:
            retention_days = max(1, int(self.retention_days_getter()))
        except Exception:
            retention_days = 30
        cutoff = time.time() - retention_days * 86400
        removed_keys = [
            key
            for key, task in self._tasks.items()
            if task.get("status") in TERMINAL_STATUSES and _timestamp(task.get("updated_at")) < cutoff
        ]
        for key in removed_keys:
            self._tasks.pop(key, None)
        return bool(removed_keys)


image_task_service = ImageTaskService(DATA_DIR / "image_tasks.json")
