from __future__ import annotations

import asyncio

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from api.image_inputs import parse_image_edit_request, read_image_sources
from api.support import require_identity, resolve_image_base_url
from services.content_filter import check_request
from services.image_task_service import image_task_service
from services.image_quota import ImageQuotaError
from services.log_service import LoggedCall


class ImageGenerationTaskRequest(BaseModel):
    client_task_id: str = Field(..., min_length=1)
    prompt: str = Field(..., min_length=1)
    model: str = "gpt-image-2"
    size: str | None = None
    resolution: str | None = None


def _parse_task_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


async def filter_or_log(call: LoggedCall, text: str) -> None:
    try:
        await run_in_threadpool(check_request, text)
    except HTTPException as exc:
        call.log("调用失败", status="failed", error=str(exc.detail))
        raise


def _raise_quota_error(exc: ImageQuotaError) -> None:
    raise HTTPException(
        status_code=429,
        detail={
            "error": str(exc),
            "required": exc.required,
            "available": exc.available,
        },
    ) from exc


def _task_error_message(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        detail = exc.detail
        if isinstance(detail, dict):
            return str(detail.get("error") or detail)
        return str(detail or "图片读取失败")
    return str(exc) or "图片读取失败"


def _raise_task_runtime_error(exc: RuntimeError) -> None:
    message = str(exc) or "图片任务提交失败"
    status_code = 503 if "队列已满" in message or "queue" in message.lower() else 502
    raise HTTPException(status_code=status_code, detail={"error": message}) from exc


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/image-tasks")
    async def list_image_tasks(
        ids: str = Query(default=""),
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        return await run_in_threadpool(image_task_service.list_tasks, identity, _parse_task_ids(ids))

    @router.post("/api/image-tasks/generations")
    async def create_generation_task(
        body: ImageGenerationTaskRequest,
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        await filter_or_log(LoggedCall(identity, "/api/image-tasks/generations", body.model, "文生图任务", request_text=body.prompt), body.prompt)
        try:
            return await run_in_threadpool(
                image_task_service.submit_generation,
                identity,
                client_task_id=body.client_task_id,
                prompt=body.prompt,
                model=body.model,
                size=body.size,
                resolution=body.resolution,
                base_url=resolve_image_base_url(request),
            )
        except ImageQuotaError as exc:
            _raise_quota_error(exc)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
        except RuntimeError as exc:
            _raise_task_runtime_error(exc)

    @router.post("/api/image-tasks/edits")
    async def create_edit_task(
        request: Request,
        authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        payload, image_sources = await parse_image_edit_request(request)
        client_task_id = str(payload.get("client_task_id") or "").strip()
        if not client_task_id:
            raise HTTPException(status_code=400, detail={"error": "client_task_id is required"})
        prompt = str(payload["prompt"])
        model = str(payload["model"])
        await filter_or_log(LoggedCall(identity, "/api/image-tasks/edits", model, "图生图任务", request_text=prompt), prompt)
        existing = await run_in_threadpool(image_task_service.get_task, identity, client_task_id)
        if existing is not None:
            return existing
        try:
            prepared = await run_in_threadpool(
                image_task_service.prepare_edit,
                identity,
                client_task_id=client_task_id,
                prompt=prompt,
                model=model,
                size=payload["size"],
                resolution=payload.get("resolution"),
                count=payload.get("n"),
            )
        except ImageQuotaError as exc:
            _raise_quota_error(exc)
        if not bool(prepared.pop("_created", False)):
            return prepared
        try:
            images = await read_image_sources(image_sources)
        except asyncio.CancelledError:
            try:
                image_task_service.fail_task(identity, client_task_id, "请求已取消，图片读取未完成")
            finally:
                raise
        except Exception as exc:
            await run_in_threadpool(image_task_service.fail_task, identity, client_task_id, _task_error_message(exc))
            raise
        try:
            return await run_in_threadpool(
                image_task_service.start_prepared_edit,
                identity,
                client_task_id=client_task_id,
                prompt=prompt,
                model=model,
                size=payload["size"],
                resolution=payload.get("resolution"),
                base_url=resolve_image_base_url(request),
                images=images,
                count=payload.get("n"),
            )
        except ImageQuotaError as exc:
            _raise_quota_error(exc)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"error": str(exc)}) from exc
        except RuntimeError as exc:
            _raise_task_runtime_error(exc)

    return router
