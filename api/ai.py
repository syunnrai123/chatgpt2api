from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from api.image_inputs import parse_image_edit_request, read_image_sources
from api.support import require_identity, resolve_image_base_url
from services.content_filter import check_request, request_text
from services.log_service import LoggedCall
from services.image_quota import ImageQuotaError, refund_image_reservation, reserve_for_image_request, run_image_handler_with_quota
from services.image_limits import MAX_IMAGES_PER_REQUEST
from services.protocol import (
    anthropic_v1_messages,
    openai_v1_chat_complete,
    openai_v1_image_edit,
    openai_v1_image_generations,
    openai_v1_models,
    openai_v1_response,
)
from utils.helper import extract_chat_prompt, has_chat_image, has_response_input_image, parse_image_count


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str = "gpt-image-2"
    n: int = Field(default=1, ge=1, le=MAX_IMAGES_PER_REQUEST)
    size: str | None = None
    resolution: str | None = None
    params: dict[str, object] | None = None
    response_format: str = "b64_json"
    history_disabled: bool = True
    stream: bool | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    prompt: str | None = None
    n: int | None = None
    stream: bool | None = None
    modalities: list[str] | None = None
    messages: list[dict[str, object]] | None = None


class ResponseCreateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    input: object | None = None
    tools: list[dict[str, object]] | None = None
    tool_choice: object | None = None
    stream: bool | None = None


class AnthropicMessageRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str | None = None
    messages: list[dict[str, object]] | None = None
    system: object | None = None
    stream: bool | None = None


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


def _chat_image_quota_request(payload: dict[str, Any]) -> tuple[str, int] | None:
    if not openai_v1_chat_complete.is_image_chat_request(payload):
        return None
    prompt = extract_chat_prompt(payload)
    if not prompt:
        raise HTTPException(status_code=400, detail={"error": "prompt is required"})
    return ("edit" if has_chat_image(payload) else "generate", parse_image_count(payload.get("n")))


def _response_image_quota_request(payload: dict[str, Any]) -> tuple[str, int] | None:
    if openai_v1_response.is_text_response_request(payload):
        return None
    prompt = openai_v1_response.extract_response_prompt(payload.get("input"))
    if not prompt:
        raise HTTPException(status_code=400, detail={"error": "input text is required"})
    mode = "edit" if has_response_input_image(payload.get("input")) else "generate"
    return mode, 1


def _content_has_image(value: object) -> bool:
    if isinstance(value, str):
        return "data:image/" in value
    if isinstance(value, list):
        return any(_content_has_image(item) for item in value)
    if isinstance(value, dict):
        return any(_content_has_image(item) for item in value.values())
    return False


def _chat_image_success(item: dict[str, Any]) -> bool:
    choices = item.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        for key in ("message", "delta"):
            content = choice.get(key)
            if isinstance(content, dict) and _content_has_image(content.get("content")):
                return True
    return False


def _response_image_output_success(item: object) -> bool:
    if not isinstance(item, dict):
        return False
    return (
        item.get("type") == "image_generation_call"
        and item.get("status") == "completed"
        and bool(str(item.get("result") or "").strip())
    )


def _response_image_success(item: dict[str, Any]) -> bool:
    if item.get("type") == "response.output_item.done" and _response_image_output_success(item.get("item")):
        return True
    response = item.get("response")
    if not isinstance(response, dict):
        return False
    output = response.get("output")
    return isinstance(output, list) and any(_response_image_output_success(output_item) for output_item in output)


def create_router() -> APIRouter:
    router = APIRouter()

    @router.get("/v1/models")
    async def list_models(authorization: str | None = Header(default=None)):
        require_identity(authorization)
        try:
            return await run_in_threadpool(openai_v1_models.list_models)
        except Exception as exc:
            raise HTTPException(status_code=502, detail={"error": str(exc)}) from exc

    @router.post("/v1/images/generations")
    async def generate_images(
            body: ImageGenerationRequest,
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        payload = body.model_dump(mode="python")
        payload["base_url"] = resolve_image_base_url(request)
        call = LoggedCall(identity, "/v1/images/generations", body.model, "文生图", request_text=body.prompt)
        await filter_or_log(call, body.prompt)
        try:
            reservation = reserve_for_image_request(identity, mode="generate", count=body.n)
        except ImageQuotaError as exc:
            _raise_quota_error(exc)
        return await call.run(
            lambda request_payload: run_image_handler_with_quota(openai_v1_image_generations.handle, request_payload, reservation),
            payload,
        )

    @router.post("/v1/images/edits")
    async def edit_images(
            request: Request,
            authorization: str | None = Header(default=None),
    ):
        identity = require_identity(authorization)
        payload, image_sources = await parse_image_edit_request(request)
        prompt = str(payload["prompt"])
        model = str(payload["model"])
        call = LoggedCall(identity, "/v1/images/edits", model, "图生图", request_text=prompt)
        await filter_or_log(call, prompt)
        try:
            reservation = reserve_for_image_request(identity, mode="edit", count=payload.get("n"))
        except ImageQuotaError as exc:
            _raise_quota_error(exc)
        try:
            payload["images"] = await read_image_sources(image_sources)
        except asyncio.CancelledError:
            try:
                refund_image_reservation(reservation)
            finally:
                raise
        except Exception:
            refund_image_reservation(reservation)
            raise
        payload["base_url"] = resolve_image_base_url(request)
        return await call.run(
            lambda request_payload: run_image_handler_with_quota(openai_v1_image_edit.handle, request_payload, reservation),
            payload,
        )

    @router.post("/v1/chat/completions")
    async def create_chat_completion(body: ChatCompletionRequest, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        payload = body.model_dump(mode="python")
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("prompt"), payload.get("messages"))
        call = LoggedCall(identity, "/v1/chat/completions", model, "文本生成", request_text=request_preview)
        await filter_or_log(call, request_preview)
        try:
            quota_request = _chat_image_quota_request(payload)
        except HTTPException as exc:
            call.log("调用失败", status="failed", error=str(exc.detail))
            raise
        if quota_request is not None:
            mode, count = quota_request
            try:
                reservation = reserve_for_image_request(identity, mode=mode, count=count)
            except ImageQuotaError as exc:
                _raise_quota_error(exc)
            return await call.run(
                lambda request_payload: run_image_handler_with_quota(
                    openai_v1_chat_complete.handle,
                    request_payload,
                    reservation,
                    success_predicate=_chat_image_success,
                ),
                payload,
            )
        return await call.run(openai_v1_chat_complete.handle, payload)

    @router.post("/v1/responses")
    async def create_response(body: ResponseCreateRequest, authorization: str | None = Header(default=None)):
        identity = require_identity(authorization)
        payload = body.model_dump(mode="python")
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("input"), payload.get("instructions"))
        call = LoggedCall(identity, "/v1/responses", model, "Responses", request_text=request_preview)
        await filter_or_log(call, request_preview)
        try:
            quota_request = _response_image_quota_request(payload)
        except HTTPException as exc:
            call.log("调用失败", status="failed", error=str(exc.detail))
            raise
        if quota_request is not None:
            mode, count = quota_request
            try:
                reservation = reserve_for_image_request(identity, mode=mode, count=count)
            except ImageQuotaError as exc:
                _raise_quota_error(exc)
            return await call.run(
                lambda request_payload: run_image_handler_with_quota(
                    openai_v1_response.handle,
                    request_payload,
                    reservation,
                    success_predicate=_response_image_success,
                ),
                payload,
            )
        return await call.run(openai_v1_response.handle, payload)

    @router.post("/v1/messages")
    async def create_message(
            body: AnthropicMessageRequest,
            authorization: str | None = Header(default=None),
            x_api_key: str | None = Header(default=None, alias="x-api-key"),
            anthropic_version: str | None = Header(default=None, alias="anthropic-version"),
    ):
        identity = require_identity(authorization or (f"Bearer {x_api_key}" if x_api_key else None))
        payload = body.model_dump(mode="python")
        model = str(payload.get("model") or "auto")
        request_preview = request_text(payload.get("system"), payload.get("messages"), payload.get("tools"))
        call = LoggedCall(identity, "/v1/messages", model, "Messages", request_text=request_preview)
        await filter_or_log(call, request_preview)
        return await call.run(anthropic_v1_messages.handle, payload, sse="anthropic")

    return router
