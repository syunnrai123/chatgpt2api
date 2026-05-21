from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Iterable, Iterator

from PIL import Image
import tiktoken

from services.account_service import account_service
from services.config import config
from services.image_storage_service import image_storage_service
from services.openai_backend_api import ImagePollTimeoutError, OpenAIBackendAPI
from utils.helper import IMAGE_MODELS, extract_image_from_message_content
from utils.log import logger


class ImageGenerationError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 502,
        error_type: str = "server_error",
        code: str | None = "upstream_error",
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.code = code
        self.param = param

    def to_openai_error(self) -> dict[str, Any]:
        return {
            "error": {
                "message": str(self),
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


def is_token_invalid_error(message: str) -> bool:
    text = str(message or "").lower()
    return (
        "token_invalidated" in text
        or "token_revoked" in text
        or "authentication token has been invalidated" in text
        or "invalidated oauth token" in text
    )


def image_stream_error_message(message: str) -> str:
    text = str(message or "")
    lower = text.lower()
    if is_token_invalid_error(text):
        return "image generation failed"
    if "curl: (35)" in lower or "tls connect error" in lower or "openssl_internal" in lower:
        return "upstream image connection failed, please retry later"
    return text or "image generation failed"


def encode_images(images: Iterable[tuple[bytes, str, str]]) -> list[str]:
    return [base64.b64encode(data).decode("ascii") for data, _, _ in images if data]


def save_image_bytes(image_data: bytes, base_url: str | None = None) -> str:
    return image_storage_service.save(image_data, base_url).url


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and str(item.get("type") or "") in {"text", "input_text", "output_text"}:
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return ""


def normalize_messages(messages: object, system: Any = None) -> list[dict[str, Any]]:
    normalized = []
    if config.global_system_prompt:
        normalized.append({"role": "system", "content": config.global_system_prompt})
    system_text = message_text(system)
    if system_text:
        normalized.append({"role": "system", "content": system_text})
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "user")
            content = message.get("content", "")
            text = message_text(content)
            images: list[tuple[bytes, str]] = []
            if role == "user":
                images.extend(extract_image_from_message_content(content))
                if isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict) or part.get("type") != "image":
                            continue
                        data = part.get("data")
                        if isinstance(data, (bytes, bytearray)):
                            images.append((bytes(data), str(part.get("mime") or "image/png")))
            if images:
                parts: list[Any] = []
                if text:
                    parts.append({"type": "text", "text": text})
                for data, mime in images:
                    parts.append({"type": "image", "data": data, "mime": mime})
                normalized.append({"role": role, "content": parts})
            else:
                normalized.append({"role": role, "content": text})
    return normalized


def prompt_with_global_system(prompt: str) -> str:
    return f"{config.global_system_prompt}\n\n{prompt}" if config.global_system_prompt else prompt


def assistant_history_text(messages: list[dict[str, Any]]) -> str:
    return "".join(str(item.get("content") or "") for item in messages if item.get("role") == "assistant")


def assistant_history_messages(messages: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("content") or "") for item in messages if item.get("role") == "assistant" and item.get("content")]


def normalize_image_resolution(value: object) -> str:
    text = str(value or "").strip().lower()
    return text if text in {"1k", "2k", "4k"} else ""


IMAGE_SIZE_MAP: dict[str, dict[str, tuple[int, int]]] = {
    "1k": {
        "1:1": (1024, 1024),
        "4:3": (1152, 864),
        "3:4": (864, 1152),
        "16:9": (1344, 768),
        "9:16": (768, 1344),
    },
    "2k": {
        "1:1": (1248, 1248),
        "3:2": (1536, 1024),
        "2:3": (1024, 1536),
        "4:3": (1440, 1088),
        "3:4": (1088, 1440),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "21:9": (1904, 816),
    },
    "4k": {
        "1:1": (2480, 2480),
        "3:2": (3056, 2032),
        "2:3": (2032, 3056),
        "4:3": (2880, 2160),
        "3:4": (2160, 2880),
        "16:9": (3312, 1872),
        "9:16": (1872, 3312),
        "21:9": (3808, 1632),
    },
}


def parse_image_size_dimensions(value: object) -> tuple[int, int] | None:
    text = str(value or "").strip().lower()
    match = re.fullmatch(r"(\d{2,5})\s*[x×]\s*(\d{2,5})", text)
    if not match:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    return (width, height) if width > 0 and height > 0 else None


def normalize_image_ratio(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9"}:
        return text
    dimensions = parse_image_size_dimensions(value)
    if dimensions is None:
        return ""
    width, height = dimensions
    ratio = width / height
    known = {
        "1:1": 1.0,
        "3:2": 3 / 2,
        "2:3": 2 / 3,
        "4:3": 4 / 3,
        "3:4": 3 / 4,
        "16:9": 16 / 9,
        "9:16": 9 / 16,
        "21:9": 21 / 9,
    }
    best = min(known, key=lambda item: abs(known[item] - ratio))
    return best if abs(known[best] - ratio) <= 0.03 else ""


def image_output_dimensions(size: object = None, resolution: object = None) -> tuple[int, int] | None:
    normalized_resolution = normalize_image_resolution(resolution)
    ratio = normalize_image_ratio(size)
    if not normalized_resolution or not ratio:
        return None
    return IMAGE_SIZE_MAP.get(normalized_resolution, {}).get(ratio)


def image_output_size(size: object = None, resolution: object = None) -> str:
    dimensions = image_output_dimensions(size, resolution)
    return f"{dimensions[0]}x{dimensions[1]}" if dimensions else ""


def infer_image_resolution_from_size(size: object) -> str:
    dimensions = parse_image_size_dimensions(size)
    if dimensions is None:
        return ""
    width, height = dimensions
    area = width * height
    if area >= 4_000_000:
        return "4k"
    if area >= 1_400_000:
        return "2k"
    return "1k"


def image_request_options(body: dict[str, Any]) -> tuple[object, object]:
    params = body.get("params")
    params = params if isinstance(params, dict) else {}
    size = body.get("size") or body.get("aspect_ratio") or params.get("size") or params.get("aspect_ratio")
    resolution = body.get("resolution") or body.get("size_tier") or params.get("resolution") or params.get("size_tier")
    resolution = resolution or infer_image_resolution_from_size(size)
    return size, resolution


def build_image_prompt(prompt: str, size: str | None, resolution: str | None = None) -> str:
    hints: list[str] = []
    if size:
        size_text = str(size).strip()
        size_dimensions = parse_image_size_dimensions(size_text)
        if size_dimensions is not None:
            hints.append(f"输出图片目标尺寸为 {size_dimensions[0]}x{size_dimensions[1]}。")
        elif size not in {"1:1", "16:9", "9:16", "4:3", "3:4"}:
            hints.append(f"输出图片，宽高比为 {size}。")
        else:
            hints.append({
                "1:1": "输出为 1:1 正方形构图，主体居中，适合正方形画幅。",
                "16:9": "输出为 16:9 横屏构图，适合宽画幅展示。",
                "9:16": "输出为 9:16 竖屏构图，适合竖版画幅展示。",
                "4:3": "输出为 4:3 比例，兼顾宽度与高度，适合展示画面细节。",
                "3:4": "输出为 3:4 比例，纵向构图，适合人物肖像或竖向场景。",
            }[size])
    normalized_resolution = normalize_image_resolution(resolution)
    if normalized_resolution:
        requested_size = image_output_size(size, normalized_resolution)
        hints.append({
            "1k": "优先尝试 1K 清晰度偏好，画面干净清晰。",
            "2k": "优先尝试 2K 清晰度偏好，保留更多纹理和细节。",
            "4k": "优先尝试 4K 清晰度偏好，尽可能提供高分辨率和精细纹理。",
        }[normalized_resolution] + (f" 目标输出尺寸为 {requested_size}。" if requested_size else ""))
    if not hints:
        return prompt
    return f"{prompt.strip()}\n\n" + "\n".join(hints)


def encoding_for_model(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        try:
            return tiktoken.get_encoding("o200k_base")
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")


def count_message_tokens(messages: list[dict[str, Any]], model: str) -> int:
    encoding = encoding_for_model(model)
    total = 0
    for message in messages:
        total += 3
        for key, value in message.items():
            if not isinstance(value, str):
                continue
            total += len(encoding.encode(value))
            if key == "name":
                total += 1
    return total + 3


def count_text_tokens(text: str, model: str) -> int:
    return len(encoding_for_model(model).encode(text))


def format_image_result(
    items: list[dict[str, Any]],
    prompt: str,
    response_format: str,
    base_url: str | None = None,
    created: int | None = None,
    message: str = "",
) -> dict[str, Any]:
    data: list[dict[str, Any]] = []
    for item in items:
        b64_json = str(item.get("b64_json") or "").strip()
        if not b64_json:
            continue
        revised_prompt = str(item.get("revised_prompt") or prompt).strip() or prompt
        if response_format == "b64_json":
            data.append({
                "b64_json": b64_json,
                "url": save_image_bytes(base64.b64decode(b64_json), base_url),
                "revised_prompt": revised_prompt,
            })
        else:
            data.append({
                "url": save_image_bytes(base64.b64decode(b64_json), base_url),
                "revised_prompt": revised_prompt,
            })
    result: dict[str, Any] = {"created": created or int(time.time()), "data": data}
    if message and not data:
        result["message"] = message
    return result


def image_byte_dimensions(image_data: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(image_data)) as image:
        return image.size


def image_resolution_min_area(resolution: object) -> int | None:
    normalized_resolution = normalize_image_resolution(resolution)
    if normalized_resolution == "2k":
        return int(1_500_000 * 0.98)
    if normalized_resolution == "4k":
        return int(6_000_000 * 0.98)
    return None


def ensure_requested_image_resolution(image_data: bytes, size: object, resolution: object) -> None:
    normalized_resolution = normalize_image_resolution(resolution)
    if normalized_resolution not in {"2k", "4k"}:
        return
    width, height = image_byte_dimensions(image_data)
    requested_dimensions = parse_image_size_dimensions(size)
    if requested_dimensions is not None:
        min_width = int(requested_dimensions[0] * 0.98)
        min_height = int(requested_dimensions[1] * 0.98)
        if width >= min_width and height >= min_height:
            return
        requested_size = f"{requested_dimensions[0]}x{requested_dimensions[1]}"
        actual_size = f"{width}x{height}"
        raise ImageGenerationError(
            f"requested resolution {normalized_resolution.upper()} was not satisfied by upstream image size "
            f"{actual_size}; requested size is {requested_size}. 已触发清晰度降级重试。",
            status_code=400,
            error_type="invalid_request_error",
            code="image_resolution_unavailable",
            param="resolution",
        )
    target = image_output_dimensions(size, normalized_resolution)
    if not target:
        min_area = image_resolution_min_area(normalized_resolution)
        if min_area is None or width * height >= min_area:
            return
        requested_size = f"{normalized_resolution.upper()} tier"
        actual_size = f"{width}x{height}"
        raise ImageGenerationError(
            f"requested resolution {normalized_resolution.upper()} was not satisfied by upstream image size "
            f"{actual_size}; requested size is {requested_size}. 已触发清晰度降级重试。",
            status_code=400,
            error_type="invalid_request_error",
            code="image_resolution_unavailable",
            param="resolution",
        )
    min_width = int(target[0] * 0.98)
    min_height = int(target[1] * 0.98)
    if width >= min_width and height >= min_height:
        return
    requested_size = f"{target[0]}x{target[1]}"
    actual_size = f"{width}x{height}"
    raise ImageGenerationError(
        f"requested resolution {normalized_resolution.upper()} was not satisfied by upstream image size "
        f"{actual_size}; requested size is {requested_size}. 已触发清晰度降级重试。",
        status_code=400,
        error_type="invalid_request_error",
        code="image_resolution_unavailable",
        param="resolution",
    )


@dataclass
class ConversationRequest:
    model: str = "auto"
    prompt: str = ""
    messages: list[dict[str, Any]] | None = None
    images: list[str] | None = None
    n: int = 1
    size: str | None = None
    resolution: str | None = None
    response_format: str = "b64_json"
    base_url: str | None = None
    message_as_error: bool = False


@dataclass
class ConversationState:
    text: str = ""
    conversation_id: str = ""
    file_ids: list[str] = field(default_factory=list)
    sediment_ids: list[str] = field(default_factory=list)
    blocked: bool = False
    tool_invoked: bool | None = None
    turn_use_case: str = ""


@dataclass
class ImageOutput:
    kind: str
    model: str
    index: int
    total: int
    created: int = field(default_factory=lambda: int(time.time()))
    text: str = ""
    upstream_event_type: str = ""
    data: list[dict[str, Any]] = field(default_factory=list)

    def to_chunk(self) -> dict[str, Any]:
        chunk: dict[str, Any] = {
            "object": "image.generation.chunk",
            "created": self.created,
            "model": self.model,
            "index": self.index,
            "total": self.total,
            "progress_text": self.text,
            "upstream_event_type": self.upstream_event_type,
            "data": [],
        }
        if self.kind == "message":
            chunk.update({
                "object": "image.generation.message",
                "message": self.text,
            })
            chunk.pop("progress_text", None)
            chunk.pop("upstream_event_type", None)
        elif self.kind == "result":
            chunk.update({
                "object": "image.generation.result",
                "data": self.data,
            })
            chunk.pop("progress_text", None)
            chunk.pop("upstream_event_type", None)
        return chunk


def assistant_message_text(message: dict[str, Any]) -> str:
    content = message.get("content") or {}
    parts = content.get("parts") or []
    if not isinstance(parts, list):
        return ""
    return "".join(part for part in parts if isinstance(part, str))


def strip_history(text: str, history_text: str = "") -> str:
    text = str(text or "")
    history_text = str(history_text or "")
    while history_text and text.startswith(history_text):
        text = text[len(history_text):]
    return text


def assistant_text(event: dict[str, Any], current_text: str = "", history_text: str = "") -> str:
    for candidate in (event, event.get("v")):
        if not isinstance(candidate, dict):
            continue
        message = candidate.get("message")
        if not isinstance(message, dict):
            continue
        role = str((message.get("author") or {}).get("role") or "").strip().lower()
        if role != "assistant":
            continue
        text = assistant_message_text(message)
        if text:
            return strip_history(text, history_text)
    return apply_text_patch(event, current_text, history_text)


def event_assistant_text(event: dict[str, Any], history_text: str = "") -> str:
    for candidate in (event, event.get("v")):
        if not isinstance(candidate, dict):
            continue
        message = candidate.get("message")
        if isinstance(message, dict) and (message.get("author") or {}).get("role") == "assistant":
            return strip_history(assistant_message_text(message), history_text)
    return ""


def apply_text_patch(event: dict[str, Any], current_text: str = "", history_text: str = "") -> str:
    if event.get("p") == "/message/content/parts/0":
        return apply_patch_op(event, current_text, history_text)

    operations = event.get("v")
    if isinstance(operations, str) and current_text and not event.get("p") and not event.get("o"):
        return current_text + operations

    if event.get("o") == "patch" and isinstance(operations, list):
        text = current_text
        for item in operations:
            if isinstance(item, dict):
                text = apply_text_patch(item, text, history_text)
        return text

    if not isinstance(operations, list):
        return current_text

    text = current_text
    for item in operations:
        if isinstance(item, dict):
            text = apply_text_patch(item, text, history_text)
    return text


def apply_patch_op(operation: dict[str, Any], current_text: str, history_text: str = "") -> str:
    op = operation.get("o")
    value = str(operation.get("v") or "")
    if op == "append":
        return current_text + value
    if op == "replace":
        return strip_history(value, history_text)
    return current_text


def add_unique(values: list[str], candidates: list[str]) -> None:
    for candidate in candidates:
        if candidate and candidate not in values:
            values.append(candidate)


def extract_conversation_ids(payload: str) -> tuple[str, list[str], list[str]]:
    conversation_match = re.search(r'"conversation_id"\s*:\s*"([^"]+)"', payload)
    conversation_id = conversation_match.group(1) if conversation_match else ""
    # Negative lookahead excludes "file-service" (URI prefix, not a real id).
    file_ids = re.findall(r"(file[-_](?!service\b)[A-Za-z0-9]+)", payload)
    sediment_ids = re.findall(r"sediment://([A-Za-z0-9_-]+)", payload)
    return conversation_id, file_ids, sediment_ids


def is_image_tool_event(event: dict[str, Any]) -> bool:
    value = event.get("v")
    message = event.get("message") or (value.get("message") if isinstance(value, dict) else None)
    if not isinstance(message, dict):
        return False
    metadata = message.get("metadata") or {}
    author = message.get("author") or {}
    content = message.get("content") or {}
    if author.get("role") != "tool":
        return False
    if metadata.get("async_task_type") == "image_gen":
        return True
    if content.get("content_type") != "multimodal_text":
        return False
    return any(
        isinstance(part, dict) and (
                part.get("content_type") == "image_asset_pointer"
                or str(part.get("asset_pointer") or "").startswith(("file-service://", "sediment://"))
        )
        for part in content.get("parts") or []
    )


def update_conversation_state(state: ConversationState, payload: str, event: dict[str, Any] | None = None) -> None:
    conversation_id, file_ids, sediment_ids = extract_conversation_ids(payload)
    if conversation_id and not state.conversation_id:
        state.conversation_id = conversation_id
    # Accept file_id / sediment_id when any of:
    #   1) event is a complete image_gen tool message
    #   2) prior server_ste_metadata already flipped tool_invoked True (in an image_gen turn)
    #   3) patch event whose payload references asset_pointer / file-service://
    # User messages (type=conversation.message) never satisfy these, so attacker-controlled
    # substrings in user input cannot inject file ids into state.
    is_patch_event = isinstance(event, dict) and event.get("o") == "patch"
    image_context = (
        (isinstance(event, dict) and is_image_tool_event(event))
        or state.tool_invoked is True
        or (is_patch_event and ("asset_pointer" in payload or "file-service://" in payload))
    )
    if image_context:
        add_unique(state.file_ids, file_ids)
        add_unique(state.sediment_ids, sediment_ids)
    if not isinstance(event, dict):
        return
    state.conversation_id = str(event.get("conversation_id") or state.conversation_id)
    value = event.get("v")
    if isinstance(value, dict):
        state.conversation_id = str(value.get("conversation_id") or state.conversation_id)
    if event.get("type") == "moderation":
        moderation = event.get("moderation_response")
        if isinstance(moderation, dict) and moderation.get("blocked") is True:
            state.blocked = True
    if event.get("type") == "server_ste_metadata":
        metadata = event.get("metadata")
        if isinstance(metadata, dict):
            if isinstance(metadata.get("tool_invoked"), bool):
                state.tool_invoked = metadata["tool_invoked"]
            state.turn_use_case = str(metadata.get("turn_use_case") or state.turn_use_case)


def conversation_base_event(event_type: str, state: ConversationState, **extra: Any) -> dict[str, Any]:
    return {
        "type": event_type,
        "text": state.text,
        "conversation_id": state.conversation_id,
        "file_ids": list(state.file_ids),
        "sediment_ids": list(state.sediment_ids),
        "blocked": state.blocked,
        "tool_invoked": state.tool_invoked,
        "turn_use_case": state.turn_use_case,
        **extra,
    }


def iter_conversation_payloads(payloads: Iterator[str], history_text: str = "",
                               history_messages: list[str] | None = None) -> Iterator[dict[str, Any]]:
    state = ConversationState()
    history_messages = history_messages or []
    history_index = 0
    for payload in payloads:
        # print(f"[upstream_sse] {payload}", flush=True)
        if not payload:
            continue
        if payload == "[DONE]":
            yield conversation_base_event("conversation.done", state, done=True)
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            update_conversation_state(state, payload)
            yield conversation_base_event("conversation.raw", state, payload=payload)
            continue
        if not isinstance(event, dict):
            yield conversation_base_event("conversation.event", state, raw=event)
            continue
        update_conversation_state(state, payload, event)
        if history_index < len(history_messages) and event_assistant_text(event, history_text) == history_messages[history_index]:
            history_index += 1
            state.text = ""
            continue
        next_text = assistant_text(event, state.text, history_text)
        if next_text != state.text:
            delta = next_text[len(state.text):] if next_text.startswith(state.text) else next_text
            state.text = next_text
            yield conversation_base_event("conversation.delta", state, raw=event, delta=delta)
            continue
        yield conversation_base_event("conversation.event", state, raw=event)


def conversation_events(
    backend: OpenAIBackendAPI,
    messages: list[dict[str, Any]] | None = None,
    model: str = "auto",
    prompt: str = "",
    images: list[str] | None = None,
    size: str | None = None,
    resolution: str | None = None,
) -> Iterator[dict[str, Any]]:
    normalized = normalize_messages(messages or ([{"role": "user", "content": prompt}] if prompt else []))
    image_model = str(model or "").strip() in IMAGE_MODELS
    history_text = "" if image_model else assistant_history_text(normalized)
    history_messages = [] if image_model else assistant_history_messages(normalized)
    final_prompt = prompt_with_global_system(build_image_prompt(prompt, size, resolution)) if image_model else prompt
    payloads = backend.stream_conversation(
        messages=normalized,
        model=model,
        prompt=final_prompt,
        images=images if image_model else None,
        system_hints=["picture_v2"] if image_model else None,
    )
    yield from iter_conversation_payloads(payloads, history_text, history_messages)


def text_backend() -> OpenAIBackendAPI:
    return OpenAIBackendAPI(access_token=account_service.get_text_access_token())


def stream_text_deltas(backend: OpenAIBackendAPI, request: ConversationRequest) -> Iterator[str]:
    attempted_tokens: set[str] = set()
    token = getattr(backend, "access_token", "")
    emitted = False
    while True:
        if token and token in attempted_tokens:
            raise RuntimeError("no available text account")
        if token:
            attempted_tokens.add(token)
        try:
            active_backend = OpenAIBackendAPI(access_token=token)
            for event in conversation_events(active_backend, messages=request.messages, model=request.model, prompt=request.prompt):
                if event.get("type") != "conversation.delta":
                    continue
                delta = str(event.get("delta") or "")
                if delta:
                    emitted = True
                    yield delta
            account_service.mark_text_used(token)
            return
        except Exception as exc:
            error_message = str(exc)
            if token and not emitted and is_token_invalid_error(error_message):
                account_service.remove_invalid_token(token, "text_stream")
                token = account_service.get_text_access_token(attempted_tokens)
                if token:
                    continue
            raise


def collect_text(backend: OpenAIBackendAPI, request: ConversationRequest) -> str:
    return "".join(stream_text_deltas(backend, request))


def stream_image_outputs(
        backend: OpenAIBackendAPI,
        request: ConversationRequest,
        index: int = 1,
        total: int = 1,
) -> Iterator[ImageOutput]:
    last: dict[str, Any] = {}
    for event in conversation_events(
            backend,
            prompt=request.prompt,
            model=request.model,
            images=request.images or [],
            size=request.size,
            resolution=request.resolution,
    ):
        last = event
        if event.get("type") == "conversation.delta":
            yield ImageOutput(
                kind="progress",
                model=request.model,
                index=index,
                total=total,
                text=str(event.get("delta") or ""),
                upstream_event_type="conversation.delta",
            )
            continue
        if event.get("type") == "conversation.event":
            raw = event.get("raw")
            raw_type = str(raw.get("type") or "") if isinstance(raw, dict) else ""
            yield ImageOutput(
                kind="progress",
                model=request.model,
                index=index,
                total=total,
                upstream_event_type=raw_type,
            )

    conversation_id = str(last.get("conversation_id") or "")
    file_ids = [str(item) for item in last.get("file_ids") or []]
    sediment_ids = [str(item) for item in last.get("sediment_ids") or []]
    message = str(last.get("text") or "").strip()
    logger.info({
        "event": "image_stream_resolve_start",
        "conversation_id": conversation_id,
        "file_ids": file_ids,
        "sediment_ids": sediment_ids,
        "tool_invoked": last.get("tool_invoked"),
        "turn_use_case": last.get("turn_use_case"),
    })
    if message and not file_ids and not sediment_ids and last.get("blocked"):
        yield ImageOutput(kind="message", model=request.model, index=index, total=total, text=message)
        return
    should_poll_for_image = bool(request.images) or last.get("turn_use_case") == "image gen"
    if message and not file_ids and not sediment_ids and not should_poll_for_image:
        yield ImageOutput(kind="message", model=request.model, index=index, total=total, text=message)
        return

    image_urls = backend.resolve_conversation_image_urls(conversation_id, file_ids, sediment_ids)
    if image_urls:
        downloaded_images = backend.download_image_bytes(image_urls)
        for image_data in downloaded_images:
            ensure_requested_image_resolution(image_data, request.size, request.resolution)
        image_items = [{"b64_json": base64.b64encode(image_data).decode("ascii")} for image_data in downloaded_images]
        data = format_image_result(
            image_items,
            request.prompt,
            request.response_format,
            request.base_url,
            int(time.time()),
        )["data"]
        if data:
            yield ImageOutput(kind="result", model=request.model, index=index, total=total, data=data)
        return

    if message:
        yield ImageOutput(kind="message", model=request.model, index=index, total=total, text=message)


def stream_image_outputs_with_pool(request: ConversationRequest) -> Iterator[ImageOutput]:
    if str(request.model or "").strip() not in IMAGE_MODELS:
        raise ImageGenerationError("unsupported image model,supported models: " + ", ".join(IMAGE_MODELS))

    emitted = False
    last_error = ""
    for index in range(1, request.n + 1):
        while True:
            try:
                token = account_service.get_available_access_token(request.resolution)
            except RuntimeError as exc:
                if emitted:
                    return
                raise ImageGenerationError(str(exc) or "image generation failed") from exc

            emitted_for_token = False
            returned_message = False
            returned_result = False
            try:
                backend = OpenAIBackendAPI(access_token=token)
                for output in stream_image_outputs(backend, request, index, request.n):
                    if output.kind == "message" and request.message_as_error:
                        raise ImageGenerationError(
                            output.text or "Image generation was rejected by upstream policy.",
                            status_code=400,
                            error_type="invalid_request_error",
                            code="content_policy_violation",
                        )
                    emitted = True
                    emitted_for_token = True
                    returned_message = output.kind == "message"
                    returned_result = returned_result or output.kind == "result"
                    yield output
                if returned_message or not returned_result:
                    account_service.mark_image_result(token, False)
                    return
                account_service.mark_image_result(token, True)
                break
            except ImagePollTimeoutError:
                raise
            except ImageGenerationError:
                account_service.mark_image_result(token, False)
                raise
            except Exception as exc:
                account_service.mark_image_result(token, False)
                last_error = str(exc)
                logger.warning({"event": "image_stream_fail", "request_token": token, "error": last_error})
                if not emitted_for_token and is_token_invalid_error(last_error):
                    account_service.remove_invalid_token(token, "image_stream")
                    continue
                raise ImageGenerationError(image_stream_error_message(last_error)) from exc

    if not emitted:
        if not last_error:
            last_error = "no account in the pool could generate images — check account quota and rate-limit status"
        raise ImageGenerationError(image_stream_error_message(last_error))


def stream_image_chunks(outputs: Iterable[ImageOutput]) -> Iterator[dict[str, Any]]:
    for output in outputs:
        yield output.to_chunk()


def collect_image_outputs(outputs: Iterable[ImageOutput]) -> dict[str, Any]:
    created = None
    data: list[dict[str, Any]] = []
    message = ""
    progress_parts: list[str] = []
    for output in outputs:
        created = created or output.created
        if output.kind == "progress" and output.text:
            progress_parts.append(output.text)
        elif output.kind == "message":
            message = output.text
        elif output.kind == "result":
            data.extend(output.data)

    result: dict[str, Any] = {"created": created or int(time.time()), "data": data}
    if not data:
        text = message or "".join(progress_parts).strip()
        if text:
            result["message"] = text
    return result
