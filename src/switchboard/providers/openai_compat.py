"""Generic OpenAI-compatible provider (OpenAI, Groq, Gemini, Ollama, vLLM, ...).

NOTE: the pure helpers (parse_retry_after, parse_completion, parse_sse_data) are unit-tested.
The HTTP calls need ``httpx`` and were NOT exercised against live services in the build sandbox.
"""
from __future__ import annotations

import email.utils
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from ..errors import (ProviderAuthError, ProviderBadRequest, ProviderTimeout, ProviderUnavailable,
                      RateLimited, StreamInterrupted)
from ..tokens import count_text, estimate_prompt_tokens
from ..types import ChatRequest, ChatResponse, Chunk, Usage
from .base import Provider


def parse_retry_after(value: str | None, now: float | None = None) -> float | None:
    """Retry-After may be delta-seconds or an HTTP date."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        dt = email.utils.parsedate_to_datetime(value)
        return max(0.0, dt.timestamp() - (now if now is not None else time.time()))
    except (TypeError, ValueError):
        return None


def error_for_status(status: int, body_text: str, retry_after: float | None):
    msg = body_text[:300]
    if status == 429:
        return RateLimited(msg, retry_after=retry_after)
    if status in (401, 403):
        return ProviderAuthError(msg)
    if status in (408, 504):
        return ProviderTimeout(msg, retry_after=retry_after)
    if status in (400, 404, 413, 422):
        return ProviderBadRequest(msg)
    return ProviderUnavailable(f"upstream {status}: {msg}", retry_after=retry_after)


def parse_completion(data: dict, req: ChatRequest) -> ChatResponse:
    try:
        choice = data["choices"][0]
        msg = choice.get("message") or {}
    except (KeyError, IndexError, TypeError) as e:
        raise ProviderUnavailable("malformed upstream response") from e
    content = msg.get("content")
    u = data.get("usage")
    if u and "prompt_tokens" in u:
        usage = Usage(int(u["prompt_tokens"]), int(u.get("completion_tokens", 0)), estimated=False)
    else:
        usage = Usage(estimate_prompt_tokens(req.messages), count_text(content or ""), estimated=True)
    return ChatResponse(content=content, usage=usage, finish_reason=choice.get("finish_reason") or "stop",
                        tool_calls=msg.get("tool_calls"))


def parse_sse_data(payload: str) -> Chunk | None:
    """Parse one SSE ``data:`` payload into a Chunk (None for [DONE]/keepalive)."""
    payload = payload.strip()
    if not payload or payload == "[DONE]":
        return None
    d = json.loads(payload)
    usage = None
    if d.get("usage"):
        u = d["usage"]
        usage = Usage(int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0)))
    choices = d.get("choices") or []
    if not choices:
        return Chunk(usage=usage) if usage else None
    ch = choices[0]
    delta = ch.get("delta") or {}
    return Chunk(content=delta.get("content"), tool_calls=delta.get("tool_calls"),
                 finish_reason=ch.get("finish_reason"), usage=usage)


class OpenAICompatProvider(Provider):
    def __init__(self, name: str, base_url: str, api_key: str | None = None, timeout: float = 60.0,
                 extra_headers: dict[str, str] | None = None) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.extra_headers = extra_headers or {}
        self._client: Any = None

    def _http(self) -> Any:
        if self._client is None:
            import httpx  # lazy: keeps the core importable without httpx
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def chat(self, req: ChatRequest, upstream_model: str) -> ChatResponse:
        import httpx
        body = req.upstream_body(upstream_model)
        body.pop("stream", None)
        body.pop("stream_options", None)
        try:
            r = await self._http().post(f"{self.base_url}/chat/completions", json=body,
                                        headers=self._headers())
        except httpx.TimeoutException as e:
            raise ProviderTimeout(str(e) or "timeout") from e
        except httpx.TransportError as e:
            raise ProviderUnavailable(f"transport error: {e}") from e
        if r.status_code >= 400:
            raise error_for_status(r.status_code, r.text, parse_retry_after(r.headers.get("retry-after")))
        return parse_completion(r.json(), req)

    async def stream(self, req: ChatRequest, upstream_model: str) -> AsyncIterator[Chunk]:
        import httpx
        body = req.upstream_body(upstream_model)
        body["stream"] = True
        started = False
        try:
            async with self._http().stream("POST", f"{self.base_url}/chat/completions", json=body,
                                           headers=self._headers()) as r:
                if r.status_code >= 400:
                    text = (await r.aread()).decode("utf-8", "replace")
                    raise error_for_status(r.status_code, text, parse_retry_after(r.headers.get("retry-after")))
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    chunk = parse_sse_data(line[5:])
                    if chunk is not None:
                        started = True
                        yield chunk
        except httpx.TimeoutException as e:
            raise (StreamInterrupted if started else ProviderTimeout)(str(e) or "timeout") from e
        except httpx.TransportError as e:
            raise (StreamInterrupted if started else ProviderUnavailable)(f"transport error: {e}") from e

    async def health(self) -> bool:
        try:
            r = await self._http().get(f"{self.base_url}/models", headers=self._headers(), timeout=5.0)
            return r.status_code < 500
        except Exception:
            return False
