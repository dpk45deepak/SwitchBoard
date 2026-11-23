"""Plain dataclasses used across the gateway (no framework dependency)."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .errors import BadRequest

PASSTHROUGH = ("top_p", "stop", "seed", "presence_penalty", "frequency_penalty", "tool_choice", "user")


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens}


@dataclass
class ChatRequest:
    model: str
    messages: list[dict]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    tools: list | None = None
    response_format: dict | None = None
    extra: dict = field(default_factory=dict)
    hints: dict = field(default_factory=dict)  # x_switchboard: never forwarded upstream
    include_usage: bool = False
    # gateway context
    api_key_id: str = "anonymous"
    path: str = "online"  # online | batch
    lane: str = "interactive"
    job_id: str | None = None
    task_key: str | None = None
    cache_bypass: bool = False
    retry_count: int = 0
    request_id: str = field(default_factory=lambda: "req_" + uuid.uuid4().hex[:16])

    @classmethod
    def from_openai(cls, body: Any, **ctx: Any) -> "ChatRequest":
        if not isinstance(body, dict):
            raise BadRequest("request body must be a JSON object")
        model, messages = body.get("model"), body.get("messages")
        if not isinstance(model, str) or not model:
            raise BadRequest("'model' is required")
        if not isinstance(messages, list) or not messages:
            raise BadRequest("'messages' must be a non-empty array")
        for m in messages:
            if not isinstance(m, dict) or "role" not in m:
                raise BadRequest("each message needs a 'role'")
        if body.get("n", 1) not in (1, None):
            raise BadRequest("only n=1 is supported")
        temp = body.get("temperature")
        if temp is not None and not isinstance(temp, (int, float)):
            raise BadRequest("'temperature' must be a number")
        mt = body.get("max_completion_tokens", body.get("max_tokens"))
        opts = body.get("stream_options") or {}
        return cls(
            model=model, messages=messages, temperature=temp, max_tokens=mt,
            stream=bool(body.get("stream")), tools=body.get("tools"),
            response_format=body.get("response_format"),
            extra={k: body[k] for k in PASSTHROUGH if k in body},
            hints=body.get("x_switchboard") or {},
            include_usage=bool(opts.get("include_usage")), **ctx,
        )

    def upstream_body(self, model_name: str) -> dict:
        b: dict[str, Any] = {"model": model_name, "messages": self.messages}
        if self.temperature is not None:
            b["temperature"] = self.temperature
        if self.max_tokens is not None:
            b["max_tokens"] = self.max_tokens
        if self.tools:
            b["tools"] = self.tools
        if self.response_format:
            b["response_format"] = self.response_format
        b.update(self.extra)
        if self.stream:
            b["stream"] = True
            b["stream_options"] = {"include_usage": True}
        return b

    def text(self) -> str:
        return "\n".join(content_text(m.get("content")) for m in self.messages)


def content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return json.dumps(content)


@dataclass
class ChatResponse:
    content: str | None
    usage: Usage
    finish_reason: str = "stop"
    tool_calls: list | None = None
    id: str = field(default_factory=lambda: "chatcmpl-" + uuid.uuid4().hex[:20])
    created: int = field(default_factory=lambda: int(time.time()))

    def to_openai(self, model: str) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return {"id": self.id, "object": "chat.completion", "created": self.created, "model": model,
                "choices": [{"index": 0, "message": msg, "finish_reason": self.finish_reason}],
                "usage": self.usage.to_dict()}


@dataclass
class Chunk:
    content: str | None = None
    tool_calls: list | None = None
    finish_reason: str | None = None
    usage: Usage | None = None

    def to_openai(self, cid: str, model: str, created: int, first: bool = False) -> dict:
        delta: dict[str, Any] = {}
        if first:
            delta["role"] = "assistant"
        if self.content is not None:
            delta["content"] = self.content
        if self.tool_calls:
            delta["tool_calls"] = self.tool_calls
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": self.finish_reason}]}
