"""POST /v1/chat/completions (JSON + SSE streaming) and GET /v1/models."""
from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..errors import SwitchboardError
from ..types import ChatRequest
from .deps import authenticate, gateway, read_json

router = APIRouter()


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


async def sse_stream(meta, chunks, include_usage: bool):
    cid, created, model = "chatcmpl-" + uuid.uuid4().hex[:20], int(time.time()), meta.model or "unknown"
    first, usage = True, None
    try:
        async for ch in chunks:
            if ch.usage:
                usage = ch.usage
            if ch.content is None and ch.finish_reason is None and not ch.tool_calls:
                continue
            yield _sse(ch.to_openai(cid, model, created, first=first))
            first = False
        if first:  # empty completion: still send a role chunk
            yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": "stop"}]})
        if include_usage and usage:
            yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [], "usage": usage.to_dict()})
    except SwitchboardError as e:  # mid-stream failure: surface it in-band, then close cleanly
        yield _sse(e.to_body())
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    gw = gateway(request)
    key = authenticate(request)
    body = await read_json(request)
    ctx = {"cache_bypass": request.headers.get("x-cache", "").lower() == "bypass"}
    rid = request.headers.get("x-request-id")
    if rid:
        ctx["request_id"] = rid[:64]
    req = ChatRequest.from_openai(body, **ctx)
    if req.stream:
        meta, chunks = await gw.handle_stream(req, key)
        return StreamingResponse(sse_stream(meta, chunks, req.include_usage), media_type="text/event-stream",
                                 headers={**meta.headers(with_cost=False), "Cache-Control": "no-cache"})
    resp, meta = await gw.handle(req, key)
    return JSONResponse(resp.to_openai(meta.model or req.model), headers=meta.headers())


@router.get("/v1/models")
async def list_models(request: Request):
    gw = gateway(request)
    reg = gw.registry
    data = [{"id": "auto", "object": "model", "created": 0, "owned_by": "switchboard"}]
    for m in reg.enabled_models():
        data.append({"id": m.id, "object": "model", "created": 0, "owned_by": m.provider, "tier": m.tier,
                     "context_window": m.context_window})
    return JSONResponse({"object": "list", "data": data})
