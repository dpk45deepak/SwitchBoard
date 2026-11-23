"""Admin endpoints (usage, savings, keys, cache, DLQ, routing log, breakers)."""
from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ..errors import BadRequest
from ..scheduler import dlq
from .deps import gateway, read_json, require_admin

router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])
public = APIRouter()


@router.get("/usage")
async def usage(request: Request, group_by: str = "model", since_seconds: float | None = None):
    since = time.time() - since_seconds if since_seconds else None
    try:
        return JSONResponse({"group_by": group_by, "data": gateway(request).ledger.usage(group_by, since)})
    except ValueError as e:
        raise BadRequest(str(e)) from e


@router.get("/savings")
async def savings(request: Request, since_seconds: float | None = None):
    since = time.time() - since_seconds if since_seconds else None
    d = gateway(request).ledger.savings(since)
    d["definition"] = "saved = (same tokens at the strongest model, no cache) - (equivalent cost actually incurred)"
    return JSONResponse(d)


@router.get("/stats")
async def stats(request: Request, window_seconds: float = 900, bucket_seconds: float = 10):
    gw = gateway(request)
    q = request.app.state.queue
    return JSONResponse({"timeseries": gw.ledger.timeseries(bucket_seconds, time.time() - window_seconds),
                         "breakers": gw.breaker_states(), "queue": q.depth(),
                         "cache": {**gw.cache.stats, "entries": len(gw.cache), "hit_rate": gw.cache.hit_rate()},
                         "savings": gw.ledger.savings(time.time() - window_seconds)})


@router.post("/keys")
async def create_key(request: Request):
    b = await read_json(request)
    token, key = gateway(request).keys.create(b.get("name", ""), b.get("rpm"), b.get("tpm"),
                                              b.get("daily_budget_usd"), b.get("soft_ratio", 0.8),
                                              bool(b.get("downgrade", False)))
    return JSONResponse({"id": key.id, "api_key": token, "note": "shown once; only a hash is stored"})


@router.get("/keys")
async def list_keys(request: Request):
    return JSONResponse({"data": gateway(request).keys.list()})


@router.post("/cache/invalidate")
async def invalidate(request: Request):
    b = await read_json(request)
    n = gateway(request).cache.invalidate(namespace=b.get("namespace"), family=b.get("model_family"))
    return JSONResponse({"invalidated": n})


@router.get("/dlq")
async def dlq_list(request: Request, job_id: str | None = None):
    return JSONResponse({"data": dlq.inspect(request.app.state.queue, job_id)})


@router.post("/dlq/requeue")
async def dlq_requeue(request: Request):
    b = await read_json(request)
    n = dlq.requeue(request.app.state.queue, b.get("job_id"), b.get("task_ids"))
    return JSONResponse({"requeued": n})


@router.get("/routing")
async def routing(request: Request, limit: int = 100):
    return JSONResponse({"data": list(gateway(request).router.log)[-limit:]})


@router.get("/breakers")
async def breakers(request: Request):
    return JSONResponse(gateway(request).breaker_states())


@public.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse((Path(__file__).resolve().parent.parent / "dashboard.html").read_text())
