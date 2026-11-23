from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from .deps import gateway

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request):
    gw = gateway(request)
    try:
        gw.db.one("SELECT 1")
        ok = bool(gw.registry.enabled_models())
    except Exception:
        ok = False
    return JSONResponse({"status": "ready" if ok else "not_ready", "providers": sorted(gw.providers)},
                        status_code=200 if ok else 503)


@router.get("/metrics")
async def metrics(request: Request):
    return PlainTextResponse(gateway(request).metrics.render(), media_type="text/plain; version=0.0.4")
