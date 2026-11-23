"""Batch jobs API: create (JSON or NDJSON), status, JSONL results, cancel, list."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..errors import BadRequest, ModelNotFound
from ..scheduler.jobs_service import parse_jsonl
from .deps import authenticate, gateway, read_json

router = APIRouter()


@router.post("/v1/jobs")
async def create_job(request: Request):
    gw = gateway(request)
    key = authenticate(request)
    svc = request.app.state.jobs
    ctype = request.headers.get("content-type", "")
    if "ndjson" in ctype or "jsonl" in ctype:  # JSONL upload; options come from query/headers
        q = request.query_params
        raw = (await request.body()).decode("utf-8", "replace")
        opts = {"priority": q.get("priority", "batch"), "model": q.get("model"),
                "max_concurrency": int(q["max_concurrency"]) if q.get("max_concurrency") else None,
                "idempotency_key": request.headers.get("idempotency-key") or q.get("idempotency_key"),
                "callback_url": q.get("callback_url")}
        reqs = parse_jsonl(raw)
    else:
        body = await read_json(request)
        reqs = body.get("requests")
        if not isinstance(reqs, list):
            raise BadRequest("'requests' must be an array of chat requests")
        opts = {k: body.get(k) for k in ("priority", "model", "max_concurrency", "idempotency_key",
                                         "callback_url", "metadata")}
        opts["priority"] = opts["priority"] or "batch"
        opts["idempotency_key"] = opts["idempotency_key"] or request.headers.get("idempotency-key")
    if opts.get("model") and opts["model"].lower() != "auto":
        try:
            gw.registry.resolve(opts["model"], gw.settings.allow_substitution)
        except ModelNotFound:
            raise
    st, created = svc.create(key.id, reqs, **opts)
    return JSONResponse(st, status_code=202 if created else 200)


@router.get("/v1/jobs")
async def list_jobs(request: Request):
    authenticate(request)
    return JSONResponse({"object": "list", "data": request.app.state.queue.list_jobs()})


@router.get("/v1/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    authenticate(request)
    st = request.app.state.jobs.status(job_id)
    if st is None:
        raise ModelNotFound("job not found")
    return JSONResponse(st)


@router.get("/v1/jobs/{job_id}/results")
async def job_results(job_id: str, request: Request):
    authenticate(request)
    if request.app.state.jobs.status(job_id) is None:
        raise ModelNotFound("job not found")
    return StreamingResponse(request.app.state.jobs.results_jsonl(job_id), media_type="application/x-ndjson")


@router.post("/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request):
    authenticate(request)
    if not request.app.state.queue.cancel(job_id):
        raise ModelNotFound("job not found")
    return JSONResponse(request.app.state.jobs.status(job_id))
