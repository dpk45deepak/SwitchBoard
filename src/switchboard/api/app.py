"""FastAPI application factory."""
from __future__ import annotations

import asyncio
import logging
import math
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..config import Settings
from ..errors import SwitchboardError
from ..gateway import Gateway, build_gateway
from ..observability.logging import request_id_var, setup_logging
from ..scheduler.jobs_service import JobsService
from ..scheduler.queue import TaskQueue
from ..scheduler.worker import Worker
from . import admin, chat, health, jobs

log = logging.getLogger("switchboard.api")


def create_app(settings: Settings | None = None, gateway: Gateway | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    setup_logging(settings.log_level, settings.log_redact_pii)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        gw = gateway or build_gateway(settings)
        app.state.gateway = gw
        app.state.queue = TaskQueue(gw.db, gw.clock)  # type: ignore[attr-defined]
        app.state.jobs = JobsService(app.state.queue, gw.ledger)
        worker_task = None
        if gw.settings.embedded_worker:
            worker = Worker(app.state.queue, gw)
            worker_task = asyncio.create_task(worker.run())
        yield
        if worker_task:
            worker.stop()
            await asyncio.wait_for(worker_task, 15)

    app = FastAPI(title="Switchboard", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(SwitchboardError)
    async def handle_err(_: Request, exc: SwitchboardError):
        headers = {"Retry-After": str(max(1, math.ceil(exc.retry_after)))} if exc.retry_after else {}
        return JSONResponse(exc.to_body(), status_code=exc.status, headers=headers)

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("x-request-id", "")[:64]
        if rid:
            request_id_var.set(rid)
        return await call_next(request)

    for r in (chat.router, jobs.router, admin.router, admin.public, health.router):
        app.include_router(r)
    return app
