"""Async batch worker. Every task runs through the SAME Gateway pipeline as online requests."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import urllib.request
import uuid

from ..clock import Clock
from ..errors import AdmissionRejected, Deferred, SwitchboardError
from ..gateway import Gateway
from ..types import ChatRequest
from .queue import Task, TaskQueue
from .retry import backoff_delay, is_retryable

log = logging.getLogger("switchboard.worker")


def post_callback(url: str, payload: dict) -> None:
    if not url.startswith(("http://", "https://")):
        return
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10).read()  # noqa: S310 (scheme checked above)
    except Exception as e:  # webhooks are best-effort
        log.warning("callback failed: %s", e)


class Worker:
    def __init__(self, queue: TaskQueue, gateway: Gateway, clock: Clock | None = None, concurrency: int = 16,
                 lease_s: float = 30.0, max_attempts: int = 5, owner: str | None = None,
                 rng: random.Random | None = None, poll_s: float = 0.05, backoff_base: float = 1.0,
                 backoff_cap: float = 60.0) -> None:
        self.queue, self.gateway = queue, gateway
        self.clock = clock or gateway.clock
        self.concurrency, self.lease_s, self.max_attempts = concurrency, lease_s, max_attempts
        self.owner = owner or f"w-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.rng = rng or random.Random()
        self.poll_s, self.backoff_base, self.backoff_cap = poll_s, backoff_base, backoff_cap
        self._inflight: dict[int, asyncio.Task] = {}
        self._stop = asyncio.Event()
        self._pause_until = 0.0  # claim pause after a deferral (breaker open / no capacity)
        self.stats = {"done": 0, "retried": 0, "dead": 0, "deferred": 0}

    def stop(self) -> None:
        self._stop.set()

    async def run(self, until_idle: bool = False, grace_s: float = 10.0) -> None:
        hb = asyncio.create_task(self._heartbeat())
        try:
            while not self._stop.is_set():
                claimed: list[Task] = []
                free = self.concurrency - len(self._inflight)
                if self.clock.now() < self._pause_until:
                    free = 0  # paused: do not churn through the queue while providers are blocked
                if free > 0:
                    claimed = self.queue.claim(self.owner, free, self.lease_s, self.max_attempts)
                    for t in claimed:
                        task = asyncio.create_task(self._run_task(t))
                        self._inflight[t.id] = task
                        task.add_done_callback(lambda _f, tid=t.id: self._inflight.pop(tid, None))
                if until_idle and not self._inflight and not claimed and self.queue.is_idle():
                    return
                await (asyncio.sleep(0) if claimed else self.clock.sleep(self.poll_s))
                if claimed:
                    await asyncio.sleep(0.001)
        finally:
            hb.cancel()
            await self._shutdown(grace_s)

    async def _shutdown(self, grace_s: float) -> None:
        """Finish in-flight work within ``grace_s``; release whatever is still running."""
        if self._inflight:
            _, pending = await asyncio.wait(list(self._inflight.values()), timeout=grace_s)
            for p in pending:
                p.cancel()  # _run_task releases its lease on cancellation
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(max(0.05, self.lease_s / 3))
            self.queue.extend(self.owner, list(self._inflight), self.lease_s)

    async def _run_task(self, t: Task) -> None:
        try:
            body = dict(t.payload.get("body", t.payload)) if isinstance(t.payload, dict) else {}
            if t.model:
                body["model"] = t.model
            body.setdefault("model", "auto")
            body["stream"] = False
            req = ChatRequest.from_openai(body, path="batch", lane=t.priority, job_id=t.job_id,
                                          task_key=f"{t.job_id}:{t.idx}", retry_count=t.attempts - 1)
            resp, meta = await self.gateway.handle(req, self.gateway.keys.get(t.api_key_id))  # type: ignore[attr-defined]
            out = resp.to_openai(meta.model or body["model"])
            out["switchboard"] = {"cache": meta.cache_status, "provider": meta.provider,
                                  "cost_equiv_usd": meta.cost_equiv, "cost_saved_usd": meta.cost_saved}
            if self.queue.complete(t.id, self.owner, out):
                self.stats["done"] += 1
            await self._maybe_callback(t.job_id)
        except asyncio.CancelledError:
            self.queue.release(t.id, self.owner)
            raise
        except (Deferred, AdmissionRejected) as e:
            self.stats["deferred"] += 1
            wait = (e.retry_after or 1.0) * (1 + 0.2 * self.rng.random())
            self._pause_until = max(self._pause_until, self.clock.now() + min(wait, 2.0))
            self.queue.release(t.id, self.owner, delay=wait)
        except SwitchboardError as e:
            if is_retryable(e) and t.attempts < self.max_attempts:
                self.stats["retried"] += 1
                delay = backoff_delay(t.attempts, self.backoff_base, self.backoff_cap, self.rng, e.retry_after)
                self.queue.retry(t.id, self.owner, delay, f"{e.code}: {e.message}")
            else:
                self.stats["dead"] += 1
                self.queue.dead(t.id, self.owner, f"{e.code}: {e.message}")
                await self._maybe_callback(t.job_id)
        except Exception as e:  # unexpected bug: bounded retries, then DLQ
            log.exception("task %s crashed", t.id)
            if t.attempts < self.max_attempts:
                self.stats["retried"] += 1
                self.queue.retry(t.id, self.owner, backoff_delay(t.attempts, 1.0, 30.0, self.rng), repr(e))
            else:
                self.stats["dead"] += 1
                self.queue.dead(t.id, self.owner, repr(e))

    async def _maybe_callback(self, job_id: str) -> None:
        cb = self.queue.claim_callback(job_id)
        if cb:
            await asyncio.to_thread(post_callback, cb["url"], cb["payload"])
