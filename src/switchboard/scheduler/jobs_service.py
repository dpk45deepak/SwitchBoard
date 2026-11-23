"""Job-level operations used by the HTTP layer: parsing, creation, status (with cost), results."""
from __future__ import annotations

import json
from collections.abc import Iterator

from ..errors import BadRequest
from ..ledger.usage import Ledger
from .queue import RANKS, TaskQueue

MAX_TASKS = 100_000


def parse_jsonl(text: str) -> list[dict]:
    """Each line: a chat body, or {"custom_id": ..., "body": {...}} (OpenAI batch style)."""
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise BadRequest(f"line {n}: invalid JSON ({e.msg})") from e
        out.append(obj)
    return out


class JobsService:
    def __init__(self, queue: TaskQueue, ledger: Ledger) -> None:
        self.queue, self.ledger = queue, ledger

    def create(self, api_key_id: str, requests: list[dict], priority: str = "batch", model: str | None = None,
               max_concurrency: int | None = None, idempotency_key: str | None = None,
               callback_url: str | None = None, metadata: dict | None = None) -> tuple[dict, bool]:
        if not requests:
            raise BadRequest("a job needs at least one request")
        if len(requests) > MAX_TASKS:
            raise BadRequest(f"too many tasks (max {MAX_TASKS})")
        if priority not in RANKS:
            raise BadRequest(f"priority must be one of {sorted(RANKS)}")
        if callback_url and not callback_url.startswith(("http://", "https://")):
            raise BadRequest("callback_url must be http(s)")
        for i, r in enumerate(requests):
            body = r.get("body", r) if isinstance(r, dict) else None
            if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
                raise BadRequest(f"request {i}: 'messages' is required")
        jid, created = self.queue.create_job(api_key_id, requests, priority, model, max_concurrency,
                                             idempotency_key, callback_url, metadata)
        return self.status(jid), created  # type: ignore[return-value]

    def status(self, job_id: str) -> dict | None:
        st = self.queue.job_status(job_id)
        if st is None:
            return None
        c = self.ledger.job_cost(job_id)
        st["cost"] = {"billed_usd": c.get("cost", 0.0), "equivalent_usd": c.get("equiv", 0.0),
                      "saved_usd": c.get("saved", 0.0), "cache_hits": int(c.get("hits", 0))}
        return st

    def results_jsonl(self, job_id: str) -> Iterator[str]:
        for r in self.queue.results(job_id):
            yield json.dumps(r) + "\n"
