"""Durable task queue on SQLite (WAL): leases, retries, dead letters, idempotent job creation.

Several worker processes may share one database file. A task is *claimed* by moving it to
``leased`` with an expiry; if the worker dies the lease expires and any worker re-claims it
(visibility timeout). Completion is only accepted from the current lease owner.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

from ..clock import Clock
from ..db.session import Database, row_dicts

RANKS = {"interactive": 0, "normal": 1, "batch": 2, "bulk": 3}
AGING_S = 30.0  # a bulk task is treated as ~90 s "older" than a batch task => cannot starve forever


@dataclass
class Task:
    id: int
    job_id: str
    idx: int
    payload: dict
    attempts: int
    priority: str
    model: str | None
    api_key_id: str


class TaskQueue:
    def __init__(self, db: Database, clock: Clock) -> None:
        self.db, self.clock = db, clock

    # -- jobs --------------------------------------------------------------------------------
    def create_job(self, api_key_id: str, requests: list[dict], priority: str = "batch", model: str | None = None,
                   max_concurrency: int | None = None, idempotency_key: str | None = None,
                   callback_url: str | None = None, metadata: dict | None = None) -> tuple[str, bool]:
        if priority not in RANKS:
            raise ValueError(f"priority must be one of {sorted(RANKS)}")
        now = self.clock.now()
        with self.db.tx() as c:
            if idempotency_key:
                row = c.execute("SELECT id FROM jobs WHERE api_key_id=? AND idempotency_key=?",
                                (api_key_id, idempotency_key)).fetchone()
                if row:
                    return row["id"], False
            jid = "job_" + uuid.uuid4().hex[:12]
            c.execute("INSERT INTO jobs (id,created,api_key_id,priority,model,max_concurrency,idempotency_key,"
                      "callback_url,metadata,total) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (jid, now, api_key_id, priority, model, max_concurrency, idempotency_key, callback_url,
                       json.dumps(metadata or {}), len(requests)))
            c.executemany("INSERT INTO tasks (job_id,idx,rank,payload,status,available_at) VALUES (?,?,?,?,?,?)",
                          [(jid, i, RANKS[priority], json.dumps(r), "pending", now) for i, r in enumerate(requests)])
        return jid, True

    def counts(self, job_id: str) -> dict[str, int]:
        out = {"pending": 0, "leased": 0, "done": 0, "dead": 0, "cancelled": 0}
        for r in self.db.query("SELECT status, COUNT(*) n FROM tasks WHERE job_id=? GROUP BY status", (job_id,)):
            out[r["status"]] = r["n"]
        return out

    def get_job(self, job_id: str) -> dict | None:
        r = self.db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
        return dict(r) if r else None

    def job_status(self, job_id: str) -> dict | None:
        job = self.get_job(job_id)
        if job is None:
            return None
        c = self.counts(job_id)
        active = c["pending"] + c["leased"]
        if job["cancelled"] and c["leased"] == 0:
            status = "cancelled"
        elif active == 0:
            status = "completed" if c["dead"] == 0 else "completed_with_errors"
        elif c["done"] + c["leased"] + c["dead"] == 0:
            status = "queued"
        else:
            status = "running"
        finished = c["done"] + c["dead"]
        elapsed = self.clock.now() - job["created"]
        eta = (elapsed / finished) * active if finished and active and not job["cancelled"] else None
        return {"id": job_id, "status": status, "created": job["created"], "priority": job["priority"],
                "total": job["total"], "counts": c, "eta_seconds": eta,
                "metadata": json.loads(job["metadata"] or "{}")}

    def list_jobs(self, limit: int = 50) -> list[dict]:
        ids = [r["id"] for r in self.db.query("SELECT id FROM jobs ORDER BY created DESC LIMIT ?", (limit,))]
        return [s for s in (self.job_status(i) for i in ids) if s]

    def cancel(self, job_id: str) -> bool:
        with self.db.tx() as c:
            if not c.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone():
                return False
            c.execute("UPDATE jobs SET cancelled=1 WHERE id=?", (job_id,))
            c.execute("UPDATE tasks SET status='cancelled' WHERE job_id=? AND status='pending'", (job_id,))
        return True

    def results(self, job_id: str):
        """Yield result dicts in index order (finished tasks only)."""
        last = -1
        while True:
            rows = self.db.query("SELECT idx,status,result,error,attempts FROM tasks WHERE job_id=? AND idx>? AND "
                                 "status IN ('done','dead','cancelled') ORDER BY idx LIMIT 500", (job_id, last))
            if not rows:
                return
            for r in rows:
                last = r["idx"]
                yield {"index": r["idx"], "status": {"done": "succeeded", "dead": "failed"}.get(r["status"], r["status"]),
                       "response": json.loads(r["result"]) if r["result"] else None, "error": r["error"],
                       "attempts": r["attempts"]}

    # -- claiming ----------------------------------------------------------------------------
    def claim(self, owner: str, limit: int, lease_s: float, max_attempts: int) -> list[Task]:
        now = self.clock.now()
        out: list[Task] = []
        with self.db.tx() as c:
            rows = c.execute(
                """SELECT t.id,t.job_id,t.idx,t.payload,t.attempts,t.status,j.priority,j.model,j.api_key_id,
                          j.max_concurrency FROM tasks t JOIN jobs j ON j.id=t.job_id
                   WHERE j.cancelled=0 AND ((t.status='pending' AND t.available_at<=?)
                                         OR (t.status='leased' AND t.lease_until<?))
                   ORDER BY t.available_at + t.rank*? , t.id LIMIT ?""", (now, now, AGING_S, limit * 8)).fetchall()
            leased = {r["job_id"]: r["n"] for r in c.execute(
                "SELECT job_id, COUNT(*) n FROM tasks WHERE status='leased' AND lease_until>=? GROUP BY job_id", (now,))}
            for r in rows:
                if len(out) >= limit:
                    break
                if r["status"] == "leased" and r["attempts"] >= max_attempts:  # poison task: crashed workers too
                    c.execute("UPDATE tasks SET status='dead', error='lease expired; max attempts reached', owner=NULL,"
                              " finished=? WHERE id=?", (now, r["id"]))
                    continue
                cap = r["max_concurrency"]
                if cap and leased.get(r["job_id"], 0) >= cap:
                    continue
                c.execute("UPDATE tasks SET status='leased', owner=?, lease_until=?, attempts=attempts+1 WHERE id=?",
                          (owner, now + lease_s, r["id"]))
                leased[r["job_id"]] = leased.get(r["job_id"], 0) + 1
                out.append(Task(r["id"], r["job_id"], r["idx"], json.loads(r["payload"]), r["attempts"] + 1,
                                r["priority"], r["model"], r["api_key_id"]))
        return out

    def extend(self, owner: str, task_ids: list[int], lease_s: float) -> None:
        if task_ids:
            q = ",".join("?" * len(task_ids))
            self.db.execute(f"UPDATE tasks SET lease_until=? WHERE owner=? AND status='leased' AND id IN ({q})",
                            [self.clock.now() + lease_s, owner, *task_ids])

    # -- outcomes (all fenced on owner + status='leased') -------------------------------------
    def _fenced(self, sql: str, params: list, task_id: int, owner: str) -> bool:
        cur = self.db.execute(sql + " WHERE id=? AND owner=? AND status='leased'", [*params, task_id, owner])
        return cur.rowcount == 1

    def complete(self, task_id: int, owner: str, response: dict) -> bool:
        return self._fenced("UPDATE tasks SET status='done', result=?, finished=?, owner=NULL, error=NULL",
                            [json.dumps(response), self.clock.now()], task_id, owner)

    def retry(self, task_id: int, owner: str, delay: float, error: str) -> bool:
        return self._fenced("UPDATE tasks SET status='pending', available_at=?, error=?, owner=NULL",
                            [self.clock.now() + delay, error[:500]], task_id, owner)

    def release(self, task_id: int, owner: str, delay: float = 0.0) -> bool:
        """Give the task back without consuming an attempt (deferral, shutdown)."""
        return self._fenced("UPDATE tasks SET status='pending', available_at=?, attempts=MAX(0,attempts-1), owner=NULL",
                            [self.clock.now() + delay], task_id, owner)

    def dead(self, task_id: int, owner: str, error: str) -> bool:
        return self._fenced("UPDATE tasks SET status='dead', error=?, finished=?, owner=NULL",
                            [error[:500], self.clock.now()], task_id, owner)

    # -- DLQ ---------------------------------------------------------------------------------
    def dead_tasks(self, job_id: str | None = None, limit: int = 100) -> list[dict]:
        sql = "SELECT id, job_id, idx, attempts, error, finished FROM tasks WHERE status='dead'"
        params: list = []
        if job_id:
            sql, params = sql + " AND job_id=?", [job_id]
        return row_dicts(self.db.query(sql + " ORDER BY finished DESC LIMIT ?", [*params, limit]))

    def requeue_dead(self, job_id: str | None = None, task_ids: list[int] | None = None) -> int:
        sql, params = "UPDATE tasks SET status='pending', attempts=0, available_at=?, error=NULL, finished=NULL WHERE status='dead'", [self.clock.now()]
        if job_id:
            sql, params = sql + " AND job_id=?", [*params, job_id]
        if task_ids:
            sql, params = sql + f" AND id IN ({','.join('?' * len(task_ids))})", [*params, *task_ids]
        return self.db.execute(sql, params).rowcount

    # -- misc --------------------------------------------------------------------------------
    def depth(self) -> dict[str, int]:
        out = {"pending": 0, "leased": 0, "dead": 0}
        for r in self.db.query("SELECT status, COUNT(*) n FROM tasks WHERE status IN ('pending','leased','dead') GROUP BY status"):
            out[r["status"]] = r["n"]
        return out

    def is_idle(self) -> bool:
        d = self.depth()
        return d["pending"] == 0 and d["leased"] == 0

    def claim_callback(self, job_id: str) -> dict | None:
        """Atomically mark the completion webhook as sent (once) if the job is terminal."""
        st = self.job_status(job_id)
        job = self.get_job(job_id)
        if not st or not job or not job["callback_url"] or job["callback_sent"]:
            return None
        if st["status"] not in ("completed", "completed_with_errors", "cancelled"):
            return None
        cur = self.db.execute("UPDATE jobs SET callback_sent=1 WHERE id=? AND callback_sent=0", (job_id,))
        return {"url": job["callback_url"], "payload": {"job_id": job_id, "status": st["status"], "counts": st["counts"]}} \
            if cur.rowcount == 1 else None
