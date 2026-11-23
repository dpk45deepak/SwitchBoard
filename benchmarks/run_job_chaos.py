"""5,000-task job against a mock provider with injected 429s and 5xx/timeouts; verifies no lost / duplicated tasks.

Usage: PYTHONPATH=src:benchmarks python benchmarks/run_job_chaos.py
The provider hard-limits 600 calls per 2 s window (300/s) (the gateway limiter is deliberately set looser, so the
provider answers real 429s) and fails 10% of other calls (80% 5xx, 20% timeouts). Two workers share the queue.
"""
from __future__ import annotations

import asyncio
import time

from common import make_gateway, save
from switchboard.limits.token_bucket import LimitSpec
from switchboard.scheduler.jobs_service import JobsService
from switchboard.scheduler.queue import TaskQueue
from switchboard.scheduler.worker import Worker

N = 5000


async def main() -> None:
    gw = make_gateway(cache=False, rpm=600, window_s=2.0, behavior_kw={"error_rate": 0.10})
    for k in gw.providers:
        gw.limiter.configure(k, LimitSpec(rpm=60000))  # looser than the provider => real provider 429s
        gw.breakers[k].base_cooldown = 0.5
    q = TaskQueue(gw.db, gw.clock)
    svc = JobsService(q, gw.ledger)
    reqs = [{"model": "mock-balanced", "temperature": 0, "messages": [{"role": "user", "content": f"chaos task {i}"}]}
            for i in range(N)]
    job, _ = svc.create("anonymous", reqs, priority="batch")
    workers = [Worker(q, gw, concurrency=32, poll_s=0.005, lease_s=60, max_attempts=30, backoff_base=0.05,
                      backoff_cap=1.0) for _ in range(2)]
    t0 = time.perf_counter()
    await asyncio.gather(*[w.run(until_idle=True) for w in workers])
    dt = time.perf_counter() - t0
    c = q.counts(job["id"])
    rows = gw.db.query("SELECT task_key, COUNT(*) n FROM usage WHERE job_id=? AND outcome='ok' GROUP BY task_key", (job["id"],))
    res = list(q.results(job["id"]))
    out = {"tasks": N, "seconds": dt, "counts": c, "distinct_charged_tasks": len(rows),
           "tasks_charged_more_than_once": sum(1 for r in rows if r["n"] > 1),
           "results_returned": len(res), "results_succeeded": sum(r["status"] == "succeeded" for r in res),
           "provider_429s": sum(p.n_429 for p in gw.providers.values()),
           "provider_injected_errors": sum(p.n_errors for p in gw.providers.values()),
           "provider_calls": sum(p.n_calls for p in gw.providers.values()),
           "tasks_needing_more_than_one_attempt": gw.db.query("SELECT COUNT(*) n FROM tasks WHERE attempts>1")[0]["n"],
           "worker_stats": [w.stats for w in workers],
           "no_lost_no_duplicated": c["done"] == N and len(rows) == N and all(r["n"] == 1 for r in rows)}
    save("job_chaos", out)
    print(out)


if __name__ == "__main__":
    asyncio.run(main())
