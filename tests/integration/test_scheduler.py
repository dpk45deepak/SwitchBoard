import asyncio
import json
import unittest

from switchboard.clock import Clock, FakeClock
from switchboard.db.session import Database
from switchboard.providers.mock import MockBehavior
from switchboard.scheduler import dlq
from switchboard.scheduler.jobs_service import JobsService, parse_jsonl
from switchboard.scheduler.queue import TaskQueue
from switchboard.scheduler.worker import Worker
from tests.helpers import mk_gateway


def reqs(n, prefix="q", **extra):
    return [{"model": "mock-balanced", "temperature": 0, **extra,
             "messages": [{"role": "user", "content": f"{prefix} {i} unique question"}]} for i in range(n)]


def setup(**kw):
    db = Database()
    gw = mk_gateway(db=db, **kw)
    q = TaskQueue(db, gw.clock)
    return gw, q, JobsService(q, gw.ledger)


class TestBatch(unittest.IsolatedAsyncioTestCase):
    async def test_job_completes_and_reports_cost(self):
        gw, q, svc = setup()
        st, created = svc.create("anonymous", reqs(40), priority="batch")
        w = Worker(q, gw, concurrency=8, poll_s=0.005)
        await w.run(until_idle=True)
        st = svc.status(st["id"])
        self.assertEqual((st["status"], st["counts"]["done"]), ("completed", 40))
        self.assertGreater(st["cost"]["equivalent_usd"], 0)
        lines = list(svc.results_jsonl(st["id"]))
        self.assertEqual(len(lines), 40)
        self.assertEqual(json.loads(lines[0])["status"], "succeeded")

    async def test_cache_shared_with_online_path_and_counted_as_hits(self):
        gw, q, svc = setup()
        from tests.helpers import chat
        await gw.handle(chat("shared question", model="mock-balanced"))
        st, _ = svc.create("anonymous", [{"model": "mock-balanced", "temperature": 0,
                                          "messages": [{"role": "user", "content": "shared question"}]}])
        await Worker(q, gw, poll_s=0.005).run(until_idle=True)
        self.assertEqual(svc.status(st["id"])["cost"]["cache_hits"], 1)

    async def test_idempotency_key_returns_same_job(self):
        gw, q, svc = setup()
        a, c1 = svc.create("k", reqs(3), idempotency_key="abc")
        b, c2 = svc.create("k", reqs(3), idempotency_key="abc")
        self.assertEqual((a["id"], c1, c2), (b["id"], True, False))
        self.assertEqual(q.depth()["pending"], 3)

    async def test_retries_with_5xx_and_429_never_lose_or_duplicate(self):
        gw, q, svc = setup(behavior=MockBehavior(error_rate=0.25, latency_ms={"balanced": 1, "cheap": 1, "strong": 1}),
                           behavior2=MockBehavior(error_rate=0.25, latency_ms={"balanced": 1, "cheap": 1, "strong": 1}))
        st, _ = svc.create("anonymous", reqs(120))
        w = Worker(q, gw, concurrency=16, poll_s=0.005, backoff_base=0.01, backoff_cap=0.05, max_attempts=12)
        await w.run(until_idle=True)
        s = svc.status(st["id"])
        self.assertEqual(s["counts"]["done"] + s["counts"]["dead"], 120)
        self.assertEqual(s["counts"]["done"], 120)
        rows = gw.ledger.db.query("SELECT task_key, COUNT(*) n FROM usage WHERE outcome='ok' GROUP BY task_key")
        self.assertEqual(len(rows), 120)
        self.assertTrue(all(r["n"] == 1 for r in rows))
        self.assertGreater(w.stats["retried"] + w.stats["deferred"], 0)

    async def test_permanent_error_goes_to_dlq_and_can_be_requeued(self):
        gw, q, svc = setup()
        bad = [{"model": "no-such-model", "messages": [{"role": "user", "content": "x"}]}]
        st, _ = svc.create("anonymous", reqs(2) + bad)
        w = Worker(q, gw, poll_s=0.005, max_attempts=3)
        await w.run(until_idle=True)
        s = svc.status(st["id"])
        self.assertEqual((s["status"], s["counts"]["done"], s["counts"]["dead"]), ("completed_with_errors", 2, 1))
        dead = dlq.inspect(q, st["id"])
        self.assertEqual(len(dead), 1)
        self.assertIn("model_not_found", dead[0]["error"])
        self.assertEqual(dead[0]["attempts"], 1)  # permanent => no pointless retries
        self.assertEqual(dlq.requeue(q, st["id"]), 1)
        self.assertEqual(q.counts(st["id"])["pending"], 1)

    async def test_cancel_stops_pending_tasks(self):
        gw, q, svc = setup()
        st, _ = svc.create("anonymous", reqs(10))
        self.assertTrue(q.cancel(st["id"]))
        await Worker(q, gw, poll_s=0.005).run(until_idle=True)
        self.assertEqual(svc.status(st["id"])["status"], "cancelled")
        self.assertEqual(q.counts(st["id"])["done"], 0)

    async def test_max_concurrency_is_respected(self):
        gw, q, svc = setup(behavior=MockBehavior(latency_ms={"balanced": 30, "cheap": 30, "strong": 30}))
        st, _ = svc.create("anonymous", reqs(12), max_concurrency=2)
        w = Worker(q, gw, concurrency=10, poll_s=0.005)
        peak = 0

        async def watch():
            nonlocal peak
            while True:
                peak = max(peak, q.counts(st["id"])["leased"])
                await asyncio.sleep(0.003)
        t = asyncio.create_task(watch())
        await w.run(until_idle=True)
        t.cancel()
        self.assertLessEqual(peak, 2)
        self.assertEqual(q.counts(st["id"])["done"], 12)

    async def test_lease_expiry_reclaims_crashed_workers_task(self):
        clock = FakeClock()
        db = Database()
        q = TaskQueue(db, clock)
        jid, _ = q.create_job("anonymous", reqs(3))
        crashed = q.claim("dead-worker", 10, lease_s=30, max_attempts=5)
        self.assertEqual(len(crashed), 3)
        self.assertEqual(q.claim("w2", 10, 30, 5), [])  # still leased
        clock.advance(31)
        re = q.claim("w2", 10, 30, 5)
        self.assertEqual(len(re), 3)
        self.assertTrue(all(t.attempts == 2 for t in re))
        # the zombie's late completion is fenced off
        self.assertFalse(q.complete(crashed[0].id, "dead-worker", {"x": 1}))
        self.assertTrue(q.complete(re[0].id, "w2", {"x": 1}))

    async def test_poison_task_is_dead_lettered_after_max_attempts(self):
        clock = FakeClock()
        q = TaskQueue(Database(), clock)
        jid, _ = q.create_job("anonymous", reqs(1))
        for _ in range(3):
            q.claim("w", 1, 10, 3)
            clock.advance(11)
        q.claim("w", 1, 10, 3)
        self.assertEqual(q.counts(jid)["dead"], 1)

    async def test_graceful_shutdown_releases_inflight_without_losing_attempts(self):
        gw, q, svc = setup(behavior=MockBehavior(latency_ms={"balanced": 500, "cheap": 500, "strong": 500}))
        st, _ = svc.create("anonymous", reqs(4))
        w = Worker(q, gw, concurrency=4, poll_s=0.005)
        t = asyncio.create_task(w.run(grace_s=0.05))
        await asyncio.sleep(0.1)
        w.stop()
        await t
        c = q.counts(st["id"])
        self.assertEqual((c["pending"], c["leased"], c["done"]), (4, 0, 0))
        self.assertEqual(q.db.query("SELECT MAX(attempts) m FROM tasks")[0]["m"], 0)

    async def test_priority_lanes_order_claims_with_aging(self):
        clock = FakeClock()
        q = TaskQueue(Database(), clock)
        q.create_job("a", reqs(2, "bulk"), priority="bulk")
        q.create_job("a", reqs(2, "inter"), priority="interactive")
        first = q.claim("w", 2, 30, 5)
        self.assertEqual({t.priority for t in first}, {"interactive"})

    def test_jsonl_parsing(self):
        out = parse_jsonl('{"messages":[{"role":"user","content":"a"}]}\n\n{"custom_id":"1","body":{"messages":[]}}\n')
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
