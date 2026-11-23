import asyncio
import unittest

from switchboard.clock import Clock
from switchboard.errors import AdmissionRejected, BadRequest, BudgetExceeded, ProviderUnavailable
from switchboard.limits.token_bucket import LimitSpec
from switchboard.providers.mock import MockBehavior
from tests.helpers import chat, mk_gateway


class TestOnline(unittest.IsolatedAsyncioTestCase):
    async def test_miss_then_exact_hit_with_cost_saved(self):
        gw = mk_gateway()
        r1, m1 = await gw.handle(chat())
        r2, m2 = await gw.handle(chat())
        self.assertEqual((m1.cache_status, m2.cache_status), ("MISS", "HIT-EXACT"))
        self.assertEqual(r1.content, r2.content)
        self.assertGreater(m2.cost_saved, 0)
        self.assertEqual(m2.cost_equiv, 0)
        self.assertIn("X-Cache", m2.headers())

    async def test_bypass_and_high_temperature_are_not_cached(self):
        gw = mk_gateway()
        await gw.handle(chat())
        r = chat(); r.cache_bypass = True
        self.assertEqual((await gw.handle(r))[1].cache_status, "MISS")
        self.assertEqual((await gw.handle(chat(temperature=1.0)))[1].cache_status, "MISS")
        self.assertEqual((await gw.handle(chat(temperature=1.0)))[1].cache_status, "MISS")

    async def test_failover_on_outage_and_breaker_opens(self):
        gw = mk_gateway(behavior=MockBehavior(outage=True))
        for i in range(8):
            resp, meta = await gw.handle(chat(f"question {i}", temperature=1.0))
            self.assertEqual(meta.provider, "mock2")  # served by the fallback every time
        self.assertEqual(gw.breakers["mock"].state, "open")
        self.assertTrue(any("breaker" in s for s in meta.skipped))

    async def test_all_providers_down_raises_503_semantics(self):
        gw = mk_gateway(behavior=MockBehavior(outage=True), behavior2=MockBehavior(outage=True))
        with self.assertRaises(ProviderUnavailable):
            await gw.handle(chat(temperature=1.0))

    async def test_provider_limit_rejects_online_with_retry_after(self):
        gw = mk_gateway(online_max_wait_s=0.0)
        for k in ("mock", "mock2"):
            gw.limiter.configure(k, LimitSpec(rpm=60))  # cap 6
        outcomes = []
        for i in range(30):
            try:
                await gw.handle(chat(f"q{i}", temperature=1.0)); outcomes.append("ok")
            except AdmissionRejected as e:
                outcomes.append("429"); self.assertGreater(e.retry_after, 0)
        self.assertIn("429", outcomes)
        self.assertLessEqual(outcomes.count("ok"), 2 * 6)

    async def test_client_limit_and_budget(self):
        gw = mk_gateway()
        _, key = gw.keys.create("c", rpm=60)  # cap 6
        n_ok = 0
        for i in range(15):
            try:
                await gw.handle(chat(f"x{i}", temperature=1.0), key); n_ok += 1
            except AdmissionRejected:
                pass
        self.assertLessEqual(n_ok, 6)
        _, poor = gw.keys.create("poor", daily_budget_usd=1e-9)
        await gw.handle(chat("spend", temperature=1.0), poor)
        with self.assertRaises(BudgetExceeded):
            await gw.handle(chat("again", temperature=1.0), poor)
        # cache hits are still served when over budget
        _, poor2 = gw.keys.create("poor2", daily_budget_usd=1e-9)
        await gw.handle(chat("cached q"), poor2)
        self.assertEqual((await gw.handle(chat("cached q"), poor2))[1].cache_status, "HIT-EXACT")

    async def test_budget_downgrade_policy(self):
        gw = mk_gateway()
        _, key = gw.keys.create("d", daily_budget_usd=1e-9, downgrade=True)
        await gw.handle(chat("burn it", temperature=1.0), key)
        _, meta = await gw.handle(chat("Design a payment ledger; analyze trade-offs step by step; implement it.",
                                       model="auto", temperature=1.0), key)
        self.assertEqual(meta.tier, "cheap")
        self.assertEqual(meta.budget, "downgraded")

    async def test_single_flight_coalesces_identical_concurrent_requests(self):
        gw = mk_gateway(behavior=MockBehavior(latency_ms={"strong": 50, "cheap": 50, "balanced": 50}))
        res = await asyncio.gather(*[gw.handle(chat("same prompt")) for _ in range(10)])
        self.assertEqual(sum(1 for _, m in res if m.cache_status == "MISS"), 1)
        self.assertEqual(gw.providers["mock"].n_calls, 1)

    async def test_explicit_constraint_not_violated_by_failover(self):
        gw = mk_gateway()
        r = chat(tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}], model="mock-strong")
        resp, meta = await gw.handle(r)  # mock models support tools -> fine
        self.assertEqual(meta.provider, "mock")
        with self.assertRaises(BadRequest):
            chat(n=3)

    async def test_streaming_reassembles_and_records_ttft(self):
        gw = mk_gateway()
        meta, it = await gw.handle_stream(chat("stream me please", stream=True, temperature=1.0))
        chunks = [c async for c in it]
        text = "".join(c.content or "" for c in chunks)
        full, _ = await gw.handle(chat("stream me please", temperature=1.0))
        self.assertEqual(text, full.content)
        self.assertEqual(chunks[-1].finish_reason, "stop")
        row = gw.ledger.db.query("SELECT ttft_ms, outcome FROM usage WHERE ttft_ms IS NOT NULL")[0]
        self.assertEqual(row["outcome"], "ok")

    async def test_stream_cache_replay(self):
        gw = mk_gateway()
        await gw.handle(chat("replay"))
        meta, it = await gw.handle_stream(chat("replay", stream=True))
        text = "".join([c.content or "" async for c in it])
        self.assertEqual(meta.cache_status, "HIT-EXACT")
        self.assertTrue(text.startswith("ANS-"))

    async def test_mid_stream_failure_is_recorded_and_not_cached(self):
        gw = mk_gateway(behavior=MockBehavior(stream_fail_after=2))
        meta, it = await gw.handle_stream(chat("long one", stream=True))
        with self.assertRaises(Exception):
            async for _ in it:
                pass
        self.assertEqual(gw.ledger.db.query("SELECT outcome FROM usage")[-1]["outcome"], "stream_error")
        self.assertEqual(len(gw.cache), 0)

    async def test_client_disconnect_mid_stream_closes_upstream(self):
        gw = mk_gateway()
        meta, it = await gw.handle_stream(chat("hang up early", stream=True, temperature=1.0))
        async for _ in it:
            break
        await it.aclose()
        self.assertEqual(gw.ledger.db.query("SELECT outcome FROM usage")[-1]["outcome"], "client_disconnect")

    async def test_timeout_is_a_failover_trigger(self):
        gw = mk_gateway(behavior=MockBehavior(latency_ms={"strong": 500}), request_timeout_s=0.05)
        _, meta = await gw.handle(chat(temperature=1.0))
        self.assertEqual(meta.provider, "mock2")  # mock2 is fast; mock timed out

    async def test_prometheus_metrics_render(self):
        gw = mk_gateway()
        await gw.handle(chat())
        txt = gw.metrics.render()
        self.assertIn("switchboard_requests_total", txt)
        self.assertIn("switchboard_breaker_state", txt)


if __name__ == "__main__":
    unittest.main()
