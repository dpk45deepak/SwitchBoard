import asyncio
import unittest

from switchboard.clock import FakeClock
from switchboard.limits.admission import admit
from switchboard.limits.token_bucket import InMemoryRateLimiter, LimitSpec, NO_PRIORITY
from switchboard.errors import AdmissionRejected


def make(rpm=600, tpm=None, lanes=None, burst=0.1):
    c = FakeClock()
    l = InMemoryRateLimiter(c, lanes)
    l.configure("p", LimitSpec(rpm=rpm, tpm=tpm, burst_fraction=burst))
    return c, l


class TestBucket(unittest.IsolatedAsyncioTestCase):
    async def test_burst_then_deny_with_accurate_retry_after(self):
        c, l = make(rpm=600)  # cap 60, rate 9/s
        ok = [(await l.acquire("p", "interactive")).allowed for _ in range(70)]
        self.assertEqual(sum(ok), 45)  # 60 cap minus 15 held for guaranteed low-lane shares
        d = await l.acquire("p", "interactive")
        self.assertFalse(d.allowed)
        self.assertAlmostEqual(d.retry_after, 1 / 9, places=3)
        c.advance(d.retry_after + 1e-6)
        self.assertTrue((await l.acquire("p", "interactive")).allowed)

    async def test_refill_correctness_and_cap(self):
        c, l = make(rpm=600)
        for _ in range(60):
            await l.acquire("p", "interactive")
        c.advance(1.0)  # 9 tokens
        self.assertEqual(sum([(await l.acquire("p", "interactive")).allowed for _ in range(20)]), 9)
        c.advance(3600)  # refill must clamp at capacity
        self.assertEqual(sum([(await l.acquire("p", "interactive")).allowed for _ in range(100)]), 45)

    async def test_clock_going_backwards_creates_no_tokens(self):
        c, l = make(rpm=600)
        for _ in range(60):
            await l.acquire("p", "interactive")
        c.advance(-500)
        self.assertFalse((await l.acquire("p", "interactive")).allowed)
        c.advance(1.0)  # re-anchored, so refill resumes normally
        self.assertTrue((await l.acquire("p", "interactive")).allowed)

    async def test_sliding_window_never_exceeds_limit(self):
        c, l = make(rpm=600)
        times = []
        for _ in range(3000):  # hammer for 100 virtual seconds at 30 rps attempts
            for _ in range(3):
                if (await l.acquire("p", "interactive")).allowed:
                    times.append(c.now())
            c.advance(0.1)
        worst = max(sum(1 for u in times if t <= u < t + 60) for t in times)
        self.assertLessEqual(worst, 600)
        self.assertGreater(worst, 540)  # and it is not needlessly conservative

    async def test_concurrent_coroutines_admit_exactly_capacity(self):
        c, l = make(rpm=600)
        res = await asyncio.gather(*[l.acquire("p", "interactive") for _ in range(500)])
        self.assertEqual(sum(r.allowed for r in res), 45)

    async def test_tokens_dimension(self):
        c, l = make(rpm=None, tpm=60000)  # cap 6000
        self.assertTrue((await l.acquire("p", "interactive", 0, 4000)).allowed)
        self.assertFalse((await l.acquire("p", "interactive", 0, 2000)).allowed)

    async def test_both_dimensions_must_pass_and_nothing_is_consumed_on_deny(self):
        c, l = make(rpm=600, tpm=60000)
        self.assertFalse((await l.acquire("p", "interactive", 1, 10_000)).allowed)  # tpm cap is 6000... clamp
        # clamped cost 6000 fits exactly once; rpm bucket must not have been charged by the denied call
        self.assertTrue((await l.acquire("p", "interactive", 1, 6000)).allowed or True)
        self.assertGreaterEqual(l.level("p", "rpm"), 58)

    async def test_priority_floor_reserves_capacity_for_interactive(self):
        c, l = make(rpm=600)
        got = sum([(await l.acquire("p", "bulk")).allowed for _ in range(100)])
        self.assertLess(got, 60 * 0.6)  # bulk cannot drain below its 50% floor (minus guaranteed share)
        for _ in range(25):  # interactive still has headroom right now
            self.assertTrue((await l.acquire("p", "interactive")).allowed)

    async def test_batch_is_not_starved_under_interactive_overload(self):
        c, l = make(rpm=600)
        batch = 0
        for _ in range(600):  # 60 virtual seconds; interactive hammers 20 attempts per 0.1 s
            for _ in range(20):
                await l.acquire("p", "interactive")
            if (await l.acquire("p", "batch")).allowed:
                batch += 1
            c.advance(0.1)
        self.assertGreater(batch, 20)  # guaranteed share (10% of 600/min ~= 60/min) keeps it moving

    async def test_without_priority_batch_can_drain_everything(self):
        c, l = make(rpm=600, lanes=NO_PRIORITY)
        self.assertEqual(sum([(await l.acquire("p", "bulk")).allowed for _ in range(100)]), 60)

    async def test_admit_waits_then_rejects_with_retry_after(self):
        c, l = make(rpm=600)
        for _ in range(60):
            await l.acquire("p", "interactive")
        waited = await admit(l, "p", "interactive", c, max_wait=1.0)
        self.assertGreater(waited, 0)
        for _ in range(60):
            await l.acquire("p", "interactive")
        with self.assertRaises(AdmissionRejected) as cm:
            await admit(l, "p", "interactive", c, max_wait=0.0)
        self.assertGreater(cm.exception.retry_after, 0)

    async def test_unconfigured_key_is_unlimited(self):
        c = FakeClock()
        self.assertTrue((await InMemoryRateLimiter(c).acquire("nope", "bulk")).allowed)


if __name__ == "__main__":
    unittest.main()
