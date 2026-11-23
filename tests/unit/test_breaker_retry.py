import random
import unittest

from switchboard.clock import FakeClock
from switchboard.errors import BadRequest, ProviderUnavailable, RateLimited
from switchboard.limits.circuit_breaker import CLOSED, HALF_OPEN, OPEN, CircuitBreaker
from switchboard.scheduler.retry import backoff_delay, is_retryable


class TestBreaker(unittest.TestCase):
    def mk(self, **kw):
        c = FakeClock()
        return c, CircuitBreaker(c, base_cooldown=5, **kw)

    def test_opens_on_consecutive_failures(self):
        c, b = self.mk(consecutive_threshold=3)
        for _ in range(3):
            self.assertTrue(b.allow())
            b.record_failure()
        self.assertEqual(b.state, OPEN)
        self.assertFalse(b.allow())

    def test_opens_on_error_rate(self):
        c, b = self.mk(consecutive_threshold=99, min_calls=10, error_rate_threshold=0.5)
        for i in range(10):
            (b.record_failure if i % 2 else b.record_success)()
        self.assertEqual(b.state, OPEN)

    def test_half_open_probe_success_closes(self):
        c, b = self.mk(consecutive_threshold=2)
        b.record_failure(); b.record_failure()
        c.advance(5.1)
        self.assertEqual(b.state, HALF_OPEN)
        self.assertTrue(b.allow())
        self.assertFalse(b.allow())  # only one probe at a time
        b.record_success()
        self.assertEqual(b.state, CLOSED)

    def test_probe_failure_reopens_with_doubled_cooldown(self):
        c, b = self.mk(consecutive_threshold=2)
        b.record_failure(); b.record_failure()
        self.assertAlmostEqual(b.retry_after(), 5, places=1)
        c.advance(5.1); b.allow(); b.record_failure()
        self.assertEqual(b.state, OPEN)
        self.assertAlmostEqual(b.retry_after(), 10, places=1)

    def test_cooldown_is_capped(self):
        c, b = self.mk(consecutive_threshold=1, max_cooldown=12)
        for _ in range(6):
            b.record_failure()
            c.advance(b.retry_after() + 0.01)
            b.allow()
        b.record_failure()
        self.assertLessEqual(b.retry_after(), 12)


class TestRetry(unittest.TestCase):
    def test_full_jitter_bounds_and_growth(self):
        rng = random.Random(1)
        for attempt in range(8):
            ceil = min(60, 2 ** attempt)
            ds = [backoff_delay(attempt, 1.0, 60.0, rng) for _ in range(200)]
            self.assertTrue(all(0 <= d <= ceil for d in ds))
        self.assertGreater(max(backoff_delay(5, 1.0, 60.0, rng) for _ in range(200)), 16)

    def test_retry_after_is_a_floor(self):
        self.assertGreaterEqual(backoff_delay(0, 1.0, 60.0, random.Random(0), retry_after=7), 7)

    def test_classification(self):
        self.assertTrue(is_retryable(RateLimited("x")))
        self.assertTrue(is_retryable(ProviderUnavailable("x")))
        self.assertFalse(is_retryable(BadRequest("x")))
        self.assertFalse(is_retryable(ValueError("x")))


if __name__ == "__main__":
    unittest.main()
