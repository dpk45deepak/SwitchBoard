"""Admission control: bounded waiting on top of a RateLimiter."""
from __future__ import annotations

import random

from ..clock import Clock
from ..errors import AdmissionRejected
from .token_bucket import RateLimiter


async def admit(limiter: RateLimiter, key: str, lane: str, clock: Clock, max_wait: float,
                requests: float = 1, tokens: float = 0, rng: random.Random | None = None) -> float:
    """Wait (bounded) until capacity is available. Returns seconds waited.

    Raises AdmissionRejected(retry_after=...) when the needed wait would exceed ``max_wait``.
    """
    rng = rng or random
    waited = 0.0
    while True:
        d = await limiter.acquire(key, lane, requests, tokens)
        if d.allowed:
            return waited
        if waited + d.retry_after > max_wait:
            raise AdmissionRejected(f"capacity exhausted for {key}", retry_after=max(d.retry_after, 0.001))
        nap = max(0.002, d.retry_after) * (1.0 + 0.1 * rng.random())
        await clock.sleep(nap)
        waited += nap
