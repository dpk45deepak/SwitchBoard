"""Retry policy: exponential backoff with full jitter; retryable vs permanent errors."""
from __future__ import annotations

import random

from ..errors import SwitchboardError


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 60.0, rng: random.Random | None = None,
                  retry_after: float | None = None) -> float:
    """Full jitter: uniform(0, min(cap, base * 2**attempt)); never shorter than a server Retry-After."""
    r = rng or random
    ceiling = min(cap, base * (2 ** max(0, attempt)))
    delay = r.uniform(0.0, ceiling)
    if retry_after:
        delay = max(delay, retry_after)
    return delay


def is_retryable(exc: BaseException) -> bool:
    return bool(getattr(exc, "retryable", False)) and isinstance(exc, SwitchboardError)
