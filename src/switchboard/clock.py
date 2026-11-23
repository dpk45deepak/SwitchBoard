"""Injectable clock so limiter/breaker/scheduler tests never sleep."""
from __future__ import annotations

import asyncio
import time


class Clock:
    def now(self) -> float:
        return time.time()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))


class FakeClock(Clock):
    """Virtual time. sleep() advances time instantly (single-flow tests)."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.t = start

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt

    async def sleep(self, seconds: float) -> None:
        self.t += max(0.0, seconds)
        await asyncio.sleep(0)
