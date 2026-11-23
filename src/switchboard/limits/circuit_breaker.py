"""Per-provider circuit breaker: closed -> open -> half-open, with exponential cool-down."""
from __future__ import annotations

from collections import deque

from ..clock import Clock

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


class CircuitBreaker:
    def __init__(self, clock: Clock | None = None, window: int = 20, min_calls: int = 10,
                 error_rate_threshold: float = 0.5, consecutive_threshold: int = 5,
                 base_cooldown: float = 5.0, max_cooldown: float = 300.0, half_open_probes: int = 1) -> None:
        self.clock = clock or Clock()
        self.window = deque(maxlen=window)
        self.min_calls = min_calls
        self.error_rate_threshold = error_rate_threshold
        self.consecutive_threshold = consecutive_threshold
        self.base_cooldown = base_cooldown
        self.max_cooldown = max_cooldown
        self.half_open_probes = half_open_probes
        self._state = CLOSED
        self._consecutive = 0
        self._trips = 0
        self._opened_at = 0.0
        self._cooldown = 0.0
        self._probes = 0

    @property
    def state(self) -> str:
        if self._state == OPEN and self.clock.now() - self._opened_at >= self._cooldown:
            self._state = HALF_OPEN
            self._probes = 0
        return self._state

    def retry_after(self) -> float:
        if self.state != OPEN:
            return 0.0
        return max(0.0, self._opened_at + self._cooldown - self.clock.now())

    def allow(self) -> bool:
        s = self.state
        if s == CLOSED:
            return True
        if s == HALF_OPEN and self._probes < self.half_open_probes:
            self._probes += 1
            return True
        return False

    def record_success(self) -> None:
        if self.state == HALF_OPEN:
            self._state = CLOSED
            self._trips = 0
            self.window.clear()
            self._probes = 0
        self.window.append(True)
        self._consecutive = 0

    def record_failure(self) -> None:
        s = self.state
        self.window.append(False)
        self._consecutive += 1
        if s == HALF_OPEN:
            self._open()
        elif s == CLOSED:
            fails = sum(1 for x in self.window if not x)
            rate_trip = len(self.window) >= self.min_calls and fails / len(self.window) >= self.error_rate_threshold
            if self._consecutive >= self.consecutive_threshold or rate_trip:
                self._open()

    def _open(self) -> None:
        self._cooldown = min(self.max_cooldown, self.base_cooldown * (2 ** self._trips))
        self._trips += 1
        self._state = OPEN
        self._opened_at = self.clock.now()
        self._probes = 0
        self._consecutive = 0
