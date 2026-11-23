"""Minimal Prometheus text-format registry (no dependency)."""
from __future__ import annotations

import threading
from collections.abc import Callable

BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)


def _lbl(labels: tuple) -> str:
    return "{" + ",".join(f'{k}="{v}"' for k, v in labels) + "}" if labels else ""


class Metrics:
    def __init__(self) -> None:
        self._c: dict[tuple[str, tuple], float] = {}
        self._g: dict[tuple[str, tuple], float] = {}
        self._h: dict[tuple[str, tuple], list[float]] = {}
        self._help: dict[str, str] = {}
        self._collectors: list[Callable[["Metrics"], None]] = []
        self._lock = threading.Lock()

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        k = (name, tuple(sorted(labels.items())))
        with self._lock:
            self._c[k] = self._c.get(k, 0.0) + value

    def gauge(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            self._g[(name, tuple(sorted(labels.items())))] = value

    def observe(self, name: str, value: float, **labels: str) -> None:
        k = (name, tuple(sorted(labels.items())))
        with self._lock:
            h = self._h.setdefault(k, [0.0] * (len(BUCKETS) + 2))  # buckets..., +Inf, sum
            for i, b in enumerate(BUCKETS):
                if value <= b:
                    h[i] += 1
            h[len(BUCKETS)] += 1
            h[len(BUCKETS) + 1] += value

    def add_collector(self, fn: Callable[["Metrics"], None]) -> None:
        self._collectors.append(fn)

    def get(self, name: str, **labels: str) -> float:
        return self._c.get((name, tuple(sorted(labels.items()))), 0.0)

    def render(self) -> str:
        for fn in self._collectors:
            fn(self)
        lines: list[str] = []
        with self._lock:
            for (n, l), v in sorted(self._c.items()):
                lines.append(f"switchboard_{n}{_lbl(l)} {v}")
            for (n, l), v in sorted(self._g.items()):
                lines.append(f"switchboard_{n}{_lbl(l)} {v}")
            for (n, l), h in sorted(self._h.items()):
                for i, b in enumerate(BUCKETS):
                    lines.append(f"switchboard_{n}_bucket{_lbl(l + (('le', str(b)),))} {h[i]}")
                lines.append(f"switchboard_{n}_bucket{_lbl(l + (('le', '+Inf'),))} {h[len(BUCKETS)]}")
                lines.append(f"switchboard_{n}_sum{_lbl(l)} {h[len(BUCKETS) + 1]}")
                lines.append(f"switchboard_{n}_count{_lbl(l)} {h[len(BUCKETS)]}")
        return "\n".join(lines) + "\n"
