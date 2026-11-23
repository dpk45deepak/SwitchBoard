from __future__ import annotations

import json
from pathlib import Path

from switchboard.clock import Clock
from switchboard.config import Settings
from switchboard.db.session import Database
from switchboard.gateway import build_gateway
from switchboard.providers.mock import MockBehavior, MockProvider
from switchboard.providers.registry import Registry
from workload_gen import Prompt

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


BENCH_RPM = 3000  # provider limit: deliberately below the 5,000-task workload, like a real free tier


def make_gateway(cache: bool = True, rpm: int | None = BENCH_RPM, window_s: float = 60.0, router_mode: str = "heuristic",
                 behavior_kw: dict | None = None, **settings_kw):
    clock = Clock()
    st = Settings(db_path=":memory:", embedder="hashing", cache_enabled=cache, router_mode=router_mode,
                  request_timeout_s=30, **settings_kw)
    reg = Registry.load(settings=st)
    mk = lambda seed: MockProvider(  # noqa: E731
        "m", clock, seed, MockBehavior(rate_limit=rpm, window_s=window_s, **(behavior_kw or {})))
    providers = {"mock": mk(1), "mock2": mk(2)}
    providers["mock"].name, providers["mock2"].name = "mock", "mock2"
    gw = build_gateway(st, clock, Database(), providers, registry=reg)
    if rpm:
        from switchboard.limits.token_bucket import LimitSpec
        for k in providers:
            gw.limiter.configure(k, LimitSpec(rpm=rpm))
    return gw


def body(p: Prompt, model: str) -> dict:
    return {"model": model, "temperature": 0, "messages": [{"role": "user", "content": p.prompt}],
            "x_switchboard": {"difficulty": p.difficulty, "answer_key": p.akey}}


def save(name: str, obj) -> None:
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=2))


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]
