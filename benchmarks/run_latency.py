"""Provider-limit compliance, interactive latency under bulk load (priority lanes on/off), and outage failover.

Usage: PYTHONPATH=src:benchmarks python benchmarks/run_latency.py [--quick]
Mock provider, 20 ms latency, provider limit 600 requests/min. Real wall-clock time.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import time

from common import pct, save
from switchboard.clock import Clock
from switchboard.config import Settings
from switchboard.db.session import Database
from switchboard.errors import SwitchboardError
from switchboard.gateway import build_gateway
from switchboard.limits.token_bucket import NO_PRIORITY, InMemoryRateLimiter, LimitSpec
from switchboard.providers.mock import MockBehavior, MockProvider
from switchboard.providers.registry import Registry
from switchboard.types import ChatRequest

RPM = 600
counter = itertools.count()


def build(priority: bool, rpm: int | None = RPM, two: bool = False, outage_cooldown: float = 5.0):
    clock = Clock()
    st = Settings(db_path=":memory:", embedder="hashing", cache_enabled=False, online_max_wait_s=2.0,
                  batch_max_wait_s=30.0, request_timeout_s=30)
    reg = Registry.load(settings=st)
    lat = {"cheap": 20, "balanced": 20, "strong": 20}
    provs = {"mock": MockProvider("mock", clock, 1, MockBehavior(latency_ms=lat, rate_limit=rpm))}
    if two:
        provs["mock2"] = MockProvider("mock2", clock, 2, MockBehavior(latency_ms=lat, rate_limit=rpm))
    else:
        reg.enabled.discard("mock2")
    lim = InMemoryRateLimiter(clock, None if priority else NO_PRIORITY)
    gw = build_gateway(st, clock, Database(), provs, lim, reg)
    if rpm:
        for k in provs:
            lim.configure(k, LimitSpec(rpm=rpm))
    for b in gw.breakers.values():
        b.base_cooldown = outage_cooldown
    return gw


def req(lane: str, path: str) -> ChatRequest:
    return ChatRequest.from_openai({"model": "mock-balanced", "temperature": 1.0,
                                    "messages": [{"role": "user", "content": f"unique {next(counter)}"}]},
                                   lane=lane, path=path)


async def phase(gw, seconds: float, interactive_rps: float, bulk_workers: int, bulk_lane: str = "bulk") -> dict:
    stop = time.perf_counter() + seconds
    lat, rej, errs = [], 0, 0
    bulk_done = 0

    async def one_interactive():
        nonlocal rej, errs
        t = time.perf_counter()
        try:
            await gw.handle(req("interactive", "online"))
            lat.append((time.perf_counter() - t) * 1000)
        except SwitchboardError as e:
            if e.status == 429:
                rej += 1
            else:
                errs += 1

    async def bulk():
        nonlocal bulk_done
        while time.perf_counter() < stop:
            try:
                await gw.handle(req(bulk_lane, "batch"))
                bulk_done += 1
            except SwitchboardError:
                await asyncio.sleep(0.05)

    bulk_tasks = [asyncio.create_task(bulk()) for _ in range(bulk_workers)]
    tasks = []
    while time.perf_counter() < stop:
        tasks.append(asyncio.create_task(one_interactive()))
        await asyncio.sleep(1 / interactive_rps)
    await asyncio.gather(*tasks)
    for t in bulk_tasks:
        t.cancel()
    await asyncio.gather(*bulk_tasks, return_exceptions=True)
    return {"seconds": seconds, "interactive_ok": len(lat), "interactive_429": rej, "interactive_errors": errs,
            "p50_ms": pct(lat, .5), "p95_ms": pct(lat, .95), "p99_ms": pct(lat, .99), "bulk_completed": bulk_done,
            "bulk_per_s": bulk_done / seconds}


def max_window(ts: list[float], w: float = 60.0) -> int:
    best, j = 0, 0
    for i, t in enumerate(ts):
        while ts[j] <= t - w:
            j += 1
        best = max(best, i - j + 1)
    return best


async def failover(duration: float = 30.0, out_from: float = 8.0, out_to: float = 20.0, rps: float = 50.0) -> dict:
    gw = build(True, rpm=None, two=True)
    gw.limiter.configure("mock", LimitSpec(rpm=100000)); gw.limiter.configure("mock2", LimitSpec(rpm=100000))
    t0 = time.perf_counter()
    events: list[tuple[float, str, float, str]] = []

    async def one():
        ts = time.perf_counter() - t0
        r = ChatRequest.from_openai({"model": "mock-strong", "temperature": 1.0,
                                     "messages": [{"role": "user", "content": f"u {next(counter)}"}]})
        try:
            _, m = await gw.handle(r)
            events.append((ts, "ok", (time.perf_counter() - t0 - ts) * 1000, m.provider))
        except SwitchboardError as e:
            events.append((ts, "fail:" + e.code, 0, "-"))

    async def chaos():
        await asyncio.sleep(out_from); gw.providers["mock"].b.outage = True
        await asyncio.sleep(out_to - out_from); gw.providers["mock"].b.outage = False

    ch = asyncio.create_task(chaos())
    tasks = []
    while time.perf_counter() - t0 < duration:
        tasks.append(asyncio.create_task(one()))
        await asyncio.sleep(1 / rps)
    await asyncio.gather(*tasks, ch)
    during = [e for e in events if out_from <= e[0] < out_to]
    after = [e for e in events if e[0] >= out_to]
    first_back = next((e[0] for e in sorted(after) if e[3] == "mock"), None)
    return {"requests": len(events), "success_rate_overall": sum(e[1] == "ok" for e in events) / len(events),
            "success_rate_during_outage": sum(e[1] == "ok" for e in during) / max(1, len(during)),
            "served_by_fallback_during_outage": sum(e[3] == "mock2" for e in during) / max(1, len(during)),
            "p95_ms_before": pct([e[2] for e in events if e[0] < out_from and e[1] == "ok"], .95),
            "p95_ms_during_outage": pct([e[2] for e in during if e[1] == "ok"], .95),
            "primary_calls_during_outage": sum(1 for e in during if e[3] == "mock"),
            "seconds_until_primary_serves_again": (first_back - out_to) if first_back else None,
            "breaker_cooldown_s": 5.0, "outage_seconds": out_to - out_from}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    q = ap.parse_args().quick
    d_alone, d_main, d_nop = (8, 15, 8) if q else (20, 70, 20)
    out: dict = {"provider_limit_rpm": RPM, "interactive_rps": 4}
    print("phase 1: interactive only", flush=True)
    out["interactive_alone"] = await phase(build(True), d_alone, 4, 0)
    print(out["interactive_alone"], flush=True)
    save("latency_partial", out)
    print("phase 2: interactive + 48 bulk workers, priority lanes ON", flush=True)
    gw = build(True)
    out["priority_on"] = await phase(gw, d_main, 4, 48)
    ts = gw.providers["mock"].accepted
    out["priority_on"]["max_requests_in_any_60s_window"] = max_window(ts)
    out["priority_on"]["provider_side_429s"] = gw.providers["mock"].n_429
    out["priority_on"]["limiter_rejections"] = dict(gw.limiter.rejections)
    print(out["priority_on"], flush=True)
    save("latency_partial", out)
    print("phase 3: same load, priority lanes OFF (all lanes equal)", flush=True)
    out["priority_off"] = await phase(build(False), d_nop, 4, 48)
    print(out["priority_off"], flush=True)
    save("latency_partial", out)
    print("failover under injected outage", flush=True)
    out["failover"] = await failover(15 if q else 30, 4 if q else 8, 10 if q else 20)
    print(out["failover"], flush=True)
    save("latency_quick" if q else "latency", out)


if __name__ == "__main__":
    asyncio.run(main())
