"""Main benchmark: naive baseline vs the gateway with features toggled on one at a time.

Usage: PYTHONPATH=src:benchmarks python benchmarks/run_bench.py [--n 5000] [--only name,name]
Everything runs in-process against the mock provider (no keys, no network). Mock latencies are 4/8/12 ms
(cheap/balanced/strong) and the mock enforces 3000 requests / 60 s (sliding window) and the gateway limiter is configured to the same limit.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

from common import BENCH_RPM, body, make_gateway, save
from switchboard.errors import RateLimited
from switchboard.ledger.pricing import equiv_cost_usd
from switchboard.providers.mock import MockBehavior, MockProvider, reference_answer
from switchboard.providers.registry import Registry
from switchboard.config import Settings
from switchboard.routing.router import Router
from switchboard.scheduler.jobs_service import JobsService
from switchboard.scheduler.queue import TaskQueue
from switchboard.scheduler.worker import Worker
from switchboard.types import ChatRequest
from workload_gen import Prompt, generate

STRONG = "mock-strong"


def correct(text: str | None, p: Prompt) -> bool:
    return bool(text) and text.startswith(reference_answer(p.akey) + " ")


def mk_req(p: Prompt, model: str = STRONG) -> ChatRequest:
    return ChatRequest.from_openai(body(p, model))


async def naive(prompts: list[Prompt], concurrency: int, rpm: int = BENCH_RPM) -> dict:
    """Direct calls to the strongest model: no cache, no routing, no limiter."""
    prov = MockProvider("mock", behavior=MockBehavior(rate_limit=rpm), seed=1)
    spec = Registry.load(settings=Settings()).models[STRONG]
    cost, n429, ok, right = 0.0, 0, 0, 0
    q: asyncio.Queue = asyncio.Queue()
    for p in prompts:
        q.put_nowait(p)

    async def worker() -> None:
        nonlocal cost, n429, ok, right
        while True:
            try:
                p = q.get_nowait()
            except asyncio.QueueEmpty:
                return
            req = mk_req(p)
            while True:
                try:
                    r = await prov.chat(req, STRONG)
                    break
                except RateLimited as e:
                    n429 += 1
                    await asyncio.sleep(e.retry_after or 1)
            cost += equiv_cost_usd(spec, r.usage)
            ok += 1
            right += correct(r.content, p)

    t0 = time.perf_counter()
    await asyncio.gather(*[worker() for _ in range(concurrency)])
    dt = time.perf_counter() - t0
    return {"tasks": len(prompts), "seconds": dt, "tasks_per_min": len(prompts) / dt * 60, "equiv_cost_usd": cost,
            "client_429s": n429, "quality": right / len(prompts), "upstream_calls": prov.n_calls}


async def gateway_run(prompts: list[Prompt], cache: bool, model: str, router_mode: str = "heuristic",
                      concurrency: int = 64, **kw) -> dict:
    gw = make_gateway(cache=cache, router_mode=router_mode, **kw)
    q = TaskQueue(gw.db, gw.clock)
    svc = JobsService(q, gw.ledger)
    job, _ = svc.create("anonymous", [body(p, model) for p in prompts], priority="batch")
    w = Worker(q, gw, concurrency=concurrency, poll_s=0.002, lease_s=120)
    t0 = time.perf_counter()
    await w.run(until_idle=True)
    dt = time.perf_counter() - t0
    res = {r["index"]: r for r in q.results(job["id"])}
    right = sum(correct(res[i]["response"]["choices"][0]["message"]["content"], p) if res[i]["response"] else False
                for i, p in enumerate(prompts))
    sv = gw.ledger.savings()
    by_model = {r["group"]: r["requests"] for r in gw.ledger.usage("model")}
    calls = sum(p.n_calls for p in gw.providers.values())
    hits = gw.cache.stats
    return {"tasks": len(prompts), "completed": sum(1 for r in res.values() if r["status"] == "succeeded"),
            "seconds": dt, "tasks_per_min": len(prompts) / dt * 60, "equiv_cost_usd": sv["equiv"],
            "ledger_baseline_usd": sv["baseline"], "cache_hit_rate": sv["cache_hit_rate"],
            "cache_exact_hits": hits["hit_exact"], "cache_semantic_hits": hits["hit_semantic"],
            "quality": right / len(prompts), "upstream_calls": calls,
            "provider_429s_seen_by_gateway": sum(p.n_429 for p in gw.providers.values()),
            "client_429s": 0, "tier_mix": by_model, "worker": w.stats}


def routing_accuracy(prompts: list[Prompt], mode: str) -> dict:
    reg = Registry.load(settings=Settings())
    r = Router(reg, Settings(router_mode=mode))
    seen, easy_cheap, hard_noncheap, hard_strong, ne, nh = set(), 0, 0, 0, 0, 0
    for p in prompts:
        if p.kind != "unique":
            continue
        t = r.route(ChatRequest("auto", [{"role": "user", "content": p.prompt}], temperature=0, path="batch")).tier
        if p.difficulty == "easy":
            ne += 1; easy_cheap += t == "cheap"
        else:
            nh += 1; hard_noncheap += t != "cheap"; hard_strong += t == "strong"
    return {"classifier": mode, "easy_to_cheap": easy_cheap / ne, "hard_not_cheap": hard_noncheap / nh,
            "hard_to_strong": hard_strong / nh, "n_easy": ne, "n_hard": nh}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    prompts = generate(a.n)
    runs = {
        "naive_sequential": lambda: naive(prompts, 1),
        "naive_parallel64": lambda: naive(prompts, 64),
        "gw_plain": lambda: gateway_run(prompts, False, STRONG),
        "gw_cache_only": lambda: gateway_run(prompts, True, STRONG),
        "gw_routing_only": lambda: gateway_run(prompts, False, "auto"),
        "gw_cache_plus_routing": lambda: gateway_run(prompts, True, "auto"),
        "gw_cache_plus_learned_routing": lambda: gateway_run(prompts, True, "auto", "learned"),
        "gw_cache_semantic_ON_hashing_embedder": lambda: gateway_run(prompts, True, STRONG, cache_semantic="on"),
    }
    only = [s for s in a.only.split(",") if s]
    out: dict = {"n": a.n, "routing_accuracy": [routing_accuracy(prompts, "heuristic"), routing_accuracy(prompts, "learned")]}
    for name, fn in runs.items():
        if only and name not in only:
            continue
        print(f"running {name} ...", flush=True)
        out[name] = await fn()
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[name].items()
                          if k not in ("tier_mix", "worker")}), flush=True)
        save("bench_main" if not only else "bench_partial", out)
    base = out.get("naive_sequential")
    if base:
        for k, v in out.items():
            if isinstance(v, dict) and "equiv_cost_usd" in v:
                v["saved_pct_vs_naive"] = 100 * (1 - v["equiv_cost_usd"] / base["equiv_cost_usd"])
                v["throughput_x_vs_naive_seq"] = v["tasks_per_min"] / base["tasks_per_min"]
        save("bench_main", out)


if __name__ == "__main__":
    asyncio.run(main())
