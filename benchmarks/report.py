"""Build docs/BENCHMARKS.md and charts from benchmarks/results/*.json (no numbers are typed by hand).

Usage: PYTHONPATH=src:benchmarks python benchmarks/report.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "benchmarks/results"
DOCS = ROOT / "docs"


def load(name: str):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def charts(main, lat, cache) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    out = []
    (DOCS / "img").mkdir(exist_ok=True, parents=True)
    if main:
        names = [k for k in main if isinstance(main[k], dict) and "equiv_cost_usd" in main[k]]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].barh(names, [main[k]["equiv_cost_usd"] for k in names], color="#2563eb")
        ax[0].set_title("Total equivalent cost (USD, lower is better)"); ax[0].invert_yaxis()
        ax[1].barh(names, [main[k]["tasks_per_min"] for k in names], color="#d97706")
        ax[1].set_title("Throughput (tasks/min, higher is better)"); ax[1].invert_yaxis(); ax[1].set_yticklabels([])
        plt.tight_layout(); plt.savefig(DOCS / "img/cost_throughput.png", dpi=130); plt.close()
        out.append("img/cost_throughput.png")
    if lat:
        ph = [("interactive_alone", "alone"), ("priority_on", "with bulk load\npriority ON"), ("priority_off", "with bulk load\npriority OFF")]
        fig, ax = plt.subplots(figsize=(7, 4))
        w = 0.25
        for i, (k, c) in enumerate((("p50_ms", "#2563eb"), ("p95_ms", "#d97706"), ("p99_ms", "#b91c1c"))):
            ax.bar([x + i * w for x in range(3)], [lat[p][k] for p, _ in ph], w, label=k[:3], color=c)
        ax.set_xticks([x + w for x in range(3)]); ax.set_xticklabels([l for _, l in ph])
        ax.set_ylabel("interactive latency (ms)"); ax.set_yscale("log"); ax.legend(); ax.set_title("Interactive latency vs bulk load")
        plt.tight_layout(); plt.savefig(DOCS / "img/latency.png", dpi=130); plt.close()
        out.append("img/latency.png")
    if cache:
        fig, ax = plt.subplots(figsize=(7, 4))
        th = [r["threshold"] for r in cache["rows"]]
        for g, ls in (("no_guard", "--"), ("guard", "-")):
            ax.plot(th, [r[g]["precision"] for r in cache["rows"]], ls, color="#2563eb", label=f"precision ({g})")
            ax.plot(th, [r[g]["recall"] for r in cache["rows"]], ls, color="#d97706", label=f"recall ({g})")
        ax.set_xlabel("similarity threshold"); ax.legend(fontsize=7); ax.set_title(f"Cache threshold study ({cache['embedder']} embedder)")
        plt.tight_layout(); plt.savefig(DOCS / "img/cache_threshold.png", dpi=130); plt.close()
        out.append("img/cache_threshold.png")
    return out


def f(x, nd=2):
    return "n/a" if x is None else (f"{x:,.{nd}f}" if isinstance(x, (int, float)) else str(x))


def main() -> None:
    main_r, lat, cache, rt = load("bench_main"), load("latency"), load("cache_threshold_hashing"), load("router_train")
    cidx = load("cache_index_hashing")
    imgs = charts(main_r, lat, cache)
    L: list[str] = ["# Benchmarks", "",
                    "Every number below was produced by the scripts in `benchmarks/` and copied here by `benchmarks/report.py`. "
                    "Raw outputs are in `benchmarks/results/*.json`. **All runs use the built-in mock provider** (no API keys, no network) "
                    "in a CPU-only sandbox, so absolute latencies are mock-scaled (4/8/12 ms for cheap/balanced/strong); "
                    "relative comparisons and the correctness properties are what matter. Costs are *equivalent* costs at the reference prices in "
                    "`models.yaml` (cheap $0.15/$0.60, balanced $0.50/$2.00, strong $2.50/$10.00 per 1M in/out tokens), not real invoices.", ""]
    if main_r:
        L += ["## 1. Cost, throughput and quality (5,000-prompt workload)", "",
              "Workload (`benchmarks/workload_gen.py`, seed 7): 5,000 synthetic prompts = 2,500 first occurrences + 1,250 exact duplicates + "
              "1,250 paraphrases (same slots, different wording); ~60% easy / ~40% hard. Provider limit: 3,000 requests/min (sliding window), "
              "below the workload size, like a real free tier. *Naive baseline* = every prompt straight to the strongest model, no cache, no gateway. "
              "Gateway runs submit the workload as one batch job (64 concurrent workers). `quality` = fraction of answers equal to the reference answer "
              "under the mock's simulated tier-quality model (see `providers/mock.py`): **this is a simulation, not an LLM-judge measurement.**", "",
              "| Config | Equiv. cost (USD) | Saved vs naive | Tasks/min | Upstream calls | Cache hit rate | Client-visible 429s | Quality |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for k, v in main_r.items():
            if isinstance(v, dict) and "equiv_cost_usd" in v:
                L.append(f"| {k} | {f(v['equiv_cost_usd'], 4)} | {f(v.get('saved_pct_vs_naive', 0), 1)}% | {f(v['tasks_per_min'], 0)} | "
                         f"{v['upstream_calls']} | {f(100 * v.get('cache_hit_rate', 0), 1)}% | {v.get('client_429s', 0)} | {f(100 * v['quality'], 1)}% |")
        L += ["", "Notes:", "- `client_429s` for the naive rows = 429 responses the client received (and had to wait out); for gateway rows the client never sees them "
              "(the limiter admits requests below the provider limit). Provider-side 429s seen *by the gateway* are in the JSON "
              "(`provider_429s_seen_by_gateway`).",
              "- Cache hits in the default rows are **exact duplicates (and single-flight coalescing)**; the semantic layer is off with the fallback hashing embedder (section 3), "
              "so paraphrases are not deduplicated. The `gw_cache_semantic_ON_hashing_embedder` row forces it on to show the damage: its quality column drops below 100% "
              "even though every request is sent to the strongest model, i.e. the loss is entirely wrong cache hits.",
              "- **Throughput caveat:** `gw_plain` is *slower* than the naive clients. The gateway deliberately stays ~10% below the provider limit (window guarantee, DECISIONS 3) to guarantee zero 429s, "
              "while the naive clients simply burst and wait out 429s. The gateway's throughput advantage comes only from not making upstream calls (cache hits), not from the limiter.",
              "- The heuristic router sends some hard prompts to cheaper tiers, which is where its quality loss comes from; the learned router is better on this workload but was trained on a generator "
              "with similar structure, so its near-perfect routing here should not be expected on real traffic.", ""]
        ra = main_r.get("routing_accuracy") or []
        if ra:
            L += ["### Router accuracy on this workload (unique prompts, labels hidden from the router)", "",
                  "| Classifier | easy -> cheap | hard -> not cheap | hard -> strong |", "|---|---:|---:|---:|"]
            for r in ra:
                L.append(f"| {r['classifier']} | {f(100 * r['easy_to_cheap'], 1)}% | {f(100 * r['hard_not_cheap'], 1)}% | {f(100 * r['hard_to_strong'], 1)}% |")
            if rt:
                L += ["", f"The learned classifier was trained on a *different* synthetic generator (`routing/dataset.py`, {rt['train_n']} prompts; "
                      f"in-distribution validation accuracy {f(100 * rt['val_accuracy'], 1)}%), so the table above is a transfer test. Real traffic is messier; "
                      "treat all of these as optimistic."]
            L.append("")
        L += ["![cost and throughput](img/cost_throughput.png)", ""]
    if lat:
        a, on, off, fo = lat["interactive_alone"], lat["priority_on"], lat["priority_off"], lat["failover"]
        L += ["## 2. Rate-limit compliance, priority lanes and failover", "",
              f"Provider limit {lat['provider_limit_rpm']} requests/min, mock latency 20 ms, interactive load {lat['interactive_rps']} req/s (open loop), "
              "bulk load = 48 closed-loop workers in the `bulk` lane.", "",
              "| Scenario | interactive ok | interactive 429 | p50 ms | p95 ms | p99 ms | bulk req/s |", "|---|---:|---:|---:|---:|---:|---:|"]
        for n, r in (("interactive alone", a), ("+ bulk load, priority lanes ON", on), ("+ bulk load, priority lanes OFF", off)):
            L.append(f"| {n} | {r['interactive_ok']} | {r['interactive_429']} | {f(r['p50_ms'], 1)} | {f(r['p95_ms'], 1)} | {f(r['p99_ms'], 1)} | {f(r['bulk_per_s'], 2)} |")
        L += ["", f"**Limit compliance:** during the {on['seconds']} s priority-ON run (interactive + saturating bulk), the maximum number of requests the provider accepted in "
              f"*any* sliding 60 s window was **{on['max_requests_in_any_60s_window']}** (limit {lat['provider_limit_rpm']}); provider-side 429s: **{on['provider_side_429s']}**.", "",
              "![latency](img/latency.png)", "",
              "### Failover under an injected outage", "",
              f"Two mock providers, 50 req/s of `mock-strong` traffic, primary hard-down for {fo['outage_seconds']} s, breaker base cool-down {fo['breaker_cooldown_s']} s.", "",
              "| Metric | Value |", "|---|---:|",
              f"| requests | {fo['requests']} |", f"| success rate (overall) | {f(100 * fo['success_rate_overall'], 2)}% |",
              f"| success rate during outage | {f(100 * fo['success_rate_during_outage'], 2)}% |",
              f"| share served by fallback during outage | {f(100 * fo['served_by_fallback_during_outage'], 1)}% |",
              f"| requests *successfully served by the primary* during its outage (must be 0) | {fo['primary_calls_during_outage']} |",
              f"| p95 latency before outage (ms) | {f(fo['p95_ms_before'], 1)} |", f"| p95 latency during outage (ms) | {f(fo['p95_ms_during_outage'], 1)} |",
              f"| seconds after recovery until the primary serves again | {f(fo['seconds_until_primary_serves_again'], 1)} |", ""]
    if cache:
        L += ["## 3. Semantic-cache threshold study", "",
              f"Embedder: **{cache['embedder']}**. {cache['paraphrase_pairs']} paraphrase pairs (should hit) and {cache['near_miss_pairs']} near-miss pairs "
              "(same wording, different number/entity; must NOT hit), synthetic, from `workload_gen.pairs()`. "
              f"Mean similarity: paraphrases {f(cache['mean_sim_paraphrase'], 3)}, near-misses {f(cache['mean_sim_near_miss'], 3)}.", "",
              "| Threshold | Precision | Recall | FP rate | Precision (+guard) | Recall (+guard) | FP rate (+guard) |", "|---:|---:|---:|---:|---:|---:|---:|"]
        for r in cache["rows"]:
            n, g = r["no_guard"], r["guard"]
            L.append(f"| {r['threshold']:.2f} | {f(n['precision'], 3)} | {f(n['recall'], 3)} | {f(n['fp_rate'], 3)} | {f(g['precision'], 3)} | {f(g['recall'], 3)} | {f(g['fp_rate'], 3)} |")
        if cidx:
            L += ["", "### In-situ replay (nearest-neighbour search over everything cached so far)", "",
                  f"The pairwise table above only compares matched pairs. Here the {cidx['n']}-prompt workload is replayed through the real `SemanticCache`; "
                  "a hit is *wrong* when the cached answer belongs to a different question. First row: semantic layer off (the default with this embedder).", "",
                  "| Semantic layer | Guard | Threshold | Exact hits (correct) | Semantic hits correct | Semantic hits WRONG | Wrong-answer rate | Paraphrase recall |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
            for r in cidx["rows"]:
                L.append(f"| {'on' if r['semantic'] else 'off'} | {'yes' if r['guard'] else 'no'} | {r['threshold']:.2f} | {r['exact_correct']} | {r['semantic_correct']} | "
                         f"{r['semantic_wrong']} | {f(100 * r['wrong_answer_rate'], 2)}% | {f(100 * r['paraphrase_recall'], 1)}% |")
        L += ["", "**Reading this honestly:** the fallback hashing embedder is lexical. It scores near-misses *higher* than paraphrases, so no threshold gives "
              "both useful recall and high precision. In the in-situ replay the semantic layer returns ~0 correct paraphrase hits and a non-trivial rate of wrong answers, "
              "so it is **disabled by default with this embedder** (`CACHE_SEMANTIC=auto`). "
              "Install `pip install -e .[embeddings]` and run `python benchmarks/eval_cache_threshold.py --embedder auto` to produce the table for `BAAI/bge-small-en-v1.5` "
              "and pick a threshold from it; **that measurement was not possible in the build sandbox** (no network, model not installed), so no numbers are claimed for it.", "",
              "![cache threshold](img/cache_threshold.png)", ""]
    jc = load("job_chaos")
    if jc:
        L += ["## 4. 5,000-task batch job under injected failures", "",
              "Two workers (32 concurrent tasks each) share one durable queue. The mock provider hard-limits 300 calls/s (the gateway limiter is set looser on purpose, so the provider "
              "returns real 429s) and fails ~10% of other calls with 5xx/timeouts.", "", "| Metric | Value |", "|---|---:|",
              f"| tasks | {jc['tasks']} |", f"| wall time (s) | {f(jc['seconds'], 1)} |", f"| tasks done / dead / pending | {jc['counts']['done']} / {jc['counts']['dead']} / {jc['counts']['pending']} |",
              f"| distinct tasks charged in the ledger | {jc['distinct_charged_tasks']} |", f"| tasks charged more than once | {jc['tasks_charged_more_than_once']} |",
              f"| results returned / succeeded | {jc['results_returned']} / {jc['results_succeeded']} |", f"| provider 429s received | {jc['provider_429s']} |",
              f"| injected 5xx/timeouts | {jc['provider_injected_errors']} |", f"| total provider calls | {jc['provider_calls']} |",
              f"| tasks that needed more than one attempt | {jc['tasks_needing_more_than_one_attempt']} |", f"| **no lost, no duplicated** | **{jc['no_lost_no_duplicated']}** |", "",
              "Process-level crash recovery (SIGKILL of a real worker mid-job, lease expiry, exactly-once charging) is covered by `tests/integration/test_crash_recovery.py`.", ""]
    L += ["## Reproduce", "", "```bash", "pip install -e '.[bench]'", "export PYTHONPATH=src:benchmarks",
          "python benchmarks/train_router.py             # trains the learned router -> router_train.json",
          "python benchmarks/eval_cache_threshold.py     # section 3 (pairwise)",
          "python benchmarks/eval_cache_index.py         # section 3 (in-situ replay)",
          "python benchmarks/run_bench.py                # section 1 (~10 min)",
          "python benchmarks/run_latency.py              # section 2 (~2.5 min; --quick for a short run)",
          "python benchmarks/run_job_chaos.py           # section 4",
          "python benchmarks/report.py                   # regenerates this file + charts", "```", "",
          "## What was NOT measured", "",
          "* Any real provider (Groq/Gemini/Ollama/OpenAI): no keys or network in the build sandbox.",
          "* Semantic-cache behaviour with a real embedding model (see section 3).",
          "* Quality with an LLM-as-judge; `quality` above is the mock's simulation.",
          "* The Redis/Lua limiter against a live Redis; the HTTP layer under load; the dashboard in a browser.", ""]
    (DOCS / "BENCHMARKS.md").write_text("\n".join(L))
    print("wrote docs/BENCHMARKS.md", imgs)


if __name__ == "__main__":
    main()
