"""Cache false-positive study: precision/recall vs similarity threshold (with and without the lexical guard).

Usage: PYTHONPATH=src python benchmarks/eval_cache_threshold.py [--embedder hashing|auto]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from switchboard.cache.embedder import get_embedder
from switchboard.cache.semantic_cache import guard_tokens
from workload_gen import pairs

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="hashing")
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    emb = get_embedder(a.embedder)
    ps = pairs(a.n)
    A, B = emb.embed([p[0] for p in ps]), emb.embed([p[1] for p in ps])
    sims = np.sum(A * B, axis=1)
    y = np.array([p[2] for p in ps])
    guard_ok = np.array([guard_tokens(p[0]) == guard_tokens(p[1]) for p in ps])
    rows = []
    for t in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99]:
        row = {"threshold": t}
        for name, mask in (("no_guard", np.ones_like(guard_ok)), ("guard", guard_ok)):
            pred = (sims >= t) & mask
            tp, fp = int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0)))
            fn = int(np.sum(~pred & (y == 1)))
            prec = tp / (tp + fp) if tp + fp else 1.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            row[name] = {"precision": prec, "recall": rec, "fp_rate": fp / int(np.sum(y == 0)),
                         "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
        rows.append(row)
    res = {"embedder": emb.name, "pairs": len(ps), "paraphrase_pairs": int(y.sum()), "near_miss_pairs": int((y == 0).sum()),
           "mean_sim_paraphrase": float(sims[y == 1].mean()), "mean_sim_near_miss": float(sims[y == 0].mean()), "rows": rows}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"cache_threshold_{emb.name.replace('/', '_')}.json").write_text(json.dumps(res, indent=2))
    print(f"embedder={emb.name} pairs={len(ps)} mean sim: paraphrase={res['mean_sim_paraphrase']:.3f} near-miss={res['mean_sim_near_miss']:.3f}")
    print("thr   | no-guard P / R / FPR     | guard P / R / FPR")
    for r in rows:
        n, g = r["no_guard"], r["guard"]
        print(f"{r['threshold']:.2f}  | {n['precision']:.3f} / {n['recall']:.3f} / {n['fp_rate']:.3f}   | {g['precision']:.3f} / {g['recall']:.3f} / {g['fp_rate']:.3f}")


if __name__ == "__main__":
    main()
