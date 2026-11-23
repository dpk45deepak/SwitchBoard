"""In-situ cache study: replay the 5,000-prompt workload through the real SemanticCache (nearest-neighbour search
over everything cached so far) and count correct hits vs WRONG hits (a hit whose cached answer belongs to a
different base prompt). This is stricter than the pairwise study, which only compares matched pairs.

Usage: PYTHONPATH=src:benchmarks python benchmarks/eval_cache_index.py [--embedder hashing|auto]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from common import RESULTS
from switchboard.cache.embedder import get_embedder
from switchboard.cache.semantic_cache import SemanticCache
from switchboard.clock import FakeClock
from switchboard.types import ChatRequest
from workload_gen import generate


def run(prompts, emb, thr: float, guard: bool, semantic: bool = True) -> dict:
    c = SemanticCache(FakeClock(), emb, sim_threshold=thr, guard=guard, semantic=semantic)
    out: Counter = Counter()
    for p in prompts:
        r = ChatRequest("m", [{"role": "user", "content": p.prompt}], temperature=0)
        k = c.prepare(r)
        h = c.lookup(k)
        if h is None:
            c.store(k, content=p.akey, usage={"prompt_tokens": 1, "completion_tokens": 1}, finish_reason="stop",
                    tool_calls=None, provider="p", model="m", equiv_cost=0)
            out["miss"] += 1
        else:
            ok = h.entry.content == p.akey
            out[f"{h.kind}_{'correct' if ok else 'WRONG'}"] += 1
    n = len(prompts)
    para = sum(1 for p in prompts if p.kind == "para")
    sc, sw = out["semantic_correct"], out["semantic_WRONG"]
    return {"threshold": thr, "guard": guard, "semantic": semantic, "exact_correct": out["exact_correct"],
            "semantic_correct": sc, "semantic_wrong": sw, "exact_wrong": out["exact_WRONG"], "wrong_answer_rate": (sw + out["exact_WRONG"]) / n,
            "paraphrases_in_workload": para, "paraphrase_recall": sc / para if para else 0.0,
            "precision_of_semantic_hits": sc / (sc + sw) if sc + sw else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="hashing")
    a = ap.parse_args()
    emb = get_embedder(a.embedder)
    prompts = generate()
    rows = [run(prompts, emb, 0.92, True, semantic=False)]
    for guard in (False, True):
        for t in (0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99):
            rows.append(run(prompts, emb, t, guard))
    res = {"embedder": emb.name, "n": len(prompts), "rows": rows}
    (RESULTS / f"cache_index_{emb.name.replace('/', '_')}.json").write_text(json.dumps(res, indent=2))
    print("embedder", emb.name)
    for r in rows:
        print(f"sem={r['semantic']!s:5} guard={r['guard']!s:5} t={r['threshold']:.2f} exact_ok={r['exact_correct']} "
              f"sem_ok={r['semantic_correct']} sem_WRONG={r['semantic_wrong']} wrong_rate={100*r['wrong_answer_rate']:.2f}% "
              f"para_recall={100*r['paraphrase_recall']:.1f}%")


if __name__ == "__main__":
    main()
