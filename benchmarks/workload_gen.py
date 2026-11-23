"""Synthetic benchmark workload (seeded, reproducible).

Mix: 5,000 prompts = 2,500 first occurrences ("unique") + 1,250 exact duplicates + 1,250 paraphrases of
earlier prompts; 60% easy / 40% hard (difficulty inherited from the base prompt). Every prompt has a hidden
`difficulty` label and an `answer_key` (its base id) used ONLY by the mock provider to simulate answer quality.
The router never sees these labels. Templates here differ in wording from routing/dataset.py (router training).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

COUNTRIES = ("France Japan Brazil Kenya Canada Egypt Norway Peru Vietnam Chile Poland Ghana Nepal Cuba Iran Spain Italy Greece "
             "Turkey Mexico Algeria Austria Belgium Bolivia Croatia Denmark Ecuador Finland Hungary Iceland Ireland Jordan Latvia "
             "Malta Nigeria Oman Panama Qatar Romania Serbia Sweden Tunisia Uganda Uruguay Yemen Zambia").split()
ADJ = "swift quiet bright heavy gentle broken golden ancient hollow fragile".split()
NOUN = "river lantern harvest bridge anchor meadow engine garden window market".split()
LANGS = ["Spanish", "German", "Italian", "French", "Dutch", "Portuguese", "Swedish", "Polish"]
FEEL = "delicious terrible lovely awful fantastic mediocre charming disappointing superb sloppy".split()
THING = "pizza service hotel laptop concert phone delivery movie coffee game".split()
TASKS = ["merges {k} sorted linked lists into one", "finds the {k} most frequent words in a text stream",
         "detects a cycle in a directed graph with {k} thousand nodes", "implements an LRU cache holding {k} entries",
         "computes the edit distance between strings of up to {k} characters", "schedules {k} jobs with dependencies"]
SYSTEMS = ["URL shortener", "chat service", "ride-matching backend", "rate limiter", "video pipeline", "payment ledger",
           "notification system", "search service", "feature-flag service", "log aggregator"]
MIN_GAP = 0.04


@dataclass
class Prompt:
    id: int
    prompt: str
    difficulty: str
    kind: str  # unique | dup | para
    base: int
    akey: str = ""  # canonical question identity (family + slots): equal questions share one correct answer


def _r(lo: int, hi: int, rng: random.Random) -> int:
    return rng.randrange(lo, hi)


EASY = {
    "arith": (lambda g: {"a": _r(100, 99999, g), "b": _r(100, 99999, g)},
              ["What is {a} plus {b}?", "Add {a} and {b}.", "Compute the sum of {a} and {b}."]),
    "unit": (lambda g: {"n": _r(2, 9999, g)},
             ["Convert {n} kilometers to miles.", "How many miles is {n} km?", "Express {n} kilometres in miles."]),
    "sent": (lambda g: {"t": g.choice(THING), "f": g.choice(FEEL), "t2": g.choice(THING), "f2": g.choice(FEEL)},
             ["Is this review positive or negative: \"The {t} was {f} and the {t2} was {f2}.\"",
              "Classify the sentiment of the following review: \"The {t} was {f} and the {t2} was {f2}.\"",
              "Tell me whether \"The {t} was {f} and the {t2} was {f2}.\" is a positive or negative review."]),
    "trans": (lambda g: {"a": g.choice(ADJ), "n": g.choice(NOUN), "l": g.choice(LANGS), "k": _r(1, 9999, g)},
              ["Translate 'the {a} {n} no. {k}' into {l}.", "How do you say 'the {a} {n} no. {k}' in {l}?",
               "Give the {l} translation of 'the {a} {n} no. {k}'."]),
    "cap": (lambda g: {"c": g.choice(COUNTRIES)},
            ["What is the capital of {c}?", "Which city is the capital of {c}?", "Name the capital city of {c}."]),
}
EASY_W = [("arith", .30), ("unit", .25), ("sent", .20), ("trans", .17), ("cap", .08)]
HARD = {
    "code": (lambda g: {"task": g.choice(TASKS).format(k=_r(3, 9999, g))},
             ["Write a Python function that {task} and explain its time complexity step by step.",
              "Implement, in Python, a function that {task}; analyze the complexity step by step and discuss edge cases.",
              "Give me a Python implementation of code that {task}, then explain the time complexity and the edge cases."]),
    "math": (lambda g: {"a": _r(20, 900, g), "b": _r(30, 99, g), "c": _r(20, 900, g), "d": _r(30, 99, g)},
             ["A train travels {a} km at {b} km/h and then {c} km at {d} km/h. Derive the average speed and explain your reasoning step by step.",
              "Show step by step how to derive the average speed of a train that goes {a} km at {b} km/h then {c} km at {d} km/h.",
              "Derive, with every step explained, the average speed over {a} km at {b} km/h followed by {c} km at {d} km/h."]),
    "proof": (lambda g: {"k": _r(1, 99999, g)},
              ["Prove by induction that 1 + 2 + ... + n = n(n+1)/2 for all n >= {k}, and justify each step rigorously.",
               "Give a rigorous step by step proof that the sum 1 + 2 + ... + n equals n(n+1)/2 for every n >= {k}.",
               "Justify rigorously, using induction, why 1 + 2 + ... + n = n(n+1)/2 holds for all n >= {k}."]),
    "design": (lambda g: {"s": g.choice(SYSTEMS), "n": _r(2, 999, g) * 100},
               ["Design a {s} that handles {n} requests per second and discuss the trade-offs between latency and consistency.",
                "Analyze the architecture of a {s} serving {n} requests per second, and evaluate the trade-offs you would make.",
                "Justify an architecture for a {s} with {n} requests per second, and critique the trade-offs involved."]),
    "debug": (lambda g: {"k": _r(2, 9999, g)},
              ["Debug this and explain why it fails on edge cases: ```python\ndef f(xs, k={k}):\n    return [sum(xs[i:i+k]) for i in range(len(xs))]\n```",
               "The following code has a bug; find it, explain step by step, and refactor it: ```python\ndef f(xs, k={k}):\n    return [sum(xs[i:i+k]) for i in range(len(xs))]\n```",
               "Explain why this function breaks on edge cases and fix it: ```python\ndef f(xs, k={k}):\n    return [sum(xs[i:i+k]) for i in range(len(xs))]\n```"]),
}
HARD_W = [("code", .30), ("math", .25), ("proof", .15), ("design", .20), ("debug", .10)]


def _pick(weights, g):
    x, acc = g.random(), 0.0
    for name, w in weights:
        acc += w
        if x < acc:
            return name
    return weights[-1][0]


def generate(n_total: int = 5000, seed: int = 7, dup_frac: float = 0.25, para_frac: float = 0.25,
             hard_frac: float = 0.40) -> list[Prompt]:
    g = random.Random(seed)
    n_dup, n_para = int(n_total * dup_frac), int(n_total * para_frac)
    n_base = n_total - n_dup - n_para
    bases: list[dict] = []
    seen: set[str] = set()
    while len(bases) < n_base:
        hard = g.random() < hard_frac
        fam = _pick(HARD_W if hard else EASY_W, g)
        slot_fn, templates = (HARD if hard else EASY)[fam]
        slots = slot_fn(g)
        t = g.randrange(3)
        text = templates[t].format(**slots)
        if text in seen:
            continue
        seen.add(text)
        bases.append({"id": len(bases), "fam": fam, "hard": hard, "slots": slots, "t": t, "text": text, "key": g.random()})
    events = [(b["key"], i, "unique", b) for i, b in enumerate(bases)]
    for kind, count in (("dup", n_dup), ("para", n_para)):
        for _ in range(count):
            b = g.choice(bases)
            text = b["text"]
            if kind == "para":
                templates = (HARD if b["hard"] else EASY)[b["fam"]][1]
                t2 = g.choice([i for i in range(3) if i != b["t"]])
                text = templates[t2].format(**b["slots"])
            events.append((b["key"] + MIN_GAP + g.random() * 0.4, b["id"], kind, {**b, "text": text}))
    events.sort(key=lambda e: e[0])
    return [Prompt(i, e[3]["text"], "hard" if e[3]["hard"] else "easy", e[2], e[3]["id"],
                   e[3]["fam"] + ":" + json.dumps(e[3]["slots"], sort_keys=True)) for i, e in enumerate(events)]


def pairs(n: int = 1000, seed: int = 11) -> list[tuple[str, str, int, str]]:
    """Labeled pairs for the cache threshold study: 1 = should hit (same meaning), 0 = must NOT hit."""
    g = random.Random(seed)
    out = []
    while len(out) < 2 * n:
        hard = g.random() < 0.4
        fam = _pick(HARD_W if hard else EASY_W, g)
        slot_fn, templates = (HARD if hard else EASY)[fam]
        s1 = slot_fn(g)
        t1, t2 = g.sample(range(3), 2)
        if len(out) % 2 == 0:  # paraphrase: same slots, different wording
            out.append((templates[t1].format(**s1), templates[t2].format(**s1), 1, "paraphrase"))
        else:  # near-miss: same wording, different slot values
            s2 = slot_fn(g)
            a, b = templates[t1].format(**s1), templates[t1].format(**s2)
            if a != b:
                out.append((a, b, 0, "near-miss"))
    return out
