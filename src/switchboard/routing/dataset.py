"""Synthetic easy/hard prompt generator used to train the learned router.

PROVENANCE: fully synthetic, produced by the templates below with a fixed seed (no external data,
no LLM labelling). Label 0 = easy (a small model suffices), 1 = hard (needs reasoning/code).
The benchmark workload (benchmarks/workload_gen.py) uses a *different* generator with different
wording, so quality numbers measure transfer between the two generators, not memorisation.
Real traffic is messier: treat accuracies here as optimistic.
"""
from __future__ import annotations

import random

COUNTRIES = "France Japan Brazil Kenya Canada Egypt Norway Peru Vietnam Chile Poland Ghana Nepal Cuba Iran".split()
WORDS = "river lantern copper harvest quiet bridge marble thunder velvet anchor meadow ember".split()
LANGS = ["Spanish", "German", "Italian", "Portuguese", "Dutch", "Swedish"]
FEELINGS = ["The delivery was fast and the packaging was lovely", "The app crashes whenever I open settings",
            "Service was okay, nothing special", "I absolutely adore this blender", "Totally overpriced and flimsy",
            "The staff went out of their way to help us"]
STRUCTS = ["LRU cache", "priority queue", "interval tree", "trie", "ring buffer", "bloom filter", "skip list"]
OPS = ["insert", "delete", "lookup", "range query", "merge", "resize", "snapshot"]
SYSTEMS = ["URL shortener", "chat service", "ride-matching backend", "rate limiter", "video transcoding pipeline",
           "payment ledger", "notification system", "search autocomplete service"]
TOPICS = [("PostgreSQL", "MongoDB"), ("Kafka", "RabbitMQ"), ("REST", "GraphQL"), ("monolith", "microservices"),
          ("Redis", "Memcached"), ("Rust", "Go"), ("TCP", "QUIC")]
VEHICLES = ["train", "cyclist", "delivery truck", "ferry", "drone"]


def _easy(rng: random.Random) -> str:
    k = rng.randrange(10)
    if k == 0:
        return f"Which city is the capital of {rng.choice(COUNTRIES)}?"
    if k == 1:
        return f"Translate the word '{rng.choice(WORDS)}' into {rng.choice(LANGS)}."
    if k == 2:
        return f"Is this review positive or negative? \"{rng.choice(FEELINGS)}.\""
    if k == 3:
        return f"What is {rng.randrange(10, 9999)} plus {rng.randrange(10, 9999)}?"
    if k == 4:
        return f"Convert {rng.randrange(2, 500)} kilometers to miles."
    if k == 5:
        return f"Give me a synonym for '{rng.choice(WORDS)}'."
    if k == 6:
        return f"Fix the grammar: 'she go to school {rng.choice(['yesterday', 'every day', 'tomorrow'])} with {rng.choice(WORDS)}'."
    if k == 7:
        return f"Summarize in one sentence: {rng.choice(FEELINGS)}, and I would order again."
    if k == 8:
        return f"How many letters are in the word '{rng.choice(WORDS)}'?"
    return f"Define the word '{rng.choice(WORDS)}' briefly."


def _hard(rng: random.Random) -> str:
    k = rng.randrange(8)
    if k == 0:
        a, b = rng.sample(OPS, 2)
        return (f"Implement a {rng.choice(STRUCTS)} in Python supporting {a} and {b}. "
                "Analyze the time complexity of each operation step by step and discuss edge cases.")
    if k == 1:
        n = rng.randrange(2, 9)
        return ("This function has a subtle bug. Find it, explain why it fails on edge cases, and rewrite it:\n"
                f"```python\ndef window_sum(xs, k={n}):\n    return [sum(xs[i:i+k]) for i in range(len(xs))]\n```")
    if k == 2:
        return (f"Design a {rng.choice(SYSTEMS)} that must handle {rng.randrange(2, 90) * 1000} requests per second. "
                "Discuss the trade-offs between consistency and availability and justify your architecture.")
    if k == 3:
        return (f"Prove that for every integer n >= {rng.randrange(1, 6)}, the sum of the first n odd numbers is "
                "n squared. Show each step of the induction rigorously.")
    if k == 4:
        v, a, b, c, d = rng.choice(VEHICLES), rng.randrange(20, 200), rng.randrange(30, 90), rng.randrange(20, 200), rng.randrange(30, 90)
        return (f"A {v} covers {a} km at {b} km/h and then {c} km at {d} km/h. Derive the average speed and "
                "explain your reasoning step by step.")
    if k == 5:
        x, y = rng.choice(TOPICS)
        return (f"Compare and contrast {x} and {y} for a high-traffic production service. Evaluate the "
                "trade-offs, analyze failure modes, and recommend one with justification.")
    if k == 6:
        return ("Optimize this SQL query and explain how the execution plan changes:\n"
                f"SELECT * FROM orders o JOIN users u ON u.id = o.user_id WHERE o.total > {rng.randrange(10, 900)} "
                "ORDER BY o.created_at;")
    return ("Analyze the concurrency issues and possible race condition in a counter shared by "
            f"{rng.randrange(2, 64)} worker threads, and implement a correct, scalable fix.")


def generate(n: int, seed: int = 0) -> tuple[list[str], list[int]]:
    rng = random.Random(seed)
    texts, labels = [], []
    for i in range(n):
        hard = i % 2
        texts.append(_hard(rng) if hard else _easy(rng))
        labels.append(hard)
    return texts, labels
