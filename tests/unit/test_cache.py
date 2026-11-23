import unittest

from switchboard.cache.embedder import HashingEmbedder
from switchboard.cache.semantic_cache import SemanticCache, guard_tokens
from switchboard.clock import FakeClock
from switchboard.types import ChatRequest


def req(text, model="gpt-4o", temp=0.0, system=None, **kw):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    return ChatRequest(model=model, messages=msgs, temperature=temp, **kw)


def mk(**kw):
    c = FakeClock()
    return c, SemanticCache(c, HashingEmbedder(), **kw)


def put(cache, r, content="A"):
    k = cache.prepare(r)
    assert cache.lookup(k) is None
    cache.store(k, content=content, usage={"prompt_tokens": 5, "completion_tokens": 5}, finish_reason="stop",
                tool_calls=None, provider="p", model="m", equiv_cost=0.1)


class TestCache(unittest.TestCase):
    def test_exact_hit_and_miss(self):
        c, ca = mk()
        put(ca, req("What is the capital of France?"))
        self.assertEqual(ca.lookup(ca.prepare(req("What is the capital of France?"))).kind, "exact")
        self.assertIsNone(ca.lookup(ca.prepare(req("Completely unrelated question about turbines"))))

    def test_cacheability_rules(self):
        _, ca = mk(temp_threshold=0.3)
        self.assertTrue(ca.cacheable(req("x", temp=0.2)))
        self.assertFalse(ca.cacheable(req("x", temp=0.9)))
        self.assertFalse(ca.cacheable(req("x", temp=None)))
        r = req("x"); r.cache_bypass = True
        self.assertFalse(ca.cacheable(r))

    def test_namespaces_isolate_system_prompt_and_model(self):
        _, ca = mk()
        put(ca, req("hello there friend", system="You are a pirate"))
        self.assertIsNone(ca.lookup(ca.prepare(req("hello there friend", system="You are a lawyer"))))
        self.assertIsNone(ca.lookup(ca.prepare(req("hello there friend", system="You are a pirate", model="other"))))

    def test_semantic_hit_on_close_paraphrase(self):
        _, ca = mk(sim_threshold=0.8)
        put(ca, req("What is the capital city of France today?"))
        h = ca.lookup(ca.prepare(req("What is the capital city of France?")))
        self.assertIsNotNone(h)
        self.assertEqual(h.kind, "semantic")
        self.assertGreaterEqual(h.similarity, 0.8)

    def test_guard_blocks_number_and_entity_swaps(self):
        _, g = mk(sim_threshold=0.5, guard=True)
        _, n = mk(sim_threshold=0.5, guard=False)
        for c in (g, n):
            put(c, req("Convert 120 kilometers to miles for Spain"))
        near = req("Convert 150 kilometers to miles for Spain")
        self.assertIsNone(g.lookup(g.prepare(near)))
        self.assertIsNotNone(n.lookup(n.prepare(near)))  # without the guard it is a false positive
        self.assertEqual(guard_tokens("Capital of France in 1999"), frozenset({"france", "1999"}))

    def test_ttl_expiry(self):
        c, ca = mk(ttl_s=10)
        put(ca, req("hello world again"))
        c.advance(11)
        self.assertIsNone(ca.lookup(ca.prepare(req("hello world again"))))

    def test_multi_turn_uses_exact_only(self):
        _, ca = mk(sim_threshold=0.1)
        r = ChatRequest("m", [{"role": "user", "content": "a b c"}, {"role": "assistant", "content": "x"},
                              {"role": "user", "content": "a b c d"}], temperature=0)
        self.assertIsNone(ca.prepare(r).text)

    def test_invalidate_by_family_and_all(self):
        _, ca = mk()
        put(ca, req("one two three", model="A"))
        put(ca, req("four five six", model="B"))
        self.assertEqual(ca.invalidate(family="A"), 1)
        self.assertIsNone(ca.lookup(ca.prepare(req("one two three", model="A"))))
        self.assertIsNotNone(ca.lookup(ca.prepare(req("four five six", model="B"))))
        self.assertEqual(ca.invalidate(), 1)
        self.assertEqual(len(ca), 0)

    def test_eviction_respects_max_entries(self):
        _, ca = mk(max_entries=20)
        for i in range(60):
            put(ca, req(f"unique prompt number {i} about topic {i * 7}"))
        self.assertLessEqual(len(ca), 20)

    def test_key_normalization_ignores_whitespace_in_semantic_text_only(self):
        _, ca = mk()
        a = ca.prepare(req("hello   world"))
        b = ca.prepare(req("hello world"))
        self.assertEqual(a.text, b.text)
        self.assertNotEqual(a.exact, b.exact)


if __name__ == "__main__":
    unittest.main()
