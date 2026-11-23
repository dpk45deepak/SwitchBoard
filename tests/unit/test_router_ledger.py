import unittest

from switchboard.config import Settings
from switchboard.db.session import Database
from switchboard.errors import ContextLengthExceeded, ModelNotFound
from switchboard.ledger.budgets import ApiKey, BudgetManager, KeyStore
from switchboard.ledger.pricing import baseline_cost_usd, cost_usd, equiv_cost_usd
from switchboard.ledger.usage import Ledger, LedgerRecord
from switchboard.clock import FakeClock
from switchboard.providers.registry import Registry
from switchboard.routing.classifier import HeuristicClassifier, LearnedClassifier, extract_features
from switchboard.routing.router import Router
from switchboard.types import ChatRequest, Usage


def R(text, model="auto", **kw):
    return ChatRequest(model=model, messages=[{"role": "user", "content": text}], temperature=0, **kw)


class TestRouter(unittest.TestCase):
    def setUp(self):
        self.reg = Registry.load(settings=Settings())
        self.router = Router(self.reg, Settings())

    def test_features(self):
        f = extract_features(R("```python\ndef f(): pass\n``` Analyze step by step why this is O(n)"))
        self.assertEqual(f["has_code"], 1.0)
        self.assertGreaterEqual(f["n_reason"], 2)

    def test_easy_goes_cheap_hard_goes_strong(self):
        self.assertEqual(self.router.route(R("What is the capital of France?")).tier, "cheap")
        hard = "Design a payment ledger and discuss the trade-offs step by step, then implement it and analyze complexity."
        self.assertEqual(self.router.route(R(hard)).tier, "strong")

    def test_explicit_model_is_honored_with_fallbacks(self):
        d = self.router.route(R("hi", model="mock-strong"))
        self.assertEqual(d.model.id, "mock-strong")
        self.assertEqual([m.id for m in d.chain][:2], ["mock-strong", "mock2-strong"])

    def test_unknown_model_and_substitution(self):
        with self.assertRaises(ModelNotFound):
            self.router.route(R("hi", model="nonexistent-9000"))
        d = self.router.route(R("hi", model="gpt-4o"))
        self.assertTrue(d.substituted)

    def test_context_constraint_error(self):
        with self.assertRaises(ContextLengthExceeded):
            self.router.route(R("word " * 700_000, model="auto"))

    def test_batch_prefers_cheapest_online_prefers_fastest(self):
        a = self.router.route(R("What is the capital of France?", path="batch"))
        self.assertEqual(a.model.id, "mock-cheap")
        self.assertIn("auto", a.requested)

    def test_budget_downgrade_caps_tier(self):
        hard = "Design a payment ledger and discuss the trade-offs step by step, then implement it."
        self.assertEqual(self.router.route(R(hard), force_tier="cheap").tier, "cheap")

    def test_decisions_are_logged(self):
        self.router.route(R("hello"))
        self.assertIn("reason", self.router.log[-1])

    def test_learned_classifier_if_model_present(self):
        lc = LearnedClassifier.load()
        if lc is None:
            self.skipTest("router_model.pkl missing")
        self.assertEqual(lc.classify(R("Which city is the capital of Peru?")).tier, "cheap")
        self.assertEqual(lc.classify(R("Prove that for every integer n >= 2 the sum of the first n odd numbers is n squared. "
                                       "Show each step of the induction rigorously.")).tier, "strong")


class TestLedger(unittest.TestCase):
    def setUp(self):
        self.reg = Registry.load(settings=Settings())
        self.db = Database()
        self.clock = FakeClock(1_700_000_000.0)
        self.ledger = Ledger(self.db)

    def rec(self, **kw):
        base = dict(ts=self.clock.now(), api_key_id="k", path="online", lane="interactive", requested_model="auto",
                    routed_model="mock-cheap", provider="mock", prompt_tokens=1000, completion_tokens=1000,
                    equiv_cost_usd=0.00075, baseline_cost_usd=0.0125)
        base.update(kw)
        return LedgerRecord(**base)

    def test_cost_arithmetic(self):
        cheap, strong = self.reg.models["mock-cheap"], self.reg.models["mock-strong"]
        u = Usage(1_000_000, 1_000_000)
        self.assertAlmostEqual(equiv_cost_usd(cheap, u), 0.15 + 0.60)
        self.assertAlmostEqual(baseline_cost_usd(strong, u), 2.5 + 10.0)
        self.assertEqual(cost_usd(cheap, u), 0.0)  # free tier: nothing billed

    def test_savings_percentage(self):
        self.ledger.record(self.rec())
        self.ledger.record(self.rec(cache_status="HIT-EXACT", equiv_cost_usd=0.0))
        s = self.ledger.savings()
        self.assertAlmostEqual(s["baseline"], 0.025)
        self.assertAlmostEqual(s["saved_pct"], 100 * (0.025 - 0.00075) / 0.025)
        self.assertAlmostEqual(s["cache_hit_rate"], 0.5)

    def test_task_key_is_charged_once(self):
        self.assertTrue(self.ledger.record(self.rec(task_key="j:1")))
        self.assertFalse(self.ledger.record(self.rec(task_key="j:1")))
        self.assertEqual(self.ledger.savings()["requests"], 1)

    def test_failed_requests_do_not_count_as_savings(self):
        self.ledger.record(self.rec(outcome="provider_unavailable", equiv_cost_usd=0, baseline_cost_usd=0))
        self.assertEqual(self.ledger.savings()["requests"], 0)

    def test_usage_grouping_and_bad_group(self):
        self.ledger.record(self.rec())
        self.assertEqual(self.ledger.usage("model")[0]["group"], "mock-cheap")
        with self.assertRaises(ValueError):
            self.ledger.usage("nonsense")

    def test_budget_states_and_keys_hashed(self):
        ks = KeyStore(self.db, self.clock)
        token, key = ks.create("t", daily_budget_usd=1.0)
        self.assertNotIn(token, str(self.db.query("SELECT * FROM api_keys")[0]["key_hash"]))
        self.assertEqual(ks.authenticate(token).id, key.id)
        self.assertIsNone(ks.authenticate("sb-wrong"))
        bm = BudgetManager(self.ledger, self.clock)
        self.assertEqual(bm.check(key).status, "ok")
        bm.note(key.id, 0.85)
        self.assertEqual(bm.check(key).status, "soft")
        bm.note(key.id, 0.2)
        self.assertEqual(bm.check(key).status, "hard")
        self.assertEqual(bm.check(ApiKey("free")).status, "ok")


if __name__ == "__main__":
    unittest.main()
