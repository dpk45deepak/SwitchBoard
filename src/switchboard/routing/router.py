"""Cost-aware router. Concrete model -> honored (with fallbacks); model="auto" -> cheapest adequate tier."""
from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from ..config import Settings
from ..errors import BadRequest, ContextLengthExceeded
from ..providers.registry import TIERS, ModelSpec, Registry
from ..tokens import estimate_prompt_tokens
from ..types import ChatRequest
from .classifier import HeuristicClassifier, LearnedClassifier


@dataclass
class RouteDecision:
    requested: str
    tier: str
    model: ModelSpec
    chain: list[ModelSpec]
    classifier: str
    score: float | None = None
    features: dict = field(default_factory=dict)
    reason: str = ""
    substituted: bool = False

    def to_log(self, request_id: str, path: str) -> dict:
        return {"ts": time.time(), "request_id": request_id, "path": path, "requested": self.requested,
                "tier": self.tier, "model": self.model.id, "chain": [m.id for m in self.chain],
                "classifier": self.classifier, "score": self.score, "features": self.features,
                "reason": self.reason}


def _price(m: ModelSpec) -> float:
    return m.equiv_in + m.equiv_out


class Router:
    def __init__(self, registry: Registry, settings: Settings, healthy: Callable[[str], bool] | None = None,
                 classifier: object | None = None) -> None:
        self.registry, self.settings = registry, settings
        self.healthy = healthy or (lambda provider: True)
        self.heuristic = HeuristicClassifier()
        if classifier is not None:
            self.classifier = classifier
        elif settings.router_mode == "learned":
            self.classifier = LearnedClassifier.load() or self.heuristic
        else:
            self.classifier = self.heuristic
        self.log: deque[dict] = deque(maxlen=2000)

    def _order(self, models: list[ModelSpec], path: str) -> list[ModelSpec]:
        if path == "batch":  # cheapest healthy model, latency is secondary
            key = lambda m: (not self.healthy(m.provider), _price(m), m.latency_ms)
        else:  # lowest latency among acceptable models
            key = lambda m: (not self.healthy(m.provider), m.latency_ms, _price(m))
        return sorted(models, key=key)

    def route(self, req: ChatRequest, force_tier: str | None = None) -> RouteDecision:
        if req.model.lower() != "auto":
            spec, sub = self.registry.resolve(req.model, self.settings.allow_substitution)
            if force_tier and TIERS.index(spec.tier) > TIERS.index(force_tier):
                return self._auto(req, force_tier, reason=f"budget downgrade from {spec.id}")
            chain = self.registry.fallback_chain(spec, req)
            d = RouteDecision(req.model, spec.tier, chain[0], chain, "explicit", reason="client-specified model"
                              + (" (substituted: provider disabled)" if sub else ""), substituted=sub)
            self.log.append(d.to_log(req.request_id, req.path))
            return d
        return self._auto(req, force_tier)

    def _auto(self, req: ChatRequest, force_tier: str | None, reason: str = "") -> RouteDecision:
        c = self.classifier.classify(req)  # type: ignore[attr-defined]
        tier = c.tier
        why = f"{c.classifier} score={c.score:.2f} -> {tier}"
        if force_tier and TIERS.index(tier) > TIERS.index(force_tier):
            tier, why = force_tier, why + f"; capped at {force_tier} ({reason or 'budget'})"
        pt = estimate_prompt_tokens(req.messages)
        i = TIERS.index(tier)
        tier_order = [TIERS[i]] + [TIERS[j] for j in range(i + 1, len(TIERS))]  # never silently go lower
        if force_tier:
            tier_order += [TIERS[j] for j in range(i - 1, -1, -1)]
        chain: list[ModelSpec] = []
        for t in tier_order:
            ok = [m for m in self.registry.tier_models(t) if self.registry.satisfies(m, req, pt) is None]
            chain += self._order(ok, req.path)
        if not chain:
            any_ctx = any(pt + (req.max_tokens or 0) > m.context_window for m in self.registry.enabled_models())
            raise (ContextLengthExceeded if any_ctx else BadRequest)("no eligible model for this request")
        d = RouteDecision("auto", chain[0].tier, chain[0], chain, c.classifier, c.score, c.features, why)
        self.log.append(d.to_log(req.request_id, req.path))
        return d
