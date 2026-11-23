"""Cost arithmetic. All prices are USD per 1M tokens."""
from __future__ import annotations

from ..providers.registry import ModelSpec
from ..types import Usage


def cost_usd(spec: ModelSpec, u: Usage) -> float:
    """What you actually pay (0 on free tiers)."""
    return (u.prompt_tokens * spec.price_in + u.completion_tokens * spec.price_out) / 1_000_000


def equiv_cost_usd(spec: ModelSpec, u: Usage) -> float:
    """Equivalent cost at the reference price (makes savings meaningful on free tiers)."""
    return (u.prompt_tokens * spec.equiv_in + u.completion_tokens * spec.equiv_out) / 1_000_000


def baseline_cost_usd(strongest: ModelSpec, u: Usage) -> float:
    """Naive baseline: same tokens, strongest model, no cache."""
    return equiv_cost_usd(strongest, u)
