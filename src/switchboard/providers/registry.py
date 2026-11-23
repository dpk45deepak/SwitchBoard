"""Model registry: loads models.yaml, resolves names/aliases, builds fallback chains."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..clock import Clock
from ..config import Settings
from ..errors import ContextLengthExceeded, ModelNotFound, BadRequest
from ..limits.token_bucket import LimitSpec
from ..tokens import estimate_prompt_tokens
from ..types import ChatRequest
from .base import Provider
from .mock import MockProvider
from .openai_compat import OpenAICompatProvider

TIERS = ("cheap", "balanced", "strong")


@dataclass
class ModelSpec:
    id: str
    provider: str
    upstream: str
    tier: str
    context_window: int = 8192
    price_in: float = 0.0
    price_out: float = 0.0
    equiv_in: float = 0.0
    equiv_out: float = 0.0
    aliases: list[str] = field(default_factory=list)
    supports_tools: bool = True
    latency_ms: float = 500.0
    rpm: float | None = None
    tpm: float | None = None


@dataclass
class ProviderSpec:
    name: str
    type: str
    base_url: str | None = None
    api_key_env: str | None = None
    base_url_env: str | None = None
    enabled_env: str | None = None
    rpm: float | None = None
    tpm: float | None = None


class Registry:
    def __init__(self, models: list[ModelSpec], providers: dict[str, ProviderSpec],
                 fallbacks: dict[str, list[str]], enabled: set[str]) -> None:
        self.models = {m.id: m for m in models}
        self.provider_specs = providers
        self.fallbacks = fallbacks
        self.enabled = enabled
        self.providers: dict[str, Provider] = {}
        self._alias: dict[str, str] = {}
        for m in models:
            for a in [m.id, m.upstream, *m.aliases]:
                self._alias.setdefault(a.lower(), m.id)

    # -- construction ------------------------------------------------------------------------
    @classmethod
    def load(cls, path: str | None = None, settings: Settings | None = None,
             env: dict[str, str] | None = None) -> "Registry":
        settings = settings or Settings()
        env = os.environ if env is None else env
        p = Path(path or settings.models_yaml or Path(__file__).resolve().parent.parent / "models.yaml")
        cfg: dict[str, Any] = yaml.safe_load(p.read_text())
        ref = cfg.get("reference_prices", {})
        provs = {n: ProviderSpec(name=n, **{k: v for k, v in d.items()}) for n, d in cfg["providers"].items()}
        models: list[ModelSpec] = []
        for m in cfg["models"]:
            m = dict(m)
            r = ref.get(m.get("tier", "balanced"), {"in": 0.0, "out": 0.0})
            m.setdefault("equiv_in", r["in"])
            m.setdefault("equiv_out", r["out"])
            models.append(ModelSpec(**m))
        real = {n for n, s in provs.items() if s.type != "mock" and s.enabled_env and env.get(s.enabled_env)}
        enable_mock = settings.enable_mock if settings.enable_mock is not None else not real
        enabled = set(real)
        if enable_mock:
            enabled |= {n for n, s in provs.items() if s.type == "mock"}
        reg = cls(models, provs, cfg.get("fallbacks", {}), enabled)
        reg._env = dict(env)  # type: ignore[attr-defined]
        return reg

    def build_providers(self, clock: Clock | None = None, timeout: float = 60.0) -> dict[str, Provider]:
        env = getattr(self, "_env", os.environ)
        for name in self.enabled:
            s = self.provider_specs[name]
            if s.type == "mock":
                from .mock import MockBehavior
                scale = float(env.get("MOCK_LATENCY_SCALE", 1))
                beh = MockBehavior(latency_ms={t: v * scale for t, v in MockBehavior().latency_ms.items()},
                                   error_rate=float(env.get("MOCK_ERROR_RATE", 0)))
                self.providers[name] = MockProvider(name=name, clock=clock, behavior=beh)
            else:
                base = (env.get(s.base_url_env) if s.base_url_env else None) or s.base_url or ""
                key = env.get(s.api_key_env) if s.api_key_env else None
                self.providers[name] = OpenAICompatProvider(name, base, key, timeout=timeout)
        return self.providers

    # -- lookup ------------------------------------------------------------------------------
    def is_enabled(self, spec: ModelSpec) -> bool:
        return spec.provider in self.enabled

    def tier_models(self, tier: str) -> list[ModelSpec]:
        return [m for m in self.models.values() if m.tier == tier and self.is_enabled(m)]

    def enabled_models(self) -> list[ModelSpec]:
        return [m for m in self.models.values() if self.is_enabled(m)]

    def strongest(self) -> ModelSpec:
        """Baseline model for savings math: the preferred enabled model of the strong tier."""
        for i in self.fallbacks.get("strong", []):
            m = self.models.get(i)
            if m and self.is_enabled(m):
                return m
        pool = self.tier_models("strong") or self.enabled_models()
        if not pool:
            raise ModelNotFound("no providers are enabled")
        return max(pool, key=lambda m: m.equiv_out + m.equiv_in)

    def resolve(self, name: str, allow_substitution: bool = True) -> tuple[ModelSpec, bool]:
        key = name.lower()
        mid = self._alias.get(key)
        if mid is None and "/" in key:
            mid = self._alias.get(key.split("/", 1)[1])
        if mid is None:
            raise ModelNotFound(f"model '{name}' is not available")
        spec = self.models[mid]
        if self.is_enabled(spec):
            return spec, False
        if allow_substitution:
            pool = self.tier_models(spec.tier)
            if pool:
                return min(pool, key=lambda m: m.equiv_out + m.equiv_in), True
        raise ModelNotFound(f"model '{name}' is not available (provider '{spec.provider}' is not enabled)")

    def limit_key(self, spec: ModelSpec) -> str:
        return f"{spec.provider}:{spec.id}" if (spec.rpm or spec.tpm) else spec.provider

    def limit_specs(self) -> list[tuple[str, LimitSpec]]:
        out = [(n, LimitSpec(rpm=self.provider_specs[n].rpm, tpm=self.provider_specs[n].tpm))
               for n in sorted(self.enabled)]
        out += [(self.limit_key(m), LimitSpec(rpm=m.rpm, tpm=m.tpm))
                for m in self.enabled_models() if m.rpm or m.tpm]
        return out

    # -- constraints & chains ----------------------------------------------------------------
    def satisfies(self, spec: ModelSpec, req: ChatRequest, prompt_tokens: int | None = None) -> str | None:
        """Return None if the model can serve the request, else the reason it can't."""
        pt = prompt_tokens if prompt_tokens is not None else estimate_prompt_tokens(req.messages)
        if pt + (req.max_tokens or 0) > spec.context_window:
            return "context"
        if req.tools and not spec.supports_tools:
            return "tools"
        return None

    def fallback_chain(self, primary: ModelSpec, req: ChatRequest) -> list[ModelSpec]:
        """primary first, then the tier's configured fallbacks. Models that cannot satisfy the
        request's explicit constraints (context length, tool support) are never used."""
        pt = estimate_prompt_tokens(req.messages)
        ids = [primary.id] + [i for i in self.fallbacks.get(primary.tier, []) if i != primary.id]
        chain = [self.models[i] for i in ids if i in self.models and self.is_enabled(self.models[i])]
        ok = [m for m in chain if self.satisfies(m, req, pt) is None]
        if not ok:
            reasons = {self.satisfies(m, req, pt) for m in chain}
            if "context" in reasons:
                raise ContextLengthExceeded("request exceeds the context window of every eligible model")
            raise BadRequest("no eligible model supports the requested features (e.g. tools)")
        return ok
