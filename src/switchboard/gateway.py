"""The shared pipeline used by BOTH the online path and the batch workers:

    client limit -> cache -> budget -> router -> (breaker, provider limiter, provider) with failover -> ledger
"""
from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from .cache.embedder import get_embedder
from .cache.semantic_cache import CacheEntry, CacheHit, SemanticCache
from .clock import Clock
from .config import Settings
from .db.session import Database
from .errors import (AdmissionRejected, BudgetExceeded, Deferred, ProviderAuthError, ProviderBadRequest,
                     ProviderError, ProviderTimeout, ProviderUnavailable, RateLimited, StreamInterrupted,
                     SwitchboardError)
from .ledger.budgets import ANONYMOUS, ApiKey, BudgetManager, KeyStore
from .ledger.pricing import baseline_cost_usd, cost_usd, equiv_cost_usd
from .ledger.usage import Ledger, LedgerRecord
from .limits.admission import admit
from .limits.circuit_breaker import OPEN, CircuitBreaker
from .limits.token_bucket import InMemoryRateLimiter, LimitSpec, RateLimiter
from .observability.metrics import Metrics
from .providers.base import Provider
from .providers.registry import ModelSpec, Registry
from .routing.router import RouteDecision, Router
from .tokens import count_text, estimate_prompt_tokens
from .types import ChatRequest, ChatResponse, Chunk, Usage

log = logging.getLogger("switchboard.gateway")


@dataclass
class Meta:
    request_id: str
    cache_status: str = "MISS"
    similarity: float | None = None
    provider: str | None = None
    model: str | None = None
    tier: str | None = None
    cost_usd: float = 0.0
    cost_equiv: float = 0.0
    cost_saved: float = 0.0
    budget: str = "ok"
    route_reason: str = ""
    skipped: list[str] = field(default_factory=list)

    def headers(self, with_cost: bool = True) -> dict[str, str]:
        h = {"X-Request-Id": self.request_id, "X-Cache": self.cache_status}
        if self.similarity is not None:
            h["X-Cache-Similarity"] = f"{self.similarity:.4f}"
        if self.provider:
            h["X-Provider"] = self.provider
        if self.model:
            h["X-Model"] = self.model
        if self.tier:
            h["X-Routing-Tier"] = self.tier
        if with_cost:
            h["X-Cost-USD"] = f"{self.cost_usd:.8f}"
            h["X-Cost-Equivalent-USD"] = f"{self.cost_equiv:.8f}"
            h["X-Cost-Saved-USD"] = f"{self.cost_saved:.8f}"
        if self.budget != "ok":
            h["X-Budget-Warning"] = self.budget
        return h


class Gateway:
    def __init__(self, settings: Settings, registry: Registry, providers: dict[str, Provider],
                 limiter: RateLimiter, ledger: Ledger, cache: SemanticCache, keys: KeyStore,
                 budgets: BudgetManager, metrics: Metrics, clock: Clock, rng: random.Random | None = None,
                 router: Router | None = None) -> None:
        self.settings, self.registry, self.providers = settings, registry, providers
        self.limiter, self.ledger, self.cache = limiter, ledger, cache
        self.keys, self.budgets, self.metrics, self.clock = keys, budgets, metrics, clock
        self.rng = rng or random.Random()
        self.breakers = {n: CircuitBreaker(clock) for n in providers}
        self.router = router or Router(registry, settings, healthy=self._healthy)
        self._inflight: dict[str, asyncio.Future] = {}
        for k, spec in registry.limit_specs():
            limiter.configure(k, spec)
        metrics.add_collector(self._collect)

    # -- helpers -----------------------------------------------------------------------------
    def _healthy(self, provider: str) -> bool:
        b = self.breakers.get(provider)
        return b is None or b.state != OPEN

    def _collect(self, m: Metrics) -> None:
        for n, b in self.breakers.items():
            m.gauge("breaker_state", {"closed": 0, "half_open": 1, "open": 2}[b.state], provider=n)
        m.gauge("cache_entries", len(self.cache))
        m.gauge("cache_hit_rate", self.cache.hit_rate())

    def _estimate(self, req: ChatRequest) -> int:
        return estimate_prompt_tokens(req.messages) + (req.max_tokens or 256)

    def _fail(self, req: ChatRequest, exc: SwitchboardError):
        if req.path == "batch":
            return Deferred(exc.message, retry_after=exc.retry_after)
        return exc

    async def _admit_client(self, req: ChatRequest, key: ApiKey) -> None:
        if not (key.rpm or key.tpm):
            return
        k = f"client:{key.id}"
        self.limiter.configure(k, LimitSpec(rpm=key.rpm, tpm=key.tpm, priority=False))
        d = await self.limiter.acquire(k, "interactive", 1, self._estimate(req))
        if not d.allowed:
            self.metrics.inc("limiter_rejections_total", scope="client")
            raise self._fail(req, AdmissionRejected("client rate limit exceeded", retry_after=d.retry_after))

    def _hit_response(self, req: ChatRequest, hit: CacheHit, meta: Meta, strongest: ModelSpec) -> ChatResponse:
        e = hit.entry
        usage = Usage(**{k: e.usage[k] for k in ("prompt_tokens", "completion_tokens")})
        meta.cache_status = "HIT-EXACT" if hit.kind == "exact" else "HIT-SEMANTIC"
        meta.similarity = hit.similarity if hit.kind == "semantic" else None
        meta.provider, meta.model = e.provider, e.model
        meta.cost_saved = baseline_cost_usd(strongest, usage)
        return ChatResponse(content=e.content, usage=usage, finish_reason=e.finish_reason, tool_calls=e.tool_calls)

    def _record(self, req: ChatRequest, meta: Meta, usage: Usage | None, spec: ModelSpec | None, t0: float,
                outcome: str = "ok", ttft_ms: float | None = None, estimated: bool = False) -> None:
        usage = usage or Usage()
        strongest = self.registry.strongest()
        hit = meta.cache_status.startswith("HIT")
        actual = 0.0 if (hit or spec is None) else cost_usd(spec, usage)
        equiv = 0.0 if (hit or spec is None) else equiv_cost_usd(spec, usage)
        base = baseline_cost_usd(strongest, usage) if outcome == "ok" else 0.0
        meta.cost_usd, meta.cost_equiv = actual, equiv
        meta.cost_saved = base - equiv if outcome == "ok" else 0.0
        rec = LedgerRecord(
            ts=self.clock.now(), api_key_id=req.api_key_id, path=req.path, lane=req.lane,
            requested_model=req.model, routed_model=meta.model or "-", provider=meta.provider or "-",
            prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
            tokens_estimated=estimated, latency_ms=(self.clock.now() - t0) * 1000, ttft_ms=ttft_ms,
            cache_status=meta.cache_status, cost_usd=actual, equiv_cost_usd=equiv, baseline_cost_usd=base,
            outcome=outcome, retry_count=req.retry_count, job_id=req.job_id,
            task_key=req.task_key if outcome == "ok" else None)
        if self.ledger.record(rec):
            self.budgets.note(req.api_key_id, equiv)
        self.metrics.inc("requests_total", path=req.path, outcome=outcome, cache=meta.cache_status)
        self.metrics.observe("request_latency_seconds", rec.latency_ms / 1000, path=req.path)
        if outcome == "ok":
            self.metrics.inc("cost_equiv_usd_total", equiv)
            self.metrics.inc("cost_saved_usd_total", meta.cost_saved)

    def _store(self, ckey, req: ChatRequest, content: str | None, usage: Usage, finish: str, tool_calls,
               spec: ModelSpec) -> CacheEntry | None:
        if ckey is None or finish not in ("stop", "tool_calls"):
            return None
        now = self.clock.now()
        from .cache.semantic_cache import guard_tokens
        e = CacheEntry(content=content, usage=usage.to_dict(), finish_reason=finish, tool_calls=tool_calls,
                       provider=spec.provider, model=spec.id, equiv_cost=equiv_cost_usd(spec, usage),
                       created=now, expires=now + self.cache.ttl_s, namespace=ckey.namespace,
                       family=ckey.family, guard=guard_tokens(ckey.text or ""))
        self.cache.store(ckey, e)
        return e

    def _budget_gate(self, req: ChatRequest, key: ApiKey, meta: Meta) -> str | None:
        st = self.budgets.check(key)
        if st.status == "hard":
            if key.downgrade:
                meta.budget = "downgraded"
                return "cheap"
            raise BudgetExceeded(f"daily budget of ${st.budget:.2f} exhausted")
        if st.status == "soft":
            meta.budget = "soft"
            log.warning("soft budget threshold reached", extra={"fields": {"api_key_id": key.id}})
        return None

    # -- provider chain ------------------------------------------------------------------------
    async def _attempt_chain(self, req: ChatRequest, decision: RouteDecision, meta: Meta, stream: bool):
        """Try each model in the chain. Returns (spec, response) or (spec, (agen, first_chunk))."""
        last_err: SwitchboardError | None = None
        blocked_limit: list[float] = []
        blocked_breaker: list[float] = []
        max_wait = self.settings.online_max_wait_s if req.path == "online" else self.settings.batch_max_wait_s
        est = self._estimate(req)
        for spec in decision.chain:
            br = self.breakers[spec.provider]
            if br.state == OPEN:
                blocked_breaker.append(br.retry_after() or 1.0)
                meta.skipped.append(f"{spec.id}:breaker")
                continue
            try:
                await admit(self.limiter, self.registry.limit_key(spec), req.lane, self.clock, max_wait,
                            tokens=est, rng=self.rng)
            except AdmissionRejected as e:
                self.metrics.inc("limiter_rejections_total", scope=spec.provider)
                blocked_limit.append(e.retry_after or 1.0)
                meta.skipped.append(f"{spec.id}:limit")
                continue
            if not br.allow():
                blocked_breaker.append(br.retry_after() or 1.0)
                meta.skipped.append(f"{spec.id}:breaker")
                continue
            provider = self.providers[spec.provider]
            try:
                if stream:
                    agen = provider.stream(req, spec.upstream)
                    try:
                        first = await asyncio.wait_for(agen.__anext__(), self.settings.request_timeout_s)
                    except StopAsyncIteration:
                        first = None
                    br.record_success()
                    return spec, (agen, first)
                resp = await asyncio.wait_for(provider.chat(req, spec.upstream), self.settings.request_timeout_s)
                br.record_success()
                return spec, resp
            except asyncio.TimeoutError:
                br.record_failure()
                last_err = ProviderTimeout(f"{spec.provider} timed out")
            except RateLimited as e:
                br.record_success()  # provider is alive; our limits are just too generous
                self.metrics.inc("provider_429_total", provider=spec.provider)
                last_err = e
            except ProviderBadRequest:
                br.record_success()
                raise
            except (ProviderUnavailable, ProviderTimeout, ProviderAuthError, StreamInterrupted) as e:
                br.record_failure()
                self.metrics.inc("provider_errors_total", provider=spec.provider, code=e.code)
                last_err = e
            meta.skipped.append(f"{spec.id}:error")
        if last_err is not None:
            raise last_err
        ra = min(blocked_limit + blocked_breaker or [1.0])
        if req.path == "batch":
            raise Deferred("no provider capacity right now", retry_after=ra)
        if blocked_limit:
            raise AdmissionRejected("provider capacity exhausted", retry_after=ra)
        raise ProviderUnavailable("all eligible providers are unavailable", retry_after=ra)

    # -- online / batch (non-streaming) ---------------------------------------------------------
    async def handle(self, req: ChatRequest, key: ApiKey | None = None) -> tuple[ChatResponse, Meta]:
        key = key or ANONYMOUS
        req.api_key_id = key.id
        t0 = self.clock.now()
        meta = Meta(req.request_id)
        await self._admit_client(req, key)
        ckey = None
        flight: asyncio.Future | None = None
        if self.cache.cacheable(req):
            ckey = self.cache.prepare(req)
            hit = self.cache.lookup(ckey)
            if hit is None and ckey.exact in self._inflight:  # single-flight: wait for the identical request
                try:
                    entry = await asyncio.shield(self._inflight[ckey.exact])
                except Exception:
                    entry = None
                if entry is not None:
                    hit = CacheHit(entry, "exact", 1.0)
            if hit is not None:
                resp = self._hit_response(req, hit, meta, self.registry.strongest())
                self._record(req, meta, resp.usage, None, t0)
                return resp, meta
            flight = asyncio.get_running_loop().create_future()
            self._inflight[ckey.exact] = flight
        entry: CacheEntry | None = None
        try:
            force = self._budget_gate(req, key, meta)
            decision = self.router.route(req, force_tier=force)
            meta.tier, meta.route_reason = decision.tier, decision.reason
            try:
                spec, resp = await self._attempt_chain(req, decision, meta, stream=False)
            except ProviderError as e:
                meta.provider = meta.provider or getattr(e, "provider", None)
                self._record(req, meta, None, None, t0, outcome=e.code)
                raise
            meta.provider, meta.model = spec.provider, spec.id
            self._record(req, meta, resp.usage, spec, t0, estimated=resp.usage.estimated)
            entry = self._store(ckey, req, resp.content, resp.usage, resp.finish_reason, resp.tool_calls, spec)
            return resp, meta
        finally:
            if flight is not None and ckey is not None:
                self._inflight.pop(ckey.exact, None)
                if not flight.done():
                    flight.set_result(entry)

    # -- streaming ------------------------------------------------------------------------------
    async def handle_stream(self, req: ChatRequest, key: ApiKey | None = None) -> tuple[Meta, AsyncIterator[Chunk]]:
        """Returns (meta, chunk iterator). Provider/model are known before the first chunk is yielded so
        headers can be sent; fail-over is only possible before the first chunk."""
        key = key or ANONYMOUS
        req.api_key_id = key.id
        t0 = self.clock.now()
        meta = Meta(req.request_id)
        await self._admit_client(req, key)
        ckey = None
        if self.cache.cacheable(req):
            ckey = self.cache.prepare(req)
            hit = self.cache.lookup(ckey)
            if hit is not None:
                resp = self._hit_response(req, hit, meta, self.registry.strongest())
                self._record(req, meta, resp.usage, None, t0)
                return meta, self._replay(resp)
        force = self._budget_gate(req, key, meta)
        decision = self.router.route(req, force_tier=force)
        meta.tier, meta.route_reason = decision.tier, decision.reason
        try:
            spec, (agen, first) = await self._attempt_chain(req, decision, meta, stream=True)
        except ProviderError as e:
            self._record(req, meta, None, None, t0, outcome=e.code)
            raise
        meta.provider, meta.model = spec.provider, spec.id
        return meta, self._relay(req, meta, spec, agen, first, ckey, t0)

    @staticmethod
    async def _replay(resp: ChatResponse) -> AsyncIterator[Chunk]:
        words = (resp.content or "").split(" ")
        for i in range(0, len(words), 4):
            yield Chunk(content=" ".join(words[i:i + 4]) + (" " if i + 4 < len(words) else ""))
        yield Chunk(finish_reason=resp.finish_reason, usage=resp.usage)

    async def _relay(self, req, meta, spec, agen, first, ckey, t0) -> AsyncIterator[Chunk]:
        parts: list[str] = []
        usage: Usage | None = None
        finish = None
        ttft = None
        outcome = "client_disconnect"  # overwritten on normal completion
        br = self.breakers[spec.provider]
        try:
            chunk = first
            while chunk is not None:
                if ttft is None and chunk.content:
                    ttft = (self.clock.now() - t0) * 1000
                if chunk.content:
                    parts.append(chunk.content)
                if chunk.usage:
                    usage = chunk.usage
                if chunk.finish_reason:
                    finish = chunk.finish_reason
                yield chunk
                try:
                    chunk = await asyncio.wait_for(agen.__anext__(), self.settings.request_timeout_s)
                except StopAsyncIteration:
                    chunk = None
            outcome = "ok"
        except (ProviderError, asyncio.TimeoutError) as e:
            outcome = "stream_error"
            br.record_failure()
            if isinstance(e, asyncio.TimeoutError):
                raise StreamInterrupted("stream timed out") from e
            raise
        finally:
            try:
                await agen.aclose()
            except Exception:
                pass
            text = "".join(parts)
            estimated = usage is None
            u = usage or Usage(estimate_prompt_tokens(req.messages), count_text(text), estimated=True)
            if outcome == "ok":
                self._record(req, meta, u, spec, t0, ttft_ms=ttft, estimated=estimated)
                self._store(ckey, req, text, u, finish or "stop", None, spec)
            else:
                self._record(req, meta, u if parts else None, spec if parts else None, t0, outcome=outcome, ttft_ms=ttft)

    # -- introspection ---------------------------------------------------------------------------
    def breaker_states(self) -> dict[str, dict]:
        return {n: {"state": b.state, "retry_after": round(b.retry_after(), 2)} for n, b in self.breakers.items()}


def build_gateway(settings: Settings | None = None, clock: Clock | None = None, db: Database | None = None,
                  providers: dict[str, Provider] | None = None, limiter: RateLimiter | None = None,
                  registry: Registry | None = None, rng: random.Random | None = None) -> Gateway:
    settings = settings or Settings()
    clock = clock or Clock()
    registry = registry or Registry.load(settings=settings)
    providers = providers if providers is not None else registry.build_providers(clock, settings.request_timeout_s)
    registry.providers = providers
    db = db or Database(settings.db_path)
    if limiter is None:
        if settings.redis_url:
            import redis.asyncio as aioredis  # optional dependency
            from .limits.token_bucket import RedisRateLimiter
            limiter = RedisRateLimiter(aioredis.from_url(settings.redis_url))
        else:
            limiter = InMemoryRateLimiter(clock)
    ledger = Ledger(db)
    embedder = get_embedder(settings.embedder, settings.embedder_model)
    semantic = settings.cache_semantic == "on" or (settings.cache_semantic == "auto" and embedder.name != "hashing")
    cache = SemanticCache(clock, embedder,
                          sim_threshold=settings.cache_sim_threshold, ttl_s=settings.cache_ttl_s,
                          temp_threshold=settings.cache_temp_threshold, max_entries=settings.cache_max_entries,
                          guard=settings.cache_guard, enabled=settings.cache_enabled, semantic=semantic)
    gw = Gateway(settings, registry, providers, limiter, ledger, cache, KeyStore(db, clock),
                 BudgetManager(ledger, clock), Metrics(), clock, rng)
    gw.db = db  # type: ignore[attr-defined]
    return gw
