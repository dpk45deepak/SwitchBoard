"""Two-layer response cache: exact hash first, then embedding similarity.

Safety properties:
  * namespaced by (requested model family, system prompt, tools, response_format): no cross-context hits
  * only cacheable when temperature <= threshold and the client has not sent ``X-Cache: bypass``
  * semantic layer only for single-turn prompts, with an optional lexical guard (numbers and proper
    nouns must agree) because embeddings happily match "capital of France" with "capital of Spain"
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

import numpy as np

from ..clock import Clock
from ..types import ChatRequest, content_text
from .embedder import Embedder

_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_CAP = re.compile(r"\b[A-Z][a-zA-Z]+\b")


def guard_tokens(text: str) -> frozenset[str]:
    nums = set(_NUM.findall(text))
    caps = set(_CAP.findall(text))
    first = text.strip().split(" ", 1)[0].strip(".,:;!?\"'")
    caps.discard(first)  # sentence-initial capital is not evidence of a proper noun
    return frozenset(nums | {c.lower() for c in caps})


def _sha(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


@dataclass
class CacheKey:
    namespace: str
    family: str
    exact: str
    text: str | None  # text to embed (None -> semantic layer disabled for this request)
    embedding: np.ndarray | None = None


@dataclass
class CacheEntry:
    content: str | None
    usage: dict
    finish_reason: str
    tool_calls: list | None
    provider: str
    model: str
    equiv_cost: float
    created: float
    expires: float
    namespace: str
    family: str
    guard: frozenset[str] = field(default_factory=frozenset)


@dataclass
class CacheHit:
    entry: CacheEntry
    kind: str  # exact | semantic
    similarity: float


class _Index:
    def __init__(self, dim: int) -> None:
        self.mat = np.zeros((64, dim), dtype=np.float32)
        self.n = 0
        self.entries: list[CacheEntry] = []

    def add(self, vec: np.ndarray, entry: CacheEntry) -> None:
        if self.n == self.mat.shape[0]:
            self.mat = np.vstack([self.mat, np.zeros_like(self.mat)])
        self.mat[self.n] = vec
        self.entries.append(entry)
        self.n += 1


class SemanticCache:
    def __init__(self, clock: Clock, embedder: Embedder, sim_threshold: float = 0.92, ttl_s: float = 3600.0,
                 temp_threshold: float = 0.3, max_entries: int = 50_000, guard: bool = True,
                 enabled: bool = True, semantic: bool = True) -> None:
        self.clock, self.embedder = clock, embedder
        self.sim_threshold, self.ttl_s, self.temp_threshold = sim_threshold, ttl_s, temp_threshold
        self.max_entries, self.guard, self.enabled = max_entries, guard, enabled
        self.semantic = semantic
        self._exact: dict[str, CacheEntry] = {}
        self._idx: dict[str, _Index] = {}
        self.stats = {"hit_exact": 0, "hit_semantic": 0, "miss": 0, "store": 0}

    # -- policy ------------------------------------------------------------------------------
    def cacheable(self, req: ChatRequest) -> bool:
        return (self.enabled and not req.cache_bypass and req.temperature is not None
                and req.temperature <= self.temp_threshold)

    def prepare(self, req: ChatRequest) -> CacheKey:
        system = "\n".join(content_text(m.get("content")) for m in req.messages if m.get("role") == "system")
        family = req.model.lower()
        ns = _sha(family, system, json.dumps(req.tools, sort_keys=True), json.dumps(req.response_format, sort_keys=True))
        body = json.dumps({"m": req.messages, "t": round(req.temperature or 0.0, 3), "mt": req.max_tokens,
                           "x": req.extra}, sort_keys=True, ensure_ascii=False)
        non_sys = [m for m in req.messages if m.get("role") != "system"]
        text = None
        if len(non_sys) == 1 and non_sys[0].get("role") == "user":
            text = " ".join(content_text(non_sys[0].get("content")).split())
        return CacheKey(ns, family, _sha(ns, body), text)

    # -- operations --------------------------------------------------------------------------
    def lookup(self, key: CacheKey) -> CacheHit | None:
        now = self.clock.now()
        e = self._exact.get(key.exact)
        if e is not None:
            if e.expires > now:
                self.stats["hit_exact"] += 1
                return CacheHit(e, "exact", 1.0)
            del self._exact[key.exact]
        idx = self._idx.get(key.namespace)
        if key.text is not None and self.semantic:
            key.embedding = self.embedder.embed([key.text])[0]
            if idx is not None and idx.n:
                sims = idx.mat[: idx.n] @ key.embedding
                order = np.argsort(-sims)[:8]
                g = guard_tokens(key.text) if self.guard else None
                for i in order:
                    s = float(sims[i])
                    if s < self.sim_threshold:
                        break
                    cand = idx.entries[int(i)]
                    if cand.expires <= now:
                        continue
                    if g is not None and cand.guard != g:
                        continue
                    self.stats["hit_semantic"] += 1
                    return CacheHit(cand, "semantic", s)
        self.stats["miss"] += 1
        return None

    def store(self, key: CacheKey, entry: CacheEntry | None = None, **kw) -> None:
        now = self.clock.now()
        e = entry or CacheEntry(created=now, expires=now + self.ttl_s, namespace=key.namespace,
                                family=key.family, guard=guard_tokens(key.text or ""), **kw)
        if len(self._exact) >= self.max_entries:
            self._evict_oldest()
        self._exact[key.exact] = e
        if key.text is not None and self.semantic:
            if key.embedding is None:
                key.embedding = self.embedder.embed([key.text])[0]
            idx = self._idx.setdefault(key.namespace, _Index(self.embedder.dim))
            idx.add(key.embedding, e)
        self.stats["store"] += 1

    def _evict_oldest(self) -> None:
        for k in sorted(self._exact, key=lambda k: self._exact[k].created)[: max(1, self.max_entries // 10)]:
            del self._exact[k]
        self._rebuild()

    def _rebuild(self) -> None:
        live = {id(e) for e in self._exact.values()}
        for ns, idx in list(self._idx.items()):
            new = _Index(self.embedder.dim)
            for i, e in enumerate(idx.entries):
                if id(e) in live:
                    new.add(idx.mat[i], e)
            if new.n:
                self._idx[ns] = new
            else:
                del self._idx[ns]

    def invalidate(self, namespace: str | None = None, family: str | None = None) -> int:
        """Drop entries by namespace hash or model family; neither given -> everything."""
        def match(e: CacheEntry) -> bool:
            return ((namespace is None or e.namespace == namespace)
                    and (family is None or e.family == family.lower()))
        dead = [k for k, e in self._exact.items() if match(e)]
        for k in dead:
            del self._exact[k]
        self._rebuild()
        return len(dead)

    def hit_rate(self) -> float:
        h = self.stats["hit_exact"] + self.stats["hit_semantic"]
        t = h + self.stats["miss"]
        return h / t if t else 0.0

    def __len__(self) -> int:
        return len(self._exact)
