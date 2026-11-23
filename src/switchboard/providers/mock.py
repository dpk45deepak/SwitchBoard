"""Deterministic mock provider: configurable latency, errors, rate limits, outages.

Answer model (used by the quality benchmark): the "reference answer" for a prompt is
``ANS-<sha1(answer_key)[:10]>``. Cheaper tiers return a ``WRONG-...`` token with a
per-(prompt, tier) deterministic probability that depends on the prompt's hidden
difficulty (``hints["difficulty"]``). This is a *simulation* of tier quality, not a
measurement of any real model.
"""
from __future__ import annotations

import hashlib
import random
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from ..clock import Clock
from ..errors import ProviderTimeout, ProviderUnavailable, RateLimited, StreamInterrupted
from ..tokens import count_text, estimate_prompt_tokens
from ..types import ChatRequest, ChatResponse, Chunk, Usage, content_text
from .base import Provider

P_WRONG = {
    "cheap": {"easy": 0.03, "hard": 0.45},
    "balanced": {"easy": 0.01, "hard": 0.15},
    "strong": {"easy": 0.0, "hard": 0.0},
}
FILLER = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor".split()


def _h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def reference_answer(key: str) -> str:
    return "ANS-" + _h(key)[:10]


@dataclass
class MockBehavior:
    latency_ms: dict[str, float] = field(default_factory=lambda: {"cheap": 4, "balanced": 8, "strong": 12})
    jitter: float = 0.2
    error_rate: float = 0.0
    rate_limit: int | None = None  # max accepted calls per window
    window_s: float = 60.0
    outage: bool = False
    stream_fail_after: int | None = None
    timeout_share: float = 0.2  # share of injected errors that are timeouts


class MockProvider(Provider):
    def __init__(self, name: str = "mock", clock: Clock | None = None, seed: int = 0,
                 behavior: MockBehavior | None = None) -> None:
        self.name = name
        self.clock = clock or Clock()
        self.rng = random.Random(seed)
        self.b = behavior or MockBehavior()
        self.accepted: list[float] = []  # timestamps of accepted calls (for compliance checks)
        self._window: deque[float] = deque()
        self.n_429 = 0
        self.n_errors = 0
        self.n_calls = 0

    @staticmethod
    def tier_of(upstream: str) -> str:
        for t in ("cheap", "balanced", "strong"):
            if upstream.endswith(t):
                return t
        return "balanced"

    async def _gate(self, tier: str) -> None:
        self.n_calls += 1
        now = self.clock.now()
        if self.b.outage:
            self.n_errors += 1
            raise ProviderUnavailable("mock outage")
        if self.b.rate_limit is not None:
            while self._window and now - self._window[0] >= self.b.window_s:
                self._window.popleft()
            if len(self._window) >= self.b.rate_limit:
                self.n_429 += 1
                raise RateLimited("mock rate limit",
                                  retry_after=max(0.001, self._window[0] + self.b.window_s - now))
            self._window.append(now)
        self.accepted.append(now)
        if self.b.error_rate and self.rng.random() < self.b.error_rate:
            self.n_errors += 1
            if self.rng.random() < self.b.timeout_share:
                await self.clock.sleep(0.002)
                raise ProviderTimeout("mock timeout")
            raise ProviderUnavailable("mock 5xx")
        lat = self.b.latency_ms.get(tier, 8) / 1000.0
        await self.clock.sleep(lat * (1 + self.b.jitter * self.rng.random()))

    def _content(self, req: ChatRequest, tier: str) -> tuple[str, Usage]:
        last_user = next((content_text(m.get("content")) for m in reversed(req.messages)
                          if m.get("role") == "user"), "")
        key = str(req.hints.get("answer_key") or last_user)
        diff = req.hints.get("difficulty", "easy")
        draw = int(_h(key + "|" + tier)[:8], 16) / 0xFFFFFFFF
        wrong = draw < P_WRONG[tier].get(diff, 0.0)
        ans = "WRONG-" + _h(key + tier)[:10] if wrong else reference_answer(key)
        n_words = int(req.hints.get("out_words") or (160 if diff == "hard" else 24))
        text = ans + " " + " ".join(FILLER[i % len(FILLER)] for i in range(n_words))
        if req.max_tokens:
            text = " ".join(text.split()[: max(1, int(req.max_tokens / 1.3))])
        usage = Usage(estimate_prompt_tokens(req.messages), count_text(text))
        return text, usage

    async def chat(self, req: ChatRequest, upstream_model: str) -> ChatResponse:
        tier = self.tier_of(upstream_model)
        await self._gate(tier)
        text, usage = self._content(req, tier)
        return ChatResponse(content=text, usage=usage)

    async def stream(self, req: ChatRequest, upstream_model: str) -> AsyncIterator[Chunk]:
        tier = self.tier_of(upstream_model)
        await self._gate(tier)
        text, usage = self._content(req, tier)
        words = text.split(" ")
        per = self.b.latency_ms.get(tier, 8) / 1000.0 / 8
        for i in range(0, len(words), 4):
            if self.b.stream_fail_after is not None and i // 4 >= self.b.stream_fail_after:
                raise StreamInterrupted("mock stream failure")
            await self.clock.sleep(per)
            piece = " ".join(words[i:i + 4]) + (" " if i + 4 < len(words) else "")
            yield Chunk(content=piece)
        yield Chunk(finish_reason="stop", usage=usage)

    def models(self) -> list[str]:
        return ["mock-cheap", "mock-balanced", "mock-strong"]
