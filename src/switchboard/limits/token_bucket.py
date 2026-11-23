"""Priority-aware dual token buckets (requests/min + tokens/min) - the core algorithm.

Design (see docs/DECISIONS.md):
  * One *main* bucket per dimension is the hard cap. Nothing is ever admitted unless the main
    bucket has the tokens, so the provider limit can never be exceeded.
  * Priority via **reserved capacity**: each lane has a ``floor`` (fraction of bucket capacity). A
    lane may only take tokens while the level stays >= its floor. Interactive has floor 0 and can
    drain the bucket; batch/bulk only get what is left over.
  * Starvation guard: low lanes also own a small *guaranteed* sub-bucket (``share`` of the refill
    rate). Guaranteed tokens are held back from other lanes, so batch keeps making progress even
    when interactive traffic would otherwise consume 100% of the capacity.
  * Window guarantee: refill rate is ``(limit - capacity) / 60`` so that in ANY 60 s window at most
    ``capacity + rate*60 = limit`` units are admitted (a plain bucket with rate=limit/60 would allow
    ~2x the limit in the first minute).

The same algorithm is implemented twice behind one interface: ``InMemoryRateLimiter`` (lite mode,
unit tests, injectable clock) and ``RedisRateLimiter`` (atomic Lua script).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from ..clock import Clock

WINDOW_S = 60.0
LANES = ("interactive", "normal", "batch", "bulk")


@dataclass(frozen=True)
class LaneConfig:
    floor: float  # fraction of capacity that must remain after this lane takes tokens
    share: float  # guaranteed fraction of the refill rate (starvation guard)


DEFAULT_LANES: dict[str, LaneConfig] = {
    "interactive": LaneConfig(0.00, 0.00),
    "normal": LaneConfig(0.15, 0.10),
    "batch": LaneConfig(0.35, 0.10),
    "bulk": LaneConfig(0.50, 0.05),
}
NO_PRIORITY: dict[str, LaneConfig] = {ln: LaneConfig(0.0, 0.0) for ln in LANES}


@dataclass(frozen=True)
class LimitSpec:
    rpm: float | None = None
    tpm: float | None = None
    burst_fraction: float = 0.10
    priority: bool = True  # False -> single FIFO-ish bucket (used for per-client limits)


@dataclass(frozen=True)
class Dim:
    name: str
    cap: float
    rate: float  # units per second

    @classmethod
    def from_limit(cls, name: str, limit: float, burst_fraction: float, min_cap: float) -> "Dim":
        cap = min(limit / 2.0, max(limit * burst_fraction, min_cap))
        cap = max(cap, 1e-9)
        return cls(name, cap, (limit - cap) / WINDOW_S)


def dims_for(spec: LimitSpec) -> list[Dim]:
    out: list[Dim] = []
    if spec.rpm:
        out.append(Dim.from_limit("rpm", spec.rpm, spec.burst_fraction, 1.0))
    if spec.tpm:
        out.append(Dim.from_limit("tpm", spec.tpm, spec.burst_fraction, 2048.0))
    return out


@dataclass
class BucketState:
    level: float
    ts: float
    g: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    retry_after: float = 0.0


def refill(d: Dim, st: BucketState, lanes: dict[str, LaneConfig], now: float) -> None:
    dt = now - st.ts
    if dt < 0:  # clock went backwards: re-anchor, never create tokens from negative time
        dt = 0.0
    st.level = min(d.cap, st.level + dt * d.rate)
    for ln, lc in lanes.items():
        if lc.share > 0:
            gcap = lc.share * d.cap
            st.g[ln] = min(gcap, st.g.get(ln, gcap) + dt * d.rate * lc.share)
    st.ts = now


def decide(d: Dim, st: BucketState, lane: str, cost: float, lanes: dict[str, LaneConfig]) -> tuple[int, float]:
    """Return (mode, wait_s). mode 0 = deny, 1 = normal path, 2 = guaranteed path."""
    lc = lanes.get(lane, LaneConfig(0.0, 0.0))
    c = min(cost, d.cap)
    others = min(sum(v for ln, v in st.g.items() if ln != lane), st.level)
    floor = lc.floor * d.cap
    if st.level - others - c >= floor - 1e-9:
        return 1, 0.0
    gcap = lc.share * d.cap
    own = st.g.get(lane, 0.0)
    if lc.share > 0 and own >= min(c, gcap) - 1e-9 and st.level >= c - 1e-9:
        return 2, 0.0
    waits = [(c + floor + others - st.level) / d.rate if d.rate > 0 else 1e9]
    if lc.share > 0 and d.rate > 0:
        waits.append(max((min(c, gcap) - own) / (d.rate * lc.share), (c - st.level) / d.rate, 0.0))
    return 0, max(min(waits), 1e-3)


def commit(d: Dim, st: BucketState, lane: str, cost: float, mode: int, lanes: dict[str, LaneConfig]) -> None:
    c = min(cost, d.cap)
    st.level -= c
    if mode == 2:
        gcap = lanes[lane].share * d.cap
        st.g[lane] = max(0.0, st.g.get(lane, 0.0) - min(c, gcap))


class RateLimiter(Protocol):
    def configure(self, key: str, spec: LimitSpec) -> None: ...
    async def acquire(self, key: str, lane: str, requests: float = 1, tokens: float = 0) -> Decision: ...


class InMemoryRateLimiter:
    """Single-process limiter. acquire() has no await between check and commit, so it is atomic
    with respect to other coroutines."""

    def __init__(self, clock: Clock | None = None, lanes: dict[str, LaneConfig] | None = None) -> None:
        self.clock = clock or Clock()
        self.lanes = DEFAULT_LANES if lanes is None else lanes
        self._specs: dict[str, LimitSpec] = {}
        self._dims: dict[str, list[Dim]] = {}
        self._state: dict[tuple[str, str], BucketState] = {}
        self.rejections: dict[str, int] = {}

    def configure(self, key: str, spec: LimitSpec) -> None:
        if self._specs.get(key) == spec:
            return
        self._specs[key] = spec
        self._dims[key] = dims_for(spec)
        for k in [k for k in self._state if k[0] == key]:
            del self._state[k]

    def _lanes_for(self, key: str) -> dict[str, LaneConfig]:
        return self.lanes if self._specs[key].priority else NO_PRIORITY

    async def acquire(self, key: str, lane: str, requests: float = 1, tokens: float = 0) -> Decision:
        if key not in self._specs:
            return Decision(True)
        now = self.clock.now()
        lanes = self._lanes_for(key)
        plan: list[tuple[Dim, BucketState, int, float]] = []
        worst = 0.0
        for d in self._dims[key]:
            cost = requests if d.name == "rpm" else tokens
            st = self._state.get((key, d.name))
            if st is None:
                st = self._state[(key, d.name)] = BucketState(d.cap, now)
            refill(d, st, lanes, now)
            if cost <= 0:
                continue
            mode, wait = decide(d, st, lane, cost, lanes)
            plan.append((d, st, mode, cost))
            worst = max(worst, wait)
        if any(m == 0 for _, _, m, _ in plan):
            self.rejections[key] = self.rejections.get(key, 0) + 1
            return Decision(False, worst)
        for d, st, mode, cost in plan:
            commit(d, st, lane, cost, mode, lanes)
        return Decision(True)

    def level(self, key: str, dim: str = "rpm") -> float:
        st = self._state.get((key, dim))
        return st.level if st else float("nan")


LUA_ACQUIRE = """
local cfg = cjson.decode(ARGV[1])
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local lane = cfg.lane
local lc = cfg.lanes[lane] or {floor = 0, share = 0}
local states = {}
local allowed = true
local wait = 0
for i, d in ipairs(cfg.dims) do
  local level = tonumber(redis.call('HGET', KEYS[1], d.name .. ':l'))
  local ts = tonumber(redis.call('HGET', KEYS[1], d.name .. ':t'))
  if level == nil then level = d.cap; ts = now end
  local dt = now - ts
  if dt < 0 then dt = 0 end
  level = math.min(d.cap, level + dt * d.rate)
  local g = {}
  local others = 0
  for ln, c in pairs(cfg.lanes) do
    if c.share > 0 then
      local gcap = c.share * d.cap
      local v = tonumber(redis.call('HGET', KEYS[1], d.name .. ':g:' .. ln))
      if v == nil then v = gcap end
      v = math.min(gcap, v + dt * d.rate * c.share)
      g[ln] = v
      if ln ~= lane then others = others + v end
    end
  end
  others = math.min(others, level)
  local cst = math.min(d.cost, d.cap)
  local floor = lc.floor * d.cap
  local mode = 0
  if cst > 0 then
    if level - others - cst >= floor - 1e-9 then
      mode = 1
    else
      local gcap = lc.share * d.cap
      local own = g[lane] or 0
      if lc.share > 0 and own >= math.min(cst, gcap) - 1e-9 and level >= cst - 1e-9 then mode = 2 end
    end
    if mode == 0 then
      allowed = false
      local w = 1e9
      if d.rate > 0 then w = (cst + floor + others - level) / d.rate end
      if lc.share > 0 and d.rate > 0 then
        local gcap = lc.share * d.cap
        local own = g[lane] or 0
        local w2 = math.max((math.min(cst, gcap) - own) / (d.rate * lc.share), (cst - level) / d.rate, 0)
        if w2 < w then w = w2 end
      end
      if w > wait then wait = w end
    end
  end
  states[i] = {level = level, g = g, cst = cst, mode = mode}
end
if allowed then
  for i, d in ipairs(cfg.dims) do
    local s = states[i]
    local newlevel = s.level - s.cst
    if s.mode == 2 then
      local gcap = lc.share * d.cap
      s.g[lane] = math.max(0, (s.g[lane] or 0) - math.min(s.cst, gcap))
    end
    redis.call('HSET', KEYS[1], d.name .. ':l', tostring(newlevel), d.name .. ':t', tostring(now))
    for ln, v in pairs(s.g) do
      redis.call('HSET', KEYS[1], d.name .. ':g:' .. ln, tostring(v))
    end
  end
  redis.call('PEXPIRE', KEYS[1], 3600000)
  return {1, '0'}
end
if wait < 0.001 then wait = 0.001 end
return {0, tostring(wait)}
"""


class RedisRateLimiter:
    """Same algorithm as InMemoryRateLimiter, executed atomically inside Redis (needs Redis >= 5).

    Uses Redis TIME as the single clock source, so multiple gateway nodes share one timeline.
    NOTE: not exercised against a live Redis in the build sandbox (no Redis available); the logic
    mirrors the unit-tested Python implementation line by line.
    """

    def __init__(self, redis_client, lanes: dict[str, LaneConfig] | None = None, prefix: str = "sb:lim:") -> None:
        self.r = redis_client
        self.lanes = DEFAULT_LANES if lanes is None else lanes
        self.prefix = prefix
        self._specs: dict[str, LimitSpec] = {}
        self._script = redis_client.register_script(LUA_ACQUIRE)

    def configure(self, key: str, spec: LimitSpec) -> None:
        self._specs[key] = spec

    async def acquire(self, key: str, lane: str, requests: float = 1, tokens: float = 0) -> Decision:
        spec = self._specs.get(key)
        if spec is None:
            return Decision(True)
        lanes = self.lanes if spec.priority else NO_PRIORITY
        dims = [{"name": d.name, "cap": d.cap, "rate": d.rate,
                 "cost": requests if d.name == "rpm" else tokens} for d in dims_for(spec)]
        cfg = {"lane": lane, "dims": dims,
               "lanes": {k: {"floor": v.floor, "share": v.share} for k, v in lanes.items()}}
        ok, wait = await self._script(keys=[self.prefix + key], args=[json.dumps(cfg)])
        return Decision(int(ok) == 1, float(wait))
