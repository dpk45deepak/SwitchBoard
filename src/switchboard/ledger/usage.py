"""Usage ledger: one row per request (online and batch), with savings versus the naive baseline."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from ..db.session import Database, row_dicts

GROUPS = {"model": "routed_model", "key": "api_key_id", "day": "day", "provider": "provider",
          "path": "path", "cache": "cache_status"}


def utc_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


@dataclass
class LedgerRecord:
    ts: float
    api_key_id: str
    path: str
    lane: str
    requested_model: str
    routed_model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_estimated: bool = False
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    cache_status: str = "MISS"
    cost_usd: float = 0.0
    equiv_cost_usd: float = 0.0
    baseline_cost_usd: float = 0.0
    outcome: str = "ok"
    retry_count: int = 0
    job_id: str | None = None
    task_key: str | None = None

    @property
    def saved_usd(self) -> float:
        return self.baseline_cost_usd - self.equiv_cost_usd if self.outcome == "ok" else 0.0


class Ledger:
    def __init__(self, db: Database) -> None:
        self.db = db

    def record(self, r: LedgerRecord) -> bool:
        """Insert a row. Returns False if ``task_key`` already exists (=> never charged twice)."""
        cur = self.db.execute(
            """INSERT OR IGNORE INTO usage (ts, day, api_key_id, path, lane, job_id, task_key, requested_model,
               routed_model, provider, prompt_tokens, completion_tokens, tokens_estimated, latency_ms, ttft_ms,
               cache_status, cost_usd, equiv_cost_usd, baseline_cost_usd, saved_usd, outcome, retry_count)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (r.ts, utc_day(r.ts), r.api_key_id, r.path, r.lane, r.job_id, r.task_key, r.requested_model,
             r.routed_model, r.provider, r.prompt_tokens, r.completion_tokens, int(r.tokens_estimated),
             r.latency_ms, r.ttft_ms, r.cache_status, r.cost_usd, r.equiv_cost_usd, r.baseline_cost_usd,
             r.saved_usd, r.outcome, r.retry_count))
        return cur.rowcount == 1

    def has_task(self, task_key: str) -> bool:
        return self.db.one("SELECT 1 FROM usage WHERE task_key=?", (task_key,)) is not None

    def spent_today(self, api_key_id: str, day: str) -> float:
        row = self.db.one("SELECT COALESCE(SUM(equiv_cost_usd),0) s FROM usage WHERE api_key_id=? AND day=?",
                          (api_key_id, day))
        return float(row["s"]) if row else 0.0

    def job_cost(self, job_id: str) -> dict:
        row = self.db.one("""SELECT COUNT(*) n, COALESCE(SUM(cost_usd),0) cost, COALESCE(SUM(equiv_cost_usd),0) equiv,
                             COALESCE(SUM(saved_usd),0) saved,
                             COALESCE(SUM(cache_status LIKE 'HIT%'),0) hits FROM usage
                             WHERE job_id=? AND outcome='ok'""", (job_id,))
        return dict(row) if row else {}

    def usage(self, group_by: str = "model", since: float | None = None) -> list[dict]:
        col = GROUPS.get(group_by)
        if col is None:
            raise ValueError(f"group_by must be one of {sorted(GROUPS)}")
        where, params = ("WHERE ts >= ?", [since]) if since else ("", [])
        return row_dicts(self.db.query(
            f"""SELECT {col} AS "group", COUNT(*) requests, SUM(prompt_tokens) prompt_tokens,
                SUM(completion_tokens) completion_tokens, SUM(cost_usd) cost_usd,
                SUM(equiv_cost_usd) equiv_cost_usd, SUM(saved_usd) saved_usd,
                SUM(cache_status LIKE 'HIT%') cache_hits, AVG(latency_ms) avg_latency_ms
                FROM usage {where} GROUP BY {col} ORDER BY requests DESC""", params))

    def savings(self, since: float | None = None) -> dict:
        where, params = ("WHERE ts >= ? AND outcome='ok'", [since]) if since else ("WHERE outcome='ok'", [])
        row = self.db.one(f"""SELECT COUNT(*) requests, COALESCE(SUM(baseline_cost_usd),0) baseline,
                              COALESCE(SUM(equiv_cost_usd),0) equiv, COALESCE(SUM(cost_usd),0) billed,
                              COALESCE(SUM(cache_status LIKE 'HIT%'),0) hits FROM usage {where}""", params)
        d = dict(row)
        d["saved_usd"] = d["baseline"] - d["equiv"]
        d["saved_pct"] = 100.0 * d["saved_usd"] / d["baseline"] if d["baseline"] else 0.0
        d["cache_hit_rate"] = d["hits"] / d["requests"] if d["requests"] else 0.0
        return d

    def timeseries(self, bucket_s: float = 10.0, since: float | None = None) -> list[dict]:
        since = since if since is not None else time.time() - 900
        rows = self.db.query("SELECT ts, latency_ms, equiv_cost_usd, saved_usd, cache_status, outcome, path "
                             "FROM usage WHERE ts >= ? ORDER BY ts", (since,))
        buckets: dict[int, list] = {}
        for r in rows:
            buckets.setdefault(int(r["ts"] // bucket_s), []).append(r)
        out = []
        for b in sorted(buckets):
            rs = buckets[b]
            lat = sorted(x["latency_ms"] for x in rs if x["path"] == "online" and x["outcome"] == "ok")
            pct = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))] if lat else None  # noqa: E731
            out.append({"t": b * bucket_s, "requests": len(rs), "equiv_cost": sum(x["equiv_cost_usd"] for x in rs),
                        "saved": sum(x["saved_usd"] for x in rs),
                        "hit_rate": sum(x["cache_status"].startswith("HIT") for x in rs) / len(rs),
                        "p50_ms": pct(0.5), "p95_ms": pct(0.95)})
        return out
