"""SQLite schema (stdlib sqlite3; WAL mode so several worker processes can share one file)."""
SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER);
CREATE TABLE IF NOT EXISTS api_keys (
  id TEXT PRIMARY KEY, name TEXT, key_hash TEXT UNIQUE, created REAL,
  rpm REAL, tpm REAL, daily_budget_usd REAL, soft_ratio REAL DEFAULT 0.8, downgrade INTEGER DEFAULT 0,
  disabled INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, day TEXT, api_key_id TEXT, path TEXT, lane TEXT,
  job_id TEXT, task_key TEXT UNIQUE, requested_model TEXT, routed_model TEXT, provider TEXT,
  prompt_tokens INTEGER, completion_tokens INTEGER, tokens_estimated INTEGER,
  latency_ms REAL, ttft_ms REAL, cache_status TEXT,
  cost_usd REAL, equiv_cost_usd REAL, baseline_cost_usd REAL, saved_usd REAL,
  outcome TEXT, retry_count INTEGER);
CREATE INDEX IF NOT EXISTS usage_ts ON usage(ts);
CREATE INDEX IF NOT EXISTS usage_key_day ON usage(api_key_id, day);
CREATE INDEX IF NOT EXISTS usage_job ON usage(job_id);
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, created REAL, api_key_id TEXT, priority TEXT, model TEXT, max_concurrency INTEGER,
  idempotency_key TEXT, callback_url TEXT, metadata TEXT, total INTEGER,
  cancelled INTEGER DEFAULT 0, callback_sent INTEGER DEFAULT 0, UNIQUE(api_key_id, idempotency_key));
CREATE TABLE IF NOT EXISTS tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, idx INTEGER, rank INTEGER, payload TEXT,
  status TEXT, attempts INTEGER DEFAULT 0, available_at REAL, lease_until REAL, owner TEXT,
  result TEXT, error TEXT, finished REAL, UNIQUE(job_id, idx));
CREATE INDEX IF NOT EXISTS tasks_claim ON tasks(status, available_at);
CREATE INDEX IF NOT EXISTS tasks_job ON tasks(job_id, status);
"""
