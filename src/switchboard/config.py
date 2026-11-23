"""Settings loaded from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _b(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass
class Settings:
    db_path: str = "switchboard.db"
    redis_url: str | None = None
    models_yaml: str | None = None
    require_auth: bool = False
    admin_key: str | None = None
    enable_mock: bool | None = None  # None = auto (only when no real provider is configured)
    cache_enabled: bool = True
    cache_temp_threshold: float = 0.3
    cache_sim_threshold: float = 0.92
    cache_ttl_s: float = 3600.0
    cache_max_entries: int = 50_000
    cache_semantic: str = "auto"  # auto (on only with a real embedding model) | on | off
    cache_guard: bool = True  # semantic hits must agree on numbers / proper nouns
    embedder: str = "auto"  # auto | hashing | sentence-transformers
    embedder_model: str = "BAAI/bge-small-en-v1.5"
    router_mode: str = "heuristic"  # heuristic | learned
    online_max_wait_s: float = 2.0
    batch_max_wait_s: float = 30.0
    request_timeout_s: float = 60.0
    max_body_bytes: int = 2_000_000
    log_prompts: bool = False
    log_redact_pii: bool = True
    log_level: str = "INFO"
    allow_substitution: bool = True  # serve e.g. "gpt-4o" from an enabled same-tier model
    embedded_worker: bool = False
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        em = os.environ.get("ENABLE_MOCK")
        return cls(
            db_path=os.environ.get("SWITCHBOARD_DB", "switchboard.db"),
            redis_url=os.environ.get("REDIS_URL") or None,
            models_yaml=os.environ.get("MODELS_YAML") or None,
            require_auth=_b("REQUIRE_AUTH", False),
            admin_key=os.environ.get("ADMIN_KEY") or None,
            enable_mock=None if em is None else _b("ENABLE_MOCK", True),
            cache_enabled=_b("CACHE_ENABLED", True),
            cache_temp_threshold=_f("CACHE_TEMP_THRESHOLD", 0.3),
            cache_sim_threshold=_f("CACHE_SIM_THRESHOLD", 0.92),
            cache_ttl_s=_f("CACHE_TTL_S", 3600),
            cache_semantic=os.environ.get("CACHE_SEMANTIC", "auto"),
            cache_guard=_b("CACHE_GUARD", True),
            embedder=os.environ.get("EMBEDDER", "auto"),
            embedder_model=os.environ.get("EMBEDDER_MODEL", "BAAI/bge-small-en-v1.5"),
            router_mode=os.environ.get("ROUTER_MODE", "heuristic"),
            online_max_wait_s=_f("ONLINE_MAX_WAIT_S", 2.0),
            batch_max_wait_s=_f("BATCH_MAX_WAIT_S", 30.0),
            request_timeout_s=_f("REQUEST_TIMEOUT_S", 60),
            max_body_bytes=int(_f("MAX_BODY_BYTES", 2_000_000)),
            log_prompts=_b("LOG_PROMPTS", False),
            log_redact_pii=_b("LOG_REDACT_PII", True),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            allow_substitution=_b("ALLOW_SUBSTITUTION", True),
            embedded_worker=_b("EMBEDDED_WORKER", False),
        )
