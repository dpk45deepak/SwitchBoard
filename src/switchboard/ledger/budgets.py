"""API keys (hashed at rest) and daily budget enforcement."""
from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass

from ..clock import Clock
from ..db.session import Database
from .usage import Ledger, utc_day


@dataclass
class ApiKey:
    id: str
    name: str = ""
    rpm: float | None = None
    tpm: float | None = None
    daily_budget_usd: float | None = None
    soft_ratio: float = 0.8
    downgrade: bool = False


ANONYMOUS = ApiKey(id="anonymous", name="anonymous")


def hash_key(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class KeyStore:
    def __init__(self, db: Database, clock: Clock) -> None:
        self.db, self.clock = db, clock

    def create(self, name: str = "", rpm: float | None = None, tpm: float | None = None,
               daily_budget_usd: float | None = None, soft_ratio: float = 0.8, downgrade: bool = False) -> tuple[str, ApiKey]:
        token = "sb-" + secrets.token_urlsafe(24)
        kid = "key_" + uuid.uuid4().hex[:10]
        self.db.execute("INSERT INTO api_keys (id,name,key_hash,created,rpm,tpm,daily_budget_usd,soft_ratio,downgrade)"
                        " VALUES (?,?,?,?,?,?,?,?,?)",
                        (kid, name, hash_key(token), self.clock.now(), rpm, tpm, daily_budget_usd, soft_ratio, int(downgrade)))
        return token, ApiKey(kid, name, rpm, tpm, daily_budget_usd, soft_ratio, downgrade)

    def authenticate(self, token: str | None) -> ApiKey | None:
        if not token:
            return None
        r = self.db.one("SELECT * FROM api_keys WHERE key_hash=? AND disabled=0", (hash_key(token),))
        if r is None:
            return None
        return ApiKey(r["id"], r["name"], r["rpm"], r["tpm"], r["daily_budget_usd"], r["soft_ratio"], bool(r["downgrade"]))

    def list(self) -> list[dict]:
        return [{k: r[k] for k in ("id", "name", "rpm", "tpm", "daily_budget_usd", "disabled")}
                for r in self.db.query("SELECT * FROM api_keys")]


@dataclass
class BudgetState:
    status: str  # ok | soft | hard
    spent: float
    budget: float | None


class BudgetManager:
    """Tracks equivalent-cost spend per key per UTC day (in-process counter seeded from the ledger)."""

    def __init__(self, ledger: Ledger, clock: Clock) -> None:
        self.ledger, self.clock = ledger, clock
        self._spent: dict[tuple[str, str], float] = {}

    def _day(self) -> str:
        return utc_day(self.clock.now())

    def spent(self, key_id: str) -> float:
        k = (key_id, self._day())
        if k not in self._spent:
            self._spent[k] = self.ledger.spent_today(key_id, k[1])
        return self._spent[k]

    def check(self, key: ApiKey) -> BudgetState:
        if key.daily_budget_usd is None:
            return BudgetState("ok", self.spent(key.id), None)
        s = self.spent(key.id)
        if s >= key.daily_budget_usd:
            return BudgetState("hard", s, key.daily_budget_usd)
        if s >= key.daily_budget_usd * key.soft_ratio:
            return BudgetState("soft", s, key.daily_budget_usd)
        return BudgetState("ok", s, key.daily_budget_usd)

    def note(self, key_id: str, amount: float) -> None:
        k = (key_id, self._day())
        self._spent[k] = self.spent(key_id) + amount


def _get_key(self: KeyStore, key_id: str) -> ApiKey:
    if key_id == ANONYMOUS.id:
        return ANONYMOUS
    r = self.db.one("SELECT * FROM api_keys WHERE id=?", (key_id,))
    if r is None:
        return ApiKey(key_id)
    return ApiKey(r["id"], r["name"], r["rpm"], r["tpm"], r["daily_budget_usd"], r["soft_ratio"], bool(r["downgrade"]))


KeyStore.get = _get_key  # type: ignore[attr-defined]
