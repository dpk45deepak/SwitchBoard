"""Structured JSON logs with request IDs and optional PII redaction. Prompts are never logged by default."""
from __future__ import annotations

import contextvars
import json
import logging
import re
import time

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{8,}\d")


def redact(text: str) -> str:
    return _PHONE.sub("[PHONE]", _EMAIL.sub("[EMAIL]", text))


class JsonFormatter(logging.Formatter):
    def __init__(self, redact_pii: bool = True) -> None:
        super().__init__()
        self.redact_pii = redact_pii

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        d = {"ts": round(time.time(), 3), "level": record.levelname, "logger": record.name,
             "request_id": request_id_var.get(), "msg": redact(msg) if self.redact_pii else msg}
        extra = getattr(record, "fields", None)
        if extra:
            d.update(extra)
        return json.dumps(d, default=str)


def setup_logging(level: str = "INFO", redact_pii: bool = True) -> None:
    h = logging.StreamHandler()
    h.setFormatter(JsonFormatter(redact_pii))
    root = logging.getLogger("switchboard")
    root.handlers[:] = [h]
    root.setLevel(level)
    root.propagate = False
