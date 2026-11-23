"""Token counting: tiktoken if available, otherwise a ~4 chars/token estimate."""
from __future__ import annotations

from .types import content_text

try:  # pragma: no cover - optional dependency
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # offline / not installed
    _ENC = None


def count_text(text: str) -> int:
    if not text:
        return 0
    if _ENC is not None:  # pragma: no cover
        return len(_ENC.encode(text))
    return max(1, (len(text) + 3) // 4)


def estimate_prompt_tokens(messages: list[dict]) -> int:
    return sum(count_text(content_text(m.get("content"))) + 4 for m in messages) + 2
