"""Local embedders. No paid API is ever used.

* ``HashingEmbedder``: dependency-free (numpy only) hashed word/bigram/char-trigram vectors.
  Deterministic and fast, but lexical: it is the lite-mode / test fallback, NOT a semantic model.
* ``SentenceTransformerEmbedder``: ``BAAI/bge-small-en-v1.5`` (or any sentence-transformers model)
  when the optional ``embeddings`` extra is installed.
"""
from __future__ import annotations

import re
import zlib
from typing import Protocol

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _add(self, v: np.ndarray, feat: str, w: float) -> None:
        h = zlib.crc32(feat.encode())
        v[h % self.dim] += w if (h >> 16) & 1 else -w

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            words = _WORD.findall(t.lower())
            for w in words:
                self._add(out[i], "w:" + w, 1.0)
                if len(w) >= 3:
                    p = f"^{w}$"
                    for j in range(len(p) - 2):
                        self._add(out[i], "c:" + p[j:j + 3], 0.35)
            for a, b in zip(words, words[1:]):
                self._add(out[i], f"b:{a}_{b}", 0.7)
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


class SentenceTransformerEmbedder:  # pragma: no cover - optional dependency
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model
        self._m = SentenceTransformer(model)
        self.dim = int(self._m.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._m.encode(texts, normalize_embeddings=True), dtype=np.float32)


def get_embedder(kind: str = "auto", model: str = "BAAI/bge-small-en-v1.5") -> Embedder:
    if kind in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder(model)
        except Exception:
            if kind != "auto":
                raise
    return HashingEmbedder()
