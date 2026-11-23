"""Difficulty classification for model="auto": v1 heuristic, v2 learned (scikit-learn)."""
from __future__ import annotations

import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..types import ChatRequest, content_text

CODE_RE = re.compile(r"```|\bdef \w+\(|\bclass \w+|\bimport \w+|#include|\bfunction\b|=>|\bSELECT\b.*\bFROM\b|\bO\(", re.I)
MATH_RE = re.compile(r"\b(integral|derivative|theorem|lemma|prove|proof|equation|probability|matrix|eigen|"
                     r"modulo|prime|induction|combinator\w*|expected value)\b", re.I)
REASON_KW = ("step by step", "step-by-step", "explain why", "explain how", "trade-off", "tradeoff", "analyze",
             "analyse", "compare and contrast", "derive", "design a", "design an", "optimize", "optimise",
             "architecture", "complexity", "edge case", "rigorous", "show your work", "refactor", "debug",
             "implement", "justify", "evaluate", "critique", "scalab", "concurrency", "race condition")
EASY_KW = ("translate", "capital of", "classify", "sentiment", "convert", "in one sentence", "what is the",
           "spell", "synonym", "define ", "rewrite", "fix the grammar", "how many", "yes or no", "plus", "minus")
FEATURE_NAMES = ["log_chars", "words100", "turns", "has_code", "has_math", "n_reason", "n_easy", "has_tools",
                 "has_schema", "nonascii", "n_questions", "digits_ratio"]


def last_user_text(req: ChatRequest) -> str:
    for m in reversed(req.messages):
        if m.get("role") == "user":
            return content_text(m.get("content"))
    return req.text()


def extract_features(req: ChatRequest) -> dict[str, float]:
    full = req.text()
    last = last_user_text(req)
    low = full.lower()
    words = len(full.split())
    return {
        "log_chars": math.log1p(len(full)) / 10.0,
        "words100": min(words / 100.0, 5.0),
        "turns": float(sum(1 for m in req.messages if m.get("role") == "user")),
        "has_code": 1.0 if CODE_RE.search(full) else 0.0,
        "has_math": 1.0 if MATH_RE.search(full) else 0.0,
        "n_reason": float(sum(1 for k in REASON_KW if k in low)),
        "n_easy": float(sum(1 for k in EASY_KW if k in low)),
        "has_tools": 1.0 if req.tools else 0.0,
        "has_schema": 1.0 if (req.response_format or {}).get("type") == "json_schema" else 0.0,
        "nonascii": sum(1 for c in last if ord(c) > 127) / max(1, len(last)),
        "n_questions": float(last.count("?")),
        "digits_ratio": sum(c.isdigit() for c in last) / max(1, len(last)),
    }


def features_from_text(text: str) -> dict[str, float]:
    return extract_features(ChatRequest(model="auto", messages=[{"role": "user", "content": text}]))


def tier_from_score(score: float, lo: float, hi: float) -> str:
    return "cheap" if score < lo else ("balanced" if score < hi else "strong")


@dataclass
class Classification:
    tier: str
    score: float
    features: dict[str, float]
    classifier: str


class HeuristicClassifier:
    """v1: hand-weighted difficulty score in [0, 1]."""
    name = "heuristic"
    LO, HI = 0.25, 0.50

    @staticmethod
    def score(f: dict[str, float]) -> float:
        s = (0.25 * min(1.0, f["words100"] / 3.0) + 0.06 * min(1.0, (f["turns"] - 1) / 6.0)
             + 0.25 * f["has_code"] + 0.15 * f["has_math"] + 0.55 * min(1.0, f["n_reason"] / 3.0)
             + 0.10 * f["has_tools"] + 0.05 * f["has_schema"] - 0.20 * min(1.0, f["n_easy"] / 2.0))
        return max(0.0, min(1.0, s))

    def classify(self, req: ChatRequest) -> Classification:
        f = extract_features(req)
        s = self.score(f)
        return Classification(tier_from_score(s, self.LO, self.HI), s, f, self.name)


class LearnedClassifier:
    """v2: logistic regression on TF-IDF + hand features -> P(hard) -> tier."""
    name = "learned"
    LO, HI = 0.30, 0.65

    def __init__(self, bundle: dict[str, Any], fallback: HeuristicClassifier | None = None) -> None:
        self.vec, self.clf = bundle["vectorizer"], bundle["clf"]
        self.fallback = fallback or HeuristicClassifier()

    @classmethod
    def load(cls, path: str | Path | None = None) -> "LearnedClassifier | None":
        p = Path(path or Path(__file__).with_name("router_model.pkl"))
        try:
            with open(p, "rb") as fh:
                return cls(pickle.load(fh))
        except Exception:
            return None

    @staticmethod
    def matrix(vec: Any, texts: list[str], feats: list[dict[str, float]]) -> Any:
        import numpy as np
        from scipy.sparse import csr_matrix, hstack

        dense = np.array([[f[n] for n in FEATURE_NAMES] for f in feats], dtype=float)
        return hstack([vec.transform(texts), csr_matrix(dense)]).tocsr()

    def p_hard(self, req: ChatRequest) -> tuple[float, dict[str, float]]:
        f = extract_features(req)
        x = self.matrix(self.vec, [last_user_text(req)], [f])
        return float(self.clf.predict_proba(x)[0, 1]), f

    def classify(self, req: ChatRequest) -> Classification:
        try:
            p, f = self.p_hard(req)
        except Exception:
            return self.fallback.classify(req)
        return Classification(tier_from_score(p, self.LO, self.HI), p, f, self.name)
