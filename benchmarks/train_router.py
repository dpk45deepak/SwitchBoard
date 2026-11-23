"""Train + evaluate the learned router (logistic regression on TF-IDF + hand features).

Usage: PYTHONPATH=src python benchmarks/train_router.py
Writes src/switchboard/routing/router_model.pkl and benchmarks/results/router_train.json.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from switchboard.routing.classifier import LearnedClassifier, features_from_text
from switchboard.routing.dataset import generate

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    xt, yt = generate(4000, seed=1)
    xv, yv = generate(1000, seed=2)  # same templates, different slot draws
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True)
    vec.fit(xt)
    Xt = LearnedClassifier.matrix(vec, xt, [features_from_text(t) for t in xt])
    Xv = LearnedClassifier.matrix(vec, xv, [features_from_text(t) for t in xv])
    clf = LogisticRegression(max_iter=2000, C=2.0).fit(Xt, yt)
    pred = clf.predict(Xv)
    res = {"train_n": len(xt), "val_n": len(xv), "val_accuracy": accuracy_score(yv, pred),
           "val_f1_hard": f1_score(yv, pred)}
    out = ROOT / "src/switchboard/routing/router_model.pkl"
    with open(out, "wb") as fh:
        pickle.dump({"vectorizer": vec, "clf": clf}, fh)
    (ROOT / "benchmarks/results").mkdir(exist_ok=True, parents=True)
    (ROOT / "benchmarks/results/router_train.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), "->", out)


if __name__ == "__main__":
    main()
