"""
TrustNet — real-label review text classifier.

Trains TF-IDF + LogisticRegression on the Kaggle CG / OR labels.
    CG = computer-generated (fake)
    OR = original (genuine)

This model answers ONE question: "does this review text read as machine-written?"
It does NOT detect rings. Rings are still found by linker.py. The two signals
are reported separately and must never be presented as one accuracy figure.

Outputs
    text_model.joblib          (~3-6 MB)
    text_model_report.json

Run
    py -3.14 train_text_model.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "FAKE.csv")
MODEL_OUT = os.path.join(HERE, "text_model.joblib")
REPORT_OUT = os.path.join(HERE, "text_model_report.json")

SEED = 42
TEST_SIZE = 0.20


def load():
    if not os.path.exists(DATA):
        sys.exit(f"FAKE.csv not found at {DATA}")
    df = pd.read_csv(DATA)
    text_col = "text_" if "text_" in df.columns else "text"
    if text_col not in df.columns:
        sys.exit(f"No text column in FAKE.csv. Columns: {list(df.columns)}")
    if "label" not in df.columns:
        sys.exit(f"No label column in FAKE.csv. Columns: {list(df.columns)}")

    df = df[[text_col, "label"]].dropna()
    df[text_col] = df[text_col].astype(str)
    df = df[df[text_col].str.strip().str.len() > 0]

    labels = df["label"].astype(str).str.strip().str.upper()
    unknown = sorted(set(labels.unique()) - {"CG", "OR"})
    if unknown:
        sys.exit(f"Unexpected label values {unknown}; expected CG / OR.")

    return df[text_col].values, (labels == "CG").astype(int).values


def build_pipeline():
    feats = FeatureUnion([
        ("word", TfidfVectorizer(
            max_features=40000, stop_words="english", ngram_range=(1, 2),
            sublinear_tf=True, min_df=2,
        )),
        ("char", TfidfVectorizer(
            max_features=40000, analyzer="char_wb", ngram_range=(3, 5),
            sublinear_tf=True, min_df=3,
        )),
    ])
    clf = LogisticRegression(
        C=4.0, max_iter=2000, solver="liblinear", class_weight="balanced",
    )
    return Pipeline([("features", feats), ("classifier", clf)])


def main():
    t0 = time.perf_counter()
    X, y = load()
    print(f"loaded {len(X):,} reviews — fake {(y == 1).sum():,} / genuine {(y == 0).sum():,}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y,
    )
    print(f"split: {len(X_tr):,} train / {len(X_te):,} held-out")

    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    print(f"fit in {time.perf_counter() - t0:.1f}s")

    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    report = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_total": int(len(X)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "test_size": TEST_SIZE,
        "seed": SEED,
        "split": "random stratified 80/20",
        "auc": float(roc_auc_score(y_te, proba)),
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred)),
        "recall": float(recall_score(y_te, pred)),
        "f1": float(f1_score(y_te, pred)),
        "caveat": (
            "Kaggle CG reviews are machine-written. Real-world paid reviews written "
            "by humans will be harder to separate than this number suggests."
        ),
    }

    joblib.dump(pipe, MODEL_OUT, compress=3)
    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n--- held-out results ---")
    for k in ("auc", "accuracy", "precision", "recall", "f1"):
        print(f"{k:<10} {report[k]:.4f}")
    print(f"\nsaved {os.path.basename(MODEL_OUT)} and {os.path.basename(REPORT_OUT)}")
    print(f"total {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()