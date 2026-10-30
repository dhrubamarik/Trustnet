"""
TrustNet — lazy loader for the real-label text classifier.

Separate from train_text_model.py so the API can import this unconditionally:
if text_model.joblib is missing, the API still starts and score_text()
returns None instead of raising.
"""
from __future__ import annotations

import json
import os
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "text_model.joblib")
REPORT_PATH = os.path.join(HERE, "text_model_report.json")

_LOCK = threading.Lock()
_MODEL = None
_LOAD_FAILED = False


def available() -> bool:
    return os.path.exists(MODEL_PATH) and not _LOAD_FAILED


def _load():
    global _MODEL, _LOAD_FAILED
    if _MODEL is not None or _LOAD_FAILED:
        return _MODEL
    with _LOCK:
        if _MODEL is not None or _LOAD_FAILED:
            return _MODEL
        try:
            import joblib
            _MODEL = joblib.load(MODEL_PATH)
            print(f"[text-model] loaded {os.path.basename(MODEL_PATH)}")
        except Exception as e:
            _LOAD_FAILED = True
            print(f"[text-model] load failed: {e}")
    return _MODEL


def score_text(text):
    """P(machine-written) in [0, 1], or None if the model is unavailable."""
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None
    model = _load()
    if model is None:
        return None
    try:
        return float(model.predict_proba([s])[0][1])
    except Exception:
        return None


def score_batch(texts):
    """score_text() over a list. Returns [float|None, ...], order preserved."""
    texts = list(texts or [])
    model = _load()
    if model is None:
        return [None] * len(texts)
    clean = [str(t).strip() if t else "" for t in texts]
    keep = [i for i, t in enumerate(clean) if t]
    out = [None] * len(clean)
    if not keep:
        return out
    try:
        p = model.predict_proba([clean[i] for i in keep])[:, 1]
        for j, i in enumerate(keep):
            out[i] = float(p[j])
    except Exception:
        pass
    return out


def model_info():
    """Report contents for the dashboard. Never raises."""
    info = {"available": available()}
    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                info.update(json.load(f))
        except Exception:
            pass
    return info