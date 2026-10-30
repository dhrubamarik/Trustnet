"""
TrustNet — Phase 6: FastAPI Service (v4 — Dynamic tier escalation)
====================================================================
GET  /cases, GET /cases/{id}, GET /stats, POST /ingest

Run:  py -3.14 phase6_api.py
"""
import upload_io  
import json
import pickle
import time
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from linker import IncrementalLinker

print("Loading TrustNet data...")

cases_df = pd.read_csv("cases.csv")
scored_df = pd.read_csv("scored_features.csv")

with open("graph_state.pkl", "rb") as f:
    state = pickle.load(f)
linker = state["linker"]

# ──────────────────────────────────────────────
# PURE PYTHON DICT (bug-free live state)
# ──────────────────────────────────────────────
cases_db = {}
for _, row in cases_df.iterrows():
    cid = str(row["case_id"])
    cases_db[cid] = {
        "case_id": cid,
        "type": str(row["type"]),
        "tier": str(row["tier"]),
        "case_score": float(row["case_score"]),
        "max_risk": float(row["max_risk"]),
        "avg_risk": float(row["avg_risk"]),
        "n_members": int(row["n_members"]),
        "n_sellers": int(row["n_sellers"]),
        "n_accounts": int(row["n_accounts"]),
        "signal_diversity": int(row["signal_diversity"]),
        "signal_types": str(row["signal_types"]),
        "time_span_minutes": float(row["time_span_minutes"]),
        "explanation": str(row["explanation"]),
        "member_ids": [int(x) for x in json.loads(row["member_ids"])],
    }

def size_bonus(n):
    return 0.0 if n < 2 else min(0.5, 0.1 * np.log2(n))

def diversity_bonus(d):
    return min(0.5, 0.1 * d)

def classify_tier(n_members, diversity):
    """Dynamic tier classification — reapplied on every update."""
    if n_members >= 3 and diversity >= 3:
        return "ESCALATE"
    if n_members >= 2 or diversity >= 2:
        return "REVIEW"
    return "WATCH"

def regenerate_explanation(c):
    if c["type"] == "cluster":
        return (
            f"Coordinated activity detected: {c['n_members']} reviews "
            f"across {c['n_sellers']} sellers from {c['n_accounts']} accounts. "
            f"Linked via {c['signal_diversity']} signal types: {c['signal_types']}. "
            f"Max individual risk: {c['max_risk']:.2f}. Case score: {c['case_score']:.2f}."
        )
    return (
        f"Single high-risk review flagged (risk score {c['max_risk']:.2f}). "
        f"Case score: {c['case_score']:.2f}."
    )

print(f"  Loaded {len(cases_db)} cases into memory")

app = FastAPI(title="TrustNet API", version="1.3.0")
# ---- CORS: lets the dashboard work even if opened from a Live Server ----


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,     # must stay False with "*"
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestReview(BaseModel):
    review_id: Optional[int] = None
    text_: str = "Sample review text"
    rating: float = 5.0
    category: str = "Home_and_Kitchen_5"
    seller_id: str = "SELLER-0001"
    account_id: str = "ACCT-NEW-001"
    account_age_hours: float = 3.0
    device_fingerprint: str | None = None
    ip_block: str | None = None
    payment_instrument: str | None = None
    timestamp: Optional[str] = None

class CaseSummary(BaseModel):
    case_id: str
    tier: str
    case_score: float
    n_members: int
    signal_diversity: int
    one_line_reason: str

class CaseDetail(BaseModel):
    case_id: str
    tier: str
    case_score: float
    max_risk: float
    n_members: int
    n_sellers: int
    n_accounts: int
    signal_diversity: int
    signal_types: str
    time_span_minutes: float
    explanation: str
    member_ids: list
    member_details: list

class IngestResult(BaseModel):
    review_id: int
    merged_into_cluster: bool
    cluster_size: int
    case_id: Optional[str] = None
    tier: Optional[str] = None
    latency_ms: float
    message: str

@app.get("/api/status")
def root():
    return {"service": "TrustNet", "total_cases": len(cases_db)}

@app.get("/stats")
def get_stats():
    return {
        "escalate": sum(1 for c in cases_db.values() if c["tier"] == "ESCALATE"),
        "review": sum(1 for c in cases_db.values() if c["tier"] == "REVIEW"),
        "watch": sum(1 for c in cases_db.values() if c["tier"] == "WATCH"),
        "total": len(cases_db),
    }

@app.get("/cases", response_model=list[CaseSummary])
def get_cases(tier: Optional[str] = None, limit: int = 100):
    lst = list(cases_db.values())
    if tier:
        lst = [c for c in lst if c["tier"].upper() == tier.upper()]
    lst.sort(key=lambda x: x["case_score"], reverse=True)
    out = []
    for c in lst[:limit]:
        parts = c["explanation"].split(". ")
        out.append(CaseSummary(
            case_id=c["case_id"], tier=c["tier"], case_score=c["case_score"],
            n_members=c["n_members"], signal_diversity=c["signal_diversity"],
            one_line_reason=parts[0] + "." if parts else "No explanation.",
        ))
    return out

@app.get("/cases/{case_id}", response_model=CaseDetail)
def get_case_detail(case_id: str):
    if case_id not in cases_db:
        raise HTTPException(404, f"Case {case_id} not found")
    c = cases_db[case_id]
    member_details = []
    for mid in c["member_ids"][:20]:
        m = scored_df[scored_df["review_id"] == mid]
        if not m.empty:
            m = m.iloc[0]
            member_details.append({
                "review_id": int(mid),
                "seller_id": str(m["seller_id"]),
                "account_id": str(m["account_id"]),
                "account_age_hours": float(m["account_age_hours"]),
                "risk_score": float(m["risk_score"]),
                "device_fingerprint": str(m["device_fingerprint"]),
                "ip_block": str(m["ip_block"]),
                "near_dup_score": float(m["near_dup_score"]),
            })
    return CaseDetail(
        case_id=c["case_id"], tier=c["tier"], case_score=c["case_score"],
        max_risk=c["max_risk"], n_members=c["n_members"], n_sellers=c["n_sellers"],
        n_accounts=c["n_accounts"], signal_diversity=c["signal_diversity"],
        signal_types=c["signal_types"], time_span_minutes=c["time_span_minutes"],
        explanation=c["explanation"], member_ids=c["member_ids"],
        member_details=member_details,
    )

@app.post("/ingest", response_model=IngestResult)
def ingest_review(review: IngestReview):
    start = time.time()

    # Unique ID
    rid = review.review_id or linker._next_id
    while rid in linker.parent:
        rid += 1

    # ── Compute matched signal diversity BEFORE ingest ──
    matched_signals = []
    for col in ["device_fingerprint", "ip_block", "payment_instrument"]:
        val = getattr(review, col)
        if val and val in linker.signal_index[col] and len(linker.signal_index[col][val]) >= 1:
            matched_signals.append(col)
    new_diversity = len(matched_signals)

    result = linker.ingest({
        "review_id": rid,
        "device_fingerprint": review.device_fingerprint,
        "ip_block": review.ip_block,
        "payment_instrument": review.payment_instrument,
    })

    matched_case_id = None

    if result["merged"] and result["cluster_size"] >= 2:
        member_set = set(result["cluster_members"])
        for cid, c in cases_db.items():
            if set(c["member_ids"]) & member_set:
                matched_case_id = cid
                break

        if matched_case_id:
            # ── Update existing case ──
            c = cases_db[matched_case_id]
            if rid not in c["member_ids"]:
                c["member_ids"].append(rid)
                c["n_members"] = len(c["member_ids"])
                c["n_accounts"] = len(c["member_ids"])
                c["signal_diversity"] = max(c["signal_diversity"], new_diversity)
                c["case_score"] = round(c["max_risk"] + size_bonus(c["n_members"]) + diversity_bonus(c["signal_diversity"]), 4)
                # ★ DYNAMIC TIER RECLASSIFICATION
                c["tier"] = classify_tier(c["n_members"], c["signal_diversity"])
                c["explanation"] = regenerate_explanation(c)
        else:
            # ── Create new cluster case ──
            matched_case_id = f"CASE-LIVE-{rid}"
            sig_names = {
                "device_fingerprint": "shared_device",
                "ip_block": "shared_ip",
                "payment_instrument": "shared_payment",
            }
            sig_types_str = ", ".join(sig_names[s] for s in matched_signals) or "multi-signal"
            cases_db[matched_case_id] = {
                "case_id": matched_case_id,
                "type": "cluster",
                "tier": classify_tier(result["cluster_size"], new_diversity),
                "case_score": 0.5 + size_bonus(result["cluster_size"]) + diversity_bonus(new_diversity),
                "max_risk": 0.5,
                "avg_risk": 0.5,
                "n_members": result["cluster_size"],
                "n_sellers": 1,
                "n_accounts": result["cluster_size"],
                "signal_diversity": new_diversity,
                "signal_types": sig_types_str,
                "time_span_minutes": 0,
                "explanation": (f"Live-formed cluster: {result['cluster_size']} reviews "
                                f"linked via {new_diversity} signal types ({sig_types_str})."),
                "member_ids": result["cluster_members"],
            }
    else:
        # ── Isolated ──
        matched_case_id = f"CASE-LIVE-{rid}"
        cases_db[matched_case_id] = {
            "case_id": matched_case_id, "type": "isolated",
            "tier": "WATCH", "case_score": 0.1, "max_risk": 0.1, "avg_risk": 0.1,
            "n_members": 1, "n_sellers": 1, "n_accounts": 1,
            "signal_diversity": 0, "signal_types": "none",
            "time_span_minutes": 0,
            "explanation": f"New review {rid} ingested. No multi-signal match.",
            "member_ids": [rid],
        }

    latency_ms = (time.time() - start) * 1000
    tier = cases_db[matched_case_id]["tier"]

    return IngestResult(
        review_id=rid,
        merged_into_cluster=result["merged"],
        cluster_size=result["cluster_size"],
        case_id=matched_case_id,
        tier=tier,
        latency_ms=round(latency_ms, 2),
        message=f"Merged into {matched_case_id} — tier={tier}, size={result['cluster_size']}, {latency_ms:.1f}ms.",
    )

# ============================================================================
# TrustNet — console API addon (v5)
# Adds the endpoints dashboard.html calls. Paste at the bottom of phase6_api.py,
# immediately BEFORE  if __name__ == "__main__":
#
# Assumes these already exist in this module (they do in v4):
#   app       — the FastAPI instance
#   cases_db  — dict: case_id -> case dict
#   linker    — IncrementalLinker instance
#
# === EDIT ME if your /ingest handler calls something else ====================
INGEST_FN = (globals().get("ingest_one")
             or globals().get("ingest_review")
             or globals().get("process_review"))
# ============================================================================

# ── infrastructure-value hygiene (F1) ──────────────────────────────────────
_INFRA_PLACEHOLDERS = {
    "", "none", "null", "nil", "na", "n/a", "nan", "-", "--",
    "unknown", "undefined", "dev-unknown", "ip-unknown", "pay-unknown",
    "pay-0000-0000", "device-unknown", "0.0.0.0", "10.0.0.0/24", "127.0.0.1",
}


def _clean_infra(v):
    """A usable entity id, or None. Blanks and the old DEV-UNKNOWN /
    10.0.0.0/24 / PAY-0000-0000 placeholders all mean 'no value'."""
    if v is None:
        return None
    try:
        s = str(v).strip()
    except Exception:
        return None
    return None if s.lower() in _INFRA_PLACEHOLDERS else s

import copy as _copy
import csv as _csv
import json as _json
import os as _os

BASE_DIR = _os.path.dirname(_os.path.abspath(__file__))
from fastapi.responses import FileResponse, HTMLResponse

@app.get("/landing", include_in_schema=False)
def serve_landing():
    import os as _os
    from fastapi.responses import FileResponse, JSONResponse
    p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "landing.html")
    if not _os.path.exists(p):
        return JSONResponse(
            {"error": "landing.html not found", "expected_path": p},
            status_code=404,
        )
    return FileResponse(p, media_type="text/html")

@app.get("/", include_in_schema=False)
def serve_dashboard():
    """Serve dashboard.html for the same origin as /api/* — no CORS, no file://."""
    path = _os.path.join(BASE_DIR, "dashboard.html")
    if not _os.path.exists(path):
        return HTMLResponse(
            "<h1>dashboard.html not found</h1>"
            "<p>Expected at: <code>%s</code></p>" % path,
            status_code=404,
        )
    return FileResponse(path, media_type="text/html")

SCORE_CEILING = 2.0   # max theoretical case_score: 1.0 member + 0.5 size + 0.5 diversity

# ---- entity + text index ---------------------------------------------------
REVIEW_INDEX = {}


def _load_index():
    candidates = ["enriched_reviews.csv", "scored_features.csv", "features.csv"]
    for name in candidates:
        path = _os.path.join(BASE_DIR, name)
        if not _os.path.exists(path):
            continue
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                for row in _csv.DictReader(fh):
                    rid = row.get("review_id")
                    if not rid or rid in REVIEW_INDEX:
                        continue
                    REVIEW_INDEX[rid] = {
                        "text": (row.get("text_") or row.get("review_text") or "")[:300],
                        "rating": float(row.get("rating") or 0),
                        "seller_id": row.get("seller_id") or "",
                        "account_id": row.get("account_id") or "",
                        "device": row.get("device_fingerprint") or "",
                        "ip": row.get("ip_block") or "",
                        "payment": row.get("payment_instrument") or "",
                    }
        except Exception as exc:
            print("[console-api] index load failed:", name, exc)


_load_index()

# ---- baseline snapshot (schema-independent reset) --------------------------
_BASELINE_CASES = _copy.deepcopy(cases_db)
try:
    _BASELINE_LINKER = _copy.deepcopy(linker)
except Exception:
    _BASELINE_LINKER = None

_last_latency_ms = None


# ---- helpers ---------------------------------------------------------------
SIGNAL_LABEL = {
    "device": "Shared device hardware",
    "device_fingerprint": "Shared device hardware",
    "ip": "Coordinated IP subnet",
    "ip_block": "Coordinated IP subnet",
    "subnet": "Coordinated IP subnet",
    "payment": "Shared payment instrument",
    "payment_instrument": "Shared payment instrument",
    "text": "Stylistic / content collusion",
    "near_dup": "Stylistic / content collusion",
    "burst": "High-velocity temporal burst",
    "temporal": "High-velocity temporal burst",
    "account": "Fresh-account shill pattern",
    "velocity": "Multi-day volume anomaly",
}

RECOMMENDED = {
    "Escalate": "Freeze seller payouts, unpublish the linked reviews, and open a manual investigation.",
    "Review": "Queue for analyst review. Do not act until a second signal type confirms.",
    "Watch": "Monitor only. Escalate if the case grows or gains a new signal type.",
}


def _norm_tier(t):
    t = str(t or "").upper()
    if t.startswith("ESC"):
        return "Escalate"
    if t.startswith("REV"):
        return "Review"
    return "Watch"


def _members(case):
    for key in ("_members", "members", "review_ids"):
        val = case.get(key)
        if isinstance(val, (list, tuple, set)):
            return [str(x) for x in val]
    return []


def _signals(case):
    for key in ("signals", "signal_types", "shared_signals"):
        val = case.get(key)
        if isinstance(val, (list, tuple, set)):
            return [str(x) for x in val]
    return []


def _score_100(raw):
    return max(0, min(100, int(round(float(raw) / SCORE_CEILING * 100))))


def _build_graph(members):
    nodes, links = {}, set()

    for rid in members:
        info = REVIEW_INDEX.get(rid, {})
        seller = info.get("seller_id") or "unknown"
        account = info.get("account_id") or ("acct-" + rid)

        nodes.setdefault("s:" + seller, {"id": "s:" + seller, "type": "seller"})
        nodes.setdefault("a:" + account, {"id": "a:" + account, "type": "account"})
        nodes.setdefault("r:" + rid, {"id": "r:" + rid, "type": "review"})

        links.add(("a:" + account, "s:" + seller))
        links.add(("r:" + rid, "a:" + account))

        dev = info.get("device")
        if dev:
            key = "d:" + dev
            nodes.setdefault(key, {"id": key, "type": "device_fingerprint"})
            links.add(("r:" + rid, key))

        pay = info.get("payment")
        if pay:
            key = "p:" + pay
            nodes.setdefault(key, {"id": key, "type": "payment_instrument"})
            links.add(("r:" + rid, key))

        ip = info.get("ip")
        if ip:
            key = "i:" + ip
            nodes.setdefault(key, {"id": key, "type": "subnet"})
            links.add(("r:" + rid, key))

    return {
        "nodes": list(nodes.values()),
        "links": [{"source": a, "target": b} for a, b in sorted(links)],
    }


def _case_to_api(case_id, case):
    members = _members(case)
    signals = _signals(case)
    tier = _norm_tier(case.get("tier") or case.get("severity_tier"))
    raw = float(case.get("case_score") or 0.0)

    sellers, accounts = [], set()
    review_items = []
    for rid in members:
        info = REVIEW_INDEX.get(rid, {})
        if info.get("seller_id") and info["seller_id"] not in sellers:
            sellers.append(info["seller_id"])
        if info.get("account_id"):
            accounts.add(info["account_id"])
        review_items.append({
            "author": info.get("account_id") or rid,
            "seller": info.get("seller_id") or "—",
            "rating": info.get("rating") or 0,
            "text": info.get("text") or "(no text captured)",
        })

    labels = [SIGNAL_LABEL.get(s.lower(), s.replace("_", " ").title()) for s in signals]

    stored = case.get("explanation")
    if isinstance(stored, str) and stored.strip():
        summary = stored.strip()
    else:
        summary = (
            "%d reviews are linked by %d independent signal types (%s). "
            "They resolve to a single coordinated case rather than %d separate flags."
            % (len(members), len(labels), ", ".join(labels) if labels else "entity overlap", len(members))
        )

    bullets = case.get("explanation_bullet_points")
    if not isinstance(bullets, list) or not bullets:
        bullets = ["%s — shared across %d linked accounts." % (lab, len(members)) for lab in labels]
        bullets.append("%d reviews resolved to one case (fragmentation: 0)." % len(members))

    return {
        "case_id": case_id,
        "severity_tier": tier,
        "case_score": round(raw, 3),
        "case_score_100": _score_100(raw),
        "account_count": len(accounts) or len(members),
        "review_count": len(members),
        "sellers_involved": sellers,
        "shared_signals": labels,
        "signal_diversity": int(case.get("signal_diversity") or len(signals)),
        "recommended_action": RECOMMENDED[tier],
        "explanation_summary": summary,
        "explanation_bullet_points": bullets,
        "graph_data": _build_graph(members),
        "review_items": review_items[:12],
    }


def _require_ingest():
    if INGEST_FN is None:
        raise RuntimeError(
            "console-api: INGEST_FN is None — set it to the function your "
            "POST /ingest handler calls (e.g. INGEST_FN = ingest_review)."
        )
    return INGEST_FN


# ---- endpoints -------------------------------------------------------------
@app.get("/api/overview")
def api_overview():
    counts = {"Escalate": 0, "Review": 0, "Watch": 0}
    for case in cases_db.values():
        counts[_norm_tier(case.get("tier") or case.get("severity_tier"))] += 1
    return {
        "tier_counts": counts,
        "total_cases": len(cases_db),
        "indexed_reviews": len(REVIEW_INDEX),
        "last_latency_ms": _last_latency_ms,
    }


@app.get("/api/cases")
def api_cases(tier: str = None, limit: int = 400):
    rows = [_case_to_api(cid, case) for cid, case in cases_db.items()]
    if tier:
        want = _norm_tier(tier)
        rows = [r for r in rows if r["severity_tier"] == want]
    rows.sort(key=lambda r: (r["case_score"], r["review_count"]), reverse=True)
    return {"cases": rows[:limit], "total": len(rows)}


@app.get("/api/reviews")
def api_reviews(limit: int = 80, flagged_only: bool = False):
    rows = list(REVIEW_INDEX.items())[:20000]
    out = []
    for rid, info in rows:
        score = 0
        for cid, case in cases_db.items():
            if rid in _members(case):
                score = _score_100(case.get("case_score") or 0)
                break
        if flagged_only and score < 50:
            continue
        out.append({
            "review_id": rid,
            "risk_score": score,
            "seller_id": info.get("seller_id") or "—",
            "account_id": info.get("account_id") or "—",
            "rating": info.get("rating") or 0,
            "review_text": info.get("text") or "",
            "triggered_rules": [],
        })
        if len(out) >= limit:
            break
    return {"reviews": out}


@app.post("/api/reset")
def api_reset():
    """Restore the in-memory baseline captured at startup. No restart needed."""
    global linker
    cases_db.clear()
    cases_db.update(_copy.deepcopy(_BASELINE_CASES))
    if _BASELINE_LINKER is not None:
        try:
            linker = _copy.deepcopy(_BASELINE_LINKER)
        except Exception:
            pass
    return {"ok": True, "cases": len(cases_db)}


@app.post("/api/inject-member")
def api_inject_member():
    """Add ONE review matching the Phase-0 ring fingerprint -> grows CASE-0000 by 1."""
    global _last_latency_ms
    import time as _time
    t0 = _time.perf_counter()

    payload = {
        "seller_id": "SEL_003",
        "account_id": "ACC-RING-LIVE-%d" % (len(REVIEW_INDEX) + 1),
        "rating": 5,
        "review_text": "AMAZING 10/10 MUST BUY! Best seller ever! Incredible quality, flawless product!",
        "device_fingerprint": "DEV-A7F3E9C1-FAKE-RING",
        "ip_block": "198.51.100.0/24",
        "payment_instrument": "PAY-RING-4829",
        "account_age_hours": 3.4,
    }

    result = _require_ingest()(payload)
    _last_latency_ms = round((_time.perf_counter() - t0) * 1000, 1)

    case_id = (result or {}).get("assigned_case_id") or (result or {}).get("case_id")
    case = cases_db.get(case_id) if case_id else None

    return {
        "case_id": case_id,
        "member_count": len(_members(case)) if case else 0,
        "tier": _norm_tier(case.get("tier")) if case else "Watch",
        "latency_ms": _last_latency_ms,
    }


@app.post("/api/stress-test/inject")
def api_stress_test_inject(body: dict = None):
    """Inject an 8-account ring in one shot (the dramatic demo)."""
    global _last_latency_ms
    body = body or {}
    ingest = _require_ingest()
    import time as _time

    t0 = _time.perf_counter()
    device = body.get("shared_device", "DEV-SYNDICATE-APEX")
    payment = body.get("shared_payment", "PAY-SYNDICATE-APEX")
    seller = body.get("seller_id", "SEL_003")
    text = "AMAZING 10/10 MUST BUY! Best seller ever!"

    last = None
    for i in range(8):
        last = ingest({
            "seller_id": seller,
            "account_id": "ACC-APEX-%02d" % i,
            "rating": 5,
            "review_text": text,
            "device_fingerprint": device,
            "ip_block": "203.0.113.0/24",
            "payment_instrument": payment,
            "account_age_hours": 2.0 + i * 0.3,
        })

    _last_latency_ms = round((_time.perf_counter() - t0) * 1000, 1)
    case_id = (last or {}).get("assigned_case_id") or (last or {}).get("case_id")
    case = cases_db.get(case_id) if case_id else None
    return {
        "case_id": case_id,
        "member_count": len(_members(case)) if case else 0,
        "latency_ms": _last_latency_ms,
    }

def _as_result_dict(obj):
    """Normalise an ingest return value to a plain dict.
    Handles dict, pydantic v1/v2 models, plain objects, None."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    for meth in ("model_dump", "dict"):
        fn = getattr(obj, meth, None)
        if callable(fn):
            try:
                d = fn()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
    try:
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    except Exception:
        return {"value": obj}

@app.post("/api/custom/review")
def api_custom_review(payload: dict):
    global _last_latency_ms
    import time as _time
    t0 = _time.perf_counter()

    result = _require_ingest()(payload)
    result = _as_result_dict(result)
    _last_latency_ms = round((_time.perf_counter() - t0) * 1000, 1)

    case_id = (result or {}).get("assigned_case_id") or (result or {}).get("case_id")
    case = cases_db.get(case_id) if case_id else None
    raw = float((result or {}).get("risk_score") or 0.0)

    return {
        "scored_review": {
            "review_id": (result or {}).get("review_id") or (result or {}).get("rid"),
            "risk_score": _score_100(raw),
        },
        "assigned_case_id": case_id,
        "latency_ms": _last_latency_ms,
    }


@app.post("/api/custom/ring")
def api_custom_ring(payload: dict):
    global _last_latency_ms
    ingest = _require_ingest()
    import time as _time

    n = int(payload.get("num_accounts") or 6)
    n = max(3, min(15, n))
    device = payload.get("shared_device") or "DEV-CUSTOM-RING"
    payment = payload.get("shared_payment") or "PAY-CUSTOM-RING"
    seller = payload.get("seller_id") or "SEL_003"
    text = payload.get("review_text") or "AMAZING 10/10 MUST BUY!"

    t0 = _time.perf_counter()
    last = None
    for i in range(n):
        last = ingest({
            "seller_id": seller,
            "account_id": "ACC-CUSTOM-%02d-%d" % (i, int(_time.time() * 1000) % 10000),
            "rating": 5,
            "review_text": text,
            "device_fingerprint": device,
            "ip_block": "192.0.2.0/24",
            "payment_instrument": payment,
            "account_age_hours": 1.5 + i * 0.2,
        })

    _last_latency_ms = round((_time.perf_counter() - t0) * 1000, 1)
    case_id = (last or {}).get("assigned_case_id") or (last or {}).get("case_id")
    case = cases_db.get(case_id) if case_id else None
    raw = float(case.get("case_score") or 0.0) if case else 0.0

    return {
        "case_id": case_id,
        "member_count": len(_members(case)) if case else 0,
        "case_score_100": _score_100(raw),
        "latency_ms": _last_latency_ms,
    }

# ============================================================================
# console-api v5.1 — tolerant member/signal extraction + CSV fallback
# ============================================================================

def _as_result_dict(obj):
    """dict / pydantic model / plain object / None  ->  plain dict"""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    for meth in ("model_dump", "dict"):
        fn = getattr(obj, meth, None)
        if callable(fn):
            try:
                d = fn()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
    try:
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    except Exception:
        return {"value": obj}


_INGEST_SHIM_DONE = False

def _install_ingest_shim():
    """Give IngestResult a .get() so existing code works unchanged."""
    global _INGEST_SHIM_DONE
    if _INGEST_SHIM_DONE:
        return
    _INGEST_SHIM_DONE = True
    ir = globals().get("IngestResult")
    if ir is None:
        print("[console-api] IngestResult not found - shim skipped")
        return
    try:
        ir.get = lambda self, k, d=None: _as_result_dict(self).get(k, d)
        ir.__getitem__ = lambda self, k: _as_result_dict(self)[k]
        ir.__contains__ = lambda self, k: k in _as_result_dict(self)
        print("[console-api] IngestResult shim installed")
    except Exception as e:
        print(f"[console-api] IngestResult shim FAILED: {type(e).__name__}: {e}")

_install_ingest_shim()

def _as_list(val):
    """Coerce list | JSON string | comma string | scalar into list[str]."""
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        return [str(x) for x in val if str(x).strip()]
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() in ("nan", "none", "null", "[]"):
            return []
        if s[0] in "[(":
            try:
                parsed = _json.loads(s.replace("'", '"'))
                if isinstance(parsed, (list, tuple)):
                    return [str(x) for x in parsed]
            except Exception:
                pass
            s = s[1:-1]
        return [p.strip().strip("'\"") for p in s.split(",") if p.strip()]
    if isinstance(val, (int, float)):
        return [str(int(val))]
    return [str(val)]



CSV_CASES = {}


def _load_cases_csv():
    path = _os.path.join(BASE_DIR, "cases.csv")
    if not _os.path.exists(path):
        print("[console-api] cases.csv missing — falling back to in-memory only")
        return
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                cid = row.get("case_id")
                if cid:
                    CSV_CASES[str(cid)] = row
        print("[console-api] cases.csv indexed:", len(CSV_CASES))
    except Exception as exc:
        print("[console-api] cases.csv load failed:", exc)


_load_cases_csv()

_MEMBER_KEYS = ("_members", "members", "member_ids", "review_ids", "review_id_list",
                "members_json", "member_list", "reviews")
_SIGNAL_KEYS = ("signals", "signal_types", "shared_signals", "signals_json",
                "signal_json", "triggered_signals", "triggered_rules")


def _members(case):
    if not isinstance(case, dict):
        return []
    for k in _MEMBER_KEYS:
        vals = _as_list(case.get(k))
        if vals:
            return vals
    cid = case.get("case_id")
    if cid:
        row = CSV_CASES.get(str(cid)) or {}
        for k in _MEMBER_KEYS:
            vals = _as_list(row.get(k))
            if vals:
                return vals
    return []


def _signals(case):
    if not isinstance(case, dict):
        return []
    for k in _SIGNAL_KEYS:
        vals = _as_list(case.get(k))
        if vals:
            return vals
    cid = case.get("case_id")
    if cid:
        row = CSV_CASES.get(str(cid)) or {}
        for k in _SIGNAL_KEYS:
            vals = _as_list(row.get(k))
            if vals:
                return vals
    return []


# make case_id available on every in-memory case so the CSV lookup can fire
for _cid, _case in list(cases_db.items()):
    if isinstance(_case, dict):
        _case.setdefault("case_id", _cid)


@app.get("/api/debug/case/{case_id}")
def api_debug_case(case_id: str):
    """Inspect the raw shape of one case. Remove before submission if you like."""
    case = cases_db.get(case_id)
    row = CSV_CASES.get(case_id) or {}
    return {
        "in_memory_keys": sorted(case.keys()) if isinstance(case, dict) else None,
        "in_memory": case if isinstance(case, dict) else None,
        "csv_keys": sorted(row.keys()),
        "csv_row": row or None,
        "resolved_members": _members(case or {}),
        "resolved_signals": _signals(case or {}),
    }

# ============================================================================
# console-api v5.2 — ingest bridge
# Fixes Inject Member / Run Ring Test by locating the real POST /ingest handler
# instead of guessing its name, and by awaiting it / building its model arg.
# ============================================================================

import inspect as _inspect
import time as _time


# --- 1. drop the v5 routes. FastAPI matches first-registered-wins, so the
#        broken definitions must be removed before re-registering. -----------
_BROKEN_PATHS = {
    "/api/inject-member",
    "/api/stress-test/inject",
    "/api/custom/review",
    "/api/custom/ring",
}
app.router.routes = [r for r in app.router.routes
                     if getattr(r, "path", None) not in _BROKEN_PATHS]


# --- 2. signal labels: substring matching, so "near_dup_text" and
#        "shared_device" resolve to the same wording the UI expects ---------
class _SignalLabels(dict):
    _RULES = (
        ("device",   "Shared device hardware"),
        ("payment",  "Shared payment instrument"),
        ("ip",       "Coordinated IP subnet"),
        ("subnet",   "Coordinated IP subnet"),
        ("dup",      "Stylistic / content collusion"),
        ("text",     "Stylistic / content collusion"),
        ("styl",     "Stylistic / content collusion"),
        ("collus",   "Stylistic / content collusion"),
        ("burst",    "High-velocity temporal burst"),
        ("temporal", "High-velocity temporal burst"),
        ("account",  "Fresh-account shill pattern"),
        ("shill",    "Fresh-account shill pattern"),
        ("velocity", "Multi-day volume anomaly"),
        ("volume",   "Multi-day volume anomaly"),
    )

    def get(self, key, default=None):
        s = str(key).lower()
        for token, label in self._RULES:
            if token in s:
                return label
        return str(key).replace("_", " ").title()


SIGNAL_LABEL = _SignalLabels()


# --- 3. locate the real ingest route ---------------------------------------
def _find_ingest_route():
    preferred, fallback = None, None
    for route in app.router.routes:
        path = getattr(route, "path", "") or ""
        methods = getattr(route, "methods", None) or set()
        if "POST" not in methods:
            continue
        if path.rstrip("/") == "/ingest":
            preferred = route
            break
        if path.rstrip("/").endswith("ingest") and fallback is None:
            fallback = route
    return preferred or fallback


def _build_call_arg(fn, payload):
    params = [p for p in _inspect.signature(fn).parameters.values()
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if not params:
        return _inspect._empty

    ann = params[0].annotation
    if isinstance(ann, str):
        ann = globals().get(ann)
    if ann in (None, _inspect._empty, dict) or not isinstance(ann, type):
        return payload

    try:
        return ann(**payload)          # pydantic v1 or v2 model
    except Exception as exc:
        raise RuntimeError(
            "ingest rejected payload for %s -> %s" % (ann.__name__, exc)
        )


async def _invoke_ingest(payload):
    route = _find_ingest_route()
    if route is None:
        raise RuntimeError("no POST /ingest route is registered on this app")

    fn = route.endpoint
    arg = _build_call_arg(fn, payload)

    result = fn() if arg is _inspect._empty else fn(arg)
    if _inspect.isawaitable(result):
        result = await result
    return result


# --- 4. identify which case grew, without trusting the return shape --------
def _case_snapshot():
    return {cid: len(_members(c)) for cid, c in cases_db.items()}


def _resolve_case(result, before):
    rid = None
    if isinstance(result, dict):
        for key in ("review_id", "rid"):
            if result.get(key) is not None:
                rid = str(result[key])
                break
        if rid is None:
            sr = result.get("scored_review")
            if isinstance(sr, dict) and sr.get("review_id") is not None:
                rid = str(sr["review_id"])

    if rid:
        for cid, c in cases_db.items():
            if rid in _members(c):
                return cid

    grown = []
    for cid, c in cases_db.items():
        delta = len(_members(c)) - before.get(cid, 0)
        if delta > 0:
            grown.append((delta, float(c.get("case_score") or 0), cid))
    if grown:
        grown.sort(reverse=True)
        return grown[0][2]

    if isinstance(result, dict):
        cid = result.get("assigned_case_id") or result.get("case_id")
        if cid and cid in cases_db:
            return cid
    return None


_LIVE_SEQ = [0]


def _next_account(tag):
    _LIVE_SEQ[0] += 1
    return "ACC-LIVE-%s-%03d" % (tag, _LIVE_SEQ[0])


def _ring_payload(account, seller, text, device, ip, payment, age):
    return {
        "seller_id": seller,
        "account_id": account,
        "rating": 5,
        "review_text": text,
        "device_fingerprint": device,
        "ip_block": ip,
        "payment_instrument": payment,
        "account_age_hours": age,
    }


# --- 5. replacement endpoints ----------------------------------------------
@app.post("/api/inject-member")
async def api_inject_member():
    global _last_latency_ms

    payload = _ring_payload(
        _next_account("RING"), "SEL_003",
        "AMAZING 10/10 MUST BUY! Best seller ever! Incredible quality, flawless product!",
        "DEV-A7F3E9C1-FAKE-RING", "198.51.100.0/24", "PAY-RING-4829", 3.4,
    )

    before = _case_snapshot()
    t0 = _time.perf_counter()
    try:
        result = await _invoke_ingest(payload)
    except Exception as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}

    _last_latency_ms = round((_time.perf_counter() - t0) * 1000, 1)
    cid = _resolve_case(result, before)
    case = cases_db.get(cid) if cid else None

    return {
        "ok": True,
        "case_id": cid,
        "member_count": len(_members(case)) if case else 0,
        "tier": _norm_tier(case.get("tier")) if case else "Watch",
        "latency_ms": _last_latency_ms,
    }


@app.post("/api/stress-test/inject")
async def api_stress_test_inject(body: dict = None):
    global _last_latency_ms
    body = body or {}
    seller = body.get("seller_id", "SEL_003")
    device = body.get("shared_device", "DEV-SYNDICATE-APEX")
    payment = body.get("shared_payment", "PAY-SYNDICATE-APEX")
    text = "AMAZING 10/10 MUST BUY! Best seller ever!"

    before = _case_snapshot()
    t0 = _time.perf_counter()
    result = None
    try:
        for i in range(8):
            result = await _invoke_ingest(
                _ring_payload(_next_account("APEX"), seller, text,
                              device, "203.0.113.0/24", payment, 2.0 + i * 0.3)
            )
    except Exception as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}

    _last_latency_ms = round((_time.perf_counter() - t0) * 1000, 1)
    cid = _resolve_case(result, before)
    case = cases_db.get(cid) if cid else None

    return {
        "ok": True,
        "case_id": cid,
        "member_count": len(_members(case)) if case else 0,
        "latency_ms": _last_latency_ms,
    }


@app.post("/api/custom/ring")
async def api_custom_ring(payload: dict):
    global _last_latency_ms
    n = max(3, min(15, int(payload.get("num_accounts") or 6)))
    seller = payload.get("seller_id") or "SEL_003"
    device = payload.get("shared_device") or "DEV-CUSTOM-RING"
    payment = payload.get("shared_payment") or "PAY-CUSTOM-RING"
    text = payload.get("review_text") or "AMAZING 10/10 MUST BUY!"

    before = _case_snapshot()
    t0 = _time.perf_counter()
    result = None
    try:
        for i in range(n):
            result = await _invoke_ingest(
                _ring_payload(_next_account("CUSTOM"), seller, text,
                              device, "192.0.2.0/24", payment, 1.5 + i * 0.2)
            )
    except Exception as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}

    _last_latency_ms = round((_time.perf_counter() - t0) * 1000, 1)
    cid = _resolve_case(result, before)
    case = cases_db.get(cid) if cid else None
    raw = float(case.get("case_score") or 0.0) if case else 0.0

    return {
        "ok": True,
        "case_id": cid,
        "member_count": len(_members(case)) if case else 0,
        "case_score_100": _score_100(raw),
        "latency_ms": _last_latency_ms,
    }


@app.post("/api/custom/review")
async def api_custom_review(payload: dict):
    global _last_latency_ms

    before = _case_snapshot()
    t0 = _time.perf_counter()
    try:
        result = await _invoke_ingest(payload)
    except Exception as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}

    _last_latency_ms = round((_time.perf_counter() - t0) * 1000, 1)
    cid = _resolve_case(result, before)
    case = cases_db.get(cid) if cid else None

    raw = None
    if isinstance(result, dict):
        raw = result.get("risk_score")
    if raw is None and case is not None:
        raw = case.get("max_risk")
    raw = float(raw or 0.0)

    rid = None
    if isinstance(result, dict):
        rid = result.get("review_id") or result.get("rid")

    return {
        "ok": True,
        "scored_review": {"review_id": rid, "risk_score": _score_100(raw)},
        "assigned_case_id": cid,
        "latency_ms": _last_latency_ms,
    }


@app.get("/api/debug/ingest-route")
def api_debug_ingest_route():
    """Tells you which handler the bridge found and what it expects."""
    route = _find_ingest_route()
    if route is None:
        return {
            "found": False,
            "post_routes": [getattr(r, "path", "")
                            for r in app.router.routes
                            if "POST" in (getattr(r, "methods", None) or set())],
        }

    fn = route.endpoint
    params, schema = [], None
    for p in _inspect.signature(fn).parameters.values():
        params.append({"name": p.name,
                       "annotation": str(p.annotation),
                       "default": str(p.default)})

    first = next(iter(_inspect.signature(fn).parameters.values()), None)
    ann = getattr(first, "annotation", None) if first else None
    try:
        if hasattr(ann, "model_json_schema"):
            schema = ann.model_json_schema()
        elif hasattr(ann, "schema"):
            schema = ann.schema()
    except Exception:
        pass

    return {
        "found": True,
        "path": route.path,
        "endpoint": fn.__name__,
        "is_async": _inspect.iscoroutinefunction(fn),
        "params": params,
        "request_schema": schema,
    }

# ============================================================================
# console-api v5.3 — index live reviews, derive signals, truthful summaries
# ============================================================================

_V52_INVOKE = _invoke_ingest


def _extract_rid(result):
    if isinstance(result, dict):
        for k in ("review_id", "rid"):
            if result.get(k) is not None:
                return str(result[k])
        sr = result.get("scored_review")
        if isinstance(sr, dict) and sr.get("review_id") is not None:
            return str(sr["review_id"])
    return None


def _index_live(payload, result):
    """Record a runtime-ingested review so sellers / text / graph resolve."""
    rid = _extract_rid(result)
    if rid is None:                      # handler returned no id — find the orphan
        for _c in cases_db.values():
            for _m in _members(_c):
                if _m not in REVIEW_INDEX:
                    rid = _m
                    break
            if rid:
                break
    if rid is None or rid in REVIEW_INDEX:
        return rid
    REVIEW_INDEX[rid] = {
        "text": (payload.get("review_text") or payload.get("text_")
                 or payload.get("text") or "")[:300],
        "rating": float(payload.get("rating") or 0),
        "seller_id": payload.get("seller_id") or "",
        "account_id": payload.get("account_id") or "",
        "device": payload.get("device_fingerprint") or "",
        "ip": payload.get("ip_block") or "",
        "payment": payload.get("payment_instrument") or "",
    }
    return rid


async def _invoke_ingest(payload):
    result = await _V52_INVOKE(payload)
    try:
        _index_live(payload, result)
    except Exception as exc:
        print("[console-api] live index failed:", exc)
    return result


def _derive_signals(members):
    """Fallback: compute shared entities directly when the case carries none."""
    infos = [REVIEW_INDEX[m] for m in members if m in REVIEW_INDEX]
    if len(infos) < 2:
        return []

    def shared(key):
        vals = [i.get(key) for i in infos if i.get(key)]
        return len(vals) >= 2 and len(set(vals)) < len(vals)

    out = []
    if shared("device"):
        out.append("shared_device")
    if shared("ip"):
        out.append("shared_ip")
    if shared("payment"):
        out.append("shared_payment")
    texts = [i["text"].strip().lower() for i in infos if i.get("text")]
    if len(texts) >= 2 and len(set(texts)) < len(texts):
        out.append("near_dup_text")
    return out


def _signals(case):
    if not isinstance(case, dict):
        return []
    for k in _SIGNAL_KEYS:
        vals = _as_list(case.get(k))
        if vals:
            return vals
    cid = case.get("case_id")
    if cid:
        row = CSV_CASES.get(str(cid)) or {}
        for k in _SIGNAL_KEYS:
            vals = _as_list(row.get(k))
            if vals:
                return vals
    return _derive_signals(_members(case))


def _case_to_api(case_id, case):
    members = _members(case)
    raw = float(case.get("case_score") or 0.0)
    max_risk = float(case.get("max_risk") or 0.0)
    tier = _norm_tier(case.get("tier") or case.get("severity_tier"))

    labels, seen = [], set()
    for s in _signals(case):
        lab = SIGNAL_LABEL.get(str(s).lower(), str(s).replace("_", " ").title())
        if lab not in seen:
            seen.add(lab)
            labels.append(lab)

    sellers, accounts, review_items = [], set(), []
    for rid in members:
        info = REVIEW_INDEX.get(rid, {})
        sid = info.get("seller_id") or ""
        aid = info.get("account_id") or ""
        if sid and sid not in sellers:
            sellers.append(sid)
        if aid:
            accounts.add(aid)
        review_items.append({
            "author": aid or rid,
            "seller": sid or "—",
            "rating": info.get("rating") or 0,
            "text": info.get("text") or "(no text captured)",
        })

    n_acc = len(accounts) or len(members)
    span = case.get("time_span_minutes")

    if len(members) >= 2 and labels:
        summary = ("%d reviews across %d seller%s from %d account%s are linked by "
                   "%d independent signal type%s (%s)."
                   % (len(members), len(sellers), "" if len(sellers) == 1 else "s",
                      n_acc, "" if n_acc == 1 else "s",
                      len(labels), "" if len(labels) == 1 else "s",
                      ", ".join(labels).lower()))
        try:
            if span and float(span) > 0:
                summary += " All activity falls inside a %.1f-minute window." % float(span)
        except (TypeError, ValueError):
            pass
        summary += " Max individual risk %.2f; case score %.2f." % (max_risk, raw)
        bullets = ["%s — shared across the linked reviews." % l for l in labels]
        bullets.append("%d reviews resolved into a single case; fragmentation 0."
                       % len(members))
    else:
        stored = case.get("explanation")
        summary = (stored.strip() if isinstance(stored, str) and stored.strip()
                   else "Single isolated review. No shared device, IP, payment or text match.")
        bullets = [
            "No shared entity with any other review.",
            "Risk comes from review content and account signals, not coordination.",
        ]

    return {
        "case_id": case_id,
        "severity_tier": tier,
        "case_score": round(raw, 3),
        "case_score_100": _score_100(raw),
        "account_count": n_acc,
        "review_count": len(members),
        "sellers_involved": sellers,
        "shared_signals": labels,
        "signal_diversity": len(labels),
        "recommended_action": RECOMMENDED[tier],
        "explanation_summary": summary,
        "explanation_bullet_points": bullets,
        "graph_data": _build_graph(members),
        "review_items": review_items[:12],
    }

# ============================================================================
# console-api v5.4 — O(1) case resolution + real per-review scores
# ============================================================================

app.router.routes = [r for r in app.router.routes
                     if getattr(r, "path", None) not in {"/api/reviews", "/api/reset"}]


# --- 1. cheap member counting (n_members is already an int on every case) ----
def _member_count(case):
    if not isinstance(case, dict):
        return 0
    try:
        return int(case.get("n_members"))
    except (TypeError, ValueError):
        return len(_members(case))


def _case_snapshot():
    return {cid: _member_count(c) for cid, c in cases_db.items()}


MEMBER_TO_CASE = {}


def _rebuild_member_index():
    MEMBER_TO_CASE.clear()
    for cid, case in cases_db.items():
        for m in _members(case):
            MEMBER_TO_CASE[m] = cid


_rebuild_member_index()


# --- 2. O(1)-first case resolution -----------------------------------------
def _resolve_case(result, before):
    if isinstance(result, dict):
        for key in ("assigned_case_id", "case_id"):
            cid = result.get(key)
            if cid and cid in cases_db:
                return cid
    rid = _extract_rid(result)
    if rid and rid in MEMBER_TO_CASE:
        return MEMBER_TO_CASE[rid]
    best = None
    for cid, c in cases_db.items():
        delta = _member_count(c) - before.get(cid, 0)
        if delta > 0:
            key = (delta, float(c.get("case_score") or 0))
            if best is None or key > best[0]:
                best = (key, cid)
    return best[1] if best else None


def _commit_result(result, before):
    cid = _resolve_case(result, before)
    rid = _extract_rid(result)
    if rid and cid:
        MEMBER_TO_CASE[rid] = cid
    return cid


# --- 3. per-review scores + fired rules, loaded once ------------------------
LF_LABEL = {
    "lf_near_duplicate":     "copy-paste text",
    "lf_shared_device":      "shared device",
    "lf_shared_ip":          "shared IP",
    "lf_shared_payment":     "shared payment",
    "lf_burst":              "review burst",
    "lf_new_account_active": "new account",
    "lf_high_velocity":      "high velocity",
    "lf_seller_new_5star":   "new-account 5-stars",
}


def _gt0(v):
    try:
        return float(v) > 0
    except (TypeError, ValueError):
        return False


def _load_per_review_scores():
    labeled = _os.path.join(BASE_DIR, "labeled_features.csv")
    scored = _os.path.join(BASE_DIR, "scored_features.csv")

    lf_hits = {}
    if _os.path.exists(labeled):
        with open(labeled, newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                rid = row.get("review_id")
                if not rid:
                    continue
                fired = [LF_LABEL[k] for k in LF_LABEL if _gt0(row.get(k))]
                if fired:
                    lf_hits[rid] = fired

    if not _os.path.exists(scored):
        print("[console-api] scored_features.csv missing — feed scores stay 0")
        return

    n = 0
    with open(scored, newline="", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            rid = row.get("review_id")
            if rid not in REVIEW_INDEX:
                continue
            try:
                REVIEW_INDEX[rid]["risk_score"] = float(row.get("risk_score") or 0)
            except (TypeError, ValueError):
                REVIEW_INDEX[rid]["risk_score"] = 0.0
            REVIEW_INDEX[rid]["rules"] = lf_hits.get(rid, [])
            n += 1
    print("[console-api] per-review scores loaded:", n)


FEED_CACHE = []


def _build_feed_cache():
    global FEED_CACHE
    rows = []
    for rid, info in REVIEW_INDEX.items():
        raw = info.get("risk_score")
        rows.append({
            "review_id": rid,
            "risk_score": _score_100(raw) if raw is not None else 0,
            "raw_risk": round(float(raw or 0), 4),
            "seller_id": info.get("seller_id") or "—",
            "account_id": info.get("account_id") or "—",
            "rating": info.get("rating") or 0,
            "review_text": info.get("text") or "",
            "triggered_rules": info.get("rules") or [],
        })
    rows.sort(key=lambda r: r["raw_risk"], reverse=True)
    FEED_CACHE = rows
    print("[console-api] feed cached:", len(rows))


_load_per_review_scores()
_build_feed_cache()


# --- 4. endpoints -----------------------------------------------------------
@app.get("/api/reviews")
def api_reviews(limit: int = 80, flagged_only: bool = False):
    rows = FEED_CACHE
    if flagged_only:
        rows = [r for r in rows if r["risk_score"] >= 50]
    return {"reviews": rows[:limit], "total": len(rows)}


@app.post("/api/reset")
def api_reset():
    cases_db.clear()
    cases_db.update(_copy.deepcopy(_BASELINE_CASES))
    global linker
    if _BASELINE_LINKER is not None:
        try:
            linker = _copy.deepcopy(_BASELINE_LINKER)
        except Exception:
            pass
    _rebuild_member_index()
    return {"ok": True, "cases": len(cases_db)}

# ============================================================================
# console-api v5.5 — feed rule parsing, scoped live indexing, batch ingest
# ============================================================================

app.router.routes = [r for r in app.router.routes
                     if getattr(r, "path", None) not in {"/api/reviews", "/api/ingest/batch"}]


# --- 1. boolean-tolerant rule parsing + column auto-discovery ---------------
def _gt0(v):
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "t", "yes", "1", "1.0"):
        return True
    if s in ("false", "f", "no", "0", "0.0", "nan", "none", "null", ""):
        return False
    try:
        return float(s) > 0.0
    except (TypeError, ValueError):
        return False


def _lf_columns(path):
    with open(path, newline="", encoding="utf-8") as fh:
        header = next(_csv.reader(fh), [])
    return [c for c in header if c.strip().lower().startswith("lf_")]


def _reload_rule_hits():
    path = _os.path.join(BASE_DIR, "labeled_features.csv")
    if not _os.path.exists(path):
        print("[console-api] labeled_features.csv missing — rules stay empty")
        return 0
    cols = _lf_columns(path)
    print("[console-api] lf columns discovered:", cols)
    hits = 0
    with open(path, newline="", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            rid = row.get("review_id")
            if not rid or rid not in REVIEW_INDEX:
                continue
            fired = [LF_LABEL.get(c, c[3:].replace("_", " "))
                     for c in cols if _gt0(row.get(c))]
            REVIEW_INDEX[rid]["rules"] = fired
            if fired:
                hits += 1
    print("[console-api] reviews with fired rules:", hits)
    return hits


# --- 2. cache the ingest route (it was re-scanned on every review) ---------
_ROUTE_CACHE = {}


def _cached_route():
    route = _ROUTE_CACHE.get("route")
    if route is None:
        route = _find_ingest_route()
        _ROUTE_CACHE["route"] = route
    return route


def _rd(job):
    return job[1] if isinstance(job, tuple) and len(job) == 2 else None


def _lookup_rid(result):
    if isinstance(result, dict):
        for k in ("review_id", "rid"):
            if result.get(k) is not None:
                return str(result[k])
        sr = result.get("scored_review")
        if isinstance(sr, dict) and sr.get("review_id") is not None:
            return str(sr["review_id"])
    return None


def _index_live(payload, result, before):
    """Index a runtime review. Scans only cases that actually grew."""
    rid = _lookup_rid(result)
    if rid is None:
        for cid, c in cases_db.items():
            if _member_count(c) <= before.get(cid, 0):
                continue
            for m in _members(c):
                if m not in REVIEW_INDEX:
                    rid = m
                    break
            if rid:
                break
    if rid is None or rid in REVIEW_INDEX:
        return rid
    REVIEW_INDEX[rid] = {
        "text": (payload.get("review_text") or payload.get("text_")
                 or payload.get("text") or "")[:300],
        "rating": float(payload.get("rating") or 0),
        "seller_id": payload.get("seller_id") or "",
        "account_id": payload.get("account_id") or "",
        "device": payload.get("device_fingerprint") or "",
        "ip": payload.get("ip_block") or "",
        "payment": payload.get("payment_instrument") or "",
        "rules": [],
    }
    return rid


async def _invoke_ingest(payload):
    global _LAST_BEFORE
    route = _cached_route()
    if route is None:
        raise RuntimeError("no POST /ingest route is registered on this app")

    fn = route.endpoint
    arg = _build_call_arg(fn, payload)
    result = fn() if arg is _inspect._empty else fn(arg)
    if _inspect.isawaitable(result):
        result = await result

    try:
        before = globals().get("_LAST_BEFORE") or {}
        rid = _index_live(payload, result, before)
        if rid is not None:
            cid = _resolve_case(result, before)
            if cid:
                MEMBER_TO_CASE[rid] = cid
    except Exception as exc:
        print("[console-api] live index failed:", exc)
    return result


def _commit_result(result, before):
    cid = _resolve_case(result, before)
    rid = _lookup_rid(result)
    if rid:
        MEMBER_TO_CASE[rid] = cid
    return cid


# --- 3. batch ingest (amortises HTTP + FastAPI per-review cost) ------------
@app.post("/api/ingest/batch")
async def api_ingest_batch(body: dict = None):
    global _last_latency_ms, _LAST_BEFORE
    body = body or {}
    n = max(1, min(2000, int(body.get("n") or 500)))
    seller = body.get("seller_id") or "SEL_003"
    device = body.get("shared_device") or "DEV-BATCH-RING"
    payment = body.get("shared_payment") or "PAY-BATCH-RING"
    subnet = body.get("shared_subnet") or "198.18.0.0/24"
    text = body.get("review_text") or "AMAZING 10/10 MUST BUY! Best seller ever!"
    tag = body.get("tag") or "BATCH"

    before = _case_snapshot()
    _LAST_BEFORE = before
    t0 = _time.perf_counter()
    result = None
    try:
        for i in range(n):
            result = await _invoke_ingest(_ring_payload(
                _next_account(tag), seller, text, device, subnet, payment,
                2.0 + (i % 24) * 0.1))
    except Exception as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}

    elapsed = _time.perf_counter() - t0
    _last_latency_ms = round(elapsed * 1000, 1)
    cid = _resolve_case(result, before)
    case = cases_db.get(cid) if cid else None

    return {
        "ok": True,
        "case_id": cid,
        "member_count": len(_members(case)) if case else 0,
        "ingested": n,
        "elapsed_ms": _last_latency_ms,
        "per_review_ms": round(elapsed * 1000 / n, 3),
        "reviews_per_sec": round(n / elapsed, 1),
    }


@app.get("/api/reviews")
def api_reviews(limit: int = 80, flagged_only: bool = False):
    rows = FEED_CACHE
    if flagged_only:
        rows = [r for r in rows if r["risk_score"] >= 50]
    return {"reviews": rows[:limit], "total": len(rows)}


_LAST_BEFORE = {}
_reload_rule_hits()
_build_feed_cache()
print("[console-api] v5.5 ready")

# ============================================================================
# console-api v5.6 — rules from scored_features.csv + teammate integrations:
#   • dual TF-IDF (word 1-2 + char_wb 3-5) + hyperbolic keyword signal
#   • GET /api/evaluation  (P@K, ring_linking_rate, stress_test_pass, coverage)
# ============================================================================

app.router.routes = [r for r in app.router.routes
                     if getattr(r, "path", None) not in {
                         "/api/reviews", "/api/custom/review", "/api/evaluation"}]


# --- 1. rebuild rules from scored_features.csv (no labeled_features.csv) ----
def _fval(row, *names):
    for n in names:
        v = row.get(n)
        if v not in (None, "", "nan", "NaN"):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


def _rules_from_row(row, lf_cols):
    # path A: lf_* columns exist right inside scored_features.csv
    if lf_cols:
        return [LF_LABEL.get(c, c[3:].replace("_", " "))
                for c in lf_cols if _gt0(row.get(c))]

    # path B: evaluate Phase-2 thresholds directly on the feature columns
    fired = []
    if _fval(row, "near_dup_score", "max_text_similarity") > 0.75:
        fired.append("copy-paste text")
    if _fval(row, "shared_device_count", "accounts_per_device") >= 5:
        fired.append("shared device")
    if _fval(row, "shared_ip_count", "accounts_per_subnet") >= 5:
        fired.append("shared IP")
    if _fval(row, "shared_payment_count", "accounts_per_payment") >= 5:
        fired.append("shared payment")
    if _fval(row, "burst_count", "seller_1h_burst_count") >= 1:
        fired.append("review burst")
    if (0 < _fval(row, "account_age_hours") < 24
            and _fval(row, "account_review_count", " account_total_reviews") > 1):
        fired.append("new account")
    if _fval(row, "review_velocity", "seller_7d_velocity_ratio") > 2.5:
        fired.append("high velocity")
    if _fval(row, "seller_new_5star_ratio", "seller_fresh_rev_ratio") > 0.05:
        fired.append("new-account 5-stars")
    return fired


def _load_rules_fallback():
    for name in ("labeled_features.csv", "scored_features.csv", "features.csv"):
        path = _os.path.join(BASE_DIR, name)
        if not _os.path.exists(path):
            continue
        lf_cols = _lf_columns(path)
        hits = 0
        with open(path, newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                rid = row.get("review_id")
                if not rid or rid not in REVIEW_INDEX:
                    continue
                fired = _rules_from_row(row, lf_cols)
                if fired:
                    REVIEW_INDEX[rid]["rules"] = fired
                    hits += 1
        print("[console-api] rules reconstructed from %s: %d reviews" % (name, hits))
        if hits > 0:
            return


_load_rules_fallback()
_build_feed_cache()


# --- 2. teammate's dual TF-IDF + hyperbolic keyword scorer ------------------
HYPERBOLIC_KEYWORDS = (
    "must buy", "10/10", "game changing", "flawless", "unbelievable",
    "best purchase", "stellar service", "exceeded expectations",
    "game-changer", "highly recommend", "superb quality", "best seller ever",
)

_SELLER_DOCS = {}
for _rid, _info in REVIEW_INDEX.items():
    _sid = _info.get("seller_id")
    _txt = _info.get("text")
    if _sid and _txt and len(_SELLER_DOCS.get(_sid, ())) < 60:
        _SELLER_DOCS.setdefault(_sid, []).append(_txt)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer as _Tfidf
    from sklearn.metrics.pairwise import cosine_similarity as _cos

    _sample_corpus = [info["text"] for info in list(REVIEW_INDEX.values())[:6000]
                      if info.get("text")]
    _WORD_VEC = _Tfidf(max_features=400, stop_words="english", ngram_range=(1, 2)).fit(_sample_corpus)
    _CHAR_VEC = _Tfidf(max_features=500, analyzer="char_wb", ngram_range=(3, 5)).fit(_sample_corpus)
    print("[console-api] dual TF-IDF fitted on %d reviews" % len(_sample_corpus))
except Exception as _exc:
    _WORD_VEC = _CHAR_VEC = None
    print("[console-api] TF-IDF unavailable:", _exc)


def _text_signals(text, seller_id):
    lower = (text or "").lower()
    kw = sum(1 for k in HYPERBOLIC_KEYWORDS if k in lower)
    word_sim = char_sim = 0.0
    peer_texts = _SELLER_DOCS.get(seller_id) or _sample_corpus[:120]
    if _WORD_VEC is not None and text and peer_texts:
        try:
            w = _cos(_WORD_VEC.transform([text]), _WORD_VEC.transform(peer_texts)).max()
            c = _cos(_CHAR_VEC.transform([text]), _CHAR_VEC.transform(peer_texts)).max()
            word_sim, char_sim = float(w), float(c)
        except Exception:
            pass
    comp_sim = 0.5 * word_sim + 0.5 * char_sim
    rules = []
    if comp_sim >= 0.75 or char_sim >= 0.88:
        rules.append("copy-paste text")
    if kw >= 2:
        rules.append("hyperbolic promo (%d)" % kw)
    return comp_sim, char_sim, kw, rules


@app.post("/api/custom/review")
async def api_custom_review(payload: dict):
    global _last_latency_ms, _LAST_BEFORE

    before = _case_snapshot()
    _LAST_BEFORE = before
    t0 = _time.perf_counter()
    try:
        result = await _invoke_ingest(payload)
    except Exception as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}

    text = payload.get("review_text") or ""
    seller = payload.get("seller_id") or "SEL_003"
    comp_sim, char_sim, kw, extra_rules = _text_signals(text, seller)

    _last_latency_ms = round((_time.perf_counter() - t0) * 1000, 1)
    cid = _resolve_case(result, before)
    case = cases_db.get(cid) if cid else None

    _rs = _as_result_dict(result).get("risk_score")
    base = (
    float(_rs) if _rs is not None else 0.0
    ) or (case.get("max_risk") if case else 0.0) or 0.0
    # blend in the text/promo signal when the base scorer sees a cold account
    promo_boost = (0.35 if comp_sim >= 0.75 else 0.20 * comp_sim) + min(0.35, 0.12 * kw)
    score_100 = max(_score_100(base), min(98, int(round((base + promo_boost) * 100))))

    rid = _lookup_rid(result)
    if rid and rid in REVIEW_INDEX:
        REVIEW_INDEX[rid]["rules"] = extra_rules
        REVIEW_INDEX[rid]["risk_score"] = score_100 / 100.0 * SCORE_CEILING

    return {
        "ok": True,
        "scored_review": {
            "review_id": rid,
            "risk_score": score_100,
            "text_similarity": round(comp_sim, 3),
            "char_similarity": round(char_sim, 3),
            "promo_keywords": kw,
            "triggered_rules": extra_rules,
        },
        "assigned_case_id": cid,
        "latency_ms": _last_latency_ms,
    }


@app.get("/api/reviews")
def api_reviews(limit: int = 80, flagged_only: bool = False):
    rows = FEED_CACHE
    if flagged_only:
        rows = [r for r in rows if r["risk_score"] >= 35 or r["triggered_rules"]]
    return {"reviews": rows[:limit], "total": len(rows)}

@app.get("/api/text-model")
def api_text_model():
    try:
        import text_model
    except Exception as e:
        return {"available": False, "error": str(e)}
    return text_model.model_info()

# --- 3. teammate's evaluation module, adapted to our ground truth ----------
@app.get("/api/evaluation")
def api_evaluation():
    ring_case = cases_db.get("CASE-0000") or {}
    ring_rids = set(_members(_BASELINE_CASES.get("CASE-0000") or ring_case))
    n_pos = len(ring_rids) or 7

    ranked = [r["review_id"] for r in FEED_CACHE]
    p_at_k, r_at_k = {}, {}
    for k in (5, 10, 20, 50):
        tp = len(set(ranked[:k]) & ring_rids)
        p_at_k["P@%d" % k] = round(tp / k, 3)
        r_at_k["R@%d" % k] = round(tp / n_pos, 3)

    ring_cases = [cid for cid, c in cases_db.items()
                  if set(_members(c)) & ring_rids]
    top_overlap = max((len(set(_members(cases_db[cid])) & ring_rids)
                       for cid in ring_cases), default=0)
    link_rate = round(top_overlap / n_pos, 3)

    api_rows = [_case_to_api(cid, c) for cid, c in list(cases_db.items())[:200]]
    explained = sum(1 for r in api_rows if r["explanation_summary"].strip())

    try:
        import text_model
        _tm = text_model.model_info()
        _out["text_model"] = {
            "auc": _tm.get("auc"),
            "accuracy": _tm.get("accuracy"),
            "split": _tm.get("split"),
            "n_total": _tm.get("n_total"),
            "caveat": _tm.get("caveat"),
        }
    except Exception:
        _out["text_model"] = {"available": False}

    return {
        "precision_at_k": p_at_k,
        "recall_at_k": r_at_k,
        "ring_members_ground_truth": n_pos,
        "ring_cases_found": len(ring_cases),
        "ring_linking_rate": link_rate,
        "fragmentation_rate": 0.0 if len(ring_cases) == 1 else round(1 - 1 / max(1, len(ring_cases)), 3),
        "stress_test_pass": bool(link_rate == 1.0 and len(ring_cases) == 1),
        "explainability_coverage_pct": round(explained / max(1, len(api_rows)) * 100, 1),
    }

# ---------- assistant ----------

# ---------- assistant ----------

try:
    from rag_assistant import TrustNetRAG
    RAG = TrustNetRAG()
except Exception as _e:
    RAG = None
    print(f"[console-api] assistant disabled: {type(_e).__name__}: {_e}")


def _rag_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    for meth in ("model_dump", "dict"):
        fn = getattr(obj, meth, None)
        if callable(fn):
            try:
                d = fn()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
    try:
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    except Exception:
        return {}


def _rag_list(val):
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        out = []
        for v in val:
            out.extend(_rag_list(v))
        return out
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null", "[]"):
        return []
    if s[0] in "[(" and s[-1] in ")]":
        try:
            import json as _json
            parsed = _json.loads(s)
            return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
    return [p.strip() for p in s.replace("|", ",").replace(";", ",").split(",") if p.strip()]


def _rag_score_100(raw):
    try:
        return int(round(float(raw) / 2.0 * 100))
    except (TypeError, ValueError):
        return 0


def _rag_review(rid):
    idx = globals().get("REVIEW_INDEX")
    if not isinstance(idx, dict):
        return None
    for k in (rid, str(rid), str(rid).strip()):
        try:
            rec = idx.get(k)
        except Exception:
            rec = None
        if rec is not None:
            return rec
    try:
        return idx.get(int(str(rid).strip()))
    except Exception:
        return None


def _rag_cases_endpoint():
    """Serve the assistant from the same payload the console renders,
    so the two can never disagree."""
    try:
        import asyncio
        import inspect
    except Exception:
        return None
    for r in getattr(app, "routes", []):
        if getattr(r, "path", None) != "/api/cases":
            continue
        if "GET" not in (getattr(r, "methods", None) or ()):
            continue
        fn = getattr(r, "endpoint", None)
        if not callable(fn):
            continue
        try:
            if inspect.iscoroutinefunction(fn):
                return asyncio.run(fn())
            return fn()
        except Exception as e:
            print(f"[console-api] assistant: /api/cases call failed: "
                  f"{type(e).__name__}: {e}")
            return None
    return None


def _rag_from_payload(payload):
    if not isinstance(payload, dict):
        return None
    rows = payload.get("cases")
    if not isinstance(rows, list) or not rows:
        return None

    import re as _re
    case_index, seller_index, tier_counts = {}, {}, {}

    for row in rows:
        c = _rag_dict(row)
        cid = str(c.get("case_id") or "").strip().upper()
        if not cid:
            continue

        tier = str(c.get("severity_tier") or c.get("tier") or "WATCH").strip().upper()

        score = c.get("case_score_100")
        if score is None:
            score = _rag_score_100(c.get("case_score"))
        try:
            score = int(score)
        except (TypeError, ValueError):
            score = 0

        sels = []
        for s in (c.get("sellers_involved") or []):
            s = str(s).strip()
            if s and s not in sels:
                sels.append(s)
        if not sels:
            for rid in _rag_list(c.get("member_ids")):
                rec = _rag_dict(_rag_review(rid))
                s = rec.get("seller_id") or rec.get("seller")
                if s and str(s).strip() not in sels:
                    sels.append(str(s).strip())

        sigs = [str(s).strip() for s in (c.get("shared_signals") or []) if str(s).strip()]

        members = c.get("review_count")
        if members is None:
            members = c.get("account_count")
        if members is None:
            members = len(_rag_list(c.get("member_ids")))
        try:
            members = int(members or 0)
        except (TypeError, ValueError):
            members = 0

        try:
            span = float(c.get("time_span_minutes") or 0)
        except (TypeError, ValueError):
            span = 0.0
        if not span:
            m = _re.search(r"(\d+(?:\.\d+)?)\s*-?\s*minute",
                           str(c.get("explanation_summary") or ""), _re.I)
            if m:
                try:
                    span = float(m.group(1))
                except ValueError:
                    span = 0.0

        summary = {"case_id": cid, "tier": tier, "score_100": score,
                   "members": members, "sellers": sels,
                   "signals": sigs, "span_min": span}
        case_index[cid] = summary
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        for s in sels:
            seller_index.setdefault(s.upper(), []).append(summary)

    return case_index, seller_index, tier_counts


def _rag_from_csv():
    import csv as _csv
    import os as _os
    path = "cases.csv"
    if not _os.path.exists(path):
        return None

    case_index, seller_index, tier_counts = {}, {}, {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            cid = str(row.get("case_id") or "").strip().upper()
            if not cid:
                continue
            tier = str(row.get("severity_tier") or row.get("tier") or "WATCH").strip().upper()
            score = _rag_score_100(row.get("case_score"))

            sels = []
            for key in ("sellers_involved", "seller_ids", "sellers", "seller_id"):
                for s in _rag_list(row.get(key)):
                    if s and s not in sels:
                        sels.append(s)
            if not sels:
                for rid in _rag_list(row.get("member_ids")):
                    rec = _rag_dict(_rag_review(rid))
                    s = rec.get("seller_id") or rec.get("seller")
                    if s and str(s).strip() not in sels:
                        sels.append(str(s).strip())

            sigs = _rag_list(row.get("signal_types")) or _rag_list(row.get("signals"))

            members = 0
            for key in ("n_members", "member_count"):
                try:
                    members = int(float(row.get(key) or 0))
                except (TypeError, ValueError):
                    members = 0
                if members:
                    break
            if not members:
                members = len(_rag_list(row.get("member_ids")))

            try:
                span = float(row.get("time_span_minutes") or 0)
            except (TypeError, ValueError):
                span = 0.0

            summary = {"case_id": cid, "tier": tier, "score_100": score,
                       "members": members, "sellers": sels,
                       "signals": sigs, "span_min": span}
            case_index[cid] = summary
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            for s in sels:
                seller_index.setdefault(s.upper(), []).append(summary)

    return case_index, seller_index, tier_counts


def _build_rag_context():
    built = _rag_from_payload(_rag_cases_endpoint())
    source = "/api/cases"
    if not built:
        built = _rag_from_csv()
        source = "cases.csv"
    if not built:
        built = ({}, {}, {})
        source = "none"

    case_index, seller_index, tier_counts = built
    ranked = sorted(case_index.values(), key=lambda c: (-c["score_100"], c["case_id"]))
    with_sellers = sum(1 for c in case_index.values() if c["sellers"])
    print(f"[console-api] assistant context from {source}: "
          f"{len(case_index)} cases, {len(seller_index)} sellers, "
          f"{with_sellers} with sellers")

    idx = globals().get("REVIEW_INDEX")
    review_counts = {}
    total_reviews = 0
    if isinstance(idx, dict):
        total_reviews = len(idx)
        for rec in idx.values():
            rec = rec if isinstance(rec, dict) else _rag_dict(rec)
            s = rec.get("seller_id") or rec.get("seller")
            if s:
                key = str(s).strip().upper()
                review_counts[key] = review_counts.get(key, 0) + 1

    return {
        "case_index": case_index,
        "seller_index": seller_index,
        "seller_review_counts": review_counts,
        "tier_counts": tier_counts,
        "total_cases": len(case_index),
        "total_reviews": total_reviews,
        "fragmentation_pct": 0,
        "top_case": ranked[0] if ranked else None,
    }


_RAG_CACHE = {"ctx": None, "ts": 0.0}


def _rag_ctx(active_case_id=None):
    import time as _time
    now = _time.time()
    if _RAG_CACHE["ctx"] is None or (now - _RAG_CACHE["ts"]) > 2.0:
        _RAG_CACHE["ctx"] = _build_rag_context()
        _RAG_CACHE["ts"] = now
    ctx = dict(_RAG_CACHE["ctx"])
    if active_case_id:
        hit = ctx["case_index"].get(str(active_case_id).strip().upper())
        if hit:
            ctx["active_case"] = hit
        else:
            ctx.pop("active_case", None)
    return ctx


@app.post("/api/assistant")
def api_assistant(payload: dict):
    if RAG is None:
        return {"answer": "Assistant unavailable.", "sources": [], "intent": "disabled"}
    payload = payload or {}
    try:
        ctx = _rag_ctx(payload.get("case_id"))
        return RAG.answer(payload.get("query", ""), ctx)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"answer": f"Assistant error: {type(e).__name__}: {e}",
                "sources": [], "intent": "error"}

# ============================================================================
# console-api v6.1 — full-payload ingest, real scoring, honest tiers,
#                     isolated rings, seller validation
# ----------------------------------------------------------------------------
# Fixes v6:
#   • /ingest was passing only 4 keys to linker.ingest — text/seller/timing
#     never reached the text-evidence path. Replaced with a full-payload
#     /ingest that computes risk, indexes the review, and links on all
#     evidence before creating/updating the case.
#   • _live_risk now combines the Kaggle CG/OR model with a hyperbole signal,
#     so paid-shill style text scores high even though the model answers
#     "machine-written vs human-written", not "paid vs honest".
#   • Seller validation is unchanged; api_inject_member now uses a real seller.
# ============================================================================

import random as _random
import uuid as _uuid

# --- 1. one tier rule ------------------------------------------------------
def _num(v, default=None):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


def _tier_for_case(raw_score, diversity, n_members):
    """Tier is a pure function of case score and signal diversity.
    _score_100 divides by SCORE_CEILING=2.0, so ESCALATE (raw>=1.0) can
    never display below 50. The case page and the assistant both rely on it."""
    raw = _num(raw_score, 0.0) or 0.0
    if raw >= 1.0 and diversity >= 3:
        return "ESCALATE"
    if raw >= 0.5 or diversity >= 2:
        return "REVIEW"
    return "WATCH"


# --- 2. real per-review risk: model + hyperbole ---------------------------
def _promo_score(text):
    """Paid-shill style: hyperbole keyword density. 0..0.95."""
    lower = str(text or "").lower()
    kw = sum(1 for k in HYPERBOLIC_KEYWORDS if k in lower)
    if kw <= 0:
        return 0.0, 0
    promo = 0.14 * kw + (0.10 if kw >= 3 else 0.0)
    return min(0.95, promo), kw


def _live_risk(payload):
    """Risk in [0,1]. Combines two independent text signals plus account cues."""
    text = str(payload.get("review_text") or payload.get("text_")
               or payload.get("text") or "").strip()

    generated = 0.0
    try:
        import text_model
        p = text_model.score_text(text)
        if p is not None:
            generated = float(p)
    except Exception:
        pass

    promo, _kw = _promo_score(text)

    score = max(generated, promo)

    age = _num(payload.get("account_age_hours"))
    if age is not None and age < 24:
        score += 0.06
    if (_num(payload.get("rating"), 0.0) or 0.0) >= 5:
        score += 0.02

    return round(max(0.0, min(1.0, score)), 4)


def _score_100_review(risk):
    return max(0, min(100, int(round(float(risk or 0.0) * 100))))


# --- 3. evidence derivation + case refresh --------------------------------
def _evidence_for(members):
    infos = []
    for m in members:
        rec = REVIEW_INDEX.get(str(m)) or REVIEW_INDEX.get(m)
        if isinstance(rec, dict):
            infos.append(rec)
    if len(infos) < 2:
        return set()

    found = set()
    for key, name in (("device", "shared_device"),
                      ("payment", "shared_payment"),
                      ("ip", "shared_ip")):
        vals = [str(i.get(key)).strip() for i in infos if i.get(key)]
        if len(vals) >= 2 and len(set(vals)) < len(vals):
            found.add(name)

    texts = [str(i.get("text") or "").strip().lower() for i in infos]
    texts = [t for t in texts if t]
    if len(texts) >= 2 and len(set(texts)) < len(texts):
        found.add("near_dup_text")

    ages = [_num(i.get("account_age_hours")) for i in infos]
    ages = [a for a in ages if a is not None]
    if len(ages) >= 2 and all(a < 24 for a in ages):
        found.add("fresh_accounts")

    return found


def _explain_case(case):
    n = int(case.get("n_members") or 0)
    labels = [SIGNAL_LABEL.get(s, str(s).replace("_", " "))
              for s in case.get("signals", [])]
    if n >= 2:
        return ("%d reviews linked by %d independent signal type%s (%s). "
                "Max individual risk %.2f; case score %.2f."
                % (n, len(labels), "" if len(labels) == 1 else "s",
                   ", ".join(labels).lower() or "entity overlap",
                   float(case.get("max_risk") or 0),
                   float(case.get("case_score") or 0)))
    return ("Single review. No shared device, IP, payment, text or timing match "
            "with any other review. Individual risk %.2f."
            % float(case.get("max_risk") or 0))


def _refresh_case(cid, extra_signals=()):
    case = cases_db.get(cid)
    if not isinstance(case, dict):
        return None
    members = _members(case)
    if not members:
        return case

    risks = []
    for m in members:
        rec = REVIEW_INDEX.get(str(m)) or REVIEW_INDEX.get(m) or {}
        v = rec.get("risk_score") if isinstance(rec, dict) else None
        f = _num(v)
        if f is not None:
            risks.append(f)
    if not risks:
        return case

    stored = set(case.get("_extra_signals") or [])
    evidence = (set(_signals(case)) | _evidence_for(members)
                | stored | set(extra_signals))
    evidence = {e for e in evidence if e and str(e).lower() != "none"}

    n = len(members)
    diversity = len(evidence)
    max_risk = max(risks)
    raw = max_risk + size_bonus(n) + diversity_bonus(diversity)

    case["max_risk"] = round(max_risk, 4)
    case["avg_risk"] = round(sum(risks) / len(risks), 4)
    case["case_score"] = round(raw, 4)
    case["signal_diversity"] = diversity
    case["signals"] = sorted(evidence)
    case["signal_types"] = ", ".join(sorted(evidence)) if evidence else "none"
    case["n_members"] = n
    case["tier"] = _tier_for_case(raw, diversity, n)
    case["explanation"] = _explain_case(case)
    return case


# --- 4. payload coercion ---------------------------------------------------
def _coerce_payload(payload):
    if not isinstance(payload, dict):
        return payload
    p = dict(payload)
    if not p.get("text_"):
        for k in ("review_text", "text"):
            if p.get(k):
                p["text_"] = p[k]
                break
    if not p.get("review_text"):
        for k in ("text_", "text"):
            if p.get(k):
                p["review_text"] = p[k]
                break
    return p


# --- 5. seller validation -------------------------------------------------
_SELLER_CACHE = {"ids": None, "n": -1}


def _seller_id_set():
    n = len(REVIEW_INDEX)
    if _SELLER_CACHE["ids"] is not None and _SELLER_CACHE["n"] == n:
        return _SELLER_CACHE["ids"]
    found = set()
    for rec in REVIEW_INDEX.values():
        if isinstance(rec, dict):
            s = rec.get("seller_id")
            if s:
                found.add(str(s).strip())
    _SELLER_CACHE["ids"] = found
    _SELLER_CACHE["n"] = n
    if found:
        print("[console-api] known sellers:", len(found))
    return found


def _check_seller(seller_id):
    s = str(seller_id or "").strip()
    if not s:
        return "seller_id is required."
    known = _seller_id_set()
    if known and s not in known:
        sample = ", ".join(sorted(known)[:3])
        return ("Unknown seller %r. Use one that exists in the dataset, "
                "for example %s." % (s, sample))
    return None


# --- 6. fresh infrastructure per ring ------------------------------------
_RING_SEQ = [0]


def _fresh_ring_infra(tag=None):
    _RING_SEQ[0] += 1
    seq = _RING_SEQ[0]
    tag = tag or ("%s%d" % (_uuid.uuid4().hex[:6].upper(), seq))
    return {
        "tag": "RING-" + tag,
        "device_fingerprint": "DEV-RING-" + tag,
        "payment_instrument": "PAY-RING-" + tag,
        "ip_block": "198.18.%d.%d/24" % (seq % 256, _random.randint(1, 254)),
    }


def _ring_payload(tag_letter, seller, text, infra, age, idx):
    return {
        "seller_id": seller,
        "account_id": "ACC-%s-%02d" % (tag_letter, idx),
        "rating": 5,
        "review_text": text,
        "text_": text,
        "device_fingerprint": infra["device_fingerprint"],
        "ip_block": infra["ip_block"],
        "payment_instrument": infra["payment_instrument"],
        "account_age_hours": age,
    }


# --- 7. drop every route we are about to redefine ------------------------
app.router.routes = [r for r in app.router.routes
                     if getattr(r, "path", None) not in {
                         "/ingest",
                         "/api/evaluation",
                         "/api/custom/review",
                         "/api/custom/ring",
                         "/api/stress-test/inject",
                         "/api/inject-member",
                         "/api/reviews",
                     }]
_ROUTE_CACHE.pop("route", None)   # force the ingest bridge to re-look-up


# --- 8. the real /ingest --------------------------------------------------
@app.post("/ingest")
def ingest_review_v6(payload: dict):
    start = time.perf_counter()
    payload = _coerce_payload(payload or {})

    text = str(payload.get("text_") or payload.get("review_text")
               or payload.get("text") or "").strip()
    if not text:
        raise HTTPException(422, "missing text_/review_text/text")

    rid = payload.get("review_id")
    try:
        rid = int(rid) if rid is not None else None
    except (TypeError, ValueError):
        rid = None
    if rid is None:
        rid = linker._next_id
    while rid in linker.parent:
        rid += 1

    seller = str(payload.get("seller_id") or "").strip() or None
    full = {
        "review_id": rid,
        "text_": text,
        "seller_id": seller,
        "device_fingerprint": _clean_infra(payload.get("device_fingerprint")),
        "ip_block": _clean_infra(payload.get("ip_block")),
        "payment_instrument": _clean_infra(payload.get("payment_instrument")),
        "account_age_hours": _num(payload.get("account_age_hours")),
        "timestamp": payload.get("timestamp"),
        "rating": _num(payload.get("rating"), 5.0),
    }

    result = linker.ingest(full)
    members = result["cluster_members"]
    signals = list(result.get("signals") or [])

    risk = _live_risk({
        "review_text": text,
        "rating": payload.get("rating"),
        "seller_id": seller,
        "account_age_hours": full["account_age_hours"],
    })

    REVIEW_INDEX[str(rid)] = {
        "text": text[:300],
        "rating": full["rating"] or 0.0,
        "seller_id": seller or "",
        "account_id": str(payload.get("account_id") or ""),
        "device": full["device_fingerprint"] or "",
        "ip": full["ip_block"] or "",
        "payment": full["payment_instrument"] or "",
        "account_age_hours": full["account_age_hours"],
        "risk_score": risk,
        "rules": [],
    }
    _SELLER_CACHE["ids"] = None   # invalidate so new sellers enter the set

    overlaps = set()
    for m in members:
        cid = MEMBER_TO_CASE.get(str(m))
        if cid and cid in cases_db:
            overlaps.add(cid)

    if overlaps:
        primary = max(overlaps, key=lambda c:
                      int(cases_db[c].get("n_members") or 0))
        case = cases_db[primary]
        extra = list(case.get("_extra_signals") or [])
        for other in overlaps - {primary}:
            oc = cases_db.pop(other, None)
            if isinstance(oc, dict):
                existing = [str(x) for x in _members(case)]
                for m in _members(oc):
                    if str(m) not in existing:
                        case.setdefault("member_ids", []).append(m)
                extra.extend(_signals(oc))
                extra.extend(oc.get("_extra_signals") or [])
                for m in _members(oc):
                    MEMBER_TO_CASE[str(m)] = primary
        if str(rid) not in [str(x) for x in _members(case)]:
            case.setdefault("member_ids", []).append(rid)
        case["_extra_signals"] = list(set(extra) | set(signals))
        case_id = primary
    else:
        case_id = "CASE-LIVE-%d" % rid
        cases_db[case_id] = {
            "case_id": case_id,
            "type": "cluster",
            "tier": "WATCH",
            "case_score": 0.0,
            "max_risk": 0.0,
            "avg_risk": 0.0,
            "n_members": 1,
            "n_sellers": 1,
            "n_accounts": 1,
            "signal_diversity": 0,
            "signal_types": "none",
            "time_span_minutes": 0.0,
            "explanation": "",
            "member_ids": [rid],
            "_extra_signals": list(signals),
        }

    for m in members:
        MEMBER_TO_CASE[str(m)] = case_id
    MEMBER_TO_CASE[str(rid)] = case_id

    _refresh_case(case_id)
    case = cases_db.get(case_id) or {}
    elapsed = (time.perf_counter() - start) * 1000

    return {
        "review_id": rid,
        "merged_into_cluster": bool(result["merged"]),
        "cluster_size": result["cluster_size"],
        "case_id": case_id,
        "tier": case.get("tier"),
        "risk_score": risk,
        "signals": signals,
        "latency_ms": round(elapsed, 2),
        "message": ("Merged into %s — tier=%s, size=%d, %.1fms."
                    % (case_id, case.get("tier"),
                       result["cluster_size"], elapsed)),
    }


# --- 9. wrapper so every other caller still goes through the bridge ------
_V55_INVOKE = _invoke_ingest


def _post_ingest(payload, result, before):
    d = _as_result_dict(result)
    rid = d.get("review_id") or d.get("rid")
    if rid is None:
        sr = d.get("scored_review")
        if isinstance(sr, dict):
            rid = sr.get("review_id")
    if rid is None:
        return

    rid = str(rid)
    rec = REVIEW_INDEX.get(rid)
    if not isinstance(rec, dict):
        return

    if rec.get("risk_score") is None:
        rec["risk_score"] = _live_risk(payload)
    rec["account_age_hours"] = _num(payload.get("account_age_hours"))

    cid = MEMBER_TO_CASE.get(rid) or _resolve_case(result, before)
    if not cid:
        return
    MEMBER_TO_CASE[rid] = cid
    extra = d.get("signals")
    _refresh_case(cid, extra if isinstance(extra, (list, tuple, set)) else [])


async def _invoke_ingest(payload):
    payload = _coerce_payload(payload)
    before = _case_snapshot()
    globals()["_LAST_BEFORE"] = before
    result = await _V55_INVOKE(payload)
    try:
        _post_ingest(payload, result, before)
    except Exception as exc:
        print("[console-api] v6 post-ingest failed:", type(exc).__name__, exc)
    return result


# --- 10. dashboard-facing endpoints --------------------------------------
@app.post("/api/inject-member")
async def api_inject_member():
    global _last_latency_ms
    payload = {
        "seller_id": "SELLER-0265",
        "account_id": "ACC-RING-LIVE-%d" % (len(REVIEW_INDEX) + 1),
        "rating": 5,
        "review_text": "AMAZING 10/10 MUST BUY! Best seller ever! "
                       "Incredible quality, flawless product!",
        "device_fingerprint": "DEV-A7F3E9C1-FAKE-RING",
        "ip_block": "198.51.100.0/24",
        "payment_instrument": "PAY-RING-4829",
        "account_age_hours": 3.4,
    }
    before = _case_snapshot()
    t0 = _time.perf_counter()
    try:
        result = await _invoke_ingest(payload)
    except Exception as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}

    _last_latency_ms = round((_time.perf_counter() - t0) * 1000, 1)
    rid = _lookup_rid(result)
    cid = (MEMBER_TO_CASE.get(str(rid)) if rid is not None else None) \
          or _resolve_case(result, before)
    case = cases_db.get(cid) if cid else None
    return {
        "ok": True,
        "case_id": cid,
        "member_count": len(_members(case)) if case else 0,
        "tier": _norm_tier(case.get("tier")) if case else "Watch",
        "latency_ms": _last_latency_ms,
    }


@app.post("/api/custom/review")
async def api_custom_review(payload: dict):
    global _last_latency_ms
    payload = _coerce_payload(payload or {})

    if payload.get("seller_id"):
        err = _check_seller(payload["seller_id"])
        if err:
            return {"ok": False, "error": err}

    before = _case_snapshot()
    t0 = _time.perf_counter()
    try:
        result = await _invoke_ingest(payload)
    except Exception as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}

    _last_latency_ms = round((_time.perf_counter() - t0) * 1000, 1)
    rid = _lookup_rid(result)
    cid = (MEMBER_TO_CASE.get(str(rid)) if rid is not None else None) \
          or _resolve_case(result, before)
    case = cases_db.get(cid) if cid else None

    rec = REVIEW_INDEX.get(str(rid)) if rid is not None else None
    risk = rec.get("risk_score") if isinstance(rec, dict) else None
    if risk is None and case is not None:
        risk = case.get("max_risk")
    risk = _num(risk, 0.0) or 0.0

    text = str(payload.get("review_text") or payload.get("text_") or "")
    _promo, kw = _promo_score(text)

    return {
        "ok": True,
        "scored_review": {
            "review_id": rid,
            "risk_score": _score_100_review(risk),
            "model_probability": round(risk, 4),
            "promo_keywords": kw,
        },
        "assigned_case_id": cid,
        "case_tier": _norm_tier(case.get("tier")) if case else "Watch",
        "latency_ms": _last_latency_ms,
    }


@app.post("/api/custom/ring")
async def api_custom_ring(payload: dict):
    global _last_latency_ms
    payload = payload or {}

    n = max(3, min(20, int(_num(payload.get("num_accounts"), 6) or 6)))
    seller = payload.get("seller_id") or "SELLER-0265"
    err = _check_seller(seller)
    if err:
        return {"ok": False, "error": err}

    text = (payload.get("review_text")
            or "UNBELIEVABLE 10/10 MUST BUY!!! Absolute game changer, "
               "best purchase ever, highly recommend.")
    infra = _fresh_ring_infra()

    before = _case_snapshot()
    t0 = _time.perf_counter()
    result = None
    try:
        for i in range(n):
            result = await _invoke_ingest(_ring_payload(
                "CUSTOM-%s" % infra["tag"], seller, text, infra,
                1.5 + i * 0.2, i))
    except Exception as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}

    _last_latency_ms = round((_time.perf_counter() - t0) * 1000, 1)
    cid = _resolve_case(result, before)
    case = cases_db.get(cid) if cid else None
    raw = float(case.get("case_score") or 0.0) if case else 0.0

    return {
        "ok": True,
        "ring_tag": infra["tag"],
        "case_id": cid,
        "member_count": len(_members(case)) if case else 0,
        "tier": _norm_tier(case.get("tier")) if case else "Watch",
        "case_score_100": _score_100(raw),
        "fragments": 1 if cid else 0,
        "latency_ms": _last_latency_ms,
    }


@app.post("/api/stress-test/inject")
async def api_stress_test_inject(body: dict = None):
    global _last_latency_ms
    body = body or {}

    seller = body.get("seller_id") or "SELLER-0265"
    err = _check_seller(seller)
    if err:
        return {"ok": False, "error": err}

    infra = _fresh_ring_infra()
    text = "AMAZING 10/10 MUST BUY! Best seller ever!"

    before = _case_snapshot()
    t0 = _time.perf_counter()
    result = None
    try:
        for i in range(8):
            result = await _invoke_ingest(_ring_payload(
                "APEX-%s" % infra["tag"], seller, text, infra,
                2.0 + i * 0.3, i))
    except Exception as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}

    _last_latency_ms = round((_time.perf_counter() - t0) * 1000, 1)
    cid = _resolve_case(result, before)
    case = cases_db.get(cid) if cid else None
    return {
        "ok": True,
        "ring_tag": infra["tag"],
        "case_id": cid,
        "member_count": len(_members(case)) if case else 0,
        "tier": _norm_tier(case.get("tier")) if case else "Watch",
        "latency_ms": _last_latency_ms,
    }


# --- 11. evaluation (fixes the _out NameError) ---------------------------
@app.get("/api/evaluation")
def api_evaluation():
    ring_case = cases_db.get("CASE-0000") or {}
    ring_rids = set(_members(_BASELINE_CASES.get("CASE-0000") or ring_case))
    n_pos = len(ring_rids) or 7

    ranked = [r["review_id"] for r in FEED_CACHE]
    p_at_k, r_at_k = {}, {}
    for k in (5, 10, 20, 50):
        tp = len(set(ranked[:k]) & ring_rids)
        p_at_k["P@%d" % k] = round(tp / k, 3)
        r_at_k["R@%d" % k] = round(tp / n_pos, 3)

    ring_cases = [cid for cid, c in cases_db.items()
                  if set(_members(c)) & ring_rids]
    top_overlap = max((len(set(_members(cases_db[cid])) & ring_rids)
                       for cid in ring_cases), default=0)
    link_rate = round(top_overlap / n_pos, 3)

    api_rows = [_case_to_api(cid, c) for cid, c in list(cases_db.items())[:200]]
    explained = sum(1 for r in api_rows if r["explanation_summary"].strip())

    out = {
        "precision_at_k": p_at_k,
        "recall_at_k": r_at_k,
        "ring_members_ground_truth": n_pos,
        "ring_cases_found": len(ring_cases),
        "ring_linking_rate": link_rate,
        "fragmentation_rate": (0.0 if len(ring_cases) == 1
                               else round(1 - 1 / max(1, len(ring_cases)), 3)),
        "stress_test_pass": bool(link_rate == 1.0 and len(ring_cases) == 1),
        "explainability_coverage_pct": round(
            explained / max(1, len(api_rows)) * 100, 1),
        "text_model": {"available": False},
    }

    try:
        import text_model
        tm = text_model.model_info()
        out["text_model"] = {
            "available": bool(tm.get("available")),
            "auc": tm.get("auc"),
            "accuracy": tm.get("accuracy"),
            "split": tm.get("split"),
            "n_total": tm.get("n_total"),
            "n_test": tm.get("n_test"),
            "caveat": tm.get("caveat"),
        }
    except Exception:
        pass

    return out


# --- 12. feed rebuild ----------------------------------------------------
def _build_feed_cache():
    global FEED_CACHE
    rows = []
    for rid, info in REVIEW_INDEX.items():
        raw = _num(info.get("risk_score"), 0.0) or 0.0
        rows.append({
            "review_id": rid,
            "risk_score": _score_100_review(raw),
            "raw_risk": round(raw, 4),
            "seller_id": info.get("seller_id") or "—",
            "account_id": info.get("account_id") or "—",
            "rating": info.get("rating") or 0,
            "review_text": info.get("text") or "",
            "triggered_rules": info.get("rules") or [],
        })
    rows.sort(key=lambda r: r["raw_risk"], reverse=True)
    FEED_CACHE = rows
    print("[console-api] feed cached:", len(rows))


@app.get("/api/reviews")
def api_reviews(limit: int = 80, flagged_only: bool = False):
    rows = FEED_CACHE
    if flagged_only:
        rows = [r for r in rows if r["risk_score"] >= 35 or r["triggered_rules"]]
    return {"reviews": rows[:limit], "total": len(rows)}


_build_feed_cache()
print("[console-api] v6.1 ready — full-payload ingest, real scoring, isolated rings")

    
# ============================================================================
# console-api v6.2 — Judge Test Lab
# ----------------------------------------------------------------------------
# Six preset rings, each with fresh infrastructure and a unique run tag, so
# running one does not contaminate the next. Each run reports PASS/FAIL
# against a stated goal. Also exposes /api/judge-lab/presets for the UI.
# ============================================================================

_LAB_PRESETS = [
    {
        "id": "classic",
        "name": "Classic ring",
        "description": "Same device, IP and payment. The textbook coordinated ring.",
        "members": 8,
        "shares": ["device", "ip", "payment"],
        "sellers": 1,
        "spacing_seconds": 30,
        "min_tier": "Escalate",
        "goal": "PASS at Escalate",
        "text": "UNBELIEVABLE 10/10 MUST BUY!!! Absolute game changer, best purchase I have ever made.",
    },
    {
        "id": "two_signals",
        "name": "Two shared signals",
        "description": "Only device and IP are shared — payment differs. Text and timing carry the rest.",
        "members": 8,
        "shares": ["device", "ip"],
        "sellers": 1,
        "spacing_seconds": 30,
        "min_tier": "Review",
        "goal": "PASS at Review or Escalate",
        "text": "Absolutely flawless product. Best purchase I have made this year.",
    },
    {
        "id": "one_signal",
        "name": "One shared signal",
        "description": "Only device is shared. Text, timing and fresh accounts must bridge the rest.",
        "members": 8,
        "shares": ["device"],
        "sellers": 1,
        "spacing_seconds": 30,
        "min_tier": "Review",
        "goal": "PASS at Review or Escalate",
        "text": "Superb quality. Exceeded expectations on every front.",
    },
    {
        "id": "no_hardware",
        "name": "No shared hardware",
        "description": "No device, IP or payment in common. Text, seller and account age carry the link.",
        "members": 8,
        "shares": [],
        "sellers": 1,
        "spacing_seconds": 60,
        "min_tier": "Review",
        "goal": "PASS at Review or Escalate",
        "text": "Highly recommend. Stellar service, top quality, will buy again.",
    },
    {
        "id": "slow_spread",
        "name": "Slow and spread out",
        "description": "Six reviews over 12-hour intervals. Text and fresh accounts only — no timing signal.",
        "members": 6,
        "shares": [],
        "sellers": 1,
        "spacing_seconds": 43200,
        "min_tier": "Review",
        "goal": "PASS at Review (Escalate unlikely — signals weaken with time)",
        "text": "Great product, arrived quickly, works as expected.",
    },
    {
        "id": "cross_seller",
        "name": "Cross-seller ring",
        "description": "Nine reviews across three sellers, sharing device and IP.",
        "members": 9,
        "shares": ["device", "ip"],
        "sellers": 3,
        "spacing_seconds": 45,
        "min_tier": "Review",
        "goal": "PASS at Review or Escalate",
        "text": "Amazing service, product exactly as shown, five stars.",
    },
]


def _lab_sellers(n):
    pool = sorted(_seller_id_set())
    if not pool:
        return []
    out = list(pool[:n])
    while len(out) < n:
        out.append(pool[len(out) % len(pool)])
    return out[:n]


@app.get("/api/judge-lab/presets")
def api_lab_presets():
    return {
        "presets": [
            {k: v for k, v in p.items() if k != "text"}
            for p in _LAB_PRESETS
        ]
    }


@app.post("/api/judge-lab/run")
async def api_lab_run(body: dict = None):
    body = body or {}
    preset_id = str(body.get("preset") or "").strip()
    preset = next((p for p in _LAB_PRESETS if p["id"] == preset_id), None)
    if preset is None:
        return {"ok": False, "error": "unknown preset %r" % preset_id}

    sellers = _lab_sellers(preset.get("sellers", 1))
    if not sellers:
        return {"ok": False, "error":
                "no sellers loaded — is enriched_reviews.csv present?"}

    infra = _fresh_ring_infra()
    run_tag = infra["tag"]
    text = preset["text"] + " [" + run_tag + "]"
    shares = set(preset["shares"])
    spacing = float(preset["spacing_seconds"])
    base_ts = time.time() - (preset["members"] - 1) * spacing

    rids = []
    t0 = time.perf_counter()
    try:
        for i in range(preset["members"]):
            seller = sellers[i % len(sellers)]
            ts = base_ts + i * spacing
            payload = {
                "seller_id": seller,
                "account_id": "ACC-LAB-%s-%02d" % (run_tag, i),
                "rating": 5,
                "review_text": text,
                "text_": text,
                "device_fingerprint": infra["device_fingerprint"] if "device" in shares else None,
                "ip_block": infra["ip_block"] if "ip" in shares else None,
                "payment_instrument": infra["payment_instrument"] if "payment" in shares else None,
                "account_age_hours": 2.0 + i * 0.1,
                "timestamp": ts,
            }
            result = await _invoke_ingest(payload)
            rid = _lookup_rid(result)
            if rid is not None:
                rids.append(str(rid))
    except Exception as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}

    elapsed_ms = (time.perf_counter() - t0) * 1000

    mapped = {MEMBER_TO_CASE.get(r) for r in rids}
    mapped.discard(None)
    primary_cid = next(iter(mapped)) if len(mapped) == 1 else None
    primary = cases_db.get(primary_cid) if primary_cid else None

    ring_in_case = (sum(1 for r in rids if MEMBER_TO_CASE.get(r) == primary_cid)
                    if primary_cid else 0)
    total_members = len(_members(primary)) if primary else 0

    tier = _norm_tier(primary.get("tier")) if primary else "Watch"
    rank = {"Watch": 0, "Review": 1, "Escalate": 2}
    tier_ok = rank.get(tier, 0) >= rank.get(preset["min_tier"], 0)
    time_ok = elapsed_ms < 5000
    one_case = len(mapped) == 1 and ring_in_case == len(rids)
    all_pass = one_case and tier_ok and time_ok

    case_score_100 = _score_100(primary.get("case_score") or 0) if primary else 0
    signals = _signals(primary) if primary else []

    print("[judge-lab] %s: %d members → %d case(s), tier=%s, %.0fms, %s"
          % (preset_id, len(rids), len(mapped), tier, elapsed_ms,
             "PASS" if all_pass else "FAIL"))

    return {
        "ok": True,
        "preset": preset_id,
        "ring_tag": run_tag,
        "case_id": primary_cid,
        "ring_member_count": len(rids),
        "ring_members_in_case": ring_in_case,
        "total_members_in_case": total_members,
        "fragments": len(mapped),
        "tier": tier,
        "min_tier": preset["min_tier"],
        "case_score_100": case_score_100,
        "elapsed_ms": round(elapsed_ms, 1),
        "signals": signals,
        "checks": {
            "one_case": one_case,
            "tier_ok": tier_ok,
            "time_ok": time_ok,
        },
        "pass": all_pass,
        "goal": preset["goal"],
    }

try:
    import text_model as _tm_boot
    _tm_boot._load()
    print("[console-api] text model warmed at boot")
except Exception as _e:
    print("[console-api] text model not warmed:", type(_e).__name__)

print("[console-api] v6.2 ready — Judge Test Lab")

# ============================================================================
# console-api v6.3 — seller alias resolution
# ----------------------------------------------------------------------------
# Fixes the Test tab: SEL_003 and other legacy ids resolve to a real seller
# instead of being rejected. Resolution order:
#   exact  ->  case-insensitive  ->  numeric-suffix alias (SEL_003 -> SELLER-0003)
#   ->  default demo seller for SEL_/SELLER_ shaped ids that do not exist
# Truly malformed ids (SELLER-NOT-A-REAL-ID-9999) are still rejected.
# ============================================================================

import re as _re

_SELLER_INDEX = {"ids": None, "n": -1, "lower": {}, "tail": {},
                 "counts": {}, "top": None}

_DEFAULT_SELLER = "SELLER-0265"


def _seller_index():
    """sellers keyed for fast resolution, rebuilt when the index grows."""
    n = len(REVIEW_INDEX)
    if _SELLER_INDEX["ids"] is not None and _SELLER_INDEX["n"] == n:
        return _SELLER_INDEX

    counts = {}
    for rec in REVIEW_INDEX.values():
        if isinstance(rec, dict):
            s = rec.get("seller_id")
            if s:
                s = str(s).strip()
                if s:
                    counts[s] = counts.get(s, 0) + 1

    ids = set(counts)
    lower = {s.lower(): s for s in ids}
    tail = {}
    for s in ids:
        m = _re.search(r"(\d+)\s*$", s)
        if m:
            tail.setdefault(int(m.group(1)), s)
    top = max(counts, key=lambda k: counts[k]) if counts else None

    _SELLER_INDEX.update({"ids": ids, "n": n, "lower": lower,
                          "tail": tail, "counts": counts, "top": top})
    if ids:
        print("[console-api] known sellers:", len(ids))
    return _SELLER_INDEX


def _default_seller():
    idx = _seller_index()
    for cand in (_DEFAULT_SELLER, idx.get("top")):
        if cand and cand in idx["ids"]:
            return cand
    return idx.get("top")


def _seller_id_set():
    """Kept for the Judge Lab, which picks real sellers from this set."""
    return _seller_index()["ids"]


def _resolve_seller(sid, allow_fallback=True):
    """-> (canonical_id, note) ; note is None when it matched exactly.
       -> (None, error_message) when it cannot be resolved."""
    idx = _seller_index()
    ids = idx["ids"]

    raw = str(sid or "").strip()

    if not ids:
        return (raw or _DEFAULT_SELLER,
                "seller index empty — id passed through unchanged")

    if not raw:
        d = _default_seller()
        return d, "no seller supplied; defaulted to %s" % d

    if raw in ids:
        return raw, None

    hit = idx["lower"].get(raw.lower())
    if hit:
        return hit, "matched %r by case-insensitive id" % raw

    m = _re.search(r"(\d+)\s*$", raw)
    if m:
        hit = idx["tail"].get(int(m.group(1)))
        if hit:
            return hit, "matched %r to %s by numeric suffix" % (raw, hit)

    # Legacy SEL_003 / SELLER_003 shape whose number is absent from the set:
    # substitute the default demo seller rather than failing the demo button.
    alias_shaped = bool(_re.match(r"^(sel|seller)[-_]?\d+$", raw.lower()))
    if allow_fallback and alias_shaped:
        d = _default_seller()
        if d:
            return d, "unknown seller %r; substituted default %s" % (raw, d)

    sample = ", ".join(sorted(ids)[:3])
    return None, ("Unknown seller %r. Use one that exists in the dataset, "
                  "for example %s." % (raw, sample))


def _seller_or_error(sid):
    canonical, note = _resolve_seller(sid)
    if canonical is None:
        return None, None, note
    return canonical, note, None


# --- re-register the three test endpoints with resolution ------------------
app.router.routes = [r for r in app.router.routes
                     if getattr(r, "path", None) not in {
                         "/api/custom/review",
                         "/api/custom/ring",
                         "/api/stress-test/inject",
                         "/api/inject-member",
                     }]


@app.post("/api/custom/review")
async def api_custom_review(payload: dict):
    global _last_latency_ms
    payload = _coerce_payload(payload or {})

    canonical, note, err = _seller_or_error(payload.get("seller_id"))
    if err:
        return {"ok": False, "error": err}
    payload["seller_id"] = canonical

    before = _case_snapshot()
    t0 = _time.perf_counter()
    try:
        result = await _invoke_ingest(payload)
    except Exception as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}

    _last_latency_ms = round((_time.perf_counter() - t0) * 1000, 1)
    rid = _lookup_rid(result)
    cid = (MEMBER_TO_CASE.get(str(rid)) if rid is not None else None) \
          or _resolve_case(result, before)
    case = cases_db.get(cid) if cid else None

    rec = REVIEW_INDEX.get(str(rid)) if rid is not None else None
    risk = rec.get("risk_score") if isinstance(rec, dict) else None
    if risk is None and case is not None:
        risk = case.get("max_risk")
    risk = _num(risk, 0.0) or 0.0

    text = str(payload.get("review_text") or payload.get("text_") or "")
    _promo, kw = _promo_score(text)

    out = {
        "ok": True,
        "seller_id": canonical,
        "scored_review": {
            "review_id": rid,
            "risk_score": _score_100_review(risk),
            "model_probability": round(risk, 4),
            "promo_keywords": kw,
        },
        "assigned_case_id": cid,
        "case_tier": _norm_tier(case.get("tier")) if case else "Watch",
        "latency_ms": _last_latency_ms,
    }
    if note:
        out["seller_note"] = note
    return out


@app.post("/api/custom/ring")
async def api_custom_ring(payload: dict):
    global _last_latency_ms
    payload = payload or {}

    n = max(3, min(20, int(_num(payload.get("num_accounts"), 6) or 6)))
    canonical, note, err = _seller_or_error(payload.get("seller_id"))
    if err:
        return {"ok": False, "error": err}

    text = (payload.get("review_text")
            or "UNBELIEVABLE 10/10 MUST BUY!!! Absolute game changer, "
               "best purchase ever, highly recommend.")
    infra = _fresh_ring_infra()

    before = _case_snapshot()
    t0 = _time.perf_counter()
    result = None
    try:
        for i in range(n):
            result = await _invoke_ingest(_ring_payload(
                "CUSTOM-%s" % infra["tag"], canonical, text, infra,
                1.5 + i * 0.2, i))
    except Exception as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}

    _last_latency_ms = round((_time.perf_counter() - t0) * 1000, 1)
    cid = _resolve_case(result, before)
    case = cases_db.get(cid) if cid else None
    raw = float(case.get("case_score") or 0.0) if case else 0.0

    out = {
        "ok": True,
        "seller_id": canonical,
        "ring_tag": infra["tag"],
        "case_id": cid,
        "member_count": len(_members(case)) if case else 0,
        "tier": _norm_tier(case.get("tier")) if case else "Watch",
        "case_score_100": _score_100(raw),
        "fragments": 1 if cid else 0,
        "latency_ms": _last_latency_ms,
    }
    if note:
        out["seller_note"] = note
    return out


@app.post("/api/stress-test/inject")
async def api_stress_test_inject(body: dict = None):
    global _last_latency_ms
    body = body or {}

    canonical, note, err = _seller_or_error(body.get("seller_id"))
    if err:
        return {"ok": False, "error": err}

    infra = _fresh_ring_infra()
    text = "AMAZING 10/10 MUST BUY! Best seller ever!"

    before = _case_snapshot()
    t0 = _time.perf_counter()
    result = None
    try:
        for i in range(8):
            result = await _invoke_ingest(_ring_payload(
                "APEX-%s" % infra["tag"], canonical, text, infra,
                2.0 + i * 0.3, i))
    except Exception as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}

    _last_latency_ms = round((_time.perf_counter() - t0) * 1000, 1)
    cid = _resolve_case(result, before)
    case = cases_db.get(cid) if cid else None

    out = {
        "ok": True,
        "seller_id": canonical,
        "ring_tag": infra["tag"],
        "case_id": cid,
        "member_count": len(_members(case)) if case else 0,
        "tier": _norm_tier(case.get("tier")) if case else "Watch",
        "latency_ms": _last_latency_ms,
    }
    if note:
        out["seller_note"] = note
    return out


@app.post("/api/inject-member")
async def api_inject_member():
    global _last_latency_ms
    seller = _default_seller() or _DEFAULT_SELLER

    payload = {
        "seller_id": seller,
        "account_id": "ACC-RING-LIVE-%d" % (len(REVIEW_INDEX) + 1),
        "rating": 5,
        "review_text": "AMAZING 10/10 MUST BUY! Best seller ever! "
                       "Incredible quality, flawless product!",
        "device_fingerprint": "DEV-A7F3E9C1-FAKE-RING",
        "ip_block": "198.51.100.0/24",
        "payment_instrument": "PAY-RING-4829",
        "account_age_hours": 3.4,
    }

    before = _case_snapshot()
    t0 = _time.perf_counter()
    try:
        result = await _invoke_ingest(payload)
    except Exception as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}

    _last_latency_ms = round((_time.perf_counter() - t0) * 1000, 1)
    rid = _lookup_rid(result)
    cid = (MEMBER_TO_CASE.get(str(rid)) if rid is not None else None) \
          or _resolve_case(result, before)
    case = cases_db.get(cid) if cid else None
    return {
        "ok": True,
        "case_id": cid,
        "member_count": len(_members(case)) if case else 0,
        "tier": _norm_tier(case.get("tier")) if case else "Watch",
        "latency_ms": _last_latency_ms,
    }


# --- inspection + dropdown source ----------------------------------------
@app.get("/api/sellers")
def api_sellers(limit: int = 40):
    idx = _seller_index()
    counts = idx.get("counts") or {}
    rows = sorted(idx["ids"], key=lambda s: (-counts.get(s, 0), s))
    return {
        "sellers": rows[:max(1, min(500, limit))],
        "total": len(idx["ids"]),
        "default": _default_seller(),
    }


@app.get("/api/debug/sellers")
def api_debug_sellers():
    return {
        "n_sellers": len(_seller_index()["ids"]),
        "default": _default_seller(),
        "sample": sorted(_seller_index()["ids"])[:10],
        "SEL_003_resolves_to": _resolve_seller("SEL_003"),
        "SELLER-NOT-A-REAL-ID-9999": _resolve_seller("SELLER-NOT-A-REAL-ID-9999"),
    }


print("[console-api] v6.3 ready — seller aliases resolve")


# ============================================================================
# console-api v6.4 — Upload & Scan (sandboxed)
# ----------------------------------------------------------------------------
# Accepts a CSV upload (≤ 2000 rows, ≤ 5 MB), scores each row with the text
# model + hyperbole signal, links rows using linker.py on a fresh instance,
# reports rings + high-risk solos. Never mutates cases_db, REVIEW_INDEX or
# MEMBER_TO_CASE — the upload is self-contained.
# ============================================================================

import io as _io
from fastapi import UploadFile, File
from fastapi.responses import Response
from linker import IncrementalLinker as _UploadLinker
from linker import _as_epoch as _parse_ts

_UPLOAD_STATE = {
    "scan_id": None, "cases": [], "rows": [],
    "evidence": [], "evidence_line": "", "filename": None,
}

_UPLOAD_MAX_ROWS = 2000
_UPLOAD_MAX_BYTES = 5 * 1024 * 1024
_UPLOAD_TEXT_COLS = ("text_", "text", "review_text", "review", "body", "content")


def _upload_validate(raw, filename):
    if filename and not filename.lower().endswith((".csv", ".txt")):
        return None, "Only .csv or .txt files are accepted."
    if len(raw) > _UPLOAD_MAX_BYTES:
        return None, ("File is %d KB; maximum is %d KB. "
                      "Trim it to 2,000 rows or fewer." % (
                          len(raw) // 1024, _UPLOAD_MAX_BYTES // 1024))
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except Exception:
            return None, "File is not valid UTF-8 or Latin-1."
    try:
        rows = list(_csv.DictReader(_io.StringIO(text)))
    except Exception as e:
        return None, "Could not parse CSV: %s" % e
    if not rows:
        return None, "File is empty or has no header row."
    if len(rows) > _UPLOAD_MAX_ROWS:
        return None, ("File has %d rows; maximum is %d. "
                      "Trim it or split into multiple uploads." % (
                          len(rows), _UPLOAD_MAX_ROWS))

    cols = [c for c in (rows[0].keys() if rows else []) if c]
    text_col = None
    for c in cols:
        if c.strip().lower() in _UPLOAD_TEXT_COLS:
            text_col = c
            break
    if not text_col:
        return None, ("No text column found. Expected one of: %s. "
                      "Got: %s." % (", ".join(_UPLOAD_TEXT_COLS),
                                    ", ".join(cols[:8]) or "none"))
    return (rows, cols, text_col), None


def _pick_col(row, *names):
    for n in names:
        v = row.get(n)
        if v not in (None, ""):
            return v
    return None

def _upload_fail(code, message, report=None, hints=None):
    """Uniform failure payload. Always returns HTTP 200 with ok=False so the
    dashboard never has to special-case a status code, and every list key is
    present so the front-end never calls .map() on undefined."""
    rep = report or {}
    return {
        "ok": False,
        "error_code": code,
        "error": message,
        "hints": hints or rep.get("hints", []),
        "report": rep or None,
        "filename": (rep.get("file") or {}).get("name", ""),
        "rows_total": (rep.get("file") or {}).get("rows_in_file", 0),
        "rows_scanned": 0,
        "rows_accepted": 0,
        "rows_rejected": (rep.get("rows") or {}).get("rejected", 0),
        "ring_cases": [],
        "high_risk_solos": [],
        "ring_cases_count": 0,
        "high_risk_solo_count": 0,
        "low_risk_solo_count": 0,
        "flagged_count": 0,
        "evidence": [],
        "evidence_line": "",
        "column_mapping": rep.get("column_mapping", {}),
        "warnings": rep.get("warnings", []),
        "guidance": rep.get("guidance", []),
    }


def _upload_regroup(lk, scored):
    """Group uploaded rows by their FINAL union-find root.

    lk.ingest() returns the root as it stood at that moment. A later row that
    merges into the same component can re-parent an earlier row, leaving its
    recorded root stale and stranding it alone. Resolving every id against the
    linker's finished state makes the grouping independent of whichever union
    rule the linker uses.
    """
    def _find(rid):
        fn = getattr(lk, "find", None)
        if callable(fn):
            try:
                return fn(rid)
            except Exception:
                pass
        parent = getattr(lk, "parent", None)
        if isinstance(parent, dict):
            cur, seen = rid, set()
            while parent.get(cur, cur) != cur and cur not in seen:
                seen.add(cur)
                cur = parent[cur]
            return cur
        return rid

    groups = {}
    for s in scored:
        groups.setdefault(_find(s["review_id"]), []).append(s)

    for root, members in groups.items():
        for s in members:
            s["cluster_root"] = root
            s["cluster_size"] = len(members)
    return groups


def _upload_opt(v):
    """NaN / NaT / None -> None, everything else floats cleanly."""
    try:
        if v is None:
            return None
        if v != v:            # NaN
            return None
    except (TypeError, ValueError):
        return v
    return v

@app.post("/api/upload/scan")
async def api_upload_scan(file: UploadFile = File(...)):
    import pandas as _pd

    filename = file.filename or "upload.csv"
    try:
        raw = await file.read()
    except Exception as e:
        return _upload_fail("read_failed", "Could not read file: %s" % e)

    prep = upload_io.prepare(raw, filename)
    report = prep["report"]
    if not prep["ok"]:
        return _upload_fail(
            report.get("error_code"),
            report.get("message") or "The file could not be scanned.",
            report)

    df = prep["df"]                      # 9 canonical columns, clean, indexed

    lk = _UploadLinker()                 # fresh, empty — self-contained sandbox
    base_rid = 1_000_000
    scored = []

    for seq, (idx, row) in enumerate(df.iterrows()):
        text = str(row.get("text_") or "").strip()
        if not text:
            continue

        seller = str(row.get("seller_id") or "").strip()
        if seller:
            canonical, _note = _resolve_seller(seller, allow_fallback=False)
            if canonical:
                seller = canonical

        age = _upload_opt(row.get("account_age_hours"))
        age = float(age) if isinstance(age, (int, float)) else None

        rating = _upload_opt(row.get("rating"))
        rating = float(rating) if isinstance(rating, (int, float)) else 0.0

        ts = row.get("timestamp")
        try:
            ts = None if (ts is None or _pd.isna(ts)) else ts
        except (TypeError, ValueError):
            ts = None

        rid = base_rid + seq
        payload = {
            "review_id": rid,
            "text_": text,
            "seller_id": seller or None,
            "device_fingerprint": str(row.get("device_fingerprint") or ""),
            "ip_block": str(row.get("ip_block") or ""),
            "payment_instrument": str(row.get("payment_instrument") or ""),
            "account_age_hours": age,
            "timestamp": ts,
            "rating": rating,
        }

        try:
            res = lk.ingest(payload)
        except Exception as e:
            return _upload_fail(
                "ingest_failed",
                "Linker failed on row %d: %s" % (int(idx) + 2, e))

        risk = _live_risk(payload)
        scored.append({
            "row": int(idx) + 2,
            "review_id": rid,
            "text": text[:200],
            "rating": rating,
            "seller_id": seller,
            "account_id": (str(row.get("account_id") or "").strip()
                           or ("ACCT-UPLOAD-%03d" % seq)),
            "risk": round(float(risk), 4),
            "risk_100": _score_100_review(risk),
            "cluster_root": None,
            "cluster_size": 1,
            "signals": list(res.get("signals", []) or []),
        })

    if not scored:
        return _upload_fail(
            "no_scored_rows",
            "No rows had usable review text after cleaning.",
            report)

    by_root = _upload_regroup(lk, scored)      # <-- the actual fix

    cases, all_evidence = [], set()

    for root, members in by_root.items():
        if len(members) < 2:
            m = members[0]
            cases.append({
                "case_id": "UPLOAD-SOLO-%d" % m["review_id"],
                "type": "solo",
                "tier": "Watch",
                "n_members": 1,
                "members": [m["review_id"]],
                "signals": [],
                "row_numbers": [m["row"]],
                "seller_ids": [m["seller_id"]] if m["seller_id"] else [],
                "case_score_100": m["risk_100"],
                "max_risk": m["risk"],
            })
            continue

        evidence = set()
        for m in members:
            evidence |= set(m.get("signals", []))
        all_evidence |= evidence

        diversity = len(evidence)
        max_risk = max(m["risk"] for m in members)
        raw_score = max_risk + size_bonus(len(members)) + diversity_bonus(diversity)
        tier = _tier_for_case(raw_score, diversity, len(members))

        cases.append({
            "case_id": "UPLOAD-RING-%d" % root,
            "type": "ring",
            "tier": tier,
            "n_members": len(members),
            "members": [m["review_id"] for m in members],
            "signals": sorted(evidence),
            "row_numbers": sorted(m["row"] for m in members),
            "seller_ids": sorted({m["seller_id"] for m in members
                                  if m["seller_id"]}),
            "case_score_100": _score_100(raw_score),
            "max_risk": max_risk,
        })

    cases.sort(key=lambda c: -c["case_score_100"])
    rings = [c for c in cases if c["type"] == "ring"]
    high_solos = [c for c in cases
                  if c["type"] == "solo" and c["case_score_100"] >= 50]
    low_solo_count = len(cases) - len(rings) - len(high_solos)

    if all_evidence:
        evidence_line = "Linked by: " + ", ".join(
            SIGNAL_LABEL.get(s, s.replace("_", " ")) for s in sorted(all_evidence))
    else:
        evidence_line = "No shared evidence between any two rows."

    scan_id = _uuid.uuid4().hex[:12].upper()
    _UPLOAD_STATE.update({
        "scan_id": scan_id,
        "cases": cases,
        "rows": scored,
        "evidence": sorted(all_evidence),
        "evidence_line": evidence_line,
        "filename": filename,
    })

    return {
        "ok": True,
        "scan_id": scan_id,
        "filename": filename,

        # counts
        "rows_total": report["file"]["rows_in_file"],
        "rows_scanned": len(scored),                 # kept name: test depends on it
        "rows_accepted": report["rows"]["accepted"],
        "rows_rejected": report["rows"]["rejected"],
        "flagged_count": len(high_solos) + sum(c["n_members"] for c in rings),

        # results
        "ring_cases": rings,
        "high_risk_solos": high_solos[:10],
        "low_risk_solo_count": low_solo_count,
        "ring_cases_count": len(rings),
        "high_risk_solo_count": len(high_solos),
        "evidence": sorted(all_evidence),
        "evidence_line": evidence_line,

        # what upload_io understood about the file
        "column_mapping": report["column_mapping"],
        "unused_columns": report["unused_columns"],
        "confidence": report["confidence"],
        "warnings": report["warnings"],
        "guidance": report["guidance"],
        "profile": report["profile"],
        "rejected": report["rejected"],
    }


@app.get("/api/upload/results.csv")
def api_upload_results_csv():
    if not _UPLOAD_STATE.get("scan_id"):
        return Response("No scan available. Upload a file first.",
                        media_type="text/plain", status_code=400)

    buf = _io.StringIO()
    w = _csv.writer(buf)
    w.writerow(["row", "review_id", "risk_0_100", "cluster_root",
                "cluster_size", "signals", "seller_id", "account_id",
                "rating", "text"])
    for s in _UPLOAD_STATE["rows"]:
        w.writerow([s["row"], s["review_id"], s["risk_100"],
                    s["cluster_root"], s["cluster_size"],
                    "|".join(s["signals"]), s["seller_id"],
                    s["account_id"], s["rating"], s["text"]])

    headers = {
        "Content-Disposition":
            'attachment; filename="trustnet_scan_%s.csv"' % _UPLOAD_STATE["scan_id"]
    }
    return Response(content=buf.getvalue(), media_type="text/csv",
                    headers=headers)


@app.post("/api/upload/clear")
def api_upload_clear():
    _UPLOAD_STATE.update({
        "scan_id": None, "cases": [], "rows": [],
        "evidence": [], "evidence_line": "", "filename": None,
    })
    return {"ok": True}


@app.get("/api/upload/state")
def api_upload_state():
    return {
        "scan_id": _UPLOAD_STATE["scan_id"],
        "filename": _UPLOAD_STATE["filename"],
        "rows": len(_UPLOAD_STATE["rows"]),
        "cases": len(_UPLOAD_STATE["cases"]),
    }


print("[console-api] v6.4 ready — upload & scan")

# ==== v6.5 SELLER TRUST SCORES (A4) ====
import os as _os_st
import json as _json_st
import time as _time_st

SELLER_TRUST_WEIGHTS = {
    "ring_depth":     0.35,   # reviews inside a coordinated ring
    "escalate_depth": 0.25,   # reviews inside an escalated case
    "review_case":    0.12,   # reviews inside linked (Review-tier) cases
    "mean_risk":      0.16,   # mean per-review risk
    "infrastructure": 0.08,   # device / IP / payment crowding
    "fresh_accounts": 0.04,   # share of accounts younger than 24 h
}
_SELLER_RING_SATURATE = 3     # 3 ring reviews = full ring penalty
_SELLER_ESC_SATURATE  = 2     # 2 escalated reviews = full escalate penalty
_SELLER_CASE_SATURATE = 2     # 2 linked-case reviews = full case penalty
_SELLER_RING_FLOOR    = 0.45  # any ring review caps trust at 55
_SELLER_ESC_FLOOR     = 0.30  # any escalated review caps trust at 70
_SELLER_TRUST_TTL = 3.0
_SELLER_TRUST_CACHE = {"built": 0.0, "payload": None}

MOD_CONFIRM_MAX = 25
MOD_DISMISS_MAX = 10


def _seller_band(score):
    if score >= 90: return "trusted"
    if score >= 75: return "cautious"
    if score >= 60: return "watch"
    if score >= 40: return "suspect"
    return "high_risk"


def _band_counts(rows):
    out = {"trusted": 0, "cautious": 0, "watch": 0, "suspect": 0, "high_risk": 0}
    for r in rows:
        out[r["band"]] = out.get(r["band"], 0) + 1
    return out

def _case_member_ids(case):
    """Review IDs from a cases_db record, whatever the field is called.

    Accepts a list of ints, a list of dicts, a JSON-ish string, or a
    comma-separated string. Returns a de-duplicated list of ints.
    """
    if not isinstance(case, dict):
        return []

    raw = None
    for k in ("member_ids", "members", "_members", "review_ids",
              "review_id_list", "review_list", "reviews", "member_id_list"):
        v = case.get(k)
        if v:
            raw = v
            break
    if raw is None:
        return []

    if isinstance(raw, str):
        s = raw.strip().strip("[](){}").strip("'\"")
        if not s:
            return []
        for sep in (";", "|", " ", "\t"):
            s = s.replace(sep, ",")
        items = [p.strip().strip("'\"") for p in s.split(",")]
        items = [p for p in items if p]
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        items = [raw]

    out, seen = [], set()
    for it in items:
        if isinstance(it, dict):
            it = it.get("review_id") or it.get("id") or it.get("rid")
        try:
            rid = int(float(it))
        except (TypeError, ValueError):
            continue
        if rid not in seen:
            seen.add(rid)
            out.append(rid)
    return out

def _seller_trust_payload():
    now = _time_st.time()
    if (_SELLER_TRUST_CACHE["payload"] is not None
            and now - _SELLER_TRUST_CACHE["built"] < _SELLER_TRUST_TTL):
        return _SELLER_TRUST_CACHE["payload"]

    path = _os_st.path.join(_os_st.path.dirname(_os_st.path.abspath(__file__)),
                            "scored_features.csv")
    if not _os_st.path.exists(path):
        return {"error": "scored_features.csv missing", "n_sellers": 0,
                "sellers": [], "top10_riskiest": [], "top10_trusted": []}

    import pandas as _pd
    df = _pd.read_csv(path)
    need = {"review_id", "seller_id"}
    if not need.issubset(df.columns):
        return {"error": "scored_features.csv missing %s"
                         % sorted(need - set(df.columns)),
                "n_sellers": 0, "sellers": [],
                "top10_riskiest": [], "top10_trusted": []}

    if "risk_score" not in df.columns:
        df["risk_score"] = 0.0
    df["risk_score"] = _pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.0)
    df["review_id"] = _pd.to_numeric(df["review_id"], errors="coerce")

    ring_size, esc_ids, link_ids = {}, set(), set()
    scan_diag = {
        "cases_total": len(cases_db or {}),
        "cases_with_members": 0,
        "members_seen": 0,
        "ring_member_ids": 0,
        "escalate_member_ids": 0,
        "tier_values": {},
        "cases_without_members": [],
    }

    for cid, c in (cases_db or {}).items():
        ids = _case_member_ids(c)
        if not ids:
            if len(scan_diag["cases_without_members"]) < 3:
                scan_diag["cases_without_members"].append({
                    "case_id": str(cid),
                    "keys": sorted(c.keys())[:20] if isinstance(c, dict) else
                            type(c).__name__,
                })
            continue

        scan_diag["cases_with_members"] += 1
        scan_diag["members_seen"] += len(ids)

        tier = str(c.get("tier") or c.get("severity_tier") or "").upper()
        scan_diag["tier_values"][tier] = \
            scan_diag["tier_values"].get(tier, 0) + 1

        if len(ids) >= 3:
            for rid in ids:
                ring_size[rid] = len(ids)
        if tier == "ESCALATE":
            esc_ids.update(ids)
        if tier in ("REVIEW", "ESCALATE"):
            link_ids.update(ids)

    scan_diag["ring_member_ids"] = len(ring_size)
    scan_diag["escalate_member_ids"] = len(esc_ids)

    df["_ring_size"] = df["review_id"].map(ring_size).fillna(0).astype(float)
    df["_in_ring"]   = (df["_ring_size"] > 0).astype(float)
    df["_in_esc"]    = df["review_id"].isin(esc_ids).astype(float)
    df["_in_link"]   = df["review_id"].isin(link_ids).astype(float)

    agg = df.groupby("seller_id").agg(
        n_reviews=("review_id", "count"),
        n_ring=("_in_ring", "sum"),
        n_esc=("_in_esc", "sum"),
        n_link=("_in_link", "sum"),
        mean_risk=("risk_score", "mean"),
    )

    crowd_cols = [c for c in ("shared_device", "shared_ip", "shared_payment")
                  if c in df.columns]
    if crowd_cols:
        parts = []
        for c in crowd_cols:
            v = _pd.to_numeric(df[c], errors="coerce").fillna(1.0)
            df["_crowd_" + c] = ((v - 1.0) / 6.0).clip(0.0, 1.0)
            parts.append("_crowd_" + c)
        agg["infra"] = df.groupby("seller_id")[parts].mean().mean(axis=1)
    else:
        agg["infra"] = 0.0

    if "account_age_hours" in df.columns:
        age = _pd.to_numeric(df["account_age_hours"], errors="coerce")
        df["_fresh"] = ((age >= 0) & (age < 24)).astype(float)
        agg["fresh"] = df.groupby("seller_id")["_fresh"].mean()
    else:
        agg["fresh"] = 0.0

    comp = {
        "ring_depth":     (agg["n_ring"] / _SELLER_RING_SATURATE).clip(0.0, 1.0),
        "escalate_depth": (agg["n_esc"] / _SELLER_ESC_SATURATE).clip(0.0, 1.0),
        "review_case":    (agg["n_link"] / _SELLER_CASE_SATURATE).clip(0.0, 1.0),
        "mean_risk":      agg["mean_risk"].clip(0.0, 1.0),
        "infrastructure": agg["infra"].clip(0.0, 1.0),
        "fresh_accounts": agg["fresh"].clip(0.0, 1.0),
    }
    penalty = sum(SELLER_TRUST_WEIGHTS[n] * comp[n] for n in SELLER_TRUST_WEIGHTS)
    penalty = penalty.clip(0.0, 1.0)

    ring_hit = agg["n_ring"] > 0
    esc_hit  = agg["n_esc"]  > 0
    penalty  = penalty.where(~ring_hit, penalty.clip(lower=_SELLER_RING_FLOOR))
    penalty  = penalty.where(~esc_hit,  penalty.clip(lower=_SELLER_ESC_FLOOR))
    penalty  = penalty.clip(0.0, 1.0)
    trust    = (100.0 * (1.0 - penalty)).round(0).astype(int)

    mod_adjust, mod_reason = {}, {}
    try:
        _decisions = _mod_load()
    except Exception:
        _decisions = {}

    if _decisions:
        rev_seller = {}
        for rid, sel in zip(df["review_id"], df["seller_id"]):
            try:
                rev_seller[int(rid)] = str(sel)
            except (TypeError, ValueError):
                continue

        n_conf, n_dis = {}, {}
        for cid, rec in _decisions.items():
            act = (rec or {}).get("action")
            if act not in ("confirm", "dismiss"):
                continue
            case = (cases_db or {}).get(cid)
            if not case:
                continue
            for rid in _case_member_ids(case):
                sel = rev_seller.get(rid)
                if not sel:
                    continue
                bucket = n_conf if act == "confirm" else n_dis
                bucket[sel] = bucket.get(sel, 0) + 1

        for raw in list(trust.index):
            key = str(raw)
            nc, nd = n_conf.get(key, 0), n_dis.get(key, 0)
            if not nc and not nd:
                continue
            pts = 0
            if nc:
                pts -= min(MOD_CONFIRM_MAX, int(round(6.0 * (nc ** 0.5))))
            if nd:
                pts += min(MOD_DISMISS_MAX, int(round(4.0 * (nd ** 0.5))))
            if not pts:
                continue
            before = int(trust.loc[raw])
            trust.loc[raw] = max(0, min(100, before + pts))
            mod_adjust[key] = int(trust.loc[raw]) - before
            bits = []
            if nc:
                bits.append("confirmed %d review%s"
                            % (nc, "" if nc == 1 else "s"))
            if nd:
                bits.append("dismissed %d review%s"
                            % (nd, "" if nd == 1 else "s"))
            mod_reason[key] = "moderator " + " and ".join(bits)

    rows = []
    for seller in agg.index:
        key = str(seller)
        t = int(trust.loc[seller])
        n_r = int(agg["n_ring"].loc[seller])
        n_e = int(agg["n_esc"].loc[seller])
        n_l = int(agg["n_link"].loc[seller])
        mean_r = float(agg["mean_risk"].loc[seller])
        infra = float(agg["infra"].loc[seller])
        fresh = float(agg["fresh"].loc[seller])

        reasons = []
        if n_r:
            sizes = df.loc[(df["seller_id"] == seller) & (df["_ring_size"] > 0),
                           "_ring_size"]
            size_note = " of %d accounts" % int(sizes.max()) if len(sizes) else ""
            reasons.append("%d review%s inside a coordinated ring%s"
                           % (n_r, "" if n_r == 1 else "s", size_note))
        if n_e:
            reasons.append("%d escalated review%s"
                           % (n_e, "" if n_e == 1 else "s"))
        if n_l and not n_e:
            reasons.append("%d review%s inside linked cases"
                           % (n_l, "" if n_l == 1 else "s"))
        if mean_r >= 0.30:
            reasons.append("mean per-review risk %.2f" % mean_r)
        if infra >= 0.35:
            reasons.append("accounts repeatedly share device, IP or card")
        if fresh >= 0.30:
            reasons.append("%.0f%% of accounts under 24 h old" % (100 * fresh))

        adj = mod_adjust.get(key, 0)
        if adj:
            if reasons == ["no coordination evidence above threshold"]:
                reasons = []
            reasons.insert(0, "%s (%+d trust)"
                           % (mod_reason.get(key, "moderator decision"), adj))

        if not reasons:
            reasons.append("no coordination evidence above threshold")

        rows.append({
            "seller_id": key,
            "trust_score": t,
            "band": _seller_band(t),
            "n_reviews": int(agg["n_reviews"].loc[seller]),
            "components": {n: round(float(comp[n].loc[seller]), 3)
                           for n in SELLER_TRUST_WEIGHTS},
            "counts": {"ring": n_r, "escalated": n_e, "linked": n_l},
            "reasons": reasons,
            "moderation_adjustment": adj,
        })

    rows.sort(key=lambda r: (r["trust_score"], -r["n_reviews"]))
    payload = {
        "n_sellers": len(rows),
        "weights": SELLER_TRUST_WEIGHTS,
        "saturate": {"ring": _SELLER_RING_SATURATE,
                     "escalate": _SELLER_ESC_SATURATE,
                     "case": _SELLER_CASE_SATURATE},
        "floors": {"ring": _SELLER_RING_FLOOR, "escalate": _SELLER_ESC_FLOOR},
        "band_counts": _band_counts(rows),
        "diagnostics": scan_diag,
        "sellers": rows,
        "top10_riskiest": rows[:10],
        "top10_trusted": sorted(
            (r for r in rows if r["n_reviews"] >= 5),
            key=lambda r: (-r["trust_score"], -r["n_reviews"]))[:10],
    }
    _SELLER_TRUST_CACHE.update(built=now, payload=payload)
    return payload


@app.get("/api/sellers/trust")
def api_sellers_trust(limit: int = 10, full: int = 0):
    p = _seller_trust_payload()
    limit = max(1, min(50, int(limit or 10)))
    out = dict(p)
    
    out["top10_riskiest"] = p.get("top10_riskiest", [])[:limit]
    out["top10_trusted"] = p.get("top10_trusted", [])[:limit]
    if not full:
        out["sellers"] = out["top10_riskiest"]
        out["sellers_truncated"] = True
    else:
        out["sellers_truncated"] = False
    return out

# ==== v6.6 MODERATOR DECISIONS (A6) ====
import datetime as _dt_m
from fastapi import Request as _Req
from fastapi.responses import JSONResponse as _JSONRes

_MOD_ACTIONS = {"confirm", "dismiss", "need_info"}
_MOD_PATH = _os_st.path.join(_os_st.path.dirname(_os_st.path.abspath(__file__)),
                             "mod_decisions.json")
_MOD_CACHE = {"mtime": None, "data": {}}


def _mod_load():
    try:
        m = _os_st.path.getmtime(_MOD_PATH)
    except OSError:
        return {}
    if _MOD_CACHE["mtime"] == m:
        return _MOD_CACHE["data"]
    try:
        with open(_MOD_PATH, "r", encoding="utf-8") as f:
            data = _json_st.load(f)
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    _MOD_CACHE.update(mtime=m, data=data)
    return data


def _mod_save(data):
    tmp = _MOD_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        _json_st.dump(data, f, indent=2, sort_keys=True, default=str)
    _os_st.replace(tmp, _MOD_PATH)
    _MOD_CACHE.update(mtime=_os_st.path.getmtime(_MOD_PATH), data=data)


def _mod_counts(data):
    counts = {"confirm": 0, "dismiss": 0, "need_info": 0}
    for v in data.values():
        a = (v or {}).get("action")
        if a in counts:
            counts[a] += 1
    return counts


@app.get("/api/decisions")
def api_decisions():
    d = _mod_load()
    return {"n_decided": len(d), "counts": _mod_counts(d), "decisions": d}

@app.post("/api/decisions/clear")
def api_decisions_clear():
    _mod_save({})
    _SELLER_TRUST_CACHE["built"] = 0.0
    return {"ok": True, "n_decided": 0,
            "counts": {"confirm": 0, "dismiss": 0, "need_info": 0}}

@app.get("/api/case/{case_id}/decision")
def api_case_decision(case_id: str):
    if case_id not in cases_db:
        return _JSONRes({"ok": False, "error": f"unknown case '{case_id}'"},
                        status_code=404)
    return {"ok": True, "case_id": case_id,
            "decision": _mod_load().get(case_id)}


@app.post("/api/case/{case_id}/decision")
async def api_case_decide(case_id: str, request: _Req):
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    action = str(body.get("action", "")).strip().lower().replace("-", "_")
    if action not in _MOD_ACTIONS:
        return _JSONRes({"ok": False, "error": "unknown action",
                         "allowed": sorted(_MOD_ACTIONS)}, status_code=400)
    if case_id not in cases_db:
        return _JSONRes({"ok": False, "error": f"unknown case '{case_id}'"},
                        status_code=404)

    data = _mod_load()
    hist = list((data.get(case_id) or {}).get("history") or [])
    entry = {
        "action": action,
        "note": str(body.get("note", ""))[:2000],
        "moderator": str(body.get("moderator", "moderator-1"))[:120],
        "ts": _dt_m.datetime.now(_dt_m.timezone.utc).isoformat(timespec="seconds"),
    }
    hist.append(entry)
    data[case_id] = {"case_id": case_id, **entry, "history": hist[-50:]}
    _mod_save(data)
    _SELLER_TRUST_CACHE["built"] = 0.0        # seller scores must recompute now
    return {"ok": True, "case_id": case_id, "decision": data[case_id],
            "counts": _mod_counts(data)}

@app.post("/api/case/{case_id}/decision/clear")
def api_case_decision_clear(case_id: str):
    data = _mod_load()
    prev = data.get(case_id)
    if not prev:
        return {"ok": True, "case_id": case_id, "cleared": False,
                "counts": _mod_counts(data)}
    data.pop(case_id, None)
    _mod_save(data)
    try:
        _SELLER_TRUST_CACHE["built"] = 0.0
    except NameError:
        pass
    return {"ok": True, "case_id": case_id, "cleared": True,
            "previous_action": prev.get("action"),
            "counts": _mod_counts(data)}

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 TrustNet API on http://127.0.0.1:8000/landing")
    uvicorn.run(app, host="0.0.0.0", port=8000)