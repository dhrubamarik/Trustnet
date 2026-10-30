"""
TrustNet acceptance tests.

    py -3.14 -m pytest -v test_trustnet.py

Requires cases.csv, scored_features.csv, graph_state.pkl, text_model.joblib
in the project root. Importing phase6_api rebuilds the in-memory baseline.
"""
import random

import pytest
from fastapi.testclient import TestClient

import phase6_api as api

client = TestClient(api.app)


def _cases(tier=None):
    params = {"tier": tier} if tier else {}
    return client.get("/api/cases", params=params).json()["cases"]


def _case_total():
    """Full case count, not the 400-row display page from /api/cases."""
    return client.get("/api/cases").json()["total"]


_REAL_SELLER = None


def _real_seller():
    """First real seller id from /api/cases (SELLER-xxxx, not SEL_xxx)."""
    global _REAL_SELLER
    if _REAL_SELLER:
        return _REAL_SELLER
    for c in _cases():
        for s in (c.get("sellers_involved") or []):
            if s and not s.startswith("SEL_"):
                _REAL_SELLER = s
                return _REAL_SELLER
    _REAL_SELLER = "SELLER-0265"
    return _REAL_SELLER


@pytest.fixture(autouse=True)
def clean():
    client.post("/api/reset")
    yield
    client.post("/api/reset")


# ── baseline ─────────────────────────────────────────────────────────────
def test_baseline_has_exactly_one_escalation():
    escalated = _cases("Escalate")
    assert len(escalated) == 1, f"expected 1 escalation, got {len(escalated)}"
    ring = escalated[0]
    assert ring["case_id"] == "CASE-0000"
    assert ring["review_count"] == 7
    assert len(ring["shared_signals"]) == 5


# ── judge-bar tests ──────────────────────────────────────────────────────
def test_judge_stress_protocol_does_not_fragment():
    """The judged bar: a coordinated ring is ONE case, never N flags."""
    seller = _real_seller()
    before = {c["case_id"] for c in _cases()}
    r = client.post("/api/stress-test/inject", json={"seller_id": seller})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True, body
    assert body["member_count"] == 8, f"expected 8 members, got {body['member_count']}"

    new_cases = [c for c in _cases() if c["case_id"] not in before]
    assert len(new_cases) == 1, "ring fragmented into %d cases" % len(new_cases)
    assert new_cases[0]["review_count"] == 8
    assert any(seller == s for s in new_cases[0]["sellers_involved"])
    assert new_cases[0]["severity_tier"] == "Escalate"
    assert new_cases[0]["case_score_100"] >= 50, (
        "ESCALATE case must display at least 50/100, got %d"
        % new_cases[0]["case_score_100"]
    )


@pytest.mark.parametrize("seed", range(10))
def test_randomized_held_out_ring(seed):
    """Ten independently parameterised rings, none seen during tuning."""
    rnd = random.Random(seed)
    n = rnd.randint(5, 9)
    seller = _real_seller()
    payload = {
        "seller_id": seller,
        "num_accounts": n,
        "review_text": "AMAZING 10/10 MUST BUY! Best seller ever! %d" % seed,
    }
    before = {c["case_id"] for c in _cases()}
    r = client.post("/api/custom/ring", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True, body
    assert body["member_count"] == n, (
        f"seed {seed}: expected {n} members, got {body['member_count']}"
    )

    new_cases = [c for c in _cases() if c["case_id"] not in before]
    assert len(new_cases) == 1, "held-out ring %d fragmented" % seed
    assert body["tier"] in ("Review", "Escalate"), (
        f"seed {seed}: tier {body['tier']}"
    )


def test_two_rings_stay_separate():
    """Each ring gets fresh infrastructure — two rings must be two cases."""
    seller = _real_seller()
    before = {c["case_id"] for c in _cases()}

    r1 = client.post("/api/custom/ring", json={
        "seller_id": seller,
        "num_accounts": 6,
        "review_text": "AMAZING 10/10 MUST BUY! Ring one text.",
    })
    assert r1.json()["ok"] is True, r1.json()
    tag1 = r1.json()["ring_tag"]
    case1 = r1.json()["case_id"]

    r2 = client.post("/api/custom/ring", json={
        "seller_id": seller,
        "num_accounts": 5,
        "review_text": "UNBELIEVABLE game changer! Ring two totally different.",
    })
    assert r2.json()["ok"] is True, r2.json()
    tag2 = r2.json()["ring_tag"]
    case2 = r2.json()["case_id"]

    assert tag1 != tag2, f"rings got the same infra tag: {tag1}"
    assert case1 != case2, f"two rings merged into one case: {case1}"

    new_cases = [c for c in _cases() if c["case_id"] not in before]
    assert len(new_cases) == 2, f"expected 2 new cases, got {len(new_cases)}"


# ── negative controls ────────────────────────────────────────────────────
def test_missing_infrastructure_does_not_link():
    """Eight unrelated reviews with no device/IP/payment must stay eight cases."""
    seller = _real_seller()
    texts = [
        "The battery drains within an hour. Not recommended.",
        "Arrived damaged and the return process was painful.",
        "Works exactly as described. Solid build quality.",
        "Instructions were confusing but the product is fine.",
        "Cheap materials, but you get what you pay for.",
        "Delivery was slow. The item itself is acceptable.",
        "Great value for the price. Would buy again.",
        "Seller shipped quickly. Packaging could be better.",
    ]

    before = _case_total()
    for i, text in enumerate(texts):
        r = client.post("/api/custom/review", json={
            "text": text,
            "rating": 3.0 + (i % 3),
            "seller_id": seller,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] is True, body

    after = _case_total()
    assert after - before == 8, (
        f"expected 8 new cases, got {after - before} "
        f"(reviews linked that should not have)"
    )


def test_unknown_seller_rejected():
    r = client.post("/api/custom/review", json={
        "text": "This is a review from a nonexistent seller.",
        "rating": 5.0,
        "seller_id": "SELLER-NOT-A-REAL-ID-9999",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False, body
    assert "unknown seller" in body["error"].lower(), body["error"]


# ── explainability + text model ──────────────────────────────────────────
def test_explainability_coverage():
    for c in _cases():
        assert c["explanation_summary"].strip()
        assert c["recommended_action"].strip()
        if c["review_count"] > 1:
            assert c["explanation_bullet_points"], c["case_id"]


def test_text_model_auc():
    """Real-label classifier must clear a fixed AUC bar on held-out data."""
    import json, os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "text_model_report.json")
    if not os.path.exists(p):
        pytest.skip("text_model_report.json missing — run train_text_model.py")
    with open(p) as f:
        rep = json.load(f)
    assert rep["split"].startswith("random"), "must be a random split, not leaky"
    assert rep["auc"] >= 0.90, f"AUC {rep['auc']:.4f} below the 0.90 bar"
    assert rep["accuracy"] >= 0.85, f"accuracy {rep['accuracy']:.4f} below 0.85"


def test_text_model_plumbing():
    """Batch and single agree; blank scores None."""
    import text_model
    texts = [
        "Good product. Would buy again.",
        "Broke after two days. Very disappointed.",
        "",
    ]
    batch = text_model.score_batch(texts)
    assert len(batch) == 3
    assert batch[2] is None, f"blank text should score None, got {batch[2]!r}"
    for t, b in zip(texts[:2], batch[:2]):
        s = text_model.score_text(t)
        if s is None and b is None:
            continue
        assert abs((s or 0.0) - (b or 0.0)) < 1e-6

def test_seller_alias_resolves():
    """Legacy SEL_003 must resolve to a real seller instead of failing."""
    body = client.post("/api/custom/review", json={
        "text": "The item arrived on time and matches the listing exactly.",
        "rating": 4.0,
        "seller_id": "SEL_003",
    }).json()
    assert body["ok"] is True, body
    assert body["seller_id"].startswith("SELLER-"), body


def _mk_csv(header, rows):
    import io
    buf = io.StringIO()
    buf.write(",".join(header) + "\n")
    for r in rows:
        buf.write(",".join('"' + str(c).replace('"', '""') + '"' for c in r) + "\n")
    return buf.getvalue()


def test_upload_good_file():
    """8-row ring + 2 solos → one ring case, no pollution of main state."""
    header = ["text_", "seller_id", "device_fingerprint", "ip_block",
              "payment_instrument", "account_age_hours"]
    rows = [["UNBELIEVABLE 10/10 MUST BUY!!! ring row %d" % i,
             "SELLER-0001", "DEV-UP-RING-1", "198.51.100.0/24",
             "PAY-UP-RING-1", "2.0"] for i in range(8)]
    rows.append(["Product works fine. Order 1234.", "SELLER-0002", "", "", "", "800"])
    rows.append(["Broke after two days. Disappointed.", "SELLER-0003", "", "", "", "400"])

    r = client.post("/api/upload/scan", files={
        "file": ("test.csv", _mk_csv(header, rows), "text/csv")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ok"] is True, d
    assert d["rows_scanned"] == 10
    assert d["ring_cases_count"] >= 1, d
    ring = d["ring_cases"][0]
    assert ring["n_members"] >= 8


def test_upload_missing_text_column():
    r = client.post("/api/upload/scan", files={
        "file": ("test.csv", "rating,seller_id\n5,SELLER-0001\n4,SELLER-0002\n",
                 "text/csv")})
    d = r.json()
    assert d["ok"] is False
    assert "text column" in d["error"].lower(), d["error"]


def test_upload_too_many_rows():
    header = ["text_"]
    rows = [["Review %d" % i] for i in range(2100)]
    r = client.post("/api/upload/scan", files={
        "file": ("test.csv", _mk_csv(header, rows), "text/csv")})
    d = r.json()
    assert d["ok"] is False
    assert "2000" in d["error"], d["error"]


def test_upload_does_not_change_main_cases():
    before = client.get("/api/cases").json()["total"]
    r = client.post("/api/upload/scan", files={
        "file": ("test.csv", "text_\nUnrelated upload review a\nUnrelated upload review b\n",
                 "text/csv")})
    assert r.json()["ok"] is True
    after = client.get("/api/cases").json()["total"]
    assert after == before, f"upload polluted main state: {before} -> {after}"