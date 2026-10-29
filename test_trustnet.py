"""
TrustNet acceptance tests.

    py -3.14 -m pytest -v test_trustnet.py

Requires cases.csv, labeled_features.csv, scored_features.csv, graph_state.pkl
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


@pytest.fixture(autouse=True)
def clean():
    client.post("/api/reset")
    yield
    client.post("/api/reset")


def test_baseline_has_exactly_one_escalation():
    escalated = _cases("Escalate")
    assert len(escalated) == 1
    ring = escalated[0]
    assert ring["case_id"] == "CASE-0000"
    assert ring["review_count"] == 7
    assert len(ring["shared_signals"]) == 5


def test_judge_stress_protocol_does_not_fragment():
    """The judged bar: a coordinated ring is ONE case, never N flags."""
    before = {c["case_id"] for c in _cases()}
    r = client.post("/api/stress-test/inject", json={"seller_id": "SEL_003"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["member_count"] == 8

    after = _cases()
    new_cases = [c for c in after if c["case_id"] not in before]
    assert len(new_cases) == 1, "ring fragmented into %d cases" % len(new_cases)
    assert new_cases[0]["review_count"] == 8
    assert any("SEL_003" == s for s in new_cases[0]["sellers_involved"])


@pytest.mark.parametrize("seed", range(10))
def test_randomized_held_out_ring(seed):
    """Ten independently parameterised rings, none seen during tuning."""
    rnd = random.Random(seed)
    n = rnd.randint(5, 9)
    payload = {
        "seller_id": "SEL_%03d" % rnd.randint(10, 20),
        "num_accounts": n,
        "review_text": "AMAZING 10/10 MUST BUY! Best seller ever! %d" % seed,
        "shared_device": "DEV-HELDOUT-%02d" % seed,
        "shared_payment": "PAY-HELDOUT-%02d" % seed,
    }
    before = {c["case_id"] for c in _cases()}
    r = client.post("/api/custom/ring", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["member_count"] == n

    new_cases = [c for c in _cases() if c["case_id"] not in before]
    assert len(new_cases) == 1, "held-out ring %d fragmented" % seed


def test_explainability_coverage():
    for c in _cases():
        assert c["explanation_summary"].strip()
        assert c["recommended_action"].strip()
        if c["review_count"] > 1:
            assert c["explanation_bullet_points"], c["case_id"]