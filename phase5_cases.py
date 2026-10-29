"""
TrustNet — Phase 5: Case Severity & Explainability (v2 — uses linker.py)
"""

import pandas as pd
import numpy as np
import pickle
import json
from collections import defaultdict
from linker import IncrementalLinker  # ← shared module

print("=" * 60)
print("PHASE 5: Case Severity & Explainability")
print("=" * 60)

df = pd.read_csv("clusters.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

with open("graph_state.pkl", "rb") as f:
    state = pickle.load(f)
edge_signals = state["edge_signals"]
clusters = state["clusters"]

print(f"\n✓ Loaded {len(df)} reviews, {len(clusters)} clusters")

# ── Severity formula ──
def size_bonus(n):
    if n < 2:
        return 0.0
    return min(0.5, 0.1 * np.log2(n))

def diversity_bonus(d):
    return min(0.5, 0.1 * d)

# ── Build cases ──
print("\nBuilding cases...")
cases = []

for cid, cluster_members in enumerate(clusters):
    members_df = df[df["review_id"].isin(cluster_members)]
    if len(members_df) == 0:
        continue

    max_risk = float(members_df["risk_score"].max())
    avg_risk = float(members_df["risk_score"].mean())
    n_members = len(members_df)
    n_sellers = int(members_df["seller_id"].nunique())
    n_accounts = int(members_df["account_id"].nunique())
    diversity = int(members_df["cluster_signal_diversity"].iloc[0])
    score = max_risk + size_bonus(n_members) + diversity_bonus(diversity)

    lf_cols = [c for c in members_df.columns if c.startswith("lf_")]
    lf_summary = {}
    for col in lf_cols:
        pct = float(members_df[col].mean() * 100)  # ← plain float
        if pct > 0:
            lf_summary[col] = pct

    signal_types = set()
    nodes = list(cluster_members)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            pair = (nodes[i], nodes[j])
            if pair in edge_signals:
                signal_types.update(edge_signals[pair])

    t_min = members_df["timestamp"].min()
    t_max = members_df["timestamp"].max()
    time_span = float((t_max - t_min).total_seconds() / 60)

    cases.append({
        "case_id": f"CASE-{cid:04d}",
        "type": "cluster",
        "tier": "",
        "case_score": round(score, 4),
        "max_risk": round(max_risk, 4),
        "avg_risk": round(avg_risk, 4),
        "n_members": n_members,
        "n_sellers": n_sellers,
        "n_accounts": n_accounts,
        "signal_diversity": diversity,
        "signal_types": ", ".join(sorted(signal_types)),
        "time_span_minutes": round(time_span, 1),
        "lf_summary": json.dumps(lf_summary),           # ← JSON, not str()
        "member_ids": json.dumps([int(x) for x in cluster_members]),  # ← plain ints
    })

# Isolated high-risk
isolated = df[df["cluster_id"] < 0]
high_risk = isolated[isolated["risk_score"] > 0.3]
for _, row in high_risk.iterrows():
    lf_cols = [c for c in df.columns if c.startswith("lf_")]
    lf_summary = {c: 100.0 for c in lf_cols if row[c] == 1}

    cases.append({
        "case_id": f"CASE-ISO-{int(row['review_id']):05d}",
        "type": "isolated",
        "tier": "",
        "case_score": round(float(row["risk_score"]), 4),
        "max_risk": round(float(row["risk_score"]), 4),
        "avg_risk": round(float(row["risk_score"]), 4),
        "n_members": 1,
        "n_sellers": 1,
        "n_accounts": 1,
        "signal_diversity": 0,
        "signal_types": "none",
        "time_span_minutes": 0,
        "lf_summary": json.dumps(lf_summary),
        "member_ids": json.dumps([int(row["review_id"])]),
    })

cases_df = pd.DataFrame(cases)
print(f"  Total cases: {len(cases_df)} "
      f"({len(cases_df[cases_df['type']=='cluster'])} clusters, "
      f"{len(cases_df[cases_df['type']=='isolated'])} isolated)")

# ── Tiers ──
ESCALATE_THRESHOLD = 1.0
REVIEW_THRESHOLD = 0.5

def assign_tier(row):
    if row["case_score"] >= ESCALATE_THRESHOLD and row["signal_diversity"] >= 3:
        return "ESCALATE"
    elif row["case_score"] >= REVIEW_THRESHOLD or row["signal_diversity"] >= 2:
        return "REVIEW"
    else:
        return "WATCH"

cases_df["tier"] = cases_df.apply(assign_tier, axis=1)
cases_df = cases_df.sort_values("case_score", ascending=False).reset_index(drop=True)

print(f"\n  Tier distribution:")
for tier in ["ESCALATE", "REVIEW", "WATCH"]:
    print(f"    {tier}: {(cases_df['tier'] == tier).sum()}")

# ── Explanations ──
print("\nGenerating explanation strings...")

LF_DESC = {
    "lf_near_duplicate": "near-duplicate review text",
    "lf_shared_device": "shared device fingerprint",
    "lf_shared_ip": "shared IP block",
    "lf_shared_payment": "shared payment instrument",
    "lf_burst": "review burst within 30 minutes",
    "lf_new_account_active": "newly created accounts (<24h) with multiple reviews",
    "lf_high_velocity": "abnormally high review velocity",
    "lf_seller_new_5star": "unusual 5-star reviews from new accounts",
}

SIG_DESC = {
    "shared_device": "device fingerprint",
    "shared_ip": "IP block",
    "shared_payment": "payment instrument",
    "near_dup_text": "near-duplicate text",
    "temporal_burst": "temporal proximity",
}

def generate_reason(row):
    parts = []
    if row["type"] == "cluster":
        parts.append(
            f"Coordinated activity detected: {row['n_members']} reviews "
            f"across {row['n_sellers']} sellers from {row['n_accounts']} accounts."
        )
    else:
        parts.append(
            f"Single high-risk review flagged (risk score {row['max_risk']:.2f})."
        )

    if row["signal_diversity"] >= 2:
        signals = [SIG_DESC.get(s, s) for s in row["signal_types"].split(", ")]
        parts.append(
            f"Linked via {row['signal_diversity']} signal types: "
            f"{', '.join(signals)}."
        )

    if 0 < row["time_span_minutes"] < 60:
        parts.append(f"All activity within {row['time_span_minutes']:.0f} minutes.")
    elif row["time_span_minutes"] >= 60:
        parts.append(f"Activity spanned {row['time_span_minutes']/60:.1f} hours.")

    lf = json.loads(row["lf_summary"])
    fired = [LF_DESC.get(k, k) for k, v in lf.items() if v >= 50]
    if fired:
        parts.append(f"Triggered rules: {', '.join(fired)}.")

    parts.append(
        f"Max individual risk: {row['max_risk']:.2f}. "
        f"Case score: {row['case_score']:.2f}."
    )
    return " ".join(parts)

cases_df["explanation"] = cases_df.apply(generate_reason, axis=1)

# ── Save ──
cases_df.to_csv("cases.csv", index=False)
print(f"\n✓ Saved cases.csv — shape: {cases_df.shape}")

# ── Acceptance ──
print("\n" + "=" * 60)
print("ACCEPTANCE CHECK: Ring Case")
print("=" * 60)

RING_DEVICE = "DEV-A7F3E9C1-FAKE-RING"
ring_reviews = set(int(x) for x in df[df["device_fingerprint"] == RING_DEVICE]["review_id"])

cases_df["_member_set"] = cases_df["member_ids"].apply(
    lambda s: set(json.loads(s))
)
ring_cases = cases_df[cases_df["_member_set"].apply(lambda x: bool(x & ring_reviews))]

print(f"\nCases containing ring members: {len(ring_cases)}")

if len(ring_cases) == 1:
    rc = ring_cases.iloc[0]
    print(f"\n✅ PASS: Ring = ONE case")
    print(f"  Case ID:    {rc['case_id']}")
    print(f"  Tier:       {rc['tier']}")
    print(f"  Score:      {rc['case_score']}")
    print(f"  Members:    {rc['n_members']}")
    print(f"  Signals:    {rc['signal_types']}")
    print(f"  Diversity:  {rc['signal_diversity']}")
    print(f"\n  Explanation:\n  {rc['explanation']}")
    missing = ring_reviews - rc["_member_set"]
    if missing:
        print(f"\n  ⚠️  Missing: {missing}")
    else:
        print(f"\n  ✅ All 7 ring members present")
else:
    print(f"\n❌ FAIL: Ring split across {len(ring_cases)} cases")

cases_df.drop(columns=["_member_set"], inplace=True)

# ── Top 10 ──
print("\n" + "=" * 60)
print("TOP 10 CASES")
print("=" * 60)
for _, c in cases_df.head(10).iterrows():
    print(f"\n  {c['case_id']} | {c['tier']} | score={c['case_score']:.3f} | "
          f"{c['n_members']} members | div={c['signal_diversity']}")
    print(f"  {c['explanation'][:150]}...")

print("\n✅ PHASE 5 COMPLETE")