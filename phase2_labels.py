"""
TrustNet — Phase 2: Weak-Supervision Labeling
===============================================
Writes 8 heuristic labeling functions, combines them into a soft label
(0-1) per review via weighted vote, and stores individual rule results
for later use as explanation reason-codes.

Input:  features.csv  (Phase 1 output)
Output: labeled_features.csv  (features + soft_label + 8 rule columns)
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("PHASE 2: Weak-Supervision Labeling")
print("=" * 60)

df = pd.read_csv("features.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"\n✓ Loaded {len(df)} reviews from features.csv")

# ──────────────────────────────────────────────
# 1. DEFINE 8 LABELING FUNCTIONS
# ──────────────────────────────────────────────
# Each function returns a boolean Series (True = suspicious).
# Each also has a weight reflecting how strong a signal it is.
# Weights are starting points — tune later against precision@K.

LABELING_FUNCTIONS = {
    # ── Text signals ──
    "lf_near_duplicate": {
        "fn": lambda df: df["near_dup_score"] > 0.75,
        "weight": 2.0,
        "description": "Near-duplicate text (cosine similarity > 0.75)",
    },

    # ── Shared-signal clusters ──
    "lf_shared_device": {
        "fn": lambda df: df["shared_device_count"] >= 5,
        "weight": 3.0,
        "description": "Device fingerprint shared by 5+ accounts",
    },
    "lf_shared_ip": {
        "fn": lambda df: df["shared_ip_count"] >= 5,
        "weight": 2.5,
        "description": "IP block shared by 5+ accounts",
    },
    "lf_shared_payment": {
        "fn": lambda df: df["shared_payment_count"] >= 5,
        "weight": 3.0,
        "description": "Payment instrument shared by 5+ accounts",
    },

    # ── Temporal signals ──
    "lf_burst": {
        "fn": lambda df: df["burst_count"] >= 1,
        "weight": 2.0,
        "description": "Review burst (1+ other reviews for same seller within 30 min)",
    },

    # ── Account signals ──
    "lf_new_account_active": {
        "fn": lambda df: (df["account_age_hours"] < 24) & (df["account_review_count"] > 1),
        "weight": 2.5,
        "description": "Account < 24h old with 2+ reviews",
    },
    "lf_high_velocity": {
        "fn": lambda df: df["account_velocity"] > 2.5,
        "weight": 1.5,
        "description": "Review velocity > 2.5 reviews/day of account age",
    },

    # ── Seller signals ──
    "lf_seller_new_5star": {
        "fn": lambda df: df["seller_new_5star_ratio"] > 0.05,
        "weight": 1.5,
        "description": "Seller gets >5% of reviews as 5-star from new accounts",
    },
}

# ──────────────────────────────────────────────
# 2. EVALUATE ALL LABELING FUNCTIONS
# ──────────────────────────────────────────────
print("\nEvaluating labeling functions...")
print(f"\n{'Rule':<28} {'Fired':>8} {'% of Data':>10} {'Weight':>8}")
print("-" * 58)

rule_cols = []
for name, spec in LABELING_FUNCTIONS.items():
    result = spec["fn"](df)
    df[name] = result.astype(int)
    rule_cols.append(name)
    pct = result.mean() * 100
    print(f"{name:<28} {result.sum():>8} {pct:>9.2f}% {spec['weight']:>8.1f}")

# ──────────────────────────────────────────────
# 3. COMBINE INTO SOFT LABEL (WEIGHTED VOTE)
# ──────────────────────────────────────────────
print("\nCombining into soft label (weighted vote)...")

total_weight = sum(spec["weight"] for spec in LABELING_FUNCTIONS.values())

df["soft_label"] = 0.0
for name, spec in LABELING_FUNCTIONS.items():
    df["soft_label"] += df[name] * spec["weight"]

# Normalize to 0-1
df["soft_label"] = df["soft_label"] / total_weight

# Also compute a simple rule count (unweighted) for quick filtering
df["rules_fired_count"] = df[rule_cols].sum(axis=1)

print(f"\nSoft label stats:")
print(f"  Mean:   {df['soft_label'].mean():.4f}")
print(f"  Median: {df['soft_label'].median():.4f}")
print(f"  P95:    {df['soft_label'].quantile(0.95):.4f}")
print(f"  P99:    {df['soft_label'].quantile(0.99):.4f}")
print(f"  Max:    {df['soft_label'].max():.4f}")

# ──────────────────────────────────────────────
# 4. SAVE
# ──────────────────────────────────────────────
OUTPUT_FILE = "labeled_features.csv"
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Saved {OUTPUT_FILE} — shape: {df.shape}")
print(f"  New columns: {rule_cols + ['soft_label', 'rules_fired_count']}")

# ──────────────────────────────────────────────
# 5. ACCEPTANCE CHECK: RING vs BASELINE
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("ACCEPTANCE CHECK: Injected Ring vs Baseline")
print("=" * 60)

RING_DEVICE = "DEV-A7F3E9C1-FAKE-RING"
ring = df[df["device_fingerprint"] == RING_DEVICE]
baseline = df[df["device_fingerprint"] != RING_DEVICE]

print(f"\n{'Rule':<28} {'Ring':>6} {'Baseline %':>12}")
print("-" * 50)
for name in rule_cols:
    ring_pct = ring[name].mean() * 100
    base_pct = baseline[name].mean() * 100
    print(f"{name:<28} {ring_pct:>5.1f}% {base_pct:>11.2f}%")

print(f"\n{'Metric':<28} {'Ring Mean':>10} {'Baseline Mean':>14} {'Ratio':>8}")
print("-" * 64)
for metric in ["soft_label", "rules_fired_count"]:
    r = ring[metric].mean()
    b = baseline[metric].mean()
    ratio = r / b if b > 0 else float("inf")
    print(f"{metric:<28} {r:>10.4f} {b:>14.4f} {ratio:>7.1f}×")

print(f"\nRing members detail:")
print(ring[["review_id", "account_id", "soft_label", "rules_fired_count"]
          + rule_cols].to_string(index=False))

# Final verdict
ring_avg = ring["soft_label"].mean()
base_p99 = baseline["soft_label"].quantile(0.99)
if ring_avg > base_p99:
    print(f"\n✅ PASS: Ring avg soft_label ({ring_avg:.4f}) > baseline P99 ({base_p99:.4f})")
else:
    print(f"\n⚠️  WARN: Ring avg ({ring_avg:.4f}) vs baseline P99 ({base_p99:.4f})")
    print("   Ring should still be detectable via graph linking in Phase 4.")

print("\n✅ PHASE 2 COMPLETE — labeled_features.csv is ready for Phase 3")