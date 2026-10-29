"""
TrustNet — Phase 3: Risk Scoring Model
========================================
Trains a Gradient Boosting Regressor on raw features to predict the
Phase 2 soft label, producing a generalized risk score per review.
Also runs Isolation Forest for comparison.

Key design: the model trains on RAW features only — NOT on the lf_*
heuristic columns or the soft_label itself as input. This ensures the
score generalizes past the exact rule thresholds.

Input:  labeled_features.csv  (Phase 2 output)
Output: scored_features.csv   (all previous cols + risk_score_gb + risk_score_if)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("PHASE 3: Risk Scoring Model")
print("=" * 60)

df = pd.read_csv("labeled_features.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"\n✓ Loaded {len(df)} reviews from labeled_features.csv")

# ──────────────────────────────────────────────
# 1. DEFINE FEATURE SET (raw features only)
# ──────────────────────────────────────────────
# EXCLUDE: IDs, text, timestamp, label, lf_* rules, soft_label, rules_fired_count
# The model must learn from raw signals, not from the heuristic outputs.

EXCLUDE_COLS = {
    "review_id", "seller_id", "account_id", "category", "label",
    "text_", "device_fingerprint", "ip_block", "payment_instrument",
    "timestamp",
    # Heuristic outputs (would be circular to use as features)
    "lf_near_duplicate", "lf_shared_device", "lf_shared_ip",
    "lf_shared_payment", "lf_burst", "lf_new_account_active",
    "lf_high_velocity", "lf_seller_new_5star",
    "soft_label", "rules_fired_count",
}

feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
print(f"\nModel features ({len(feature_cols)}): {feature_cols}")

X = df[feature_cols].copy()
y = df["soft_label"].values

# Handle any NaN/inf
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

# ──────────────────────────────────────────────
# 2. GRADIENT BOOSTING REGRESSOR
# ──────────────────────────────────────────────
print("\n[1/3] Training Gradient Boosting Regressor...")
print("  (Predicts soft_label from raw features — 5-fold CV to avoid overfitting)")

gbr = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=20,
    random_state=42,
)

# Use cross-validated predictions so scores aren't overfitted to training data
df["risk_score_gb"] = cross_val_predict(gbr, X, y, cv=5, n_jobs=-1)

# Clip to [0, 1]
df["risk_score_gb"] = df["risk_score_gb"].clip(0, 1)

# Also fit on full data for later use on new incoming reviews
gbr.fit(X, y)

print(f"  CV score range: [{df['risk_score_gb'].min():.4f}, {df['risk_score_gb'].max():.4f}]")
print(f"  CV score mean:  {df['risk_score_gb'].mean():.4f}")

# ──────────────────────────────────────────────
# 3. ISOLATION FOREST (unsupervised comparison)
# ──────────────────────────────────────────────
print("\n[2/3] Training Isolation Forest (unsupervised)...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso = IsolationForest(
    n_estimators=200,
    contamination=0.05,  # expect ~5% anomalous
    max_samples="auto",
    random_state=42,
    n_jobs=-1,
)
iso.fit(X_scaled)

# Isolation Forest returns -1 (anomaly) or 1 (normal)
# decision_function: lower = more anomalous
raw_if_scores = iso.decision_function(X_scaled)

# Invert and normalize to [0, 1] so higher = more risky
df["risk_score_if"] = 1.0 - (raw_if_scores - raw_if_scores.min()) / (raw_if_scores.max() - raw_if_scores.min())

print(f"  IF score range: [{df['risk_score_if'].min():.4f}, {df['risk_score_if'].max():.4f}]")
print(f"  IF score mean:  {df['risk_score_if'].mean():.4f}")

# ──────────────────────────────────────────────
# 4. CHOOSE PRIMARY SCORE
# ──────────────────────────────────────────────
# Use GB as primary (supervised on soft labels, better calibrated).
# IF is kept as a secondary signal.
df["risk_score"] = df["risk_score_gb"]

# ──────────────────────────────────────────────
# 5. SAVE
# ──────────────────────────────────────────────
OUTPUT_FILE = "scored_features.csv"
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Saved {OUTPUT_FILE} — shape: {df.shape}")

# ──────────────────────────────────────────────
# 6. PRECISION@K CHECK
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("PRECISION@K CHECK (against known ring)")
print("=" * 60)

RING_DEVICE = "DEV-A7F3E9C1-FAKE-RING"
ring_ids = set(df[df["device_fingerprint"] == RING_DEVICE]["review_id"].values)
RING_SIZE = len(ring_ids)

print(f"\nRing size: {RING_SIZE} members")

for score_col in ["risk_score_gb", "risk_score_if", "soft_label"]:
    ranked = df.sort_values(score_col, ascending=False)
    print(f"\n--- {score_col} ---")
    for k in [10, 20, 50, 100]:
        top_k_ids = set(ranked.head(k)["review_id"].values)
        hits = len(top_k_ids & ring_ids)
        precision = hits / k
        recall = hits / RING_SIZE
        print(f"  Top-{k:>3}: {hits}/{RING_SIZE} ring members found  "
              f"(precision={precision:.2%}, recall={recall:.0%})")

# ──────────────────────────────────────────────
# 7. RING vs BASELINE DISTRIBUTION
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("RING vs BASELINE SCORE DISTRIBUTION")
print("=" * 60)

ring = df[df["device_fingerprint"] == RING_DEVICE]
baseline = df[df["device_fingerprint"] != RING_DEVICE]

print(f"\n{'Score':<20} {'Ring Mean':>10} {'Ring Min':>10} {'Base Mean':>10} {'Base P99':>10}")
print("-" * 64)
for col in ["risk_score_gb", "risk_score_if", "soft_label"]:
    print(f"{col:<20} {ring[col].mean():>10.4f} {ring[col].min():>10.4f} "
          f"{baseline[col].mean():>10.4f} {baseline[col].quantile(0.99):>10.4f}")

# ──────────────────────────────────────────────
# 8. CRITICAL NOTE
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("⚠️  IMPORTANT: AT THIS STAGE...")
print("=" * 60)
print("""
Each ring member has a HIGH individual risk score, but they still
appear as SEPARATE, UNCONNECTED flags. This is EXPECTED.

Example: 7 individual "high risk" reviews scattered across 4 sellers.
A human analyst would not yet see them as a coordinated ring.

The GRAPH LINKING in Phase 4 is what connects them into ONE case.
That is the core differentiator of TrustNet.
""")

print("✅ PHASE 3 COMPLETE — scored_features.csv is ready for Phase 4")