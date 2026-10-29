"""
TrustNet — Phase 1: Feature Engineering
=========================================
Reads enriched_reviews.csv (Phase 0 output) and computes review-level,
seller-level, and account-level features into a single flat table.

Input:  enriched_reviews.csv
Output: features.csv  (one row per review, one column per feature)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. LOAD
# ──────────────────────────────────────────────
print("=" * 60)
print("PHASE 1: Feature Engineering")
print("=" * 60)

df = pd.read_csv("enriched_reviews.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["review_id"] = range(len(df))  # stable key for all later phases
print(f"\n✓ Loaded {len(df)} reviews, {df['seller_id'].nunique()} sellers, "
      f"{df['account_id'].nunique()} accounts")

# ──────────────────────────────────────────────
# 2. REVIEW-LEVEL FEATURES
# ──────────────────────────────────────────────
print("\n[1/6] Computing near-duplicate scores (TF-IDF + NearestNeighbors)...")

# TF-IDF with capped vocabulary for speed on 40K rows
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
tfidf_matrix = tfidf.fit_transform(df["text_"].fillna(""))

# Find each review's most similar neighbor (n_neighbors=2: self + 1 nearest)
nn = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute")
nn.fit(tfidf_matrix)
distances, indices = nn.kneighbors(tfidf_matrix)

# Cosine similarity = 1 - cosine distance; take the nearest OTHER review
df["near_dup_score"] = 1.0 - distances[:, 1]  # 0 = unrelated, 1 = identical
df["near_dup_flag"] = (df["near_dup_score"] > 0.75).astype(int)
print(f"  Near-duplicate flags raised: {df['near_dup_flag'].sum()}")

# ── Burstiness ──
print("[2/6] Computing burstiness (reviews per seller in 1-hour window)...")

df = df.sort_values(["seller_id", "timestamp"]).reset_index(drop=True)

def compute_burst(group):
    """Count how many other reviews for this seller fall within ±30 min."""
    times = group["timestamp"].values
    bursts = []
    for t in times:
        window = np.abs((times - t) / np.timedelta64(1, "m")) <= 30
        bursts.append(int(window.sum()) - 1)  # exclude self
    group["burst_count"] = bursts
    return group

df = df.groupby("seller_id", group_keys=False).apply(compute_burst)
print(f"  Max burst count: {df['burst_count'].max()}")

# ── Sentiment-Rating Mismatch ──
print("[3/6] Computing sentiment-rating mismatch (keyword-based)...")

POS_WORDS = {"good", "great", "excellent", "love", "amazing", "best",
             "perfect", "wonderful", "fantastic", "recommend", "awesome",
             "superb", "outstanding", "brilliant", "impressed", "happy"}
NEG_WORDS = {"bad", "terrible", "worst", "hate", "awful", "poor",
             "horrible", "disappointing", "broken", "waste", "ugly",
             "defective", "useless", "cheap", "rubbish", "angry"}

def keyword_sentiment(text):
    words = set(str(text).lower().split())
    pos = len(words & POS_WORDS)
    neg = len(words & NEG_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total

df["text_sentiment"] = df["text_"].apply(keyword_sentiment)

# Mismatch: high rating (4-5) with negative text, or low rating (1-2) with positive text
def mismatch(row):
    r = row["rating"]
    s = row["text_sentiment"]
    if r >= 4 and s < -0.2:
        return 1.0
    elif r <= 2 and s > 0.2:
        return 1.0
    elif r >= 4 and s < 0:
        return 0.5
    elif r <= 2 and s > 0:
        return 0.5
    return 0.0

df["sentiment_rating_mismatch"] = df.apply(mismatch, axis=1)

# ── Account Age vs Volume ──
print("[4/6] Computing account age vs review volume ratio...")

account_review_counts = df["account_id"].value_counts().to_dict()
df["account_review_count"] = df["account_id"].map(account_review_counts)
# High ratio = many reviews from a very new account → suspicious
df["age_volume_ratio"] = df["account_review_count"] / (df["account_age_hours"] + 1)

# ── Rating Deviation from Category Baseline ──
print("[5/6] Computing rating deviation from category baseline...")

cat_mean = df.groupby("category")["rating"].transform("mean")
cat_std = df.groupby("category")["rating"].transform("std").replace(0, 1)
df["rating_deviation"] = ((df["rating"] - cat_mean) / cat_std).abs()

# ──────────────────────────────────────────────
# 3. SELLER-LEVEL FEATURES
# ──────────────────────────────────────────────
print("[6/6] Computing seller-level and account-level aggregates...")

seller_agg = df.groupby("seller_id").agg(
    seller_total_reviews=("review_id", "count"),
    seller_avg_rating=("rating", "mean"),
    seller_rating_std=("rating", "std"),
    seller_5star_ratio=("rating", lambda x: (x == 5).mean()),
).reset_index()
seller_agg["seller_rating_std"] = seller_agg["seller_rating_std"].fillna(0)

# New-account 5-star ratio per seller
new_account_mask = df["account_age_hours"] < 48
new_5star = df[new_account_mask & (df["rating"] == 5)]
seller_new_5star = new_5star.groupby("seller_id").size().reset_index(name="seller_new_5star_count")
seller_agg = seller_agg.merge(seller_new_5star, on="seller_id", how="left")
seller_agg["seller_new_5star_count"] = seller_agg["seller_new_5star_count"].fillna(0)
seller_agg["seller_new_5star_ratio"] = (
    seller_agg["seller_new_5star_count"] / seller_agg["seller_total_reviews"]
)

df = df.merge(seller_agg, on="seller_id", how="left")

# ──────────────────────────────────────────────
# 4. ACCOUNT / SHARED-SIGNAL FEATURES
# ──────────────────────────────────────────────

# Shared device count: how many distinct accounts use this device?
device_account_counts = df.groupby("device_fingerprint")["account_id"].nunique().to_dict()
df["shared_device_count"] = df["device_fingerprint"].map(device_account_counts)

# Shared IP count: how many distinct accounts share this IP block?
ip_account_counts = df.groupby("ip_block")["account_id"].nunique().to_dict()
df["shared_ip_count"] = df["ip_block"].map(ip_account_counts)

# Shared payment count
payment_account_counts = df.groupby("payment_instrument")["account_id"].nunique().to_dict()
df["shared_payment_count"] = df["payment_instrument"].map(payment_account_counts)

# Account review velocity: reviews per day of account age
df["account_velocity"] = df["account_review_count"] / (df["account_age_hours"] / 24 + 0.1)

# ──────────────────────────────────────────────
# 5. SELECT FINAL FEATURE COLUMNS & SAVE
# ──────────────────────────────────────────────
FEATURE_COLS = [
    "review_id", "seller_id", "account_id", "category", "rating", "label",
    "text_", "device_fingerprint", "ip_block", "payment_instrument",
    "timestamp", "account_age_hours",
    # Review-level
    "near_dup_score", "near_dup_flag", "burst_count",
    "text_sentiment", "sentiment_rating_mismatch",
    "age_volume_ratio", "rating_deviation",
    # Seller-level
    "seller_total_reviews", "seller_avg_rating", "seller_rating_std",
    "seller_5star_ratio", "seller_new_5star_ratio",
    # Account/signal-level
    "shared_device_count", "shared_ip_count", "shared_payment_count",
    "account_review_count", "account_velocity",
]

out = df[FEATURE_COLS].copy()
out.to_csv("features.csv", index=False)
print(f"\n✓ Saved features.csv — shape: {out.shape}")

# ──────────────────────────────────────────────
# 6. ACCEPTANCE CHECK: RING vs BASELINE
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("ACCEPTANCE CHECK: Injected Ring vs Baseline")
print("=" * 60)

RING_DEVICE = "DEV-A7F3E9C1-FAKE-RING"
ring = out[out["device_fingerprint"] == RING_DEVICE]
baseline = out[out["device_fingerprint"] != RING_DEVICE]

check_features = [
    "near_dup_score", "burst_count", "age_volume_ratio",
    "shared_device_count", "shared_ip_count", "shared_payment_count",
    "account_velocity", "seller_new_5star_ratio",
]

print(f"\n{'Feature':<28} {'Ring Mean':>10} {'Baseline Mean':>14} {'Ring Stands Out?':>18}")
print("-" * 74)
for f in check_features:
    r_mean = ring[f].mean()
    b_mean = baseline[f].mean()
    # "Stands out" if ring mean > 2× baseline mean (or baseline is near zero)
    standout = "✅ YES" if (b_mean == 0 and r_mean > 0) or (r_mean > 2 * b_mean) else "⚠️  mild"
    print(f"{f:<28} {r_mean:>10.3f} {b_mean:>14.3f} {standout:>18}")

print(f"\nRing members ({len(ring)} rows):")
print(ring[["review_id", "account_id", "seller_id", "near_dup_score",
            "burst_count", "shared_device_count", "account_age_hours"]].to_string(index=False))

print("\n✅ PHASE 1 COMPLETE — features.csv is ready for Phase 2")