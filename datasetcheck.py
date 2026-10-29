"""
TrustNet — Phase 0 v2: Synthetic Entity Layer (FIXED POOLS)
=============================================================
Fix: device/IP/payment pools are now large enough that normal sharing
is 1-2 accounts. The ring's 7-way sharing becomes a clear outlier.

Input:  FAKE.csv
Output: enriched_reviews.csv
"""

import pandas as pd
import numpy as np
import hashlib
import random
from datetime import datetime, timedelta

print("=" * 60)
print("PHASE 0 v2: Synthetic Entity Layer (FIXED)")
print("=" * 60)

df = pd.read_csv("FAKE.csv")
print(f"\n✓ Loaded {len(df)} reviews from FAKE.csv")

# --- Column detection ---
text_col = "text_" if "text_" in df.columns else "text"
label_col = "label" if "label" in df.columns else None
rating_col = "rating" if "rating" in df.columns else None
category_col = "category" if "category" in df.columns else None

print(f"  Text: '{text_col}', Label: '{label_col}', "
      f"Rating: '{rating_col}', Category: '{category_col}'")

# ──────────────────────────────────────────────
# CONFIG — POOLS ARE NOW REALISTICALLY LARGE
# ──────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

N = len(df)
NUM_SELLERS = 300
NUM_ACCOUNTS = 2000

# ★ KEY FIX: pools large enough that random sharing is rare
#   40K reviews / 35K devices ≈ 1.15 per device (most unique)
#   40K reviews / 20K IPs ≈ 2 per IP (some household sharing)
#   40K reviews / 35K payments ≈ 1.15 per payment (most unique)
NUM_DEVICES = 35000
NUM_IP_BLOCKS = 20000
NUM_PAYMENTS = 35000

END_DATE = datetime(2025, 1, 15, 23, 59)
START_DATE = END_DATE - timedelta(days=180)

RING_SIZE = 7
RING_DEVICE = "DEV-A7F3E9C1-FAKE-RING"
RING_IP = "198.51.100.0/24"
RING_PAYMENT = "PAY-RING-4829"

# ──────────────────────────────────────────────
# GENERATE ENTITY POOLS
# ──────────────────────────────────────────────
print("\nGenerating realistic entity pools...")

seller_pool = [f"SELLER-{i:04d}" for i in range(1, NUM_SELLERS + 1)]
account_pool = [f"ACCT-{i:05d}" for i in range(1, NUM_ACCOUNTS + 1)]

# Account ages: ~8% new (<48h), rest established
account_ages = {}
for acct in account_pool:
    if random.random() < 0.08:
        account_ages[acct] = round(random.uniform(1, 48), 1)
    else:
        account_ages[acct] = round(random.uniform(72, 8760), 1)

# Device fingerprints
device_pool = [
    "DEV-" + hashlib.md5(str(i).encode()).hexdigest()[:12].upper()
    for i in range(NUM_DEVICES)
]

# IP blocks
ip_pool = [
    f"{random.randint(10,223)}.{random.randint(0,255)}.{random.randint(0,255)}.0/24"
    for _ in range(NUM_IP_BLOCKS)
]

# Payment instruments
payment_pool = [
    f"PAY-{random.randint(1000,9999)}-{random.randint(1000,9999)}"
    for _ in range(NUM_PAYMENTS)
]

# ──────────────────────────────────────────────
# ASSIGN ENTITIES
# ──────────────────────────────────────────────
print("Assigning entities to reviews...")

# Sellers: cluster by category
if category_col:
    categories = df[category_col].unique()
    cat_to_sellers = {}
    sellers_per_cat = max(5, NUM_SELLERS // max(len(categories), 1))
    idx = 0
    for cat in categories:
        cat_to_sellers[cat] = seller_pool[idx : idx + sellers_per_cat]
        idx = (idx + sellers_per_cat) % NUM_SELLERS
    df["seller_id"] = df[category_col].apply(
        lambda c: random.choice(cat_to_sellers.get(c, seller_pool))
    )
else:
    df["seller_id"] = [random.choice(seller_pool) for _ in range(N)]

df["account_id"] = [random.choice(account_pool) for _ in range(N)]
df["account_age_hours"] = df["account_id"].map(account_ages)
df["device_fingerprint"] = [random.choice(device_pool) for _ in range(N)]
df["ip_block"] = [random.choice(ip_pool) for _ in range(N)]
df["payment_instrument"] = [random.choice(payment_pool) for _ in range(N)]

total_seconds = int((END_DATE - START_DATE).total_seconds())
df["timestamp"] = [
    START_DATE + timedelta(seconds=random.randint(0, total_seconds))
    for _ in range(N)
]

# ──────────────────────────────────────────────
# INJECT COORDINATED FRAUD RING
# ──────────────────────────────────────────────
print(f"\n🔴 Injecting coordinated fraud ring ({RING_SIZE} members)...")

ring_indices = random.sample(range(N), RING_SIZE)
ring_sellers = random.sample(seller_pool, 4)

base_text = str(df.loc[ring_indices[0], text_col])[:200]
ring_texts = [
    base_text,
    base_text.replace(".", "!") + " Highly recommend.",
    base_text[:150] + " Great product, fast shipping!",
    base_text.replace("the", "this") + " Will buy again.",
    base_text[:120] + " Absolutely love it, 5 stars!",
    base_text.replace("good", "excellent") + " Best purchase ever.",
    base_text[:180] + " Exceeded my expectations, thank you!",
][:RING_SIZE]

ring_base_time = START_DATE + timedelta(days=90, hours=14, minutes=30)

for i, idx in enumerate(ring_indices):
    df.loc[idx, "seller_id"] = ring_sellers[i % len(ring_sellers)]
    df.loc[idx, "account_id"] = f"ACCT-RING-{i:03d}"
    df.loc[idx, "account_age_hours"] = round(random.uniform(2, 10), 1)
    df.loc[idx, "device_fingerprint"] = RING_DEVICE
    df.loc[idx, "ip_block"] = RING_IP
    df.loc[idx, "payment_instrument"] = RING_PAYMENT
    df.loc[idx, "timestamp"] = ring_base_time + timedelta(minutes=i * 1.7)
    df.loc[idx, text_col] = ring_texts[i]
    if rating_col:
        df.loc[idx, rating_col] = 5
    if label_col:
        df.loc[idx, label_col] = "CG"

print(f"  Ring device:  {RING_DEVICE}")
print(f"  Ring IP:      {RING_IP}")
print(f"  Ring payment: {RING_PAYMENT}")
print(f"  Ring sellers: {ring_sellers}")

# ──────────────────────────────────────────────
# SAVE & VALIDATE
# ──────────────────────────────────────────────
df.to_csv("enriched_reviews.csv", index=False)
print(f"\n✓ Saved enriched_reviews.csv — shape: {df.shape}")

ring_check = df[df["device_fingerprint"] == RING_DEVICE]
assert len(ring_check) == RING_SIZE, "❌ Ring injection failed!"

# Quick pool sanity check
print(f"\n--- Pool Sharing Stats (should be ~1-2 for baseline) ---")
baseline = df[df["device_fingerprint"] != RING_DEVICE]
print(f"  Avg accounts per device (baseline): "
      f"{baseline.groupby('device_fingerprint')['account_id'].nunique().mean():.2f}")
print(f"  Avg accounts per IP (baseline): "
      f"{baseline.groupby('ip_block')['account_id'].nunique().mean():.2f}")
print(f"  Avg accounts per payment (baseline): "
      f"{baseline.groupby('payment_instrument')['account_id'].nunique().mean():.2f}")
print(f"  Ring device accounts: "
      f"{ring_check['account_id'].nunique()}  ← should be 7")

print(f"\n--- Dataset Summary ---")
print(f"Total reviews:       {len(df)}")
print(f"Unique sellers:      {df['seller_id'].nunique()}")
print(f"Unique accounts:     {df['account_id'].nunique()}")
print(f"Unique devices:      {df['device_fingerprint'].nunique()}")
print(f"Unique IP blocks:    {df['ip_block'].nunique()}")
print(f"Unique payments:     {df['payment_instrument'].nunique()}")

print("\n✅ PHASE 0 v2 COMPLETE — re-run Phase 1 now")