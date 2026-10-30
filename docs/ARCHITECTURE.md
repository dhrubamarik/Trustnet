# Architecture

Nine stages, five artifacts, one API.

```
FAKE.csv (40,432)
  └─ phase1_features.py   → enriched_reviews.csv
  └─ phase2_labels.py     → labeled_features.csv
  └─ phase3_scoring.py    → scored_features.csv (40 × 40,432)
  └─ phase4_fix.py        → clusters.csv + graph_state.pkl
  └─ phase5_cases.py      → cases.csv (1,627 × 15)
  └─ train_text_model.py  → text_model.joblib + text_model_report.json
  └─ phase6_api.py        → serves everything
```

`labeled_features.csv` is an intermediate and may be absent; the API tolerates
its absence and prints a single benign warning at boot.

---

## 1. Feature extraction

Nineteen raw features per review: text-derived (`near_dup_score`, `burst_count`,
`sentiment_rating_mismatch`), seller-derived (`seller_5star_ratio`,
`seller_new_5star_ratio`), and account-derived (`account_age_hours`,
`account_velocity`).

Kaggle supplies **text, rating, category, CG/OR label** only. Sellers, accounts,
devices, IPs, payments and timestamps are synthesized and tuned so that baseline
infrastructure sharing stays low while a ring's collides:

| Metric | Baseline | Ring |
|---|---|---|
| Mean shared devices | 2.15 | 7.0 |
| Mean shared IPs | 3.04 | 7.0 |
| Mean shared payment instruments | 2.16 | 7.0 |
| Burst count | 0.03 | 0.86 |
| Near-duplicate score | 0.499 | 0.916 |

That 8× separation is the reason the linker has something to find.

## 2. Weak supervision

Eight labeling functions produce a soft label. Ring mean **0.7976** against a
baseline mean of **0.0834** — a **9.6× separation** with no human labels.

| Function | Weight | Fires when |
|---|---|---|
| `lf_shared_device` | 3.0 | ≥5 reviews share a device |
| `lf_shared_payment` | 3.0 | ≥5 reviews share a payment instrument |
| `lf_shared_ip` | 2.5 | ≥5 reviews share an IP block |
| `lf_new_account_active` | 2.5 | Account <24h old with >1 review |
| `lf_near_duplicate` | 2.0 | Near-dup score >0.75 |
| `lf_burst` | 2.0 | ≥1 same-seller burst |
| `lf_high_velocity` | 1.5 | >2.5 reviews/day |
| `lf_seller_new_5star` | 1.5 | >5% of seller's reviews are new-account 5-star |

## 3. Risk scoring

`GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
subsample=0.8, min_samples_leaf=20)`, 5-fold `cross_val_predict` over the 19 raw
features. The ring scores **mean 0.8087, min 0.7911**.

Isolation Forest was evaluated and recovered only 2 of the 7 ring members. It is
retained as an `iforest_score` column and otherwise unused.

## 4. Linking

`linker.py` is standalone — it has no dependency on the Kaggle pipeline and runs
on anything posted to `/ingest`.

| Signal | Weight | Condition |
|---|---|---|
| `shared_device` | 3.0 | Same device fingerprint |
| `shared_payment` | 3.0 | Same payment instrument |
| `shared_ip` | 2.5 | Same IP block |
| `exact_dup_text` | 3.5 | Jaccard ≥ 0.95 |
| `near_dup_text` | 2.0 | Jaccard ≥ 0.85 |
| `temporal_burst` | 1.5 | Same seller, within ±30 min |
| `fresh_accounts` | 1.0 | Both accounts <24h old |

`MIN_EDGE_WEIGHT = 4.0`, `MAX_SELLER_PEERS = 60`, `MAX_EDGES_PER_INGEST = 400`.

The floor is the single most important line in the system. Without it, the
linker produced **one 39,704-member cluster holding 98% of the corpus**. With
it, the corpus breaks into **2 clusters** — the ring at exactly 7 members,
diversity 5, and **zero false inclusions**.

A single shared device scores 3.0 and does not clear the floor on its own. That
is deliberate: an office or a family shares an IP and must not become a case.
The `one_signal` preset in the Judge Lab still links because freshness and timing
accumulate on top of the shared signal across all eight accounts.

The near-duplicate matcher is a composite: `0.50 × word TF-IDF cosine + 0.50 ×
char-3..5 TF-IDF cosine`, which survives punctuation and word-order edits that
defeat a plain word-vector comparison.

## 5. Case construction

```
raw = max_member_risk
    + min(0.5, 0.1 · log₂(n_members))
    + min(0.5, 0.1 · n_signal_types)

ESCALATE  if raw ≥ 1.0 AND n_signal_types ≥ 3
REVIEW    if raw ≥ 0.5 OR  n_signal_types ≥ 2
WATCH     otherwise

score_100 = round(raw / 2.0 × 100)
```

One rule, one place. Invariant: **ESCALATE implies score ≥ 50.** Reviewers and
chairs both check this first, and it holds.

Output: 1,627 cases — 1 ESCALATE, 45 REVIEW, 1,581 WATCH.

## 6. Text model

`FeatureUnion(word TF-IDF 40K, char_wb 3–5 TF-IDF 40K)` +
`LogisticRegression(C=4.0, liblinear, class_weight=balanced)`.

| Metric | Value |
|---|---|
| AUC | 0.9848 |
| Accuracy | 0.9405 |
| Precision | 0.9448 |
| Recall | 0.9357 |
| F1 | 0.9402 |
| Train / test | 32,345 / 8,087, stratified 80/20, seed 42 |

Caveat carried in `text_model_report.json` and surfaced at `/api/text-model`:
CG reviews are machine-written. Paid reviews written by humans will be harder to
separate than this number suggests.

Live risk combines independently:

```
risk = max(model_prob, promo_score)
promo_score = min(0.95, 0.14 × hyperbole_count + 0.10 if count ≥ 3)
live risk += 0.06 if account_age < 24h
live risk += 0.02 if rating ≥ 5
clamped to [0, 1]
```

## 7. API

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/landing` | Landing page |
| GET | `/api/status` | Boot state |
| GET | `/api/overview` | Tier counts |
| GET | `/api/cases` | All cases |
| GET | `/api/case/{id}` | One case, full detail |
| GET | `/api/case/{id}/timeline` | Member timestamps + gaps |
| GET | `/api/reviews` | Review feed |
| GET | `/api/evaluation` | Precision@K, linking rate, fragmentation |
| GET | `/api/text-model` | Trained metrics + caveat |
| GET | `/api/sellers/trust` | Ledger, `full=0\|1` |
| GET | `/api/decisions` | All moderator decisions |
| POST | `/ingest` | Live single review |
| POST | `/api/custom/review` | Demo: one review |
| POST | `/api/custom/ring` | Demo: six-bot ring |
| POST | `/api/ingest/batch` | Batched ingest |
| POST | `/api/judge-lab/run` | Six-shape ring test |
| POST | `/api/upload/scan` | CSV upload and scan |
| POST | `/api/case/{id}/decision` | Confirm / dismiss / need-info |
| POST | `/api/decisions/clear` | Clear all decisions |
| POST | `/api/reset` | Reset demo state |

FastAPI resolves routes **first-registered-wins**, so `/api/case/{id}/timeline`
must be registered before any `/api/case/{id}/{suffix}` catch-all.

OpenAPI at `/docs` is enabled on purpose. Every endpoint is inspectable.

## 8. Seller trust

```
weights: ring_depth 0.35 · escalate_depth 0.25 · review_case 0.12
         mean_risk 0.16 · infrastructure 0.08 · fresh_accounts 0.04

saturation: ring 3 · escalate 2 · review case 2
floors:     any ring review      → trust ≤ 55
            any escalate review  → trust ≤ 70

moderator:  confirm → −min(25, round(6·√n))
            dismiss → +min(10, round(4·√n))

bands: ≥90 trusted · ≥75 cautious · ≥60 watch · ≥40 suspect · else high_risk
```

The first version used ring reviews as a **fraction** of a seller's ~163
reviews. Two ring reviews contributed 0.012 and every seller scored 96–97 —
technically correct, operationally useless. Absolute counts with saturation
produce the visible spread the Sellers tab now shows:
`293 trusted · 3 cautious · 0 watch · 1 suspect · 3 high_risk`.

Dismissal is capped lower than confirmation (±10 vs −25) on purpose: a false
positive must be correctable without letting a real ring launder its record.

## 9. Assistant

Retrieval only. **No LLM.** Every sentence in its output is either a row from
`rag_knowledge.json` or a live value fetched from the same endpoints the
dashboard reads.

That is a deliberate trade. An LLM would produce better prose and occasionally
invent a case ID, a score, or a seller on stage. Refusal guards handle unknown
cases, unknown sellers, and off-topic input. When it does not know, it says so.