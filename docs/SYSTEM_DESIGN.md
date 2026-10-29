# TrustNet — System Design

**Version:** 5.7 · Phase 9
**Target:** SYNAPSE 1.0 Round 1 (IEEE SMC KGEC)

---

## 1. Judged acceptance bar

> When a coordinated fake-review ring is injected into the live feed, the
> system must surface it as **ONE linked, high-priority case — not
> scattered low/medium flags**. Ring fragmentation rate must be **0%**.

Everything in this document serves that single line.

---

## 2. Problem framing

Per-review classifiers fail on rings because each review is genuine-looking:

- Text is fluent, stars are 5.0, purchase history is plausible
- The signal is **relational**: many accounts sharing one device, one
  payment token, one IP subnet, one posting burst
- Amazon's own public disclosure confirms > 200 million suspected fake
  reviews blocked per year — the industry consensus is that text models
  alone do not solve it

TrustNet's answer: **represent the reviews as a graph of entities**,
link accounts by shared infrastructure, and roll those links into a single
weighted case.

---

## 3. Design principles

| Principle | Why |
|---|---|
| **One ring → one case** | The judged bar. Union-find components guarantee this by construction. |
| **Signal diversity, not signal count** | The edge-weight floor of 4.0 requires ≥2 independent signal types to link. This turned a 39,704-account mega-cluster (98% of the dataset) into 2 clean clusters. |
| **Explainable by edge, not by score** | Every edge carries a named reason (`shared_device`, `shared_payment`). A neural embedding gives a number; an edge gives a sentence. |
| **Human in the loop** | The console ranks and explains; a moderator decides. |
| **Retrieval, not generation, for the assistant** | "Cannot hallucinate" is a provable property only if there is no generative model. |

---

## 4. High-level architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ INGEST                                                           │
│ Kaggle FAKE.csv (40,432 reviews)                                 │
│ → synthesized seller / account / device / IP / payment / ts      │
│ → enriched_reviews.csv (40,439 × 12)                             │
└──────────────────────────────────────────────────────────────────┘
↓
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 1 — Feature engineering (19 raw features)                  │
│ • Dual-space TF-IDF (word 1–2 gram + char_wb 3–5 gram)           │
│ • Entity-sharing counts (device, IP, payment)                    │
│ • Temporal velocity, burst detection                             │
│ • Near-duplicate similarity (cosine on char TF-IDF)              │
│ → features.csv (40,432 × 29)                                     │
└──────────────────────────────────────────────────────────────────┘
↓
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 2 — Weak supervision (8 labeling functions)                │
│ lf_shared_device (w 3.0)          lf_shared_payment (3.0)        │
│ lf_shared_ip (2.5)                lf_new_account_active (2.5)    │
│ lf_near_duplicate (2.0)           lf_burst (2.0)                 │
│ lf_high_velocity (1.5)            lf_seller_new_5star (1.5)      │
│ → labeled_features.csv (weak soft_label)                         │
└──────────────────────────────────────────────────────────────────┘
↓
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 3 — Gradient Boosting regressor                            │
│ 200 trees, depth 4, lr 0.05, subsample 0.8                       │
│ 5-fold cross_val_predict on 19 raw features                      │
│ → scored_features.csv (risk_score ∈ [0,1])                       │
└──────────────────────────────────────────────────────────────────┘
↓
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 4 — Union-find graph linking                               │
│ Entity graph: accounts ↔ device / IP / payment / text cluster    │
│ MIN_EDGE_WEIGHT = 4.0 (≥2 independent signal types required)     │
│ → clusters.csv + graph_state.pkl                                 │
└──────────────────────────────────────────────────────────────────┘
↓
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 5 — Case rollup + tiering                                  │
│ case_score = max(member risk)                                    │
│   + min(0.5, 0.1·log₂ n_members)                                 │
│   + min(0.5, 0.1·signal_diversity)                               │
│ ESCALATE ≥1.0 AND diversity ≥3 · REVIEW ≥0.5 OR diversity ≥2     │
│ → cases.csv (1,627 cases: 1 / 45 / 1,581)                        │
└──────────────────────────────────────────────────────────────────┘
↓
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 6 — FastAPI console                                        │
│ /api/overview · /api/cases · /api/reviews                        │
│ /api/inject-member · /api/stress-test/inject · /api/reset        │
│ /api/custom/review · /api/custom/ring · /api/evaluation          │
│ /api/assistant · / (dashboard.html)                              │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Phase-by-phase design rationale

### Phase 1 — Feature engineering

**What:** 19 numeric features from review text, entity-sharing, and time.

**Why these features:**

| Feature | Rationale |
|---|---|
| `shared_device`, `shared_ip`, `shared_payment` | The three infrastructure signals that a text model cannot see |
| `near_dup_score` | Copy-paste rings reuse templates with light rewording — cosine on char TF-IDF catches this where word-TF-IDF misses |
| `burst_count`, `velocity` | Rings post within a tight window; organic reviewers spread out |
| `account_age_hours` | Compromised accounts are typically freshly created |
| `seller_new_5star_ratio` | Rings target one seller's recent window |

**Dual vectorizer choice:** word 1–2 gram TF-IDF (`max_features=400`) plus
char_wb 3–5 gram TF-IDF (`max_features=500`), composite = 0.5·word + 0.5·char.
Word captures phrase reuse; char captures pattern reuse across light rewording.
Measured ring vs baseline: near-dup 0.916 vs 0.499.

### Phase 2 — Weak supervision

**Why weak supervision:** no public review dataset carries a `is_fraud`
column, because platforms never release fraud labels. Eight labeling
functions encode the ring signatures we already know, then vote.

Result:
- Baseline (ordinary reviewers) soft_label mean = **0.0834**, P99 = **0.3889**
- Injected ring mean = **0.7976** — a **9.6× separation**

`lf_new_account_active` fires 0% on the ring (each ring account has
exactly one review), so the two-weakness-first design was validated as
correct rather than over-engineered.

### Phase 3 — Risk scoring

**Why Gradient Boosting, not deep learning:**
- 40K samples, 19 features → GB's bias-variance sweet spot
- Fast inference (17.7 ms engine cost on this machine)
- Feature importances extractable for explainability
- Deep nets need 10⁵+ labelled samples; we have soft labels, not ground truth

**Why `cross_val_predict`:** prevents leakage. Each sample's score comes
from a model that did not see it during training.

**Why Isolation Forest was not chosen:** recovered only **2 of 7** ring
members — it detects point outliers, not coordinated clusters. Retained as
a single input column (`iforest_score`) alongside the GB score.

### Phase 4 — Graph linking (the critical phase)

**The mega-cluster problem.** The first pass used a single-signal edge rule
(any shared entity links the two accounts). Result: **39,704 members in
one cluster** — 98% of the dataset chained together through one shared
office Wi-Fi.

**The fix.** `MIN_EDGE_WEIGHT = 4.0`, with weights:

| Signal | Weight |
|---|---|
| `shared_device` | 3.0 |
| `shared_payment` | 3.0 |
| `shared_ip` | 2.5 |
| `near_duplicate_text` | 2.0 |
| `temporal_burst` | 1.5 |

4.0 means **at least two independent signal types must agree** before
accounts link. Device + payment (6.0) ✓. IP + burst (4.0) ✓. Single
IP alone (2.5) ✗.

**Before vs after:**

| Metric | Single-signal | Weighted (≥4.0) |
|---|---|---|
| Clusters on the dataset | 1 (39,704-member mega-cluster) | **2** |
| Ring detected as one cluster | coincidental | **yes** — size 7, diversity 5, 0 false inclusions |

### Phase 5 — Case rollup and tiering

**Why the score formula:**

```
case_score = max(member risk)
           + min(0.5, 0.1 · log₂ n_members)
           + min(0.5, 0.1 · signal_diversity)
```

- `max(member risk)` — the worst actor's score is the case's floor
- Size bonus (log-scaled, capped at 0.5) — more members, higher priority,
  but with diminishing returns so a 500-account flood does not outrank a
  small true ring
- Diversity bonus (capped at 0.5) — five signal types agreeing is much
  stronger evidence than five accounts sharing one IP

**Result on baseline:** 1,627 cases → **1 ESCALATE / 45 REVIEW / 1,581 WATCH**.

### Phase 6 — Console API

Chose FastAPI over Flask because:
- Automatic OpenAPI docs at `/docs` (useful for the demo)
- Async ingest path
- Pydantic validation on mutation endpoints
- Same server can host the dashboard at `/` — no separate file server

### Phase 8 — Judge stress protocol

**What:** 10 back-to-back injection rounds; each round injects a ring on
fresh device/IP/payment identifiers and checks after each round that:
- the ring is one case
- no fragmentation into sub-cases
- the tier is ≥ REVIEW
- latency < 200 ms

**Result:** 10/10 passed. Avg latency 7.8 ms, max 11.4 ms. Fragmentation 0%.

### Phase 9 — Retrieval assistant

**Why retrieval-only, no LLM:**
- Zero install risk at judging time
- **Provable** non-hallucination — every sentence traces to a JSON row
  or a value read from the live case store
- Sub-10 ms response, consistent with the throughput story
- Deterministic refusals — a prompted LLM refusal is 95% reliable, a regex
  is 100%

Extraction strategies the assistant uses, in order:
1. Explicit seller ID → seller report
2. Explicit case ID → case dossier
3. Hard-routed knowledge questions → curated answer
4. Live counters → status readout
5. Active case follow-up → case dossier of the selected case
6. TF-IDF retrieval → top-scoring knowledge entry

---

## 6. Tech stack decisions

| Layer | Chosen | Rejected | Why |
|---|---|---|---|
| Graph | **NetworkX + union-find** | GNN, Neo4j | 7.8 ms vs seconds; explainable edges; no cold-start labelled graph |
| Risk model | **Gradient Boosting** | XGBoost, deep net | sklearn is the base install; GB matches XGB at this scale |
| Website | **FastAPI + vanilla JS** | Flask, React | Auto-docs, no build step, single-file dashboard, judge sees the source |
| Assistant | **TF-IDF + regex guards** | LLM API | Zero install, proofs of non-hallucination, deterministic |
| Entity graph storage | **pickle + CSV** | Postgres | One-machine demo; pickle loads in 200 ms |
| Theme | **Light/dark CSS variables** | Third-party UI lib | Zero deps, swiss-design compliance |

---

## 7. What we explicitly did not build

Each of these was considered and rejected — say so on stage:

- **Transactions / refunds layer** — distinct problem, doubles scope
- **LLM-based assistant** — non-deterministic refusals, install risk
- **Synthetic dataset as a runtime fallback** — the Kaggle text IS real;
  synthesizing text would undermine the "real data" answer
- **Streaming feature pipeline** — sparse sampling of positions risks
  missing ring members; batch then serve from cached CSVs

---

## 8. Limitations (state these on stage — they build trust)

1. Detects coordinated rings, **not** lone bad reviews. A single fake
   review with no shared infrastructure will not escalate — by design.
2. Risk model trained on weak labels derived from the same signals it
   evaluates. Scores are consistent, not independently validated against
   real platform fraud labels (which do not exist publicly).
3. Device, IP, payment, timestamps are synthesized. Text, rating, category,
   CG/OR labels are real.
4. Assesses **review integrity**, not product quality. No access to specs,
   return rates, or shipping data.

---

## 9. Future work

- Real-time streaming ingest with sliding-window linking
- Human feedback loop (Confirm / Dismiss) to convert weak labels into
  ground truth — designed, not implemented
- Explainability per-edge for deeper dossiers
- Deployed scale test (10M reviews/day) — measured only at ring scale and
  under a 500-account flood; not claimed beyond that

---

## 10. File map

See [`FILE_REFERENCE.md`](FILE_REFERENCE.md) for every script, CSV, and
HTTP endpoint.
