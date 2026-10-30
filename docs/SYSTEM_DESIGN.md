# TrustNet — System Design

**Version 6.7 · Phase 9**
**Target:** SYNAPSE 1.0 Round 1 (IEEE SMC KGEC)

---

## 1. Judged acceptance bar

> When a coordinated fake-review ring is injected into the live feed, the system
> must surface it as **one linked, high-priority case — not scattered
> low/medium flags.** Ring fragmentation rate must be **0%**.

Everything below serves that line.

---

## 2. Problem framing

Per-review classifiers fail on rings because each review is individually
plausible: fluent text, 5.0 stars, a believable purchase history.

The signal is **relational**. Many accounts share one device, one payment
token, one IP block, one posting window. Amazon's own public disclosure puts
blocked suspected fake reviews above 200 million a year; the industry consensus
is that text alone does not solve it.

TrustNet represents reviews as a graph of entities, links accounts by shared
evidence, and rolls those links into a single weighted case a human can act on.

---

## 3. Design principles

| Principle | Consequence |
|---|---|
| **One ring → one case** | Union-find components guarantee it by construction, not by tuning |
| **Evidence floor, not signal count** | `MIN_EDGE_WEIGHT = 4.0` requires ≥2 independent signal types to link; this turned a 39,704-account mega-cluster into 2 clean clusters |
| **Explainable by edge, not by embedding** | Every edge carries a name — `shared_device`, `near_dup_text` — so every case can state *why* in English |
| **Human decides, machine ranks** | No automatic ban, freeze or unpublish. Ever. |
| **Retrieval, not generation** | "Cannot hallucinate" is provable only if no generative model exists |

---

## 4. High-level architecture

```
FAKE.csv (40,432 Kaggle reviews: text, rating, category, CG/OR)
  │
  ├── phase1_features.py   → synthesize entity layer, 19 raw features
  │                          enriched_reviews.csv · features.csv
  ├── phase2_labels.py     → 8 weak-supervision functions
  │                          labeled_features.csv
  ├── phase3_scoring.py    → GB regressor, cross_val_predict
  │                          scored_features.csv (42 cols)
  ├── phase4_fix.py        → weighted union-find
  │                          clusters.csv · graph_state.pkl
  ├── phase5_cases.py      → rollup + tiering
  │                          cases.csv (1,627)
  ├── train_text_model.py  → text classifier on CG/OR
  │                          text_model.joblib
  └── phase6_api.py        → FastAPI: console, landing, assistant,
                             upload, judge lab, seller trust, decisions
```

Two scorers run in parallel and are combined at case level, not averaged:

- **Text scorer** — machine-written vs human-written, AUC 0.985
- **Entity scorer** — infrastructure sharing, weak supervision + GB, ring mean
  risk 0.8087

A case's tier depends on **signal diversity**, not on either score alone.

---

## 5. Phase-by-phase rationale

### 5.1 Phase 1 — Feature engineering

Nineteen numeric features from text, entity sharing, and time.

| Feature | Why it exists |
|---|---|
| `shared_device`, `shared_ip`, `shared_payment` | The three signals a text model structurally cannot see |
| `near_dup_score` | Rings reuse templates with light rewording; char-level cosine catches what word-level misses |
| `burst_count`, `account_velocity` | Rings post in a tight window; organic reviewers spread out |
| `account_age_hours` | Compromised accounts are freshly created |
| `seller_new_5star_ratio` | Rings target one seller's recent window |

Dual vectorizer: word 1–2 gram (`max_features=400`) plus `char_wb` 3–5 gram
(`max_features=500`), composite `0.5 × word + 0.5 × char`. Measured ring versus
baseline near-duplicate: 0.916 against 0.499.

**Kaggle supplies text, rating, category and the CG/OR label. Nothing else.**
Seller, account, device, IP, payment and timestamp are synthesized and tuned so
baseline infrastructure sharing stays at 2.15 while a ring collides at 7.0.

### 5.2 Phase 2 — Weak supervision

No public review dataset carries an `is_fraud` column; platforms never release
fraud labels. Eight labeling functions encode known ring signatures and vote.

| Function | Weight | Fires when |
|---|---|---|
| `lf_shared_device` | 3.0 | ≥5 reviews share a device |
| `lf_shared_payment` | 3.0 | ≥5 share a payment instrument |
| `lf_shared_ip` | 2.5 | ≥5 share an IP block |
| `lf_new_account_active` | 2.5 | Account <24 h with >1 review |
| `lf_near_duplicate` | 2.0 | Near-dup score >0.75 |
| `lf_burst` | 2.0 | ≥1 same-seller burst |
| `lf_high_velocity` | 1.5 | >2.5 reviews/day |
| `lf_seller_new_5star` | 1.5 | >5% new-account 5-star |

Result: baseline mean 0.0834, P99 0.3889; ring mean **0.7976** — a **9.6×**
separation with zero human labels.

`lf_new_account_active` fires 0% on the ring because each ring account posts
exactly one review. That validated the two-weakness-first design instead of
leaving it as an untested assumption.

### 5.3 Phase 3 — Entity risk scoring

**Gradient Boosting, not deep learning.** 40K samples over 19 features is the
bias-variance sweet spot for trees, inference is milliseconds, and feature
importances are extractable for explanation. A deep net wants 10⁵ labelled
samples; we have soft labels.

**`cross_val_predict`** so each review is scored by a model that never saw it.

**Isolation Forest was evaluated and rejected** — it recovered 2 of 7 ring
members, because it finds point outliers rather than coordinated clusters. It
remains as the `risk_score_if` column and influences nothing.

### 5.4 Text model — the parallel scorer

`FeatureUnion(word TF-IDF 40K, char_wb 3–5 TF-IDF 40K)` +
`LogisticRegression(C=4.0, liblinear, class_weight="balanced")`, trained on the
CG/OR label over a stratified 80/20 split.

AUC 0.9848 · accuracy 0.9405 · F1 0.9402.

**What it answers:** machine-written versus human-written. **What it does not
answer:** paid versus honest. That distinction is carried in
`text_model_report.json`, returned by `/api/text-model`, and rendered on the
dashboard. Quoting 0.985 without it would be misleading.

At ingest, live risk combines the two independently:

```
risk        = max(model_prob, promo_score)
promo_score = min(0.95, 0.14 × hyperbole_count + (0.10 if count ≥ 3))
live risk  += 0.06 if account_age < 24 h
live risk  += 0.02 if rating ≥ 5
clamped to [0, 1]
```

### 5.5 Phase 4 — Linking, and the mega-cluster

**The failure.** The first pass linked any pair sharing any entity. Result:
**39,704 members in one cluster** — 98% of the dataset — chained through one
shared office IP.

**The fix.** An evidence floor of 4.0 against weighted signals:

| Signal | Weight |
|---|---|
| `exact_dup_text` (Jaccard ≥ 0.95) | 3.5 |
| `shared_device` | 3.0 |
| `shared_payment` | 3.0 |
| `shared_ip` | 2.5 |
| `near_dup_text` (Jaccard ≥ 0.85) | 2.0 |
| `temporal_burst` (±30 min, same seller) | 1.5 |
| `fresh_accounts` (both <24 h) | 1.0 |

Device + payment (6.0) links. IP + burst (4.0) links. One shared IP alone (2.5)
does not. That single number is what keeps an office or a family card from
becoming a case.

**Before versus after:**

| Metric | Single-signal | Floor 4.0 |
|---|---|---|
| Clusters | 1 (39,704 members) | **2** |
| Ring as one cluster | coincidental | **yes** — size 7, diversity 5, 0 false inclusions |

`MAX_SELLER_PEERS = 60` and `MAX_EDGES_PER_INGEST = 400` bound the work per
ingest so one seller cannot fan out unboundedly.

### 5.6 Phase 5 — Case rollup and tiering

```
raw = max(member risk)
    + min(0.5, 0.1 · log₂ n_members)
    + min(0.5, 0.1 · signal_diversity)

ESCALATE  if raw ≥ 1.0 AND signal_diversity ≥ 3
REVIEW    if raw ≥ 0.5 OR  signal_diversity ≥ 2
WATCH     otherwise

score_100 = round(raw / 2.0 × 100)
```

- `max(member risk)` sets the floor — the worst actor defines the case
- **Size bonus, log-scaled and capped** so a 500-account flood cannot outrank a
  small true ring
- **Diversity bonus, capped** — five signal types agreeing is far stronger
  evidence than five accounts sharing one IP

**Invariant: ESCALATE implies score ≥ 50.** One rule, one place, checked first
by anyone reviewing the code.

Output: 1,627 cases — **1 ESCALATE / 45 REVIEW / 1,581 WATCH**.

### 5.7 Phase 6 — Console API

FastAPI over Flask because OpenAPI at `/docs` is free and inspectable, the
ingest path is async, mutation bodies get Pydantic validation, and the same
process serves the dashboard at `/` so there is no second file server to fail.

FastAPI resolves routes **first-registered-wins**, which is why
`/api/case/{id}/timeline` is registered ahead of any catch-all suffix route.

### 5.8 Live scoring path

A review arriving at `/ingest` is scored immediately — no placeholder values.
The full review payload reaches `linker.ingest()`, which returns an
`IngestResult` carrying the union-find root **at that moment**. A later merge
can move the root, so any caller building a case must re-resolve every root
against the final union-find state before grouping. Skipping that step strands
the earliest ring member in a one-member group; it produced a 7-of-8 result
during development and is the reason the upload path carries an explicit
regrouping step.

### 5.9 Upload & Scan

Any CSV is parsed, column-mapped, validated, scored and linked.

Limits: 5 MB, 2,000 rows, 5–8,000 characters of text. Rejection codes cover
`empty_file`, `too_large`, `not_a_table` (PDF, xlsx, HTML, binary),
`too_many_rows`, `no_text_column`, `all_rows_invalid`, and more. Failures return
**HTTP 200 with `ok: false`** plus every list key present so the front-end never
crashes on a missing array.

This tab is the dataset-agnosticism proof. The text scorer touches Kaggle. The
linker does not — it runs on whatever arrives.

A spreadsheet with no review text never reaches scoring: it fails the text
column heuristic (mean length ≥ 25) and is rejected with a full column-mapping
report and operator hints.

### 5.10 Judge Test Lab

Six ring shapes executed over HTTP against the live pipeline, plus the
`check_linking.py` negative control at the linker level.

| Shape | Members | Tier | Score |
|---|---|---|---|
| Classic ring | 8/8 | Escalate | 77 |
| Two shared signals | 8/8 | Escalate | 58 |
| One shared signal | 8/8 | Escalate | 53 |
| No shared hardware | 8/8 | Review | 48 |
| Slow and spread out | 6/6 | Review | 34 |
| Cross-seller ring | 9/9 | Review | 45 |

All six fragment to zero. Three reach Review rather than Escalate, which is the
correct ceiling for infrastructure-based linking and is shown live rather than
hidden.

### 5.11 Moderator loop and seller trust

Three verdicts — confirm, dismiss, need-more-info — written to
`mod_decisions.json` with a timestamp, reviewer and append-only history. The
file survives server restart; the dashboard reads it back, so the badge and the
trust movement are both real state, not a demo illusion.

Seller trust:

```
weights:   ring_depth 0.35 · escalate_depth 0.25 · review_case 0.12
           mean_risk 0.16 · infrastructure 0.08 · fresh_accounts 0.04
saturation: ring 3 · escalate 2 · review case 2
floors:     any ring review      → trust ≤ 55
            any escalate review  → trust ≤ 70
moderator:  confirm → −min(25, round(6·√n))
            dismiss → +min(10, round(4·√n))
bands:      ≥90 trusted · ≥75 cautious · ≥60 watch · ≥40 suspect · else high_risk
```

The loop closes: a confirmation moves four sellers from 37/37/37/55 to
29/29/29/49, live, with the reason written under the number.

Dismissal is deliberately capped lower than confirmation. A false positive must
be correctable without letting a real ring launder its record.

### 5.12 Phase 8 — Judge stress protocol

Ten rounds, each injecting a ring on fresh device/IP/payment identifiers,
asserting one case per round, zero fragmentation, tier ≥ REVIEW, latency < 200 ms.

Result: **10/10**, fragmentation **0%**, avg 7.8 ms, max 11.4 ms.

### 5.13 Assistant

Retrieval only. **No LLM.** Every sentence is a `rag_knowledge.json` row or a
live value from an endpoint the dashboard also reads.

Why that trade:

- Zero install risk at judging time
- **Provable** non-hallucination — there is no generative component to hallucinate
- Sub-10 ms responses, consistent with the throughput story
- Regex refusals are deterministic; a prompted refusal is 95% reliable at best

Routing order: refusal guards → explicit seller → explicit case → hard routes →
live counters → active-case follow-up → TF-IDF retrieval.

Intents: `kb` · `case` · `case_missing` · `seller` · `seller_clean` · `status` ·
`refused` · `fallback` · `error`

---

## 6. Tech stack decisions

| Layer | Chosen | Rejected | Why |
|---|---|---|---|
| Graph | Weighted union-find | GNN, Neo4j | 7.8 ms vs seconds; named edges; no labelled entity graph to train on |
| Risk model | Gradient Boosting | XGBoost, deep net | sklearn is the base install; matches XGB at this scale |
| Text model | TF-IDF + LogisticRegression | BERT, LLM | Trains in 35 s, 6 MB, no GPU, explainable per-term |
| Server | FastAPI | Flask | Free OpenAPI at `/docs`, async ingest, Pydantic |
| Front-end | Vanilla JS, one file | React | No build step; a judge can read the source |
| Assistant | TF-IDF + regex guards | LLM API | Zero install, provable refusals, deterministic |
| State | pickle + CSV + JSON | Postgres | One-machine demo; pickle loads in 200 ms |
| Graph rendering | 2D canvas, zero deps | vis.js, 3d-force-graph | 20–25 nodes do not justify a CDN dependency or non-deterministic layout |
| Theme | CSS custom properties | UI library | Zero deps, swiss-design compliance |

---

## 7. What we explicitly did not build

Say these on stage. Each was considered and cut for a reason.

- **Transactions / refunds layer** — a different problem, doubles scope
- **LLM assistant** — non-deterministic refusals, install risk, offline risk
- **Synthetic dataset as a runtime fallback** — the Kaggle text *is* real;
  synthesizing it would undermine the one honest answer about provenance
- **3D / force-directed graph** — 20–25 nodes, CDN dependency, projector
  geometry; a 2D canvas with hover and pin is strictly more readable
- **Stylometric paraphrase detection** — the honest next layer, not built
- **Retraining on adjudicated labels** — labels are captured, retraining is not
- **Streaming feature pipeline** — sparse sampling risks missing ring members;
  batch, then serve from cached artifacts
- **`phase6_api.py` refactor** — deferred; 23 green tests are worth more than
  tidier imports the night before submission

---

## 8. Limitations

State these before they are asked. Each is reproducible from the shipped
artifacts.

1. **The entity layer is synthetic.** Kaggle supplies text, rating, category and
   the CG/OR label. Sellers, accounts, devices, IPs, payments and timestamps are
   generated. Ring detection is therefore validated against constructed rings
   with known ground truth, not against independently observed fraud.

2. **The text model answers a narrower question than it appears to.** AUC 0.985
   separates machine-written from human-written text. A human shill writing
   their own reviews scores as human.

3. **Rings with no shared infrastructure reach Review, not Escalate.** The Judge
   Lab shows this live: `no_hardware` 48, `slow_spread` 34, `cross_seller` 45.
   All fragment to zero, none auto-escalate. That is the correct behaviour for a
   system that never bans, and it is a real ceiling.

4. **A single review cannot escalate.** It can score 82/100 on text alone.
   Escalation requires three independent signal types across at least two
   accounts. This produces visible false negatives by design.

5. **Paraphrase defeats near-duplicate matching.** `exact_dup_text` and
   `near_dup_text` catch copy-paste, punctuation edits and word-order shuffles.
   They do not catch a competent rewrite of every review in a ring.

6. **Risk scores are consistent, not independently validated.** The model is
   trained on weak labels derived from signals it also evaluates. No public
   labelled ring dataset exists to validate against.

7. **The system assesses review integrity, not product quality.** A case says
   "these reviews are linked and suspicious". It does not say the product is bad.

8. **No production load test.** Measured at ring scale and under a 500-account
   flood. Union-find across 40,432 reviews fits in memory; across 10M/day it does
   not, and sharding by seller is the hardest unsolved problem here.

9. **Persistence is a pickle and a JSON file.** Production needs Postgres and a
   graph store.

10. **Every endpoint is unauthenticated.** Deliberate for judging. Production
    needs authn, authz and rate limiting.

---

## 9. Future work

- **Shard union-find by seller** above ~10M reviews/day — the real scaling problem
- **Retrain on adjudicated decisions.** `mod_decisions.json` is append-only with
  timestamps; it is exactly the labelled data a supervised ring classifier needs.
  This submission delivers the label-capture half only.
- Stylometric and cross-seller temporal co-occurrence features, to move
  `no_hardware` and `slow_spread` from Review to Escalate
- Persistence to Postgres plus a graph store, with the audit trail preserved
- Per-edge dossiers in the UI (partially present via `signal_types`)

---

## 10. File map

| Document | Contents |
|---|---|
| `FILE_REFERENCE.md` | Every script, artifact, endpoint and state variable |
| `RESULTS.md` | Every measured number, source-attributed |
| `SYSTEM_DESIGN.md` | This document — design rationale and limitations |
| `../README.md` | Quickstart, verification, the honest claim |
| `../DEMO.md` | Eight-beat script with the number that moves at each step |