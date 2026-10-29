# TrustNet — File Reference

---

## 1. Pipeline scripts

### `datasetcheck.py`
Sanity check on the raw Kaggle dataset. Prints row counts, label balance
(CG 20,216 / OR 20,216), and missing-value rates. No outputs.

### `phase1_features.py`
Feeds `FAKE.csv` into the enrichment pipeline.
- **Input:** `FAKE.csv`
- **Output:** `enriched_reviews.csv` (40,439 × 12), `features.csv` (40,432 × 29)
- **Runtime:** ~4 minutes

### `phase2_labels.py`
Applies the 8 weighted labeling functions to produce a weak label.
- **Input:** `features.csv`
- **Output:** `labeled_features.csv` (40,432 × 39)
- **Runtime:** ~1 minute

### `phase3_scoring.py`
Trains a Gradient Boosting regressor via `cross_val_predict`.
- **Input:** `features.csv` + `labeled_features.csv`
- **Output:** `scored_features.csv` (40,432 × 40)
- **Runtime:** ~3 minutes

### `phase4_fix.py`
Union-find graph linking with `MIN_EDGE_WEIGHT = 4.0`. Supersedes
`phase4_graph.py`.
- **Input:** `scored_features.csv`
- **Output:** `clusters.csv`, `graph_state.pkl`
- **Runtime:** ~30 seconds

### `phase5_cases.py`
Rolls clusters into cases and assigns tiers.
- **Input:** `scored_features.csv` + `clusters.csv` + `graph_state.pkl`
- **Output:** `cases.csv` (1,627 × 15)
- **Runtime:** ~20 seconds

### `linker.py`
Shared class used by both offline graph linking and live incremental
linking at API runtime. Contains `IncrementalLinker` with `add_review()`
and `_link_new()` methods. Also defines `MIN_EDGE_WEIGHT` and the
signal-weight table.

### `phase8_stress_test.py`
Judge stress protocol. Runs 10 injection rounds and asserts one-case
delivery, 0% fragmentation, latency bound.
- **Input:** live server on `127.0.0.1:8000`
- **Output:** console report only
- **Runtime:** ~10 seconds

### `phase4_graph.py`
**DEPRECATED.** Superseded by `phase4_fix.py`. Kept for git history only.

---

## 2. Server and UI

### `phase6_api.py`
FastAPI server. Combines:
- Data loaders (`scored_features.csv`, `cases.csv`, `REVIEW_INDEX`)
- Console endpoints (see § 5)
- Dashboard served at `/`
- Assistant integration (`/api/assistant`)

### `dashboard.html`
5-tab console UI: Cases, Test, Feed, Architecture, Assistant.
Vanilla JS, no build step. Served from FastAPI at `/`.

### `landing.html`
Standalone marketing page. Open directly or serve at `/landing`.

---

## 3. Assistant

### `rag_assistant.py`
Retrieval-only assistant. 15-entry knowledge base + live case store.
Zero generative components.

### `rag_knowledge.json`
15 curated Q&A entries + hard routes. Every answer cites its `id` as source.

---

## 4. Tests and benchmarks

### `test_trustnet.py`
13 tests:
- `test_baseline_has_exactly_one_escalation`
- `test_judge_stress_protocol_does_not_fragment`
- `test_randomized_held_out_ring` (10 parametrized seeds)
- `test_explainability_coverage`
- **Runtime:** ~35 seconds

### `bench_throughput.py`
Sequential HTTP benchmark — 200 calls, reports mean/median/p95.

### `bench_components.py`
Engine vs transport decomposition — sequential 200 + batched 500, reports
per-call cost split.

---

## 5. HTTP endpoints

All endpoints `GET` unless noted. Base URL: `http://127.0.0.1:8000`.

### `GET /`
Serves `dashboard.html`. Main UI entry point.

### `GET /api/status`
One-line service status JSON. (Kept from early dev; not used by the dashboard.)

### `GET /api/overview`
Summary metrics for the top bar.

```json
{
  "tier_counts": {"Escalate": 1, "Review": 45, "Watch": 1581},
  "last_latency_ms": 7.4,
  "total_cases": 1627
}
```

### `GET /api/cases`
Full ranked case list. Backs the Cases tab.

```json
{
  "cases": [
    {
      "case_id": "CASE-0000",
      "severity_tier": "Escalate",
      "case_score": 1.6009,
      "case_score_100": 81,
      "review_count": 7,
      "account_count": 7,
      "sellers_involved": ["SELLER-0265", "SELLER-0039", "SELLER-0182", "SELLER-0221"],
      "shared_signals": ["shared_device", "shared_ip", "shared_payment", "near_dup_text", "temporal_burst"],
      "explanation_summary": "...",
      "explanation_bullet_points": ["..."],
      "recommended_action": "...",
      "review_items": ["..."],
      "graph_data": {"nodes": ["..."], "links": ["..."]}
    }
  ],
  "total": 1627
}
```

### `GET /api/reviews?limit=N&flagged_only=BOOL`
Feed of scored reviews.

```json
{
  "reviews": [
    {
      "review_id": 6293,
      "risk_score": 41,
      "seller_id": "SELLER-0221",
      "account_id": "ACCT-RING-006",
      "rating": 5.0,
      "review_text": "...",
      "triggered_rules": ["copy-paste text", "shared device", "..."]
    }
  ]
}
```

### `GET /api/case/{case_id}`
Single case detail.

### `GET /api/debug/case/{case_id}`
Raw ground-truth view (used during development).

### `POST /api/inject-member`
Adds one review using the known ring parameters.

- Body: empty
- Returns: `{"case_id": "CASE-0000", "member_count": 8, "latency_ms": 7.4}`

### `POST /api/stress-test/inject`
Injects 8 reviews using fresh device/IP/payment identifiers.

- Body: `{"seller_id": "SEL_003", "ring_name": "SYNDICATE-APEX"}`
- Returns: `{"case_id": "CASE-LIVE-40512", "member_count": 8, "latency_ms": 12.1}`

### `POST /api/reset`
Restores baseline state from CSVs. Clears live ring growth.

### `POST /api/custom/review`
Scores a single review through the live pipeline.

Body:
```json
{
  "seller_id": "SEL_003",
  "account_id": "ACC-TEST-XYZ",
  "rating": 5,
  "review_text": "...",
  "device_fingerprint": "DEV-MANUAL-TEST"
}
```
Returns: `{"scored_review": {...}, "assigned_case_id": "CASE-LIVE-..."}`

### `POST /api/custom/ring`
Injects a synthetic ring with custom parameters.

- Body: `{"seller_id": "...", "ring_name": "...", "num_accounts": 6, "review_text": "...", "shared_device": "...", "shared_payment": "..."}`

### `GET /api/evaluation`
Returns precision/recall/ring-linking metrics. Full payload in
RESULTS.md.

### `POST /api/assistant`
Retrieval-only chat.

- Body: `{"query": "...", "case_id": "CASE-0000"}` (`case_id` optional)
- Returns: `{"answer": "...", "sources": ["..."], "intent": "..."}`

Intents: `kb` · `case` · `case_missing` · `seller` · `seller_clean` ·
`status` · `refused` · `fallback` · `error`

---

## 6. CSV artifacts

### `enriched_reviews.csv` (40,439 × 12)
`review_id, review_text, rating, category, label (CG/OR), seller_id,
account_id, account_age_hours, device_fingerprint, ip_block,
payment_instrument, timestamp`

### `features.csv` (40,432 × 29)
The 19 numeric features used by the risk model plus identifying columns.
Key features: `shared_device`, `shared_ip`, `shared_payment`,
`near_dup_score`, `burst_count`, `velocity`, `account_age_hours`,
`seller_new_5star_ratio`.

### `labeled_features.csv` (40,432 × 39) — DELETED at demo time
Weak supervision output. The API tolerates its absence by
reconstructing rules directly from `scored_features.csv` feature columns.

### `scored_features.csv` (40,432 × 40)
Adds `risk_score`, `iforest_score`, and 8 `lf_*` columns.

### `clusters.csv`
`cluster_id, size, signal_diversity, member_ids`.

### `cases.csv` (1,627 × 15)
`case_id, case_score, max_risk, avg_risk, n_members, n_sellers,
signal_diversity, signal_types, time_span_minutes, tier,
member_ids, seller_ids, explanation_summary, recommended_action`.

### `graph_state.pkl`
Pickled `IncrementalLinker` instance. Loaded at API startup for live
incremental linking.

---

## 7. Session state (in-memory, not persisted)

| Name | Purpose |
|---|---|
| `REVIEW_INDEX` | review_id → seller, text, rating, risk, rules |
| `REVIEW_TEXT_IDX` | flexible key → review record (assistant only) |
| `CASES_DB` | case_id → mutable case dict (survives resets) |
| `FEED_CACHE` | sorted by risk descending |
| `_RAG_CACHE` | assembled assistant context, 2-second TTL |
