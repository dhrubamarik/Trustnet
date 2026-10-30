# TrustNet — File Reference

Every script, artifact, endpoint and state variable in the running system.

---

## 1. Pipeline scripts

| Script | Input | Output | Runtime |
|---|---|---|---|
| `datasetcheck.py` | `FAKE.csv` | console only | <1 s |
| `phase1_features.py` | `FAKE.csv` | `enriched_reviews.csv`, `features.csv` | ~4 min |
| `phase2_labels.py` | `features.csv` | `labeled_features.csv` | ~1 min |
| `phase3_scoring.py` | `features.csv` + `labeled_features.csv` | `scored_features.csv` | ~3 min |
| `phase4_fix.py` | `scored_features.csv` | `clusters.csv`, `graph_state.pkl` | ~30 s |
| `phase5_cases.py` | above three | `cases.csv` | ~20 s |
| `train_text_model.py` | `FAKE.csv` | `text_model.joblib`, `text_model_report.json` | ~35 s |

Run in that order. `phase6_api.py` last.

### `datasetcheck.py`
Validates `FAKE.csv`: 40,432 rows, `label` balance CG 20,216 / OR 20,216, missing-value rates. Reads only.

### `phase1_features.py`
Synthesizes the entity layer (seller, account, device, IP block, payment
instrument, timestamp) onto the Kaggle text, then engineers 19 numeric
features.

- `enriched_reviews.csv` — 40,439 × 12
- `features.csv` — 40,432 × 29

**Both are regenerable intermediates and are absent from the current demo
workspace.** Nothing at demo time requires them; run this script only to
rebuild from `FAKE.csv`.

### `phase2_labels.py`
Applies the eight weighted labeling functions.

- `labeled_features.csv` — 40,432 × 39

**Deleted at demo time by design.** `phase6_api.py` tolerates its absence and
prints one benign warning (`labeled_features.csv missing — rules stay empty`),
then reconstructs rule names from `scored_features.csv`.

### `phase3_scoring.py`
Trains `GradientBoostingRegressor` via 5-fold `cross_val_predict`.

- `scored_features.csv` — 40,432 rows × 42 columns

Adds `risk_score`, `risk_score_gb`, `risk_score_if`, `soft_label`,
`rules_fired_count` and all eight `lf_*` columns.

### `phase4_fix.py`
Union-find graph linking with `MIN_EDGE_WEIGHT = 4.0`.

- `clusters.csv` — 28.8 MB
- `graph_state.pkl` — pickled linker, loads in ~200 ms

**Supersedes `phase4_graph.py`, which has been deleted.** Running the old
script produces the 39,704-member mega-cluster.

### `phase5_cases.py`
Rolls clusters into cases and assigns tiers.

- `cases.csv` — 1,627 × 15

### `train_text_model.py`
Trains the text classifier on the CG/OR label.

- `text_model.joblib` — ~3–6 MB
- `text_model_report.json` — metrics plus the mandatory caveat string

Fit 30.6 s, total 34.9 s.

### `text_model.py`
Loads the joblib and exposes `_load()` and per-text scoring. Warmed at boot by
`phase6_api.py`, which is why the first Judge Lab row is not 1 second slower
than the rest.

### `linker.py`
Standalone. **No dependency on the Kaggle pipeline** — it runs on whatever
arrives at `/ingest`, which is why Upload & Scan works on a corpus the model
never saw.

Public surface: `ingest()`, `find()`, `parent`, `MIN_EDGE_WEIGHT`,
`MAX_SELLER_PEERS`, `MAX_EDGES_PER_INGEST`.

| Signal | Weight | Condition |
|---|---|---|
| `shared_device` | 3.0 | Same device fingerprint |
| `shared_payment` | 3.0 | Same payment instrument |
| `shared_ip` | 2.5 | Same IP block |
| `exact_dup_text` | 3.5 | Jaccard ≥ 0.95 |
| `near_dup_text` | 2.0 | Jaccard ≥ 0.85 |
| `temporal_burst` | 1.5 | Same seller, within ±30 min |
| `fresh_accounts` | 1.0 | Both accounts <24 h old |

`MIN_EDGE_WEIGHT = 4.0` · `MAX_SELLER_PEERS = 60` · `MAX_EDGES_PER_INGEST = 400`

`ingest()` returns an `IngestResult` object, not a dict, and returns the
union-find root **at ingest time**. A later merge can move the root, so callers
that need the final grouping must re-resolve with `find()` or a `parent` walk.

### `upload_io.py`
CSV parsing, validation, column mapping. `prepare(raw, filename)` returns
`{"ok": bool, "df": …, "report": …}`.

| Limit | Value |
|---|---|
| Max bytes | 5 MB |
| Max rows | 2,000 |
| Max text length | 8,000 chars |
| Min text length | 5 chars |
| Rows quoted in the report | 25 |

Error codes: `empty_file` `too_large` `not_a_table` `parse_failed`
`empty_table` `too_many_rows` `no_text_column` `all_rows_invalid`
`read_failed` `ingest_failed` `no_scored_rows`.

Failures return **HTTP 200 with `ok: false`** and every list key present
(`ring_cases: []`, `high_risk_solos: []`, …) so the front-end never calls
`.map()` on `undefined`. This is deliberate and covered by tests.

### `check_linking.py`
Injects six ring shapes and one negative control directly into `linker.py`,
bypassing HTTP. Prints `OVERALL: PASS`.

### `phase8_stress_test.py`
Ten back-to-back injection rounds on fresh device/IP/payment identifiers.
Asserts one case per round, zero fragmentation, latency bound. Console only.

### `make_sample_upload.py`
Generates `sample_upload.csv` from the synthesized sellers — 300 rows
containing one 8-member ring at rows 24, 39, 83, 87, 268, 271, 275, 297.

### `verify_demo.py`
Optional pre-flight. `--reset` calls both `/api/reset` and
`/api/decisions/clear`, then asserts the eight demo invariants. Present only if
generated; not required to run the system.

---

## 2. Server and UI

### `phase6_api.py`
FastAPI server. Data loaders, console endpoints, dashboard at `/`, landing at
`/landing`, assistant at `/api/assistant`, moderator decisions, seller trust,
Upload & Scan, Judge Test Lab, forensic timeline.

**Import gotcha:** this file imports `os` as `_os`. New code must use `_os.path`,
not `os.path`, or it raises `NameError` at request time.

FastAPI resolves routes **first-registered-wins**. `/api/case/{id}/timeline`
must be registered before any `/api/case/{id}/{suffix}` catch-all.

### `dashboard.html`
Single-page console, vanilla JS, no build step, served from FastAPI at `/`.

Tab order: Cases · Feed · Test · Judge Lab · Upload · Architecture · Sellers ·
Assistant. Sellers carries `idx 07`.

**Never open this file from the filesystem.** Every `fetch` fails on `file://`.

### `landing.html`
Standalone marketing page, served at `/landing`.

---

## 3. Assistant

### `rag_assistant.py`
Retrieval-only. 15-entry knowledge base plus the live case store. Zero
generative components — every sentence traces to a `rag_knowledge.json` row or
a value read from an endpoint the dashboard also reads.

### `rag_knowledge.json`
15 curated Q&A entries plus hard routes.

---

## 4. Tests and benchmarks

### `test_trustnet.py`
**23 tests, ~23 s.** Covers baseline escalation count, the stress protocol, ten
randomized held-out rings, explainability coverage, upload acceptance and
rejection paths, text-model plumbing, seller trust payload shape, and decision
persistence.

```
py -3.14 -m pytest -v test_trustnet.py
23 passed, 15 warnings in 22.62s
```

The warnings are cosmetic: an httpx deprecation inside Starlette's test client,
and a pytest plugin notice.

### `bench_throughput.py`
Sequential HTTP benchmark, 200 calls, mean / median / p95 / p99.

### `bench_components.py`
Engine-versus-transport decomposition: sequential 200 + batched 500.

---

## 5. HTTP endpoints

Base URL `http://127.0.0.1:8000`. **Use `127.0.0.1`, never `localhost`** — the
IPv6 fallback costs ~2 s per call on Windows.

OpenAPI at `/docs` is enabled on purpose.

### Read

| Route | Purpose |
|---|---|
| `GET /` | Dashboard |
| `GET /landing` | Landing page |
| `GET /api/status` | Boot state |
| `GET /api/overview` | Tier counts + last latency |
| `GET /api/cases` | Full ranked case list |
| `GET /api/case/{id}` | One case, full detail |
| `GET /api/case/{id}/timeline` | Member timestamps + gaps |
| `GET /api/reviews?limit=N&flagged_only=B` | Scored review feed |
| `GET /api/evaluation` | Precision@K, linking rate, fragmentation, text model |
| `GET /api/text-model` | Trained metrics + caveat |
| `GET /api/sellers?limit=N` | Seller list |
| `GET /api/sellers/trust?limit=N&full=0\|1` | Trust ledger + band counts |
| `GET /api/decisions` | All moderator decisions |
| `GET /api/case/{id}/decision` | One case's decision |
| `GET /api/judge-lab/presets` | Six ring shapes |
| `GET /api/upload/state` | Last upload result |
| `GET /api/upload/results.csv` | Scored rows as a download |
| `GET /api/debug/case/{id}` | Raw ground truth |
| `GET /api/debug/ingest-route` | Route resolution check |
| `GET /api/debug/sellers` | Seller alias resolution |

`/api/sellers/trust` defaults to `full=0`, returning `top10_riskiest` plus
`sellers_truncated: true`. Pass `full=1` for all 300.

### Write

| Route | Purpose |
|---|---|
| `POST /ingest` | Live single review, full payload |
| `POST /api/custom/review` | Demo: one review |
| `POST /api/custom/ring` | Demo: custom ring |
| `POST /api/ingest/batch` | Batched ingest |
| `POST /api/inject-member` | Add one known-ring review |
| `POST /api/stress-test/inject` | Inject 8 fresh-identity reviews |
| `POST /api/judge-lab/run` | Run one of the six ring shapes |
| `POST /api/upload/scan` | Multipart CSV upload and scan |
| `POST /api/upload/clear` | Clear upload state |
| `POST /api/assistant` | Retrieval chat |
| `POST /api/case/{id}/decision` | Confirm / dismiss / need_info |
| `POST /api/case/{id}/decision/clear` | Undo one decision |
| `POST /api/decisions/clear` | Clear all decisions |
| `POST /api/reset` | Reset live state |

### `/api/reset` does not touch `mod_decisions.json`

Reset restores the graph and case store. Moderator decisions are separate and
survive it. Clear them with `POST /api/decisions/clear` before any demo run that
depends on the seller ledger moving.

### `/api/overview`

```json
{
  "tier_counts": {"Escalate": 1, "Review": 45, "Watch": 1581},
  "last_latency_ms": 7.4,
  "total_cases": 1627
}
```

### `/api/cases`

Field names below are as returned by the route. `cases.csv` column names differ
— see § 6.

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
      "recommended_action": "...",
      "graph_data": {"nodes": ["..."], "links": ["..."]}
    }
  ],
  "total": 1627
}
```

### `/api/reviews?limit=N&flagged_only=B`

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
      "triggered_rules": ["copy-paste text", "shared device"]
    }
  ]
}
```

### `/api/case/{id}/timeline`

```json
{
  "ok": true,
  "case_id": "CASE-0000",
  "n_timed": 7,
  "n_members": 7,
  "span_minutes": 10.2,
  "from": "2024-08-14T14:29:00",
  "to": "2024-08-14T14:39:12",
  "points": [
    {"review_id": 32386, "ts": "2024-08-14T14:29:00", "offset_min": 0.0}
  ]
}
```

Returns `n_timed: 0` plus a `note` when no timestamps resolve. Backed by a
one-time index over `scored_features.csv` (`review_id` + `timestamp`).

### `POST /api/case/{id}/decision`

```json
{"action": "confirm", "reviewer": "moderator", "note": "optional"}
```

Writes to `mod_decisions.json`, invalidates the seller trust cache.

### `POST /api/assistant`

```json
{"query": "Why was CASE-0000 flagged?", "case_id": "CASE-0000"}
```

Returns `{"answer", "sources", "intent"}`.
Intents: `kb` · `case` · `case_missing` · `seller` · `seller_clean` · `status` ·
`refused` · `fallback` · `error`

---

## 6. CSV artifacts

### Present in the demo workspace

| File | Size | Shape |
|---|---|---|
| `FAKE.csv` | 14.6 MB | 40,432 × 5 |
| `scored_features.csv` | 28.8 MB | 40,432 × 42 |
| `clusters.csv` | 28.8 MB | cluster_id, size, signal_diversity, member_ids |
| `cases.csv` | 0.6 MB | 1,627 × 15 |
| `sample_upload.csv` | 40 KB | 300 rows, one 8-member ring |

### `scored_features.csv` — 42 columns

Identifiers and raw fields: `review_id`, `seller_id`, `account_id`, `category`,
`rating`, `label` (CG/OR), `text_`, `device_fingerprint`, `ip_block`,
`payment_instrument`, `timestamp`, `account_age_hours`.

Engineered: `near_dup_score`, `near_dup_flag`, `burst_count`, `text_sentiment`,
`sentiment_rating_mismatch`, `age_volume_ratio`, `rating_deviation`,
`seller_total_reviews`, `seller_avg_rating`, `seller_rating_std`,
`seller_5star_ratio`, `seller_new_5star_ratio`, `shared_device_count`,
`shared_ip_count`, `shared_payment_count`, `account_review_count`,
`account_velocity`.

Labels and scores: all eight `lf_*` columns, `soft_label`,
`rules_fired_count`, `risk_score_gb`, `risk_score_if`, `risk_score`.

`timestamp` is the only timestamp source in the workspace and is what backs
`/api/case/{id}/timeline`. Range: 2024-07-20 → 2025-01-15, zero nulls.

### `cases.csv` — 15 columns, exactly these

```
case_id, type, tier, case_score, max_risk, avg_risk, n_members, n_sellers,
n_accounts, signal_diversity, signal_types, time_span_minutes, lf_summary,
member_ids, explanation
```

`signal_types` is a comma-separated string. `member_ids` is a JSON-ish list.
`type` distinguishes baseline clusters from live-injected cases.

### Regenerable, absent at demo time

| File | Produced by | Needed? |
|---|---|---|
| `enriched_reviews.csv` (40,439 × 12) | `phase1_features.py` | No |
| `features.csv` (40,432 × 29) | `phase1_features.py` | No |
| `labeled_features.csv` (40,432 × 39) | `phase2_labels.py` | No |

### `graph_state.pkl`
Pickled linker instance. Loads at startup: `reviews 40432, components 40425,
largest 7, top5 [7, 2, 1, 1, 1]`.

### `mod_decisions.json`
Append-only moderator record: case_id → action, timestamp, reviewer, note.
**Survives server restart.** This is what makes the moderation loop a claim
rather than a demo trick.

---

## 7. In-memory state

Not persisted; rebuilt at boot.

| Name | Purpose |
|---|---|
| `cases_db` | case_id → mutable case dict |
| `REVIEW_INDEX` | review_id → seller, text, rating, risk, rules |
| `REVIEW_TEXT_IDX` | flexible key → review record (assistant only) |
| `FEED_CACHE` | reviews sorted by risk descending |
| `_RAG_CACHE` | assembled assistant context, 2-second TTL |
| `_CASE_TS` | review_id → `Timestamp`, built once from `scored_features.csv` |
| `_CASE_TS_BUILT` | index-built guard; set even on failure so a bad read cannot retry per request |
| `_SELLER_TRUST_CACHE` | `{"built": 0.0}`; reset to invalidate on any decision change |
| `_UPLOAD_STATE` | last upload report and scored frame |

### Cache invalidation rule

Any write to `mod_decisions.json` **must** be followed by
`_SELLER_TRUST_CACHE["built"] = 0.0`. Three routes do this: decision, decision
clear, decisions clear. A fourth route that writes decisions without this line
will show stale trust on the Sellers tab and it will look like a bug in the
formula.

---

## 8. Expected boot log

```
Loading TrustNet data...
  Loaded 1627 cases into memory
[console-api] IngestResult shim installed
[console-api] cases.csv indexed: 1627
[console-api] per-review scores loaded: 40432
[console-api] labeled_features.csv missing — rules stay empty
[console-api] rules reconstructed from scored_features.csv: 23071 reviews
[console-api] dual TF-IDF fitted on 6000 reviews
[assistant] knowledge base: 15 entries
[console-api] v6.1 ready — full-payload ingest, real scoring, isolated rings
[text-model] loaded text_model.joblib
[console-api] text model warmed at boot
[console-api] v6.4 ready — upload & scan
🚀 TrustNet API on http://127.0.0.1:8000

[timeline] timestamps indexed: 40432     ← on first case open, not at boot
```

Three lines are benign and appear on every run: the `labeled_features.csv`
warning, `GET /.well-known/appspecific/com.chrome.devtools.json 404` (a Chrome
probe), and a Starlette warning suggesting `httpx2`.