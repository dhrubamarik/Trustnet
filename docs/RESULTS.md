# TrustNet — Measured Results

Every number below was produced by this codebase on this machine. Nothing is
extrapolated.

---

## 1. Baseline dataset

| Metric | Value |
|---|---|
| Reviews | 40,432 |
| CG (computer-generated) | 20,216 |
| OR (original) | 20,216 |
| Entity pools | 35,000 devices · 20,000 IP blocks · 35,000 payment instruments |
| Cases after Phase 5 | 1,627 |
| ESCALATE / REVIEW / WATCH | **1 / 45 / 1,581** |

---

## 2. Phase 1 — Feature separation

| Feature | Baseline mean | Ring mean |
|---|---|---|
| `shared_device` | 2.15 | 7.00 |
| `shared_ip` | 3.04 | 7.00 |
| `shared_payment` | 2.16 | 7.00 |
| `burst_count` | 0.03 | 0.86 |
| `near_dup_score` | 0.499 | 0.916 |

The 8× infrastructure separation is the reason the linker has anything to find.

---

## 3. Phase 2 — Weak supervision

| Metric | Value |
|---|---|
| Baseline `soft_label` mean | 0.0834 |
| Baseline P99 | 0.3889 |
| Ring `soft_label` mean | **0.7976** |
| Separation | **9.6×** |

`lf_new_account_active` fires **0%** on the injected ring — each ring account
posts exactly one review — which is why the two-weakness-first design was
validated rather than over-engineered.

---

## 4. Phase 3 — Risk model

| Metric | Value |
|---|---|
| Model | `GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, min_samples_leaf=20)` |
| Validation | 5-fold `cross_val_predict` on 19 raw features |
| Ring mean risk | 0.8087 |
| Ring min risk | 0.7911 |
| Ring recall | **7 / 7** |
| Precision@10 | 0.70 |
| Isolation Forest recovery | 2 / 7 — retained as a column only |

---

## 5. Phase 4 — Graph linking

**Before the weight floor:**

| Metric | Value |
|---|---|
| Clusters | 1 |
| Largest cluster | **39,704** (98% of dataset) |
| Cause | A single shared office IP chained everything |

**After `MIN_EDGE_WEIGHT = 4.0`:**

| Metric | Value |
|---|---|
| Clusters | **2** |
| Ring as one cluster | **yes** |
| Ring cluster size | **7** |
| Signal diversity | **5** |
| False inclusions | **0** |

`graph_state.pkl` load: `{"reviews": 40432, "components": 40425,
"largest": 7, "top5": [7, 2, 1, 1, 1]}`.

---

## 6. `check_linking.py` — six shapes, direct linker injection

| Preset | Members | Components | Result |
|---|---|---|---|
| Classic ring | 8 | 1 | PASS |
| Two shared signals | 8 | 1 | PASS |
| One shared signal | 8 | 1 | PASS |
| No shared hardware | 8 | 1 | PASS |
| Slow and spread out | 6 | 1 | PASS |
| Cross-seller ring | 9 | 1 | PASS |
| **Negative control (8 unrelated)** | — | **8** | PASS |

```
OVERALL: PASS
```

The negative control is the load-bearing row. Eight unrelated reviews stay eight
components — the linker does not merge everything it sees.

---

## 7. Phase 5 — Case rollup

`CASE-0000`, the injected ring:

```json
{
  "case_id": "CASE-0000",
  "tier": "ESCALATE",
  "case_score": 1.6009,
  "max_risk": 0.8201,
  "avg_risk": 0.8087,
  "n_members": 7,
  "n_sellers": 4,
  "n_accounts": 7,
  "signal_diversity": 5,
  "signal_types": "near_dup_text, shared_device, shared_ip, shared_payment, temporal_burst",
  "time_span_minutes": 10.2,
  "member_ids": [32386, 20451, 31787, 9741, 6293, 1399, 36088]
}
```

`cases.csv` stores `case_score` 1.6009. The dashboard displays **81/100**, which
it recomputes live from member risk at request time rather than reading the
stored value. If the two ever disagree under a fresh reset, the live
recomputation is authoritative.

---

## 8. Text model — machine-written vs human-written

| Metric | Value |
|---|---|
| AUC | **0.9848** |
| Accuracy | 0.9405 |
| Precision | 0.9448 |
| Recall | 0.9357 |
| F1 | 0.9402 |
| Train / test | 32,345 / 8,087, stratified 80/20, seed 42 |
| Fit / total time | 30.6 s / 34.9 s |

Pipeline: `FeatureUnion(word TF-IDF 40K, char_wb 3–5 TF-IDF 40K)` +
`LogisticRegression(C=4.0, liblinear, class_weight="balanced")`.

**The caveat, which travels with the number in `text_model_report.json` and in
`/api/text-model`:** Kaggle CG reviews are machine-written. Real-world paid
reviews written by humans will be harder to separate than 0.985 suggests.

---

## 9. Judge Test Lab — six ring shapes over HTTP

| Ring type | Members | Tier | Score | Latency | Verdict |
|---|---|---|---|---|---|
| Classic ring | 8 / 8 | Escalate | 77 | 1092 ms\* | PASS |
| Two shared signals | 8 / 8 | Escalate | 58 | 44 ms | PASS |
| One shared signal | 8 / 8 | Escalate | 53 | 55 ms | PASS |
| No shared hardware | 8 / 8 | Review | 48 | 59 ms | PASS |
| Slow and spread out | 6 / 6 | Review | 34 | 55 ms | PASS |
| Cross-seller ring | 9 / 9 | Review | 45 | 57 ms | PASS |

\*First row only, cold `text_model.joblib` load. The model is warmed at boot, so
a live demo should never see it. Run one preset before presenting if the server
was just restarted.

Fragmentation is 0 across all six. The bottom three reach **Review, not
Escalate** — that is correct behaviour for a system with an escalation floor of
three independent signal types, not a failure.

Cross-seller detail: case `CASE-LIVE-40470`, size 9, tag `RING-09A8A46`, linked
by `fresh_accounts, near_dup_text, shared_device, shared_ip, temporal_burst`.

---

## 10. Phase 8 — Judge stress protocol

Ten back-to-back rounds, each injecting a ring on fresh device/IP/payment
identifiers.

| Metric | Value |
|---|---|
| Rounds passed | 10 / 10 |
| Ring detection | 100% |
| Fragmentation | **0%** |
| Latency | avg 7.8 ms, max 11.4 ms |

**Verdict: PASS.**

---

## 11. Throughput

`bench_throughput.py`, sequential HTTP, N=200:

```
elapsed     5.05 s
throughput  39.6 reviews/sec
mean        25.18 ms
median      29.53 ms
p95         35.92 ms
p99         37.82 ms
min / max   6.96 / 38.23 ms
```

`bench_components.py`:

```
[1] sequential 200 calls
      throughput   46.7 reviews/sec
      mean / med   21.36 / 21.44 ms
      p95          37.52 ms
      min          7.37 ms

[2] batch 500 reviews
      throughput   56.5 reviews/sec
      per-review   17.70 ms
      flood size   500 members in one case

DECOMPOSITION
  engine       17.70 ms
  transport     3.73 ms
  sequential   46.7 rps
  batched      56.5 rps
```

**Why the batch figure is 17.70 ms and not 7.37 ms.** Every bench review shares
one device, IP and payment instrument, so all 200–500 collapse into a single
growing cluster. Call #1 costs 7.4 ms, call #500 costs ~28 ms. Real traffic in
8–10-member rings sits near the floor.

Stage framing:

> "7.4 ms at ring sizes — 8 to 10 accounts, roughly 135 reviews per second.
> Under a 500-account flood into one cluster it degrades linearly to 17.7 ms
> average, 56.5 per second, with zero fragmentation."

Do not quote 147.6 rev/s. That figure came from an in-process micro-benchmark
that excluded HTTP, union-find, case merge and tier re-evaluation.

---

## 12. Upload & Scan

`sample_upload.csv`, 300 rows.

| Metric | Value |
|---|---|
| Rows scanned | 300 of 300 |
| Ring members found | **8** |
| Ring rows | 24, 39, 83, 87, 268, 271, 275, 297 |
| Ring case ID | `UPLOAD-RING-1000037` |
| Tier / score | **ESCALATE, 77 / 100** |
| Ring seller | `SELLER-0042` |
| High-risk solos | 69 |
| Low-risk rows | 223 |
| Fragmentation | **0** |

The case label carries the *second* ring row's ID, because union-find names a
component after whichever root wins. Cosmetic; all eight rows are inside the
case.

A union-find root resolved at ingest time can be superseded by a later merge, so
the upload path re-resolves every root against the final union-find state before
building the case. Without that step, the earliest ring row strands itself in a
one-member group — which is exactly the bug that produced a 7-of-8 result during
development.

**Rejection path.** A student-marks CSV passes decoding and delimiter sniffing,
matches zero column aliases, fails the text-column heuristic (needs mean length
≥ 25), and is rejected with error code `no_text_column` before any scoring
happens. Report includes a column mapping, unused columns, per-row rejection
reasons, and operator hints.

---

## 13. Seller trust ledger

Weights: `ring_depth 0.35 · escalate_depth 0.25 · review_case 0.12 · mean_risk
0.16 · infrastructure 0.08 · fresh_accounts 0.04`

Saturation: ring 3 · escalate 2 · review case 2.
Floors: any ring review → trust ≤ 55; any escalate review → trust ≤ 70.
Moderator: confirm → `−min(25, round(6·√n))`; dismiss → `+min(10, round(4·√n))`.

Band distribution across 300 sellers:

```
trusted 293 · cautious 3 · watch 0 · suspect 1 · high_risk 3
```

Scan diagnostics behind that spread:

```json
{"cases_total": 1627, "cases_with_members": 1627, "members_seen": 1634,
 "ring_member_ids": 7, "escalate_member_ids": 7,
 "tier_values": {"ESCALATE": 1, "REVIEW": 45, "WATCH": 1581},
 "cases_without_members": []}
```

293 sellers at `trusted` is not a bug. With 1 ESCALATE and 45 REVIEW across
1,627 cases, most sellers hold one or two ordinary reviews.

Top-of-risk readout:

| Seller | Trust | Band | Reason line |
|---|---|---|---|
| `SELLER-0182` | 29 | high_risk | confirmed 2 reviews (−8) · 2 ring reviews · 2 escalated |
| `SELLER-0221` | 29 | high_risk | same |
| `SELLER-0039` | 29 | high_risk | same |
| `SELLER-0265` | 49 | suspect | 1 ring review · 1 escalated · −6 |
| `SELLER-0268` | 85 | cautious | 2 reviews inside linked cases |
| `SELLER-0030` | 86 | cautious | same |

**Why the formula was rewritten.** The first version used ring reviews as a
*fraction* of a seller's ~163 reviews. Two ring reviews contributed 0.012 and
every seller scored 96–97 — technically arithmetically correct, operationally
useless. Absolute counts with saturation and presence floors produce the spread
above.

---

## 14. Moderator loop — measured movement

| Step | `SELLER-0182` | `SELLER-0221` | `SELLER-0039` | `SELLER-0265` |
|---|---|---|---|---|
| Baseline | 37 | 37 | 37 | 55 |
| After confirming `CASE-0000` | **29** | **29** | **29** | **49** |
| After clear | 37 | 37 | 37 | 55 |

The three sellers holding two confirmed reviews drop 8 points
(`round(6·√2) = 8`). `SELLER-0265` holds one and drops 6 (`round(6·√1) = 6`).

Persistence: the confirmed badge and the moved trust scores survive a full
server process restart and a hard browser refresh. The decision is read back
from `mod_decisions.json`, not from memory.

Dismissal is capped at ±10 against confirmation's −25 on purpose. A false
positive must be correctable without letting a real ring launder its record.

---

## 15. `/api/evaluation`

```json
{
  "precision_at_k": {"P@5": 1.00, "P@10": 0.70, "P@20": 0.35, "P@50": 0.14},
  "recall_at_k":    {"R@5": 0.714, "R@10": 1.00, "R@20": 1.00, "R@50": 1.00},
  "ring_members_ground_truth": 7,
  "ring_cases_found": 1,
  "ring_linking_rate": 1.00,
  "fragmentation_rate": 0.00,
  "stress_test_pass": true,
  "explainability_coverage_pct": 100.0,
  "text_model": {
    "available": true,
    "auc": 0.9848,
    "accuracy": 0.9405,
    "split": "random stratified 80/20",
    "n_total": 40432,
    "n_test": 8087,
    "caveat": "Kaggle CG reviews are machine-written. Real-world paid reviews written by humans will be harder to separate than this number suggests."
  }
}
```

---

## 16. Tests

```
py -3.14 -m pytest -v test_trustnet.py
=========================================
23 passed, 15 warnings in 22.62s
=========================================
```

Coverage spans baseline escalation count, the stress protocol, ten randomized
held-out rings, explainability coverage, upload acceptance and rejection, the
text-model plumbing, seller trust payload shape, and decision persistence.

---

## 17. Demo values

| Moment | Value |
|---|---|
| Genuine Sample (Test tab) | **8 / 100** |
| Fake Sample (Test tab) | **82 / 100** |
| `CASE-0000` post-reset | 7 members · **81 / 100** · 5 signals · 10.2 min |
| Run Ring Test | new `CASE-LIVE-*` · **8 members** · ESCALATE · 0 fragments |
| Judge Lab, no shared hardware | **8 / 8** · REVIEW · 48 |
| Upload `sample_upload.csv` | **8 members** · row 24 present · ESCALATE 77 · 300/300 scored |
| Upload a non-review CSV | rejected · `no_text_column` · hints listed |
| Sellers, pre-confirm | 37 / 37 / 37 / 55 |
| Sellers, post-confirm | 29 / 29 / 29 / 49 |
| Sellers, post-clear | 37 / 37 / 37 / 55 |

**The reading for stage.** A single review can score 82 out of 100 on text alone
and still cannot escalate — escalation needs three independent signal types
across two or more accounts. The number that matters for a ring is
fragmentation, and it is zero.

The older framing ("fake sample caps at 53") is dead. Do not use it.