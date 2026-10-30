<div align="center">

# TrustNet

### Coordinated review ring detection for online marketplaces.

**One ring → one case → zero fragmentation.**

[![Python](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/tests-23%20passed-brightgreen.svg)]()
[![Fragmentation](https://img.shields.io/badge/ring%20fragmentation-0%25-success.svg)]()

</div>

---

## 📑 Contents

- [The problem](#the-problem)
- [Results at a glance](#results-at-a-glance)
- [What the console does](#what-the-console-does)
- [Architecture](#architecture)
- [Quickstart — virtual environment](#quickstart--virtual-environment-recommended)
- [Quickstart — no virtual environment](#quickstart--no-virtual-environment)
- [First run on a new machine](#first-run-on-a-new-machine)
- [Verify](#verify)
- [Demo](#demo)
- [The honest claim](#the-honest-claim)
- [Known production gaps](#known-production-gaps)
- [Project layout](#project-layout)
- [Real vs synthetic](#real-vs-synthetic)
- [License](#license)

---

## The problem

Fake review rings work by making each individual review look ordinary. A
per-review classifier reads *"Great product, 10/10"* and has nothing to work
with. The signal is not in the text — it is in the **infrastructure shared
behind the reviews**: one device, one payment token, one IP subnet, one posting
burst.

TrustNet links accounts by that shared evidence, rolls each linked group into
**one high-priority case**, explains it in English, and records what a human
decided. Never dozens of scattered flags.

---

## Results at a glance

| Metric | Value |
|---|---|
| Ring detection rate (10 held-out rings) | **100%** |
| Ring fragmentation rate | **0%** |
| Precision@10 | **0.70** |
| Recall@10 | **1.00** (7 of 7 members recovered) |
| Explainability coverage | **100%** |
| Ingest latency at ring size (8–10 accounts) | **~7.4 ms ≈ 135 rev/s** |
| Text classifier AUC (machine vs human) | **0.9848** |
| Automated tests passing | **23 / 23** |

Full per-phase breakdown and honest caveats: [`docs/RESULTS.md`](docs/RESULTS.md).

---

## What the console does

Eight tabs, one server, no build step.

| Tab | Purpose |
|---|---|
| **01 Cases** | Ranked queue with tier, score, member count, signal types |
| **02 Feed** | Scored review stream, flagged-only filter |
| **03 Test** | Score a single review through the live pipeline |
| **04 Judge Lab** | Six ring shapes, one click each, live scoreboard |
| **05 Upload** | Bring your own CSV: parse, map columns, score, link |
| **06 Architecture** | Phase-by-phase design and the honest limitations |
| **07 Sellers** | Trust ledger, 300 sellers ranked, banded, with reasons |
| **08 Assistant** | Retrieval-only Q&A — no LLM, every sentence sourced |

A case detail panel carries the graph, per-review evidence, a forensic
timeline, and three moderator verdicts — confirm, dismiss, need more
information. Verdicts persist to `mod_decisions.json` across restarts and move
the implicated sellers' trust scores live.

---

## Architecture

```text
FAKE.csv (40,432 reviews: real text, rating, category, CG/OR label)
   │
   ▼
phase1   Feature engineering     → 19 raw features, synthesized entity layer
phase2   Weak supervision        → 8 labeling functions
phase3   Gradient Boosting       → risk_score per review
phase4   Weighted union-find     → clusters.csv + graph_state.pkl
phase5   Case rollup + tiering   → cases.csv (1,627 cases: 1 / 45 / 1,581)
text     Machine vs human scorer → text_model.joblib (AUC 0.985)
api      FastAPI console         → dashboard at 127.0.0.1:8000
stress   10 randomized rings     → 10/10 passed, 0% fragmentation
```

Two scorers run in parallel and are combined at **case** level, not averaged:
text answers *machine or human*, entity linking answers *coordinated or not*.
A case tier depends on signal diversity, not on either score alone.

Full design rationale: [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md).
File-by-file and endpoint reference: [`docs/FILE_REFERENCE.md`](docs/FILE_REFERENCE.md).

---

## Quickstart — virtual environment (recommended)

### Windows

```bat
cd Trustnet
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
py -3.14 phase6_api.py
```

### macOS / Linux

```bash
cd Trustnet
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python phase6_api.py
```

Open **http://127.0.0.1:8000/** in a browser.

> [!WARNING]
> Never use `localhost` — the IPv6 fallback adds roughly 2 seconds per call on
> Windows. Never open `dashboard.html` from the filesystem — every fetch
> fails on `file://` and the console renders empty.

> [!IMPORTANT]
> `python-multipart` is a hard requirement, not optional: `UploadFile` raises at
> import time without it, and that is the whole Upload tab.

---

## Quickstart — no virtual environment

Only if you accept polluting your global Python install.

```bat
py -3.14 -m pip install --user -r requirements.txt
py -3.14 phase6_api.py
```

---

## First run on a new machine

You need the generated artifacts. Either copy them, or rebuild from `FAKE.csv`.

### Option A — copy the artifacts (fastest, demo-ready)

Transfer these into the project root:

| File | Size |
|---|---|
| `scored_features.csv` | 28.8 MB |
| `clusters.csv` | 28.8 MB |
| `cases.csv` | 0.6 MB |
| `graph_state.pkl` | ~1 MB |
| `text_model.joblib` | ~3–6 MB |
| `text_model_report.json` | <1 KB |
| `sample_upload.csv` | 40 KB |

Then `py -3.14 phase6_api.py`. Without `text_model.joblib` the server still
starts, but the first Judge Lab row pays a cold model load of about one second.

### Option B — rebuild from raw data

Needs `FAKE.csv`.

```bat
py -3.14 datasetcheck.py
py -3.14 phase1_features.py      :: ~4 min
py -3.14 phase2_labels.py        :: ~1 min
py -3.14 phase3_scoring.py       :: ~3 min
py -3.14 phase4_fix.py           :: ~30 s
py -3.14 phase5_cases.py         :: ~20 s
py -3.14 train_text_model.py     :: ~35 s
py -3.14 phase6_api.py
```

`enriched_reviews.csv`, `features.csv` and `labeled_features.csv` are
intermediates. The API runs fine without `labeled_features.csv` and prints one
benign warning at boot.

---

## Verify

```bat
py -3.14 -m pytest -v test_trustnet.py
py -3.14 check_linking.py
```

Expect **23 passed** and **`OVERALL: PASS`**.

`check_linking.py` injects six ring shapes and one negative control straight
into `linker.py`, bypassing HTTP:

| Preset | Members | Components | Result |
|---|---|---|---|
| Classic ring | 8 | 1 | ✅ PASS |
| Two shared signals | 8 | 1 | ✅ PASS |
| One shared signal | 8 | 1 | ✅ PASS |
| No shared hardware | 8 | 1 | ✅ PASS |
| Slow and spread out | 6 | 1 | ✅ PASS |
| Cross-seller ring | 9 | 1 | ✅ PASS |
| Negative control (8 unrelated) | — | 8 | ✅ PASS |

The negative control is the row that matters. Eight unrelated reviews stay eight
components — the linker does not merge everything it sees.

- **Live metrics:** http://127.0.0.1:8000/api/evaluation
- **Swagger:** http://127.0.0.1:8000/docs — enabled on purpose, every endpoint is
  inspectable.

---

## Demo

Eight beats, about three minutes, with the number that must move at each step:
[`DEMO.md`](DEMO.md).

| # | Action | Number that moves |
|---|---|---|
| 1 | Open CASE-0000 | 7 reviews · 4 sellers · 5 signals · 10.2 min |
| 2 | Feed → Run Ring Test | new case · 8 members · 0 fragments |
| 3 | Judge Lab → No shared hardware | 8 / 8 · Review · 48 |
| 4 | Upload `sample_upload.csv` | 8 members · row 24 · ESCALATE 77 |
| 5 | Upload a non-review CSV | rejected · `no_text_column` · hints shown |
| 6 | Confirm → Sellers tab | 37 → 29 |
| 7 | Clear → Sellers tab | 29 → 37 |
| 8 | Assistant → "What are the limitations?" | four limits, no invention |

Reset with `POST /api/reset` and `POST /api/decisions/clear` before
presenting. Reset does not touch moderator decisions; only the second call does.

The old `POST /api/inject-member` sequence is still available, but it grows
CASE-0000 permanently for the session (7 → 8 → 9 → 10) and then requires a
reset. It is not in the scripted demo for that reason.

---

## The honest claim

Ring detection rests on the entity layer — device, IP, payment, timing,
account age. Kaggle supplies text, rating, category and the CG/OR label.
Everything else is synthesized, and the generator is tuned so baseline
infrastructure sharing stays at **2.15** while a ring collides at **7.0**.

The text model separates machine-written from human-written (AUC 0.985).
It is **not** a general paid-shill detector. That caveat ships inside
`text_model_report.json` and is returned by `/api/text-model`.

A single review can score 82/100 on text alone and still cannot escalate.
Escalation requires **≥3 independent signal types across ≥2 accounts**.
A shared office IP, a family card, a hotel wifi — each creates an edge and
none clears the 4.0 evidence floor. Without that floor the linker produced one
39,704-member cluster holding 98% of the corpus.

The system ranks and explains. It never bans a seller, freezes a payout,
or unpublishes a review.

Weights are explicit starting points, calibrated on the known ring and
validated on ten randomized held-out rings. They are not learned from
labelled fraud data, because no public labelled ring dataset exists.

---

## Known production gaps

Stated up front, because they are the first questions worth asking.

| Gap | Current state | Production requirement |
|---|---|---|
| Persistence | pickle, CSV, one JSON file | Postgres + a graph store |
| Union-find scale | Fine at 40,432 reviews | Sharding by seller above ~10M/day |
| Auth | All endpoints open, deliberate for judging | Authn, authz, rate limiting |
| Retraining | Label capture only (`mod_decisions.json`) | Retrain on adjudicated verdicts |
| Adversarial text | Per-review paraphrase defeats near-dup matching | Stylometric or cross-seller temporal |
| Load test | Peak measured under a 500-review flood | Sustained multi-node benchmark |

---

## Project layout

```text
Trustnet/
├── datasetcheck.py            Validates FAKE.csv, prints class balance
├── phase1_features.py         Entity synthesis + 19 engineered features
├── phase2_labels.py           8 weak-supervision labeling functions
├── phase3_scoring.py          Gradient Boosting risk model
├── phase4_fix.py              Weighted union-find linking (active)
├── phase5_cases.py            Case rollup and tier assignment
├── linker.py                  Standalone weighted linker — dataset-agnostic
├── train_text_model.py        Trains the machine-vs-human classifier
├── text_model.py              Loads and serves text_model.joblib
├── upload_io.py               CSV parsing, validation, column mapping
├── phase6_api.py              FastAPI server + all routes
├── phase8_stress_test.py      10-round judge stress protocol
├── check_linking.py           Six ring shapes + negative control
├── make_sample_upload.py      Generates sample_upload.csv
├── verify_demo.py             Optional pre-flight invariant check
├── rag_assistant.py           Retrieval-only assistant
├── rag_knowledge.json         15-entry knowledge base + hard routes
├── test_trustnet.py           23 automated tests
├── bench_throughput.py        Sequential HTTP benchmark
├── bench_components.py        Engine vs transport decomposition
├── dashboard.html             Console UI, 8 tabs
├── landing.html               Landing page
├── requirements.txt
├── README.md
├── DEMO.md
└── docs/
    ├── SYSTEM_DESIGN.md
    ├── FILE_REFERENCE.md
    └── RESULTS.md
```

---

## Real vs synthetic

| Data | Source |
|---|---|
| Review text | **Real** — Kaggle Fake Reviews Dataset |
| Star rating | **Real** |
| Category | **Real** |
| CG / OR labels | **Real** (20,216 / 20,216) |
| Seller, account, device, IP, payment, timestamp | **Synthesized** |

Platforms never release infrastructure identifiers. Without them, coordinated
rings are structurally undetectable at the review level — the generator exists
because the field needed to solve this does not exist publicly.

The linker itself is dataset-agnostic: it runs on whatever arrives at
`/ingest`. That is why Upload & Scan works on a corpus the text model never saw.
The text scorer touches Kaggle. The linker does not.

Full disclosure and calibration rationale: [`docs/RESULTS.md`](docs/RESULTS.md).

---

## License

Academic submission — **SYN26-CS12**, IEEE SMC KGEC, SYNAPSE 1.0 Round 1.
