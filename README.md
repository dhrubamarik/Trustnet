# TrustNet

**Coordinated review ring detection for online marketplaces.**

One ring → one case → zero fragmentation.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/tests-13%20passed-brightgreen.svg)]()
[![Fragmentation](https://img.shields.io/badge/ring%20fragmentation-0%25-success.svg)]()

---

## The problem

Fake review rings work by making each individual review look ordinary. A
per-review classifier reads *"Great product, 10/10"* and has nothing to
work with. The signal isn't in the text — it's in the **infrastructure
shared behind the reviews**: one device, one payment token, one IP subnet,
one posting burst.

TrustNet links accounts by that shared infrastructure and collapses the
whole ring into **one high-priority case** for a human moderator. Never
dozens of scattered flags.

## Results at a glance

| Metric | Value |
|---|---|
| Ring detection rate (10 held-out rings) | **100%** |
| Ring fragmentation rate | **0%** |
| Precision@10 | **0.70** |
| Recall@10 | **1.00** (7 of 7 members recovered) |
| Explainability coverage | **100%** |
| Ingest latency at ring size (8–10 accounts) | **~7.4 ms ≈ 135 rev/s** |
| Automated tests passing | **13 / 13** |

Full numbers, per-phase breakdown, and honest caveats: [`docs/RESULTS.md`](docs/RESULTS.md).

---

## Architecture

```
FAKE.csv (40,432 reviews, real text + CG/OR labels)
│
▼
Phase 1  Feature engineering   → enriched_reviews.csv → features.csv (29 cols)
Phase 2  Weak supervision       → labeled_features.csv (8 LFs)
Phase 3  Gradient Boosting      → scored_features.csv (risk_score)
Phase 4  Union-find graph linking → clusters.csv + graph_state.pkl
Phase 5  Case rollup + tiering  → cases.csv (1,627 cases)
Phase 6  FastAPI console        → dashboard at 127.0.0.1:8000
Phase 8  Judge stress test      → 10/10 rounds, 0% fragmentation
Phase 9  Assistant + docs       → RAG tab, this README
```

Full design rationale and tech-stack choices: [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md).
File-by-file API reference: [`docs/FILE_REFERENCE.md`](docs/FILE_REFERENCE.md).

---

## Quickstart — with virtual environment (recommended)

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
python3.14 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python phase6_api.py
```

Then open `http://127.0.0.1:8000` in a browser.
**Never use `localhost`** (Windows IPv6 fallback adds ~2 s per call).

## Quickstart — without virtual environment

Only do this if you accept polluting your global Python install.

```bat
py -3.14 -m pip install --user -r requirements.txt
py -3.14 phase6_api.py
```

If `--user` is blocked by policy, drop it. Some environments require
`--break-system-packages` on Linux; prefer venv in that case.

## First-time setup on a new machine

You need two things to run the demo:

- The **generated CSVs** — `scored_features.csv`, `clusters.csv`,
  `cases.csv`, `graph_state.pkl` (4 files, ~80 MB total)
- `FAKE.csv` — only if you want to re-run the pipeline from scratch

**Option A — copy the CSVs (fastest, demo-ready):**

Zip and transfer these four files into the project root, then run
`py -3.14 phase6_api.py`.

**Option B — rebuild from raw data (needs FAKE.csv):**

```bat
py -3.14 datasetcheck.py
py -3.14 phase1_features.py      :: ~4 min
py -3.14 phase2_labels.py        :: ~1 min
py -3.14 phase3_scoring.py       :: ~3 min
py -3.14 phase4_fix.py           :: ~30 s
py -3.14 phase5_cases.py         :: ~20 s
py -3.14 phase6_api.py
```

Skip `phase4_graph.py` — deprecated, superseded by `phase4_fix.py`.

## Demo tour (2 minutes)

1. Open `http://127.0.0.1:8000` → **01 Cases** tab
2. Click **CASE-0000** — 81/100, ESCALATED, 7 accounts, 5 signal types
3. Click **Inject Member** 3× — watch the same case grow 7 → 8 → 9 → 10
   without ever fragmenting
4. Click **Run Ring Test** — a fresh ring on an unknown device/IP/payment
   cluster surfaces as one new case (generalisation proof)
5. **02 Test** tab — Fake Sample ≈ 53/100 ("Text alone cannot escalate"),
   Genuine Sample ≈ 16/100
6. **05 Assistant** tab — *"Why was CASE-0000 flagged?"* — every answer
   cites its source; nothing is generated

## Tests

```bat
py -3.14 -m pytest -v test_trustnet.py
```

Expect **13 passed**: 3 core + 10 parametrized held-out rings.

Live evaluation endpoint: `http://127.0.0.1:8000/api/evaluation`

## Project layout

```
Trustnet/
├── phase1_features.py          Feature engineering
├── phase2_labels.py            8 weak-supervision labeling functions
├── phase3_scoring.py           Gradient Boosting risk model
├── phase4_fix.py               Union-find graph linking (active)
├── phase4_graph.py             (deprecated)
├── phase5_cases.py             Case rollup and tier assignment
├── linker.py                   Shared IncrementalLinker class
├── phase6_api.py               FastAPI server + console API
├── phase8_stress_test.py       Judge stress protocol
├── rag_assistant.py            Retrieval-only assistant
├── rag_knowledge.json          Assistant knowledge base
├── test_trustnet.py            13 automated tests
├── bench_throughput.py         Sequential HTTP benchmark
├── bench_components.py         Engine vs transport decomposition
├── dashboard.html              Console UI (5 tabs)
├── landing.html                Marketing landing page
├── docs/
│   ├── SYSTEM_DESIGN.md
│   ├── FILE_REFERENCE.md
│   └── RESULTS.md
├── requirements.txt
└── README.md
```

## Real vs synthetic

| Data | Source |
|---|---|
| Review text | Real — Kaggle Fake Reviews Dataset |
| Star rating | Real |
| Category | Real |
| CG / OR labels | Real (20,216 / 20,216) |
| Seller, account, device, IP, payment, timestamp | **Synthesized** |

Platforms never release infrastructure identifiers. Without them,
coordinated rings are structurally undetectable — the generator exists
because the field needed to solve the problem does not exist publicly.

Full disclosure and calibration rationale: [`docs/RESULTS.md`](docs/RESULTS.md).

## License

Academic submission — SYN26-CS12, IEEE SMC KGEC, SYNAPSE 1.0 Round 1.
