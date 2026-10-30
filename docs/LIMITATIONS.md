# Limitations

Written to be read before the questions are asked. Every claim here is
reproducible from the shipped artifacts.

## 1. The dataset is half synthetic, and we say which half

Kaggle's *Fake Reviews Dataset* — 40,432 reviews, 20,216 CG (machine-written) and
20,216 OR (human-written) — contains **text, rating, category, and a binary
label. Nothing else.**

It has no seller IDs, no accounts, no devices, no IPs, no payment instruments,
and no timestamps. Every one of those fields is synthesized by
`phase1_features.py`.

This is not a hidden detail. It is the reason a ring can be detected at all: the
real corpus has no entity layer, so the entity layer had to be built, and it was
built to resemble what a marketplace actually emits.

**Consequence:** the linker is validated against rings we constructed. Its
zero-fragmentation result is real, but it is a result on generated rings with
known ground truth, not on independently observed fraud.

## 2. The text model answers a narrower question than it appears to

AUC 0.985 separates **machine-written** text from **human-written** text.

It does not separate **paid** from **honest** text. A human shill writing their
own reviews scores as human, and the model gives them a low probability.

This is stated in `text_model_report.json`, returned by `/api/text-model`, and
shown on the dashboard. Quoting 0.985 without that caveat would be misleading.

## 3. Rings with no shared infrastructure are caught less confidently

The Judge Lab six shapes, honestly:

| Shape | Members | Tier | Score |
|---|---|---|---|
| Classic ring | 8/8 | **Escalate** | 77 |
| Two shared signals | 8/8 | **Escalate** | 58 |
| One shared signal | 8/8 | **Escalate** | 53 |
| No shared hardware | 8/8 | Review | 48 |
| Slow and spread out | 6/6 | Review | 34 |
| Cross-seller ring | 9/9 | Review | 45 |

All six fragment to zero — every member lands in one case. But the bottom three
reach **Review, not Escalate**. An adversary who avoids device and IP sharing,
or who spreads reviews over weeks, or who splits across sellers, is caught as a
case that a human must look at — not as an automatic escalation.

That is the correct behaviour for a system that never auto-bans, and it is a
real ceiling on what infrastructure-based linking can achieve.

## 4. A single review cannot escalate, by design

A review can score 82/100 on text alone. It will not escalate. Escalation
requires three independent signal types across at least two accounts.

This produces visible false negatives: an obviously fake single review sits at
Review with a high score. That is the intended trade — one review is one data
point, and three signal types is the minimum at which we are willing to raise a
case to the top of a human queue.

## 5. One shared signal is not a case

`MIN_EDGE_WEIGHT = 4.0` drops any pair whose combined evidence is weaker. A
shared office IP, a family payment card, a hotel wifi network — all produce
edges, none clear the floor alone.

Without this floor the linker produced a single **39,704-member cluster holding
98% of the corpus**. That failure is documented, not hidden: it is why
`phase4_fix.py` exists and why the negative control in `check_linking.py` is a
permanent test rather than a one-time measurement.

## 6. The near-duplicate matcher is defeated by paraphrase

`exact_dup_text` (Jaccard ≥ 0.95) and `near_dup_text` (Jaccard ≥ 0.85) catch
copy-paste, punctuation edits, and word-order shuffling. They do not catch a
competent rewrite of every review in a ring.

A ring that writes unique text per account and avoids sharing hardware is
invisible to the current signal set. Stylometry, temporal co-occurrence across
sellers, and cross-ring graph structure would be the next layer. We did not
build it.

## 7. The system assesses review integrity, not product quality

A case says "these reviews are linked and suspicious". It does not say the
product is bad, the seller is fraudulent, or the money should be refunded. Those
are different questions requiring a different data source.

## 8. No production load test

Measured throughput, honestly decomposed:

| Measurement | Value |
|---|---|
| Single-review HTTP latency, ring-sized traffic | ~7.4 ms |
| Sequential, 200 calls | 46.7 rev/s, mean 21.36 ms |
| Batched, 500 calls | 56.5 rev/s, mean 17.70 ms |
| Phase 8 stress test, 10 rounds | 100% detection, avg 7.8 ms, max 11.4 ms |

The 500-call batch is slower per call than the 7.4 ms floor because all bench
reviews share one device, IP and payment instrument, so the cluster grows with
every ingest and edge resolution costs more each time. Real traffic in 8–10
member rings sits near the floor.

None of this is a 10M-reviews-per-day test. Union-find across 40,432 reviews fits
in memory; union-find across 10M/day does not, and sharding it by seller is the
hardest unsolved problem in this project.

## 9. Retraining is designed, not built

The moderator verdicts are captured: `mod_decisions.json` is append-only, with
timestamps, and it survives restarts. Confirmed and dismissed cases are exactly
the labelled data a supervised ring classifier would need.

Those labels are not fed back into anything. Retraining on adjudicated decisions
is the next phase, and this submission delivers only the label-capture half.

## 10. Deliberately out of scope

- Real-time streaming infrastructure (Kafka, Flink)
- Multi-tenant isolation
- Appeals workflow for sellers
- Any automated enforcement action

The last one is a design choice, not a gap. The system ranks and explains. A
human decides.