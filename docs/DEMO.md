# Demo

Three minutes. Eight beats. The number in the right column is the one that must
move — if it does not move, stop and say so rather than narrating past it.

## Pre-flight (60 seconds before you present)

1. `py -3.14 phase6_api.py` — already running, do not start it twice.
2. Open `/docs` in a pinned tab. Execute `POST /api/reset`, then
   `POST /api/decisions/clear`.
3. Open the dashboard, hard-refresh (Ctrl+F5).
4. Confirm `CASE-0000` reads **7 members, 81/100**.
5. Confirm the Sellers tab reads **37 / 37 / 37 / 55** for
   `SELLER-0265`, `SELLER-0039`, `SELLER-0182`, `SELLER-0221`.

If step 5 shows 29/29/29/49, the decisions were not cleared — repeat step 2.

---

## The eight beats

| # | Click | What must move |
|---|---|---|
| 1 | Open `CASE-0000` | 7 reviews · 4 sellers · 5 signal types · 10.2 min span · timeline shows 6 of 6 gaps under 30 min |
| 2 | Feed → **Run Ring Test** | New `CASE-LIVE-*` · **8 members** · Escalate · **0 fragments** |
| 3 | Judge Lab → **No shared hardware** → Run | **8 / 8** · Review · **48** |
| 4 | Upload & Scan → `sample_upload.csv` | **8 members** · row **24** present · ESCALATE **77** · 300 of 300 rows scored |
| 5 | Upload any spreadsheet that is not reviews | Rejected · `no_text_column` · hints listed in red |
| 6 | Back to `CASE-0000` → **Confirm** → Sellers tab | **37 → 29**, red −8 tag, reason reads "moderator confirmed 1 review" |
| 7 | Sellers tab → re-open case → **Dismiss** | **29 → 37**, badge updates |
| 8 | Assistant → "What are the limitations?" | Four limits, no invented numbers |

**Beat 1 script.** "Seven reviews, four different sellers, five independent
signal types, all within ten minutes of each other. This is one case, not seven
flags. A flat queue would have made it seven tickets and seven moderators, and
none of them would have seen the pattern."

**Beat 4 script.** "This CSV has three hundred rows. The system puts eight of
them — row 24 among them — into a single ring, scores it 77 out of 100, and
tells you which rows and why. The other 292 are ranked as solos. We never
trained on this file."

**Beat 6 script.** "I confirm this case. Every seller involved drops eight
points of trust, and the reason is written under the number. Dismiss it and the
points come back. The decision survives a server restart — it is on disk, not in
memory."

**Closing line.** "A single review can score 82 out of 100 on text alone and
still cannot escalate — escalation needs three independent signal types across
two or more accounts. The metric that matters for a ring is fragmentation, and
ours is zero."

---

## Questions, with the answer we have actually verified

**"Is this real data or synthetic?"**
The review text, ratings, categories and the fake/real labels are Kaggle, 40,432
rows. Every seller, account, device, IP, payment instrument and timestamp is
synthesized, because Kaggle has none of those columns. The generator is tuned so
baseline infrastructure sharing stays at 2.15 while a ring collides at 7.0. The
linker itself is dataset-agnostic — that is why Upload & Scan works on a file it
never saw.

**"Why these weights — 3.0 for device, 1.5 for burst?"**
They are explicit starting points, calibrated against the one ring whose ground
truth we know, then validated on ten randomized held-out rings with 100%
detection and zero fragmentation. We are not claiming they were learned from
labelled fraud data, because no public labelled ring dataset exists.

**"Why a graph and not a GNN?"**
Three reasons. Latency: 7.8 ms per case versus seconds for inference.
Explainability: we can name the edge — `shared_device`, `near_dup_text` — where a
GNN gives an embedding. Cold start: there is no labelled entity-layer data to
train a GNN on, and our weak supervision supplies exactly that structure.

**"Doesn't an office wifi flag every employee?"**
A single shared signal creates an edge at 3.0, below the 4.0 floor, so it does
not become a case. That floor is the single most load-bearing number in the
system — without it the linker produced one 39,704-member cluster containing 98%
of the corpus. With it, two clean clusters.

**"What would break at ten million reviews a day?"**
Union-find. It fits in memory at 40,432 and does not at 10M/day. Sharding by
seller is the hardest real problem we have and we have not solved it. Also
missing: persistence beyond a pickle and a JSON file, auth on every endpoint,
and retraining on the adjudicated decisions.

**"What can't it catch?"**
A ring that rewrites every review in original text and avoids sharing hardware,
IP, or payment. Our near-duplicate matcher defeats copy-paste; it does not defeat
a competent paraphrase. Rings that spread slowly or cross sellers still get
caught as one case, but only reach Review, not Escalate. Both are shown live in
the Judge Lab.

---

## If something goes wrong

**Ring test shows 14 members, not 8.** The button was clicked twice. Two
independent rings both linked to `SELLER-0003` through the same default phrase
and merged, which is correct behaviour. Say so, then reset.

**First Judge Lab row takes ~1 second.** Cold model load. It is warmed at boot;
if you see it, the server was just restarted. Run one preset before you present.

**Sellers tab shows no movement after Confirm.** `/api/decisions/clear` was not
run in pre-flight, or the page was not refreshed after the POST. Check the badge
on the case panel — if it is already there before you clicked, the state was
dirty.

**Assistant gives a generic answer.** It takes two to three seconds to warm its
case cache on first call. Ask one throwaway question before the real one.