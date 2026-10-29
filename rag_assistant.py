"""
TrustNet retrieval assistant.

No generative model. Every returned sentence is either a row from
rag_knowledge.json or a value read from the live case store at request time.
That property is what makes the "cannot hallucinate" claim defensible.
"""

from __future__ import annotations

import json
import os
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MIN_SIM = 0.12
KW_BOOST = 0.30
SECONDARY_SIM = 0.55
MAX_QUERY = 600
MAX_SELLERS_SHOWN = 6

WELCOME = (
    "I answer from the case store and a fixed knowledge base. "
    "There is no generative model in this tab, so I will not invent a number.\n\n"
    "Try: *can I buy from SEL_003?* \u00b7 *why was CASE-0000 flagged?* \u00b7 "
    "*how fast is it?* \u00b7 *what are the limitations?*"
)

# ---------- signal names ----------

SIG_LABEL = {
    "near_dup_text": "copy-pasted text",
    "shared_device": "shared device",
    "shared_ip": "shared IP subnet",
    "shared_payment": "shared payment instrument",
    "temporal_burst": "review burst",
    "burst": "review burst",
    "shared_account": "shared account",
    "high_velocity": "high velocity",
    "seller_new_5star": "new-seller 5-star concentration",
    "new_account": "new accounts",
}

VERDICT = {
    "ESCALATE": ("Do not buy from any seller in this case until it is cleared. "
                 "A coordinated ring is attached to them."),
    "REVIEW": ("Buy with care. This seller is linked to an open case that is "
               "still being assessed."),
    "WATCH": ("No escalation found. This seller is only on a watchlist, so "
              "ordinary purchase risk applies."),
}

# ---------- guards ----------

REFUSALS = [
    (re.compile(r"\b(i love you|love u|marry me|be my (girlfriend|boyfriend)|"
                r"are you single|do you love me|date me)\b", re.I),
     "I'm the TrustNet fraud console assistant \u2014 personal questions aren't "
     "something I handle. Ask me about a case, a seller, or how detection works."),
    (re.compile(r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|above|prior|"
                r"earlier|system)\b.{0,30}\b(instruction|prompt|rule|direction)", re.I),
     "I can't change my operating rules. Ask me about TrustNet."),
    (re.compile(r"\b(system prompt|your instructions|jailbreak|dan mode|"
                r"developer mode|act as a different|pretend you are)\b", re.I),
     "I don't discuss my own configuration. Ask me about TrustNet."),
    (re.compile(r"\b(tell me a joke|write (me )?(a|an) (poem|story|essay)|weather|"
                r"recipe|who is the president|homework|translate|sing)\b", re.I),
     "That's outside what this console covers. I answer questions about "
     "cases, sellers and how TrustNet works."),
]

CODE_LANG = re.compile(
    r"\b(java|python|javascript|typescript|c\+\+|c#|golang|rust|sql|html|css|"
    r"react|django|flask|node\.?js|pandas|numpy|kotlin|swift)\b", re.I)
CODE_WORD = re.compile(
    r"\b(code|snippet|script|program|function|class|compile|syntax|"
    r"refactor|debug|stack ?trace|regex|algorithm)\b", re.I)

SELLER_RE = re.compile(r"\b((?:SEL|SELLER)[-_][A-Za-z0-9]+)\b", re.I)
CASE_RE = re.compile(r"\b(CASE-[A-Za-z0-9-]+)\b", re.I)

STATUS_WORDS = ("how many", "current", "live", "right now", "queue",
                "status", "in total", "how much")
CASE_WORDS = ("this case", "why flagged", "flagged", "evidence", "clue",
              "action", "what happened", "explain", "dossier", "summary")

# direct hits for the four questions that get asked every time
HARD_ROUTES = [
    (re.compile(r"\b(limitation|limit|weakness|drawback|fail|wrong|mistake|"
                r"can'?t do|cannot do|not good at|honest|blind spot)\b", re.I),
     "limitations"),
    (re.compile(r"\b(kaggle|synthetic|real data|dataset|where.{0,20}data|"
                r"data source)\b", re.I),
     "dataset_real_vs_synthetic"),
    (re.compile(r"\b(how fast|latency|throughput|speed|per second|"
                r"reviews per|performance)\b", re.I),
     "speed_and_throughput"),
    (re.compile(r"\b(precision|recall|results|accuracy|evaluation|metrics|"
                r"p@10|p at 10|test pass)\b", re.I),
     "evaluation_results"),
]


class TrustNetRAG:
    def __init__(self, knowledge_path: str | None = None):
        if knowledge_path is None:
            knowledge_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "rag_knowledge.json")
        self.knowledge_path = knowledge_path
        self.documents: list[dict] = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.doc_vectors = None
        self._by_id: dict[str, dict] = {}
        self._load()

    # ---------- setup ----------

    def _load(self):
        if not os.path.exists(self.knowledge_path):
            print(f"[assistant] knowledge base missing: {self.knowledge_path}")
            return
        with open(self.knowledge_path, encoding="utf-8") as f:
            self.documents = json.load(f)
        self._by_id = {d["id"]: d for d in self.documents if d.get("id")}
        corpus = [
            f"{d.get('question','')} {' '.join(d.get('keywords', []))} {d.get('answer','')}"
            for d in self.documents
        ]
        if corpus:
            self.doc_vectors = self.vectorizer.fit_transform(corpus)
        print(f"[assistant] knowledge base: {len(self.documents)} entries")

    # ---------- guards ----------

    def _refusal(self, q: str):
        for pattern, message in REFUSALS:
            if pattern.search(q):
                return message
        if CODE_LANG.search(q) and CODE_WORD.search(q):
            return ("I don't write or review code. Ask me about a case, a "
                    "seller, or how the detection pipeline works.")
        return None

    # ---------- retrieval ----------

    def retrieve(self, query: str, top_k: int = 2):
        if not self.documents or self.doc_vectors is None:
            return []
        q = query.lower().strip()
        sims = cosine_similarity(self.vectorizer.transform([q]), self.doc_vectors)[0]
        for i, doc in enumerate(self.documents):
            for kw in doc.get("keywords", []):
                # plural-tolerant: 'limitation' also matches 'limitations'
                stem = re.escape(kw.lower()).rstrip("s")
                if re.search(r"\b" + stem + r"s?\b", q):
                    sims[i] += KW_BOOST
        order = sims.argsort()[::-1][:top_k]
        return [{"doc": self.documents[i], "score": float(sims[i])}
                for i in order if sims[i] >= MIN_SIM]

    # ---------- entry point ----------

    def answer(self, query: str, ctx: dict | None = None) -> dict:
        ctx = ctx or {}
        q = (query or "").strip()
        if not q:
            return {"answer": WELCOME, "sources": [], "intent": "empty"}
        if len(q) > MAX_QUERY:
            q = q[:MAX_QUERY]

        blocked = self._refusal(q)
        if blocked:
            return {"answer": blocked, "sources": [], "intent": "refused"}

        # 1. explicit seller
        seller = self._find_seller(q)
        if seller:
            return self._seller_report(seller, ctx)

        # 2. explicit case id
        case_id = self._find_case(q)
        if case_id:
            return self._case_report(case_id, ctx)

        # 3. fixed questions
        for pattern, doc_id in HARD_ROUTES:
            if pattern.search(q):
                doc = self._by_id.get(doc_id)
                if doc:
                    return {"answer": doc["answer"], "sources": [doc_id],
                            "intent": "kb"}

        # 4. live counters
        if any(w in q.lower() for w in STATUS_WORDS):
            hit = self.retrieve(q, top_k=1)
            if hit and hit[0]["score"] >= SECONDARY_SIM:
                return {"answer": hit[0]["doc"]["answer"],
                        "sources": [hit[0]["doc"]["id"]], "intent": "kb"}
            return {"answer": self._live_status(ctx),
                    "sources": ["live case store"], "intent": "status"}

        # 5. follow-up about the case on screen
        if ctx.get("active_case") and any(w in q.lower() for w in CASE_WORDS):
            return self._case_report(ctx["active_case"].get("case_id"), ctx,
                                     fallback=ctx["active_case"])

        # 6. general knowledge
        hit = self.retrieve(q, top_k=2)
        if not hit:
            return {"answer": WELCOME, "sources": [], "intent": "fallback"}

        text = hit[0]["doc"]["answer"]
        sources = [hit[0]["doc"]["id"]]
        if len(hit) > 1 and hit[1]["score"] >= SECONDARY_SIM:
            text += "\n\nAlso relevant: " + hit[1]["doc"]["answer"]
            sources.append(hit[1]["doc"]["id"])
        return {"answer": text, "sources": sources, "intent": "kb"}

    # ---------- extractors ----------

    @staticmethod
    def _find_seller(q: str):
        m = SELLER_RE.search(q)
        return m.group(1).upper() if m else None

    @staticmethod
    def _find_case(q: str):
        m = CASE_RE.search(q)
        return m.group(1).upper() if m else None

    # ---------- formatters ----------

    @staticmethod
    def _format_signals(signals):
        if not signals:
            return "none recorded"
        return ", ".join(SIG_LABEL.get(s, str(s).replace("_", " ")) for s in signals)

    @staticmethod
    def _format_sellers(sellers):
        if not sellers:
            return ""
        shown = list(sellers)[:MAX_SELLERS_SHOWN]
        extra = len(sellers) - len(shown)
        out = ", ".join(shown)
        return out + (f" (+{extra} more)" if extra > 0 else "")

    @staticmethod
    def _tail(s: str):
        """Numeric tail of a seller id, for alias matching."""
        m = re.search(r"(\d+)\s*$", str(s))
        if not m:
            return None
        return m.group(1).lstrip("0") or "0"

    @staticmethod
    def _live_status(ctx: dict) -> str:
        counts = ctx.get("tier_counts") or {}
        total = ctx.get("total_cases", 0)
        lines = [
            "**Live console state**",
            f"\u2022 {total} cases in the queue",
            f"\u2022 {counts.get('ESCALATE', 0)} escalated \u00b7 "
            f"{counts.get('REVIEW', 0)} in review \u00b7 "
            f"{counts.get('WATCH', 0)} watchlisted",
        ]
        top = ctx.get("top_case")
        if top:
            lines.append(f"\u2022 Top case {top.get('case_id', '?')} at "
                         f"{top.get('score_100', 0)}/100 "
                         f"({top.get('tier', 'WATCH')})")
        lines.append(f"\u2022 {ctx.get('total_reviews', 0)} reviews scored, "
                     f"fragmentation {ctx.get('fragmentation_pct', 0)}%")
        return "\n".join(lines)

    # ---------- reports ----------

    def _case_report(self, case_id, ctx: dict, fallback=None):
        summaries = ctx.get("case_index") or {}
        key = str(case_id or "").upper()
        case = summaries.get(key) or fallback
        if not case:
            return {"answer": f"I don't have a case matching **{case_id}** in the "
                              f"current queue. Check the Cases tab for the exact ID.",
                    "sources": [], "intent": "case_missing"}

        cid = case.get("case_id", key)
        tier = case.get("tier", "WATCH")
        score = case.get("score_100", 0)
        members = case.get("members", 0)
        sellers = case.get("sellers") or []
        signals = case.get("signals") or []
        span = case.get("span_min") or 0

        body = f"**{cid}** \u00b7 {tier} \u00b7 {score}/100\n"
        body += f"\u2022 {members} linked account{'' if members == 1 else 's'}"
        if sellers:
            body += (f" across {len(sellers)} seller"
                     f"{'' if len(sellers) == 1 else 's'}: "
                     f"{self._format_sellers(sellers)}")
        body += "\n"
        body += f"\u2022 Evidence ({len(signals)}): {self._format_signals(signals)}\n"
        if span:
            body += f"\u2022 Activity window: {span:.1f} minutes\n"

        if len(signals) >= 2:
            body += (f"\n**Why it holds together:** {len(signals)} independent "
                     "signal types agree on the same account set. A single shared "
                     "signal would not clear the edge-weight floor of 4.0, so on "
                     "its own it would not link.\n")
        body += (f"\n**Recommended action:** "
                 f"{VERDICT.get(tier, 'Review manually.')}")
        return {"answer": body, "sources": [f"case store:{cid}"], "intent": "case"}

    def _seller_hits(self, seller: str, ctx: dict):
        index = ctx.get("seller_index") or {}
        want = str(seller).upper()

        if want in index:
            return list(index[want])

        for k, v in index.items():
            if str(k).upper() == want:
                return list(v)

        tail = self._tail(want)
        if tail:
            for k, v in index.items():
                if self._tail(str(k).upper()) == tail:
                    return list(v)

        return [c for c in (ctx.get("case_index") or {}).values()
                if want in [str(s).upper() for s in (c.get("sellers") or [])]]

    def _seller_report(self, seller: str, ctx: dict):
        hits = self._seller_hits(seller, ctx)

        if not hits:
            body = (f"**{seller}** \u2014 no coordinated ring detected.\n\n"
                    "No shared device, shared payment or posting burst was found "
                    "around this seller, so no case was opened against them.\n\n"
                    "That is not a guarantee of authenticity. TrustNet catches "
                    "coordinated rings, not lone bad reviews \u2014 a single fake "
                    "review with no shared infrastructure will not surface here.\n\n"
                    "One more limit worth stating plainly: this is a review-integrity "
                    "check, not a product check. There is no access to product specs, "
                    "return rates or shipping data, so it cannot tell you whether the "
                    "item itself is any good.")
            return {"answer": body, "sources": ["live case store"],
                    "intent": "seller_clean"}

        rank = {"ESCALATE": 0, "REVIEW": 1, "WATCH": 2}
        hits.sort(key=lambda c: (rank.get(c.get("tier", "WATCH"), 3),
                                 -c.get("score_100", 0)))
        top = hits[0]

        cid = top.get("case_id", "?")
        tier = top.get("tier", "WATCH")
        body = (f"**{seller}** \u2014 {len(hits)} linked case"
                f"{'' if len(hits) == 1 else 's'}.\n\n"
                f"**{cid}** \u00b7 {tier} \u00b7 {top.get('score_100', 0)}/100\n"
                f"\u2022 {top.get('members', 0)} linked accounts\n"
                f"\u2022 Evidence: {self._format_signals(top.get('signals') or [])}\n")
        if top.get("span_min"):
            body += f"\u2022 Activity window: {top['span_min']:.1f} minutes\n"

        if len(hits) > 1:
            others = ", ".join(f"{c.get('case_id', '?')} ({c.get('tier', 'WATCH')})"
                               for c in hits[1:5])
            body += f"\nOther cases: {others}\n"

        body += f"\n**Recommendation:** {VERDICT.get(tier, 'Review manually.')}\n\n"
        body += ("This is a review-integrity assessment, not a product assessment "
                 "\u2014 it says nothing about whether the item is any good.")
        return {"answer": body,
                "sources": [f"case store:{c.get('case_id', '?')}" for c in hits[:3]],
                "intent": "seller"}