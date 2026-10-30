"""
TrustNet — Shared Incremental Linker
=====================================
Union-Find with weighted multi-signal evidence.

A new review links to an existing one only when the evidence they share adds
up to MIN_EDGE_WEIGHT (4.0). One shared office Wi-Fi is 2.5 — not enough.
Device + IP is 5.5 — enough. Identical text + a 30-minute window is 5.0.

Evidence and weight
    shared_device      3.0   same device fingerprint
    shared_payment     3.0   same payment instrument
    shared_ip          2.5   same /24 subnet  (weak: offices share Wi-Fi)
    exact_dup_text     3.5   same seller, text Jaccard >= 0.95
    near_dup_text      2.0   same seller, text Jaccard >= 0.85
    temporal_burst     1.5   same seller, posted within 30 minutes
    fresh_accounts     1.0   both accounts under 24 hours old

Text and timing are compared ONLY against recent reviews of the same seller,
capped at 60 peers. That cap is what stops a templated seller from chaining
into a mega-cluster — the 39,704-member failure from Phase 4.

Usage:  from linker import IncrementalLinker
"""

from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

# ── evidence weights ──────────────────────────────────────────────────────
SIGNAL_WEIGHTS = {
    "shared_device": 3.0,
    "shared_payment": 3.0,
    "shared_ip": 2.5,
    "exact_dup_text": 3.5,
    "near_dup_text": 2.0,
    "temporal_burst": 1.5,
    "fresh_accounts": 1.0,
}

MIN_EDGE_WEIGHT = 4.0
NEAR_DUP_JACCARD = 0.85
EXACT_DUP_JACCARD = 0.95
BURST_WINDOW_MINUTES = 30.0
MAX_SELLER_PEERS = 60
MAX_EDGES_PER_INGEST = 400

_ENTITY_COLS = (
    ("device_fingerprint", "shared_device"),
    ("payment_instrument", "shared_payment"),
    ("ip_block", "shared_ip"),
)

_BAD_VALUES = {
    "", "none", "null", "nil", "na", "n/a", "nan", "-", "--", "unknown",
    "undefined", "dev-unknown", "ip-unknown", "pay-unknown", "pay-0000-0000",
    "device-unknown", "0.0.0.0", "10.0.0.0/24", "127.0.0.1",
}

_JUNK = re.compile(r"[^a-z0-9 ]+")


def _clean(v):
    """A usable entity id, or None. Placeholders mean 'no value'."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in _BAD_VALUES:
        return None
    return s


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _as_epoch(v):
    if v is None:
        return None
    f = _num(v)
    if f is not None:
        return f
    try:
        import datetime as _dt
        return _dt.datetime.fromisoformat(
            str(v).strip().replace("Z", "+00:00")
        ).timestamp()
    except Exception:
        return None


def _shingles(text, k=5):
    s = " ".join(_JUNK.sub(" ", str(text or "").lower()).split())
    if not s:
        return set()
    if len(s) < k:
        return {s}
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / (len(a) + len(b) - inter)


class IncrementalLinker:
    """
    Incremental union-find for entity linking.

    The baseline graph is loaded once (or unpickled). Everything after that is
    incremental — a new review is compared against existing index buckets and
    against recent reviews of the same seller. No rebuild.
    """

    def __init__(self, existing_df=None, filtered_graph=None):
        self.parent = {}
        self.rank = {}
        self.signal_index = {
            "device_fingerprint": defaultdict(set),
            "ip_block": defaultdict(set),
            "payment_instrument": defaultdict(set),
        }
        self._next_id = 0
        self._min_weight = MIN_EDGE_WEIGHT

        self._seller_peers = defaultdict(list)   # seller -> [(rid, text, ts, age)]
        self._review_time = {}
        self._review_seller = {}
        self._review_age = {}
        self._review_text = {}
        self._comp = defaultdict(set)            # root -> member ids
        self._comp_ready = True

        if existing_df is not None and filtered_graph is not None:
            self._load_baseline(existing_df, filtered_graph)

    # ── baseline ──────────────────────────────────────────────────────────
    def _load_baseline(self, existing_df, filtered_graph):
        for rid in existing_df["review_id"].values:
            rid = int(rid)
            self.parent[rid] = rid
            self.rank[rid] = 0
            self._comp[rid].add(rid)

        for _, row in existing_df.iterrows():
            rid = int(row["review_id"])
            for col in self.signal_index:
                if col not in row:
                    continue
                val = row[col]
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    continue
                s = _clean(val)
                if s:
                    self.signal_index[col][s].add(rid)

        for u, v in filtered_graph.edges():
            self.union(int(u), int(v))

        self._next_id = int(existing_df["review_id"].max()) + 1
        self._comp_ready = True

    # ── compatibility with pickles written by the previous class ──────────
    def _ensure_state(self):
        if not hasattr(self, "signal_index") or not isinstance(self.signal_index, dict):
            self.signal_index = {}
        for col in ("device_fingerprint", "ip_block", "payment_instrument"):
            idx = self.signal_index.get(col)
            if not isinstance(idx, defaultdict):
                self.signal_index[col] = defaultdict(set, idx or {})

        for attr in ("parent", "rank", "_review_time", "_review_seller",
                     "_review_age", "_review_text"):
            if not hasattr(self, attr):
                setattr(self, attr, {})
        if not hasattr(self, "_seller_peers"):
            self._seller_peers = defaultdict(list)
        if not hasattr(self, "_comp"):
            self._comp = defaultdict(set)
        if not hasattr(self, "_min_weight"):
            self._min_weight = MIN_EDGE_WEIGHT
        if not hasattr(self, "_next_id"):
            self._next_id = (max(self.parent) + 1) if self.parent else 0

        if not getattr(self, "_comp_ready", False):
            self._comp = defaultdict(set)
            for rid in list(self.parent):
                self._comp[self.find(rid)].add(rid)
            self._comp_ready = True

    # ── union-find ────────────────────────────────────────────────────────
    def find(self, x):
        x = int(x)
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        root = x
        while self.parent.get(root, root) != root:
            root = self.parent[root]
        while self.parent.get(x, x) != x:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return rx
        if self.rank.get(rx, 0) < self.rank.get(ry, 0):
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank.get(rx, 0) == self.rank.get(ry, 0):
            self.rank[rx] = self.rank.get(rx, 0) + 1
        moved = self._comp.pop(ry, set())
        if moved:
            self._comp.setdefault(rx, set()).update(moved)
        return rx

    # ── ingest ────────────────────────────────────────────────────────────
    def ingest(self, new_review: dict) -> dict:
        self._ensure_state()

        rid = int(new_review.get("review_id", self._next_id))
        self._next_id = max(self._next_id, rid + 1)

        if rid not in self.parent:
            self.parent[rid] = rid
            self.rank[rid] = 0
            self._comp[rid].add(rid)

        text = str(
            new_review.get("text_")
            or new_review.get("text")
            or new_review.get("review_text")
            or ""
        ).strip()
        seller = str(new_review.get("seller_id") or "").strip()
        ts = _as_epoch(new_review.get("timestamp"))
        age = _num(new_review.get("account_age_hours"))

        weights = defaultdict(float)
        reasons = defaultdict(set)

        # 1. shared infrastructure
        for col, name in _ENTITY_COLS:
            val = _clean(new_review.get(col))
            if not val:
                continue
            for other in list(self.signal_index[col].get(val, ())):
                other = int(other)
                if other == rid:
                    continue
                weights[other] += SIGNAL_WEIGHTS[name]
                reasons[other].add(name)

        # 2. text + timing, same seller only, capped peer window
        if seller:
            mine = _shingles(text) if text else set()
            for peer in self._seller_peers.get(seller, ()):
                other, other_text, other_ts, other_age = peer
                other = int(other)
                if other == rid:
                    continue

                if mine and other_text:
                    j = _jaccard(mine, _shingles(other_text))
                    if j >= EXACT_DUP_JACCARD:
                        weights[other] += SIGNAL_WEIGHTS["exact_dup_text"]
                        reasons[other].add("near_dup_text")
                    elif j >= NEAR_DUP_JACCARD:
                        weights[other] += SIGNAL_WEIGHTS["near_dup_text"]
                        reasons[other].add("near_dup_text")

                if ts is not None and other_ts is not None:
                    if abs(ts - other_ts) <= BURST_WINDOW_MINUTES * 60:
                        weights[other] += SIGNAL_WEIGHTS["temporal_burst"]
                        reasons[other].add("temporal_burst")

                if age is not None and other_age is not None:
                    if age < 24 and other_age < 24:
                        weights[other] += SIGNAL_WEIGHTS["fresh_accounts"]
                        reasons[other].add("fresh_accounts")

        # 3. union on total evidence
        ranked = sorted(weights.items(), key=lambda kv: -kv[1])[:MAX_EDGES_PER_INGEST]
        merged = False
        evidence = set()
        joined = []
        for other, w in ranked:
            if w >= self._min_weight:
                self.union(rid, other)
                merged = True
                joined.append(int(other))
                evidence |= reasons[other]

        # 4. index this review
        for col, _name in _ENTITY_COLS:
            val = _clean(new_review.get(col))
            if val:
                self.signal_index[col][val].add(rid)

        if seller:
            peers = self._seller_peers[seller]
            peers.append((rid, text, ts, age))
            if len(peers) > MAX_SELLER_PEERS:
                del peers[: len(peers) - MAX_SELLER_PEERS]

        self._review_seller[rid] = seller
        self._review_time[rid] = ts
        self._review_age[rid] = age
        self._review_text[rid] = text

        root = self.find(rid)
        component = sorted(self._comp.get(root, {rid}))

        return {
            "review_id": rid,
            "cluster_root": root,
            "cluster_size": len(component),
            "cluster_members": component,
            "merged": merged,
            "signals": sorted(evidence),
            "joined": joined,
        }

    # ── introspection ─────────────────────────────────────────────────────
    def stats(self):
        self._ensure_state()
        sizes = sorted((len(v) for v in self._comp.values()), reverse=True)
        return {
            "reviews": len(self.parent),
            "components": len(self._comp),
            "largest": sizes[0] if sizes else 0,
            "top5": sizes[:5],
        }