"""
TrustNet — Shared Incremental Linker
======================================
Union-Find data structure for incremental graph linking.
Import this in any script that needs to load graph_state.pkl
or do live ingestion.

Usage:  from linker import IncrementalLinker
"""

from collections import defaultdict
import pandas as pd


class IncrementalLinker:
    """
    Incremental union-find for entity linking.
    When a new review arrives, merges it into existing components
    based on shared signals — no full graph rebuild needed.
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
        self._min_weight = 4.0

        if existing_df is not None and filtered_graph is not None:
            for rid in existing_df["review_id"].values:
                self.parent[int(rid)] = int(rid)
                self.rank[int(rid)] = 0
            for _, row in existing_df.iterrows():
                rid = int(row["review_id"])
                for col in self.signal_index:
                    val = row[col]
                    if pd.notna(val):
                        self.signal_index[col][val].add(rid)
            for u, v in filtered_graph.edges():
                self.union(int(u), int(v))
            self._next_id = int(existing_df["review_id"].max()) + 1

    def find(self, x):
        x = int(x)
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return rx
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return rx

    def ingest(self, new_review: dict) -> dict:
        """
        Ingest one new review. Returns the cluster it joined.
        new_review must have: device_fingerprint, ip_block,
                              payment_instrument, review_id
        """
        rid = int(new_review.get("review_id", self._next_id))
        self._next_id = max(self._next_id, rid + 1)
        self.parent[rid] = rid
        self.rank[rid] = 0

        match_counts = defaultdict(int)
        for col in self.signal_index:
            val = new_review.get(col)
            if val and val in self.signal_index[col]:
                for existing_rid in self.signal_index[col][val]:
                    match_counts[existing_rid] += 1

        merged = False
        for existing_rid, count in match_counts.items():
            if count >= 2:
                self.union(rid, existing_rid)
                merged = True

        for col in self.signal_index:
            val = new_review.get(col)
            if val:
                self.signal_index[col][val].add(rid)

        root = self.find(rid)
        component = [k for k in self.parent if self.find(k) == root]

        return {
            "review_id": rid,
            "cluster_root": root,
            "cluster_size": len(component),
            "cluster_members": component,
            "merged": merged,
        }