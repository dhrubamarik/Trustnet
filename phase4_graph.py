"""
TrustNet — Phase 4: Graph Linking Layer
=========================================
Builds a multi-signal entity graph, finds connected components (clusters),
and provides an incremental union-find for live ingestion.

This is the CORE DIFFERENTIATOR of TrustNet: individual high-risk flags
become ONE linked case when they share multiple signals.

Input:  scored_features.csv  (Phase 3 output)
Output: clusters.csv         (one row per review, with cluster_id + cluster_size)
        graph_state.pkl      (saved graph + union-find for incremental use)
"""

import pandas as pd
import numpy as np
import networkx as nx
import pickle
from collections import defaultdict
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("PHASE 4: Graph Linking Layer")
print("=" * 60)

df = pd.read_csv("scored_features.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"\n✓ Loaded {len(df)} reviews from scored_features.csv")

# ──────────────────────────────────────────────
# 1. SIGNAL DEFINITIONS & EDGE WEIGHTS
# ──────────────────────────────────────────────
# Each signal type has:
#   - column: the DataFrame column to group by
#   - weight: edge weight (higher = stronger evidence of coordination)
#   - max_group: skip groups larger than this (likely legitimate, e.g. office Wi-Fi)

SIGNALS = {
    "shared_device": {
        "column": "device_fingerprint",
        "weight": 3.0,
        "max_group": 15,
        "description": "Same device fingerprint",
    },
    "shared_ip": {
        "column": "ip_block",
        "weight": 2.0,
        "max_group": 20,
        "description": "Same IP block",
    },
    "shared_payment": {
        "column": "payment_instrument",
        "weight": 3.0,
        "max_group": 15,
        "description": "Same payment instrument",
    },
    "near_dup_text": {
        "column": None,  # handled separately via near_dup_score
        "weight": 2.5,
        "max_group": 20,
        "description": "Near-duplicate review text",
    },
    "temporal_burst": {
        "column": None,  # handled separately via timestamp proximity
        "weight": 1.5,
        "max_group": 25,
        "description": "Same seller, reviews within 30 minutes",
    },
}

# ──────────────────────────────────────────────
# 2. BUILD THE GRAPH
# ──────────────────────────────────────────────
G = nx.Graph()
G.add_nodes_from(df["review_id"].values)

# Track which signal types connect each edge (for diversity bonus later)
edge_signals = defaultdict(set)

print("\nBuilding edges from shared signals...")

# --- Signal 1-3: Group-based (device, IP, payment) ---
for sig_name, sig in SIGNALS.items():
    if sig["column"] is None:
        continue

    col = sig["column"]
    groups = df.groupby(col)["review_id"].apply(list)
    edges_added = 0

    for val, members in groups.items():
        if len(members) < 2 or len(members) > sig["max_group"]:
            continue  # skip singletons and mega-groups
        # Connect all pairs within this group
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                u, v = members[i], members[j]
                if G.has_edge(u, v):
                    G[u][v]["weight"] += sig["weight"]
                else:
                    G.add_edge(u, v, weight=sig["weight"])
                edge_signals[(u, v)].add(sig_name)
                edge_signals[(v, u)].add(sig_name)
                edges_added += 1

    print(f"  {sig_name:<20}: {edges_added:>8} edges added")

# --- Signal 4: Near-duplicate text ---
# Use the near_dup_score from Phase 1. We need actual pairs, not just scores.
# Reconstruct: for each review with near_dup_score > 0.70, find its nearest
# neighbor in the TF-IDF space. For speed, we use a simpler approach:
# group reviews by truncated text hash (first 100 chars).
print("  near_dup_text       : computing text-similarity pairs...")

# Quick text clustering: hash the first 80 chars (lowered, stripped)
df["_text_hash"] = df["text_"].fillna("").str.lower().str.strip().str[:80].apply(hash)
text_groups = df[df["near_dup_score"] > 0.70].groupby("_text_hash")["review_id"].apply(list)

nd_edges = 0
for h, members in text_groups.items():
    if len(members) < 2 or len(members) > SIGNALS["near_dup_text"]["max_group"]:
        continue
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            u, v = members[i], members[j]
            w = SIGNALS["near_dup_text"]["weight"]
            if G.has_edge(u, v):
                G[u][v]["weight"] += w
            else:
                G.add_edge(u, v, weight=w)
            edge_signals[(u, v)].add("near_dup_text")
            edge_signals[(v, u)].add("near_dup_text")
            nd_edges += 1
print(f"  {'':>20}  {nd_edges:>8} edges added")

# --- Signal 5: Temporal burst (same seller, within 30 min) ---
print("  temporal_burst      : computing temporal pairs...")

df_sorted = df.sort_values(["seller_id", "timestamp"])
tb_edges = 0
for seller, group in df_sorted.groupby("seller_id"):
    if len(group) < 2:
        continue
    times = group["timestamp"].values
    ids = group["review_id"].values
    # Only check pairs within 30 min (sliding window)
    for i in range(len(ids)):
        for j in range(i + 1, min(i + 20, len(ids))):  # cap lookahead
            diff_min = (times[j] - times[i]) / np.timedelta64(1, "m")
            if diff_min > 30:
                break
            u, v = ids[i], ids[j]
            w = SIGNALS["temporal_burst"]["weight"]
            if G.has_edge(u, v):
                G[u][v]["weight"] += w
            else:
                G.add_edge(u, v, weight=w)
            edge_signals[(u, v)].add("temporal_burst")
            edge_signals[(v, u)].add("temporal_burst")
            tb_edges += 1
print(f"  {'':>20}  {tb_edges:>8} edges added")

# Clean up temp column
df.drop(columns=["_text_hash"], inplace=True)

print(f"\n  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ──────────────────────────────────────────────
# 3. FIND CONNECTED COMPONENTS
# ──────────────────────────────────────────────
print("\nFinding connected components...")

components = list(nx.connected_components(G))
# Only keep components with 2+ members (singletons aren't "clusters")
clusters = [c for c in components if len(c) >= 2]

# Assign cluster IDs
review_to_cluster = {}
for cid, cluster in enumerate(clusters):
    for rid in cluster:
        review_to_cluster[rid] = cid

df["cluster_id"] = df["review_id"].map(review_to_cluster).fillna(-1).astype(int)
df["cluster_size"] = df["cluster_id"].map(
    {cid: len(c) for cid, c in enumerate(clusters)}
).fillna(1).astype(int)

# Count distinct signal types per cluster
cluster_diversity = {}
for cid, cluster in enumerate(clusters):
    signals_in_cluster = set()
    cluster_nodes = list(cluster)
    for i in range(len(cluster_nodes)):
        for j in range(i + 1, len(cluster_nodes)):
            pair = (cluster_nodes[i], cluster_nodes[j])
            if pair in edge_signals:
                signals_in_cluster.update(edge_signals[pair])
    cluster_diversity[cid] = len(signals_in_cluster)

df["cluster_signal_diversity"] = df["cluster_id"].map(cluster_diversity).fillna(0).astype(int)

print(f"  Total components (size ≥ 2): {len(clusters)}")
print(f"  Largest cluster: {max(len(c) for c in clusters)} members")
print(f"  Reviews in clusters: {(df['cluster_id'] >= 0).sum()} / {len(df)}")

# ──────────────────────────────────────────────
# 4. INCREMENTAL UNION-FIND (for live ingestion)
# ──────────────────────────────────────────────
print("\nBuilding incremental Union-Find structure...")

class IncrementalLinker:
    """
    Union-Find data structure for incremental graph linking.
    When a new review arrives, it merges into existing components
    based on shared signals — no full graph rebuild needed.
    """

    def __init__(self, existing_df, signals_config):
        self.parent = {}
        self.rank = {}
        self.signals_config = signals_config

        # Signal value → set of review_ids (for fast lookup)
        self.signal_index = {
            "device_fingerprint": defaultdict(set),
            "ip_block": defaultdict(set),
            "payment_instrument": defaultdict(set),
        }

        # Initialize from existing data
        for rid in existing_df["review_id"].values:
            self.parent[rid] = rid
            self.rank[rid] = 0

        # Build signal index from existing data
        for _, row in existing_df.iterrows():
            rid = row["review_id"]
            for col in self.signal_index:
                val = row[col]
                if pd.notna(val):
                    self.signal_index[col][val].add(rid)

        # Merge existing edges (replay the graph)
        for u, v in G.edges():
            self.union(u, v)

        self._next_id = int(existing_df["review_id"].max()) + 1

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return rx
        # Union by rank
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return rx

    def ingest(self, new_review: dict) -> dict:
        """
        Ingest one new review. Returns the cluster it joined (or new cluster).
        new_review must have: device_fingerprint, ip_block, payment_instrument,
                              review_id (or None for auto-assign)
        """
        rid = new_review.get("review_id", self._next_id)
        self._next_id = max(self._next_id, rid + 1)
        self.parent[rid] = rid
        self.rank[rid] = 0

        merged_into = set()

        # Check each signal for existing matches
        for col in self.signal_index:
            val = new_review.get(col)
            if val and val in self.signal_index[col]:
                for existing_rid in self.signal_index[col][val]:
                    self.union(rid, existing_rid)
                    merged_into.add(self.find(rid))

        # Add to signal index
        for col in self.signal_index:
            val = new_review.get(col)
            if val:
                self.signal_index[col][val].add(rid)

        # Find final component
        root = self.find(rid)
        component = [k for k, v in self.parent.items() if self.find(k) == root]

        return {
            "review_id": rid,
            "cluster_root": root,
            "cluster_size": len(component),
            "cluster_members": component,
            "merged": len(merged_into) > 0,
        }


linker = IncrementalLinker(df, SIGNALS)
print(f"  Union-Find initialized: {len(linker.parent)} entities")

# ──────────────────────────────────────────────
# 5. SAVE
# ──────────────────────────────────────────────
df.to_csv("clusters.csv", index=False)
print(f"\n✓ Saved clusters.csv — shape: {df.shape}")

# Save graph state for Phase 5/6
graph_state = {
    "graph": G,
    "edge_signals": dict(edge_signals),
    "linker": linker,
    "clusters": clusters,
    "signals_config": SIGNALS,
}
with open("graph_state.pkl", "wb") as f:
    pickle.dump(graph_state, f)
print("✓ Saved graph_state.pkl")

# ──────────────────────────────────────────────
# 6. ACCEPTANCE CHECK: RING = ONE COMPONENT?
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("ACCEPTANCE CHECK: Ring Resolution")
print("=" * 60)

RING_DEVICE = "DEV-A7F3E9C1-FAKE-RING"
ring = df[df["device_fingerprint"] == RING_DEVICE]
ring_cluster_ids = ring["cluster_id"].unique()

print(f"\nRing members: {len(ring)}")
print(f"Distinct cluster IDs among ring members: {ring_cluster_ids}")

if len(ring_cluster_ids) == 1 and ring_cluster_ids[0] >= 0:
    cid = ring_cluster_ids[0]
    cluster_members = df[df["cluster_id"] == cid]
    print(f"\n✅ PASS: Ring is ONE cluster (ID={cid})")
    print(f"  Cluster size: {len(cluster_members)}")
    print(f"  Signal diversity: {cluster_members['cluster_signal_diversity'].iloc[0]}")
    print(f"  All ring members in cluster? "
          f"{set(ring['review_id']) <= set(cluster_members['review_id'])}")

    # Check for false inclusions
    non_ring = cluster_members[cluster_members["device_fingerprint"] != RING_DEVICE]
    print(f"  Non-ring members pulled in: {len(non_ring)}")
    if len(non_ring) > 0:
        print(f"    (IDs: {list(non_ring['review_id'].values[:10])})")
else:
    print(f"\n❌ FAIL: Ring fragmented into {len(ring_cluster_ids)} clusters!")
    print("  This needs debugging before Phase 5.")

# ──────────────────────────────────────────────
# 7. INCREMENTAL TEST
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("INCREMENTAL INGESTION TEST")
print("=" * 60)

# Simulate ingesting a new ring member
test_review = {
    "review_id": 99999,
    "device_fingerprint": RING_DEVICE,
    "ip_block": "198.51.100.0/24",
    "payment_instrument": "PAY-RING-4829",
}
result = linker.ingest(test_review)
print(f"\nIngested test review 99999 (ring device + IP + payment)")
print(f"  Merged into existing cluster? {result['merged']}")
print(f"  Cluster size after merge: {result['cluster_size']}")
print(f"  Expected: merged=True, size=8 (7 original + 1 new)")

if result["merged"] and result["cluster_size"] >= 8:
    print("\n✅ INCREMENTAL LINKING WORKS")
else:
    print("\n⚠️  Incremental linking may need adjustment")

print("\n✅ PHASE 4 COMPLETE — clusters.csv + graph_state.pkl ready for Phase 5")