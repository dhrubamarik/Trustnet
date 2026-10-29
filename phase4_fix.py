"""
TrustNet — Phase 4.1: Fix Mega-Cluster (v3 — uses linker.py)
"""

import pandas as pd
import numpy as np
import networkx as nx
import pickle
from collections import defaultdict
from linker import IncrementalLinker  # ← shared module

print("=" * 60)
print("PHASE 4.1: Fixing Mega-Cluster")
print("=" * 60)

with open("graph_state.pkl", "rb") as f:
    state = pickle.load(f)

G = state["graph"]
edge_signals = state["edge_signals"]
SIGNALS = state["signals_config"]

df = pd.read_csv("scored_features.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"\n✓ Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# ── Filter edges ──
MIN_EDGE_WEIGHT = 4.0
G_filtered = nx.Graph()
G_filtered.add_nodes_from(G.nodes())

edges_kept = 0
edges_removed = 0
for u, v, data in G.edges(data=True):
    if data["weight"] >= MIN_EDGE_WEIGHT:
        G_filtered.add_edge(u, v, weight=data["weight"])
        edges_kept += 1
    else:
        edges_removed += 1

print(f"\nEdge filtering (min_weight={MIN_EDGE_WEIGHT}):")
print(f"  Kept:    {edges_kept} (multi-signal)")
print(f"  Removed: {edges_removed} (single-signal)")

# ── Connected components ──
components = list(nx.connected_components(G_filtered))
clusters = [c for c in components if len(c) >= 2]

review_to_cluster = {}
for cid, cluster in enumerate(clusters):
    for rid in cluster:
        review_to_cluster[rid] = cid

df["cluster_id"] = df["review_id"].map(review_to_cluster).fillna(-1).astype(int)
df["cluster_size"] = df["cluster_id"].map(
    {cid: len(c) for cid, c in enumerate(clusters)}
).fillna(1).astype(int)

cluster_diversity = {}
for cid, cluster in enumerate(clusters):
    signals_in_cluster = set()
    nodes = list(cluster)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            pair = (nodes[i], nodes[j])
            if pair in edge_signals:
                signals_in_cluster.update(edge_signals[pair])
    cluster_diversity[cid] = len(signals_in_cluster)

df["cluster_signal_diversity"] = df["cluster_id"].map(cluster_diversity).fillna(0).astype(int)

print(f"\nFixed components:")
print(f"  Total clusters (size >= 2): {len(clusters)}")
if clusters:
    sizes = [len(c) for c in clusters]
    print(f"  Largest cluster: {max(sizes)} members")
    print(f"  Reviews in clusters: {(df['cluster_id'] >= 0).sum()} / {len(df)}")

# ── Rebuild linker ──
print("\nRebuilding incremental linker...")
linker = IncrementalLinker(df, G_filtered)
print(f"  Linker ready: {len(linker.parent)} entities")

# ── Save ──
df.to_csv("clusters.csv", index=False)
print(f"\n✓ Saved clusters.csv")

state["graph"] = G_filtered
state["linker"] = linker
state["clusters"] = clusters
with open("graph_state.pkl", "wb") as f:
    pickle.dump(state, f)
print("✓ Saved graph_state.pkl (now uses linker.IncrementalLinker)")

# ── Acceptance ──
print("\n" + "=" * 60)
print("ACCEPTANCE CHECK")
print("=" * 60)

RING_DEVICE = "DEV-A7F3E9C1-FAKE-RING"
ring = df[df["device_fingerprint"] == RING_DEVICE]
ring_cids = ring["cluster_id"].unique()
print(f"\nRing cluster IDs: {ring_cids}")

if len(ring_cids) == 1 and ring_cids[0] >= 0:
    cid = ring_cids[0]
    cluster = df[df["cluster_id"] == cid]
    non_ring = cluster[cluster["device_fingerprint"] != RING_DEVICE]
    print(f"✅ Ring = ONE cluster (size={len(cluster)}, "
          f"diversity={cluster['cluster_signal_diversity'].iloc[0]})")
    print(f"   False inclusions: {len(non_ring)}")

test = {
    "review_id": 99999,
    "device_fingerprint": RING_DEVICE,
    "ip_block": "198.51.100.0/24",
    "payment_instrument": "PAY-RING-4829",
}
result = linker.ingest(test)
print(f"\nIncremental test: merged={result['merged']}, size={result['cluster_size']}")
print("\n✅ PHASE 4.1 COMPLETE")