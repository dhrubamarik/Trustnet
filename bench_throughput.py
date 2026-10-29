"""
bench_throughput.py — honest sustained-throughput measurement for TrustNet.

Requires phase6_api.py running on http://127.0.0.1:8000
Restores the clean baseline when it finishes.

    py -3.14 bench_throughput.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
N = 200
WARMUP = 25


def post(path, body=None):
    data = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        sys.exit("API unreachable at %s — start phase6_api.py (%s)" % (BASE, exc))
    return payload, (time.perf_counter() - t0) * 1000.0


def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main():
    post("/api/reset")

    print("warm-up: %d ingests (discarded)" % WARMUP)
    for _ in range(WARMUP):
        post("/api/inject-member")
    post("/api/reset")

    print("measuring: %d sequential ingest round-trips" % N)
    lat = []
    t0 = time.perf_counter()
    for i in range(N):
        _, ms = post("/api/inject-member")
        lat.append(ms)
        if (i + 1) % 50 == 0:
            print("  %d / %d" % (i + 1, N))
    elapsed = time.perf_counter() - t0

    print()
    print("RESULTS — %d sequential HTTP ingests, one at a time" % N)
    print("  elapsed     %.2f s" % elapsed)
    print("  throughput  %.1f reviews/sec" % (N / elapsed))
    print("  mean        %.2f ms" % statistics.mean(lat))
    print("  median      %.2f ms" % statistics.median(lat))
    print("  p95         %.2f ms" % pct(lat, 95))
    print("  p99         %.2f ms" % pct(lat, 99))
    print("  min / max   %.2f / %.2f ms" % (min(lat), max(lat)))
    print()
    print("Scope: network + FastAPI + scoring + union-find link +")
    print("case merge + tier re-evaluation. End-to-end over HTTP.")
    print("Do not compare this to in-process scoring benchmarks.")

    post("/api/reset")
    print("\nbaseline restored — CASE-0000 back to 7 members.")


if __name__ == "__main__":
    main()