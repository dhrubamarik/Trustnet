"""
bench_components.py — decompose TrustNet ingest throughput.

Separates fixed transport overhead from true engine cost by comparing
per-review HTTP calls against a single batched call.

Requires phase6_api.py on http://127.0.0.1:8000
    py -3.14 bench_components.py
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
SEQ_N = 200
BATCH_N = 500


def req(path, body=None):
    data = json.dumps(body or {}).encode("utf-8")
    r = urllib.request.Request(BASE + path, data=data,
                               headers={"Content-Type": "application/json"},
                               method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        sys.exit("API unreachable at %s — start phase6_api.py (%s)" % (BASE, exc))
    return payload, (time.perf_counter() - t0) * 1000.0


def main():
    print("warming up…")
    for _ in range(20):
        req("/api/inject-member")
    req("/api/reset")

    print("\n[1] sequential per-review HTTP — %d calls" % SEQ_N)
    lat, t0 = [], time.perf_counter()
    for _ in range(SEQ_N):
        _, ms = req("/api/inject-member")
        lat.append(ms)
    seq_elapsed = time.perf_counter() - t0
    seq_rps = SEQ_N / seq_elapsed
    lat.sort()
    print("    throughput   %.1f reviews/sec" % seq_rps)
    print("    mean / med   %.2f / %.2f ms" % (sum(lat) / len(lat), lat[len(lat) // 2]))
    print("    p95 / min    %.2f / %.2f ms" % (lat[int(.95 * len(lat))], lat[0]))

    req("/api/reset")

    print("\n[2] single batched call — %d reviews" % BATCH_N)
    body, total_ms = req("/api/ingest/batch", {"n": BATCH_N, "tag": "BENCH"})
    if not body.get("ok"):
        sys.exit("batch failed: %s" % body.get("error"))
    batch_rps = body["reviews_per_sec"]
    per_rev = body["per_review_ms"]
    print("    throughput   %.1f reviews/sec" % batch_rps)
    print("    per review   %.2f ms" % per_rev)
    print("    members      %d in %s" % (body["member_count"], body["case_id"]))

    req("/api/reset")

    overhead = (1000.0 / seq_rps) - per_rev
    print("\n" + "=" * 62)
    print("  DECOMPOSITION")
    print("  engine cost (scoring + link + merge + tier)  %.2f ms" % per_rev)
    print("  fixed transport / API overhead               %.2f ms" % max(overhead, 0))
    print("  sequential end-to-end                        %.1f rev/sec" % seq_rps)
    print("  batched end-to-end                           %.1f rev/sec" % batch_rps)
    print("=" * 62)
    print("\nbaseline restored.")


if __name__ == "__main__":
    main()