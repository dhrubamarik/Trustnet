"""
TrustNet — Phase 8: Stress-Test Validation (v2)
================================================
Inject a NEW ring via /ingest, verify ONE case, ESCALATE tier,
within 10s latency budget. Run 10 rounds with fresh random seeds.

Run:  py -3.14 phase8_stress_test.py
"""

import requests
import random
import time
import string
from datetime import datetime

API = "http://127.0.0.1:8000"   # ★ IPv4 — avoids 2s IPv6 fallback on Windows
SESSION = requests.Session()

def random_id(prefix, length=10):
    return f"{prefix}-{''.join(random.choices(string.ascii_uppercase + string.digits, k=length))}"

def run_stress_test(ring_size=6, seed=None, verbose=True):
    if seed:
        random.seed(seed)

    ring_device = random_id("DEV-HELDOUT")
    ring_ip = f"{random.randint(1, 223)}.{random.randint(0,255)}.{random.randint(0,255)}.0/24"
    ring_payment = random_id("PAY-HELDOUT")
    ring_sellers = [f"SELLER-{i:04d}" for i in random.sample(range(1, 300), 3)]

    if verbose:
        print(f"\n{'='*60}\nSTRESS TEST — Seed {seed}\n{'='*60}")
        print(f"  Ring device:  {ring_device}")
        print(f"  Ring IP:      {ring_ip}")
        print(f"  Ring payment: {ring_payment}")
        print(f"  Ring sellers: {ring_sellers}\n")

    # BEFORE — all cases
    before = SESSION.get(f"{API}/cases?limit=99999").json()
    before_counts = {c["case_id"]: c["n_members"] for c in before}

    latencies = []
    case_ids_seen = set()

    for i in range(ring_size):
        payload = {
            "device_fingerprint": ring_device,
            "ip_block": ring_ip,
            "payment_instrument": ring_payment,
            "account_id": f"ACCT-HELDOUT-{i:03d}-{seed}",
            "account_age_hours": round(random.uniform(1.0, 10.0), 1),
            "rating": 5.0,
            "seller_id": ring_sellers[i % len(ring_sellers)],
            "text_": f"Amazing! Exceeded expectations! {random.choice(['Highly recommend', '5 stars all the way'])}",
            "timestamp": datetime.now().isoformat(),
        }

        t_start = time.time()
        resp = SESSION.post(f"{API}/ingest", json=payload).json()
        latency_ms = (time.time() - t_start) * 1000
        latencies.append(latency_ms)

        if resp.get("case_id"):
            case_ids_seen.add(resp["case_id"])

        if verbose:
            status = "✓" if resp["merged_into_cluster"] else "✗"
            print(f"  [{i+1}/{ring_size}] {status} cluster_size={resp['cluster_size']:>3} "
                  f"case={resp.get('case_id','—'):<20} "
                  f"tier={resp.get('tier','?'):<9} "
                  f"latency={latency_ms:>6.1f}ms")

    # AFTER — all cases
    time.sleep(0.1)
    after = SESSION.get(f"{API}/cases?limit=99999").json()
    after_counts = {c["case_id"]: c["n_members"] for c in after}

    # Identify the case that absorbed the ring
    absorbed_case = None
    for cid, after_n in after_counts.items():
        before_n = before_counts.get(cid, 0)
        if after_n >= before_n + ring_size - 1:  # ring_size - 1 merged (first was isolated)
            absorbed_case = cid
            break

    escalated_case = next((c for c in after if c["case_id"] == absorbed_case), None)

    # Checks
    case_found = absorbed_case is not None
    fragmentation_ok = len(case_ids_seen) == 1
    tier_ok = escalated_case is not None and escalated_case["tier"] == "ESCALATE"
    latency_ok = max(latencies) < 10_000
    size_ok = escalated_case is not None and escalated_case["n_members"] >= ring_size

    if verbose and escalated_case:
        print(f"\n  --- RESULTS ---")
        print(f"  Absorbed into:  {absorbed_case}")
        print(f"  Cluster size:   {escalated_case['n_members']}")
        print(f"  Signal div:     {escalated_case['signal_diversity']}")
        print(f"  Tier:           {escalated_case['tier']}")
        print(f"  Score:          {escalated_case['case_score']}")
        print(f"\n  --- LATENCY ---")
        print(f"  Avg:  {sum(latencies)/len(latencies):.1f}ms  Max: {max(latencies):.1f}ms")

    checks = {
        "Ring resolved":      case_found,
        "No fragmentation":   fragmentation_ok,
        "Tier = ESCALATE":    tier_ok,
        "All members in":     size_ok,
        "Latency < 10s":      latency_ok,
    }
    passed = all(checks.values())

    if verbose:
        print(f"\n  --- CHECKS ---")
        for check, ok in checks.items():
            print(f"  {'✅' if ok else '❌'}  {check}")
        print(f"\n  {'🎉 PASS' if passed else '🚨 FAIL'}")

    return {"passed": passed, "checks": checks, "latencies": latencies}

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 8: Stress-Test Validation")
    print("=" * 60)

    results = []
    for i in range(10):
        seed = random.randint(1, 100_000)
        try:
            r = run_stress_test(ring_size=6, seed=seed, verbose=(i == 0))
            results.append(r)
            s = "✅" if r["passed"] else "🚨"
            avg = sum(r["latencies"]) / len(r["latencies"])
            print(f"  Round {i+1:>2} (seed {seed:>6}): {s} (avg {avg:.1f}ms)")
        except Exception as e:
            print(f"  Round {i+1:>2} (seed {seed:>6}): 🚨 ERROR — {e}")
            results.append({"passed": False, "latencies": [10000]})

    print(f"\n{'='*60}\nFINAL REPORT\n{'='*60}")
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    rates = {
        "Detection":     passed / total * 100,
    }
    all_lat = [l for r in results for l in r["latencies"]]
    print(f"  Rounds:             {total}")
    print(f"  Passed:             {passed}")
    print(f"  Ring detection:     {rates['Detection']:.0f}%  (target 100%)")
    print(f"  Avg latency:        {sum(all_lat)/len(all_lat):.1f}ms")
    print(f"  Max latency:        {max(all_lat):.1f}ms")
    print(f"  Fragmentation rate: {sum(1 for r in results if not r['checks'].get('No fragmentation', False)) / total * 100:.0f}%  (target 0%)")

    if passed == total:
        print(f"\n  🎉🎉🎉 100% DETECTION RATE — DEMO READY 🎉🎉🎉")
    else:
        print(f"\n  ⚠️  {total - passed} rounds failed")