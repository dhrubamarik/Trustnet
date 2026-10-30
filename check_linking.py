"""
TrustNet — linking regression check.

Verifies the linker does two opposite things:
  1. every ring recipe collapses into ONE cluster
  2. unrelated reviews with no shared infrastructure never link

Also loads graph_state.pkl to prove the pickle still unpickles after the
linker rewrite, and prints baseline cluster counts (should be 2 components,
largest 7).

Run:  py -3.14 check_linking.py
"""

import pickle
import random

from linker import IncrementalLinker

TEMPLATE = ("UNBELIEVABLE 10/10 MUST BUY!!! Absolute game changer, best "
            "purchase I have ever made. Highly recommend!!!")

PRESETS = [
    # name            n   shared                     spacing   sellers
    ("classic",       8,  ["device", "ip", "payment"],      30,  1),
    ("two_signals",   8,  ["device", "ip"],                 30,  1),
    ("one_signal",    8,  ["device"],                       30,  1),
    ("no_hardware",   8,  [],                               30,  1),
    ("slow_spread",   6,  [],                           43200,  1),
    ("cross_seller",  9,  ["device", "ip"],                 30,  3),
]


def run_preset(name, n, share, spacing_s, nsellers):
    lk = IncrementalLinker()
    tag = "%s%d" % (name.upper(), random.randint(1000, 9999))
    device = "DEV-" + tag
    payment = "PAY-" + tag
    ip = "192.0.2.%d/24" % random.randint(1, 254)
    sellers = ["SELLER-%04d" % (260 + i) for i in range(nsellers)]
    t0 = 1_700_000_000.0

    rids = []
    for i in range(n):
        rid = 900000 + i
        rids.append(rid)
        lk.ingest({
            "review_id": rid,
            "device_fingerprint": device if "device" in share else None,
            "payment_instrument": payment if "payment" in share else None,
            "ip_block": ip if "ip" in share else None,
            "seller_id": sellers[i % nsellers],
            "text_": TEMPLATE,
            "account_age_hours": 2.0 + 0.1 * i,
            "timestamp": t0 + i * spacing_s,
        })

    roots = {lk.find(r) for r in rids}
    sizes = sorted((len(lk._comp[r]) for r in roots), reverse=True)
    return len(roots), sizes


def check_unrelated():
    """8 reviews, no shared infrastructure, different sellers and text."""
    lk = IncrementalLinker()
    for i in range(8):
        lk.ingest({
            "review_id": 800000 + i,
            "seller_id": "SELLER-%04d" % (400 + i),
            "text_": "Product %d arrived on time and works as described." % i,
            "account_age_hours": 900.0 + i,
            "timestamp": 1_700_000_000.0 + i * 86400,
        })
    return lk.stats()


def main():
    print("=" * 72)
    print("RING RECIPES")
    print("=" * 72)
    ok = True
    for name, n, share, spacing, nsellers in PRESETS:
        comps, sizes = run_preset(name, n, share, spacing, nsellers)
        verdict = "PASS" if comps == 1 and sizes[0] == n else "FAIL"
        if verdict == "FAIL":
            ok = False
        print("%-14s %d reviews -> %d component(s) %-22s %s"
              % (name, n, comps, str(sizes), verdict))

    print()
    print("=" * 72)
    print("NEGATIVE CONTROL — unrelated reviews must not link")
    print("=" * 72)
    stats = check_unrelated()
    verdict = "PASS" if stats["components"] == 8 and stats["largest"] == 1 else "FAIL"
    if verdict == "FAIL":
        ok = False
    print("8 unrelated reviews -> %d components, largest %d   %s"
          % (stats["components"], stats["largest"], verdict))

    print()
    print("=" * 72)
    print("BASELINE PICKLE")
    print("=" * 72)
    try:
        with open("graph_state.pkl", "rb") as f:
            state = pickle.load(f)
        base = state["linker"]
        print("graph_state.pkl loads:", base.stats())
    except Exception as e:
        ok = False
        print("graph_state.pkl FAILED:", type(e).__name__, e)

    print()
    print("OVERALL:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()