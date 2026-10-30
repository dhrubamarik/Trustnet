"""verify_demo.py — one command, every demo number checked.

    py -3.14 verify_demo.py            read-only
    py -3.14 verify_demo.py --reset    restore baseline first, then check

Run immediately before presenting. Non-zero exit = something drifted.
"""

import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8000"
RESET = "--reset" in sys.argv

passed = 0
failed = 0


def _req(path, payload=None):
    if payload is None:
        req = urllib.request.Request(BASE + path)
    else:
        req = urllib.request.Request(
            BASE + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode("utf-8"))


def dig(obj, *keys, default=None):
    cur = obj
    for k in keys:
        if isinstance(cur, dict):
            if k not in cur:
                return default
            cur = cur[k]
        else:
            return default
    return cur


def check(label, got, want, cmp="eq"):
    global passed, failed
    if cmp == "ge":
        ok = isinstance(got, (int, float)) and got >= want
    elif cmp == "between":
        lo, hi = want
        ok = isinstance(got, (int, float)) and lo <= got <= hi
    else:
        ok = got == want
    if ok:
        passed += 1
    else:
        failed += 1
    print("  [%s] %-38s got=%r want=%r" % ("PASS" if ok else "FAIL", label, got, want))


def main():
    print("TrustNet demo verification -> %s\n" % BASE)

    try:
        _req("/api/status")
    except Exception as exc:
        print("  [FAIL] API unreachable: %s" % exc)
        print("\nStart phase6_api.py first.")
        return 2

    if RESET:
        print("  ... POST /api/reset")
        _req("/api/reset", {})
        print("  ... POST /api/decisions/clear")
        _req("/api/decisions/clear", {})
    print()

    cases = _req("/api/cases")
    check("cases total", dig(cases, "total"), 1627)

    ov = _req("/api/overview")
    tc = dig(ov, "tier_counts", default={}) or {}
    check("escalated cases", tc.get("Escalate"), 1)
    check("cases in review", tc.get("Review"), 45)
    check("watchlist cases", tc.get("Watch"), 1581)

    c0 = _req("/api/case/CASE-0000")
    members = (dig(c0, "review_count") or dig(c0, "n_members")
               or len(dig(c0, "members", default=[]) or []))
    check("CASE-0000 members", members, 7)
    check("CASE-0000 score", dig(c0, "case_score_100"), 81)

    ev = _req("/api/evaluation")
    check("ring_linking_rate", dig(ev, "ring_linking_rate"), 1)
    check("fragmentation_rate", dig(ev, "fragmentation_rate"), 0)
    check("text model AUC", dig(ev, "text_model", "auc"), 0.9848)

    presets = dig(_req("/api/judge-lab/presets"), "presets", default=[]) or []
    check("judge lab presets", len(presets), 6)

    st = _req("/api/sellers/trust?limit=10")
    check("sellers in ledger", dig(st, "n_sellers"), 300)
    riskiest = dig(st, "top10_riskiest", default=[]) or []
    check("top10 populated", len(riskiest), 10)
    if riskiest:
        lowest = min(int(r.get("trust_score", 100)) for r in riskiest)
        check("riskiest trust in range", lowest, (0, 55), cmp="between")
        adjusted = [r for r in riskiest if r.get("moderation_adjustment")]
        print("  [info] top-10 sellers carrying a moderator adjustment: %d"
              % len(adjusted))

    dec = _req("/api/decisions")
    check("decisions cleared", dig(dec, "n_decided"), 0)
    print("  [info] decisions on record: %s"
          % (dig(dec, "counts", default={}),))

    print("\n%d passed, %d failed" % (passed, failed))
    if failed:
        print("\nBaseline drift? Run with --reset.")
        print("CASE-0000 at 8 members means a live injection is still in state.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())