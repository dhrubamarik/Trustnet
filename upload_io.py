"""upload_io.py — validation and normalisation for Upload & Scan.

Takes the raw uploaded bytes, decides whether the file is a review table at
all, maps whatever column names the user chose onto TrustNet's canonical
columns, drops rows that cannot be scored, and returns a JSON-safe report
describing everything understood, guessed, or discarded.

    import upload_io
    prep = upload_io.prepare(raw, filename)
    if not prep["ok"]:
        return prep["report"]                      # already has http_status
    df  = prep["df"]                               # 9 canonical columns, cleaned
    out.update(prep["report"])                     # merge the audit trail
"""

from __future__ import annotations

import csv
import io
import json
import re

import pandas as pd

MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 2000
MAX_TEXT_CHARS = 8000
MIN_TEXT_CHARS = 5
REJECT_SAMPLE = 25

CANONICAL = [
    "text_", "rating", "seller_id", "account_id", "account_age_hours",
    "device_fingerprint", "ip_block", "payment_instrument", "timestamp",
]

ALIASES = {
    "text_": ["text_", "text", "review_text", "reviewtext", "review", "content",
              "body", "comment", "comments", "review_body", "reviewbody",
              "review_content", "description", "feedback", "sentence", "message"],
    "rating": ["rating", "stars", "star", "score", "review_rating",
               "reviewrating", "star_rating", "num_stars", "overall",
               "overall_rating", "rate"],
    "seller_id": ["seller_id", "sellerid", "seller", "store_id", "storeid",
                  "store", "vendor_id", "vendorid", "vendor", "merchant_id",
                  "merchantid", "merchant", "shop_id", "shopid", "shop"],
    "account_id": ["account_id", "accountid", "account", "user_id", "userid",
                   "user", "reviewer_id", "reviewerid", "reviewer", "author_id",
                   "authorid", "author", "customer_id", "customerid",
                   "profile_id", "username"],
    "account_age_hours": ["account_age_hours", "accountagehours", "account_age",
                          "account_age_hrs", "age_hours", "agehours",
                          "account_age_days", "age_days", "agedays",
                          "account_age_minutes", "age_minutes"],
    "device_fingerprint": ["device_fingerprint", "devicefingerprint", "device_id",
                           "deviceid", "device", "fingerprint", "device_hash",
                           "hardware_id", "hardwareid"],
    "ip_block": ["ip_block", "ipblock", "ip", "ip_address", "ipaddress",
                 "ipv4", "ip_subnet", "subnet", "client_ip"],
    "payment_instrument": ["payment_instrument", "paymentinstrument", "payment_id",
                           "paymentid", "payment", "card_id", "cardid", "card",
                           "card_hash", "cardhash", "payment_method",
                           "paymentmethod", "payment_hash"],
    "timestamp": ["timestamp", "time", "date", "datetime", "created_at",
                  "createdat", "review_date", "reviewdate", "review_time",
                  "posted_at", "postedat", "review_timestamp", "date_time"],
}

ID_COLS = ["seller_id", "account_id", "device_fingerprint", "ip_block",
           "payment_instrument"]

_NULLS = {"nan", "none", "null", "n/a", "na", "-", "--", ""}

_UNITS = {
    "account_age_days": 24.0,
    "age_days": 24.0,
    "agedays": 24.0,
    "account_age_minutes": 1.0 / 60.0,
    "age_minutes": 1.0 / 60.0,
}


def _key(s):
    k = re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower())
    return re.sub(r"_+", "_", k).strip("_")


_ALIAS_INDEX = {}
for _canon, _names in ALIASES.items():
    for _n in _names:
        _ALIAS_INDEX[_key(_n)] = _canon

_TEXT_KEYS = {_key(n) for n in ALIASES["text_"]}


def _num(v, nd=2):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return round(f, nd)


def _new_report(filename, raw):
    return {
        "ok": False,
        "error_code": None,
        "message": None,
        "hints": [],
        "http_status": 400,
        "file": {
            "name": filename or "",
            "bytes": len(raw or b""),
            "format": None,
            "encoding": None,
            "delimiter": None,
            "rows_in_file": 0,
            "columns_in_file": 0,
        },
        "column_mapping": {},
        "unused_columns": [],
        "confidence": None,
        "rows": {"parsed": 0, "accepted": 0, "rejected": 0},
        "rejected": [],
        "warnings": [],
        "profile": {},
        "guidance": [],
    }


def prepare(raw, filename=""):
    """Validate and normalise an uploaded file.

    Returns {"ok": bool, "df": DataFrame|None, "report": dict}.
    The report always carries "http_status" and is safe to serialise.
    """
    rep = _new_report(filename, raw)
    try:
        return _prepare(raw, filename, rep)
    except Exception as exc:
        rep.update(
            ok=False,
            error_code="parse_failed",
            http_status=400,
            message=f"Could not read this file ({type(exc).__name__}: {exc}).",
            hints=["Export as a plain CSV with a header row.",
                   "From Excel: File > Save as > CSV UTF-8."],
        )
        return {"ok": False, "df": None, "report": rep}


def _fail(rep, code, status, message, hints=None):
    rep.update(ok=False, error_code=code, http_status=status,
               message=message, hints=hints or [])
    return {"ok": False, "df": None, "report": rep}


def _prepare(raw, filename, rep):
    if not raw:
        return _fail(rep, "empty_file", 400, "The uploaded file is empty.",
                     ["Choose a CSV file that contains at least one review."])

    if len(raw) > MAX_BYTES:
        mb = len(raw) / (1024 * 1024)
        return _fail(rep, "too_large", 413,
                     f"File is {mb:.1f} MB. The limit is 5 MB.",
                     ["Split the file and scan it in batches."])

    head = raw[:4096]
    if head[:4] == b"PK\x03\x04":
        return _fail(rep, "not_a_table", 415,
                     "This looks like an Excel (.xlsx) or ZIP file, not a CSV.",
                     ["Open it in Excel and use File > Save as > CSV UTF-8."])
    if head[:4] in (b"\xd0\xcf\x11\xe0", b"%PDF"):
        return _fail(rep, "not_a_table", 415,
                     "This looks like a PDF or legacy Excel file, not a CSV.",
                     ["Save it as CSV UTF-8 and upload that file."])
    if b"\x00" in head:
        return _fail(rep, "not_a_table", 415,
                     "This file contains binary data, not text.",
                     ["Upload a CSV, TSV or JSON file of reviews."])

    text, encoding = _decode(raw)
    if text is None:
        return _fail(rep, "not_a_table", 415,
                     "The file encoding could not be determined.",
                     ["Re-export as CSV UTF-8."])
    rep["file"]["encoding"] = encoding

    stripped = text.lstrip()
    skipped = False

    if stripped[:1] in ("{", "["):
        rep["file"]["format"] = "json"
        df = _read_json(text)
        if df is None:
            return _fail(rep, "parse_failed", 400,
                         "The file looks like JSON but could not be parsed as a "
                         "list of review objects.",
                         ["Upload CSV, or JSON shaped as a list of objects."])
    else:
        if stripped[:5].lower() in ("<html", "<?xml", "<!doc"):
            return _fail(rep, "not_a_table", 415,
                         "This looks like an HTML or XML page saved as a file.",
                         ["Open the page, copy the table, save it as UTF-8 CSV."])
        delimiter = _sniff(text)
        rep["file"]["delimiter"] = delimiter
        rep["file"]["format"] = "tsv" if delimiter == "\t" else "csv"
        df, skipped = _read_csv(text, delimiter)
        if df is None:
            return _fail(rep, "parse_failed", 400,
                         "The file could not be read as a table.",
                         ["Check that every row has the same number of columns."])
        if any(len(str(c)) > 40 for c in df.columns):
            df, _ = _read_csv(text, delimiter, header=None)
            if df is None:
                return _fail(rep, "parse_failed", 400,
                             "The file could not be read as a table.", [])
            df.columns = [f"column_{i + 1}" for i in range(df.shape[1])]
            rep["warnings"].append(
                "No header row detected — columns were named column_1, "
                "column_2, ...")

    if skipped:
        rep["warnings"].append(
            "Some rows had a different number of columns and were skipped.")
    if df is None or df.shape[0] == 0:
        return _fail(rep, "empty_table", 400,
                     "The file has a header but no data rows.",
                     ["Add at least one review row and upload again."])
    if df.shape[0] > MAX_ROWS:
        return _fail(rep, "too_many_rows", 400,
                     f"The file has {df.shape[0]} rows. The limit is "
                     f"{MAX_ROWS} rows.",
                     [f"Scan the first {MAX_ROWS} rows, then the next batch.",
                      "The limit keeps the scan fast enough to stay "
                      "interactive."])

    rep["file"]["rows_in_file"] = int(df.shape[0])
    rep["file"]["columns_in_file"] = int(df.shape[1])

    mapping, unused = _map_columns(df, rep)
    age_src = next((k for k, v in mapping.items()
                    if v == "account_age_hours"), None)
    age_unit = _UNITS.get(_key(age_src), 1.0) if age_src else 1.0
    df = df.rename(columns=mapping)

    if "text_" not in df.columns:
        guess = _guess_text_column(df)
        if guess is None:
            found = [str(c) for c in list(df.columns)[:12]]
            return _fail(
                rep, "no_text_column", 400,
                "No review text column was found, so nothing can be scored.",
                [f"Columns present: {', '.join(found) or 'none'}.",
                 "Rename the review column to text_, text or review_text.",
                 "The engine runs entirely on review text — it cannot proceed "
                 "without it."])
        df = df.rename(columns={guess: "text_"})
        mapping[guess] = "text_"
        rep["warnings"].append(
            f"No recognised text column name — used '{guess}' because it holds "
            "the longest values in the file.")

    rep["column_mapping"] = mapping
    rep["unused_columns"] = unused
    rep["confidence"] = _confidence_for(mapping)

    clean = _clean(df, age_unit)
    _med = float(clean["text_"].str.len().median())
    _pct = float(clean["text_"].str.contains(r"[.!?]", regex=True).mean())
    if _med < 40 or _pct < 0.35:
        rep["warnings"].append(
            "This does not look like review text: median length %d characters, "
            "only %.0f%% contain sentence punctuation. The scan will still run, "
            "but treat the result as unvalidated." % (_med, 100 * _pct))
    kept, reason = _triage(clean)

    rep["rows"] = {"parsed": int(len(clean)),
                   "accepted": int(kept.sum()),
                   "rejected": int((~kept).sum())}
    rep["rejected"] = [
        {"row": int(i) + 2, "reason": str(r)}
        for i, r in reason[~kept].head(REJECT_SAMPLE).items()
    ]

    if kept.sum() == 0:
        counts = reason[reason != ""].value_counts().to_dict()
        return _fail(rep, "all_rows_invalid", 400,
                     "Every row was rejected before scoring.",
                     [f"{v} row(s): {k}" for k, v in counts.items()])

    rep["profile"] = _profile(clean[kept])
    rep["guidance"] = _guidance(rep)
    rep["ok"] = True
    rep["http_status"] = 200
    return {"ok": True, "df": clean[kept], "report": rep}


# --------------------------------------------------------------- readers ---

def _decode(raw):
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return None, None


def _sniff(text):
    sample = text[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except Exception:
        counts = {d: sample.count(d) for d in ",;\t|"}
        best = max(counts, key=counts.get)
        return best if counts[best] else ","


def _read_csv(text, delimiter, header=0):
    for mode in ("error", "skip"):
        try:
            df = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str,
                             keep_default_na=False, header=header,
                             engine="python", on_bad_lines=mode)
            return df, mode == "skip"
        except Exception:
            continue
    return None, False


def _read_json(text):
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if isinstance(obj, dict):
        for k in ("reviews", "data", "rows", "records", "items"):
            if isinstance(obj.get(k), list):
                obj = obj[k]
                break
    if not isinstance(obj, list) or not obj:
        return None
    rows = [r for r in obj if isinstance(r, dict)]
    if not rows:
        return None
    return pd.DataFrame(rows).astype(str)


# --------------------------------------------------------------- columns ---

def _map_columns(df, rep):
    mapping, used = {}, set()
    for col in df.columns:
        canon = _ALIAS_INDEX.get(_key(col))
        if canon and canon not in used:
            mapping[col] = canon
            used.add(canon)
    unused = [str(c) for c in df.columns if c not in mapping]
    if unused:
        rep["warnings"].append(
            f"{len(unused)} column(s) not used by the engine: "
            f"{', '.join(unused[:8])}{'…' if len(unused) > 8 else ''}.")
    return mapping, unused


def _guess_text_column(df):
    best, best_len = None, 0.0
    for col in df.columns:
        s = df[col].fillna("").astype(str)
        lens = s.str.len()
        if (lens > 15).mean() >= 0.5 and lens.mean() > best_len:
            best, best_len = col, float(lens.mean())
    return best if best_len >= 25 else None


def _confidence_for(mapping):
    text_src = next((k for k, v in mapping.items() if v == "text_"), None)
    named = text_src is not None and _key(text_src) in _TEXT_KEYS
    others = sum(1 for v in mapping.values() if v != "text_")
    if named and others >= 2:
        return "high"
    if named or others >= 2:
        return "medium"
    return "low"


# ----------------------------------------------------------------- clean ---

def _clean(df, age_unit=1.0):
    out = pd.DataFrame(index=df.index)

    txt = df["text_"].fillna("").astype(str).str.strip()
    txt = txt.str.replace(r"\s+", " ", regex=True)
    out["text_"] = txt.str.slice(0, MAX_TEXT_CHARS)

    for c in ID_COLS:
        if c in df.columns:
            s = df[c].fillna("").astype(str).str.strip()
            out[c] = s.mask(s.str.lower().isin(_NULLS), "")
        else:
            out[c] = ""

    if "rating" in df.columns:
        r = pd.to_numeric(
            df["rating"].fillna("").astype(str).str.replace(",", ".", regex=False),
            errors="coerce")
        if r.notna().any():
            rr = r.dropna()
            if rr.max() > 5 and rr.between(0, 10).mean() > 0.95:
                r = r / 2.0
        out["rating"] = r.where(r.between(0, 5)).astype(float)
    else:
        out["rating"] = float("nan")

    if "account_age_hours" in df.columns:
        a = pd.to_numeric(df["account_age_hours"].fillna("").astype(str),
                          errors="coerce") * age_unit
        out["account_age_hours"] = a.where(a >= 0).astype(float)
    else:
        out["account_age_hours"] = float("nan")

    if "timestamp" in df.columns:
        out["timestamp"] = _parse_time(df["timestamp"].fillna("").astype(str))
    else:
        out["timestamp"] = pd.NaT

    return out[CANONICAL]


def _parse_time(s):
    if len(s) == 0:
        return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.8:
        v = num.copy()
        if v.dropna().max() > 1e11:
            v = v / 1000.0
        try:
            return pd.to_datetime(v, unit="s", errors="coerce", utc=True)
        except Exception:
            return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")
    try:
        return pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):
        try:
            return pd.to_datetime(s, errors="coerce", utc=True)
        except Exception:
            return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")


def _triage(clean):
    txt = clean["text_"]
    reason = pd.Series("", index=clean.index, dtype=object)

    too_short = txt.str.len() < MIN_TEXT_CHARS
    reason[too_short] = f"text shorter than {MIN_TEXT_CHARS} characters"

    has_letter = txt.str.contains(r"[^\W\d_]", regex=True, na=False)
    bare = reason.eq("") & ~has_letter
    reason[bare] = "text contains no letters"

    return reason.eq(""), reason


# --------------------------------------------------------------- profile ---

def _profile(d):
    infra = {}
    for c in ID_COLS:
        filled = int((d[c] != "").sum())
        infra[c] = _num(filled / len(d), 3) if len(d) else 0.0

    sellers = d.loc[d["seller_id"] != "", "seller_id"]
    accounts = d.loc[d["account_id"] != "", "account_id"]

    ratings = None
    if d["rating"].notna().any():
        r = d["rating"].dropna()
        hist = {str(int(k)): int(v) for k, v in
                r.round().clip(0, 5).value_counts().sort_index().items()}
        ratings = {"n": int(r.shape[0]), "mean": _num(r.mean()), "hist": hist}

    tr = None
    ts = d["timestamp"].dropna()
    if ts.shape[0] >= 2:
        span = (ts.max() - ts.min()).total_seconds() / 3600.0
        tr = {"from": ts.min().isoformat(), "to": ts.max().isoformat(),
              "span_hours": _num(span, 1)}

    lens = d["text_"].str.len()
    return {
        "rows": int(len(d)),
        "sellers": int(sellers.nunique()),
        "accounts": int(accounts.nunique()),
        "text_length": {"mean": _num(lens.mean(), 1),
                        "median": int(lens.median()),
                        "max": int(lens.max())},
        "ratings": ratings,
        "infrastructure_coverage": infra,
        "time_range": tr,
        "duplicate_rows": int(d.duplicated(subset=["text_", "seller_id"]).sum()),
    }


def _guidance(rep):
    out = []
    prof = rep.get("profile") or {}
    cov = prof.get("infrastructure_coverage") or {}
    mapped = set(rep.get("column_mapping", {}).values())

    hard = [c for c in ("device_fingerprint", "ip_block", "payment_instrument")
            if c in mapped]
    if not hard:
        out.append(
            "No device, IP or payment columns. Rings here can only be linked by "
            "text and timing. Adding any one of them makes ring detection much "
            "sharper.")
    elif min(cov.get(c, 0.0) for c in hard) < 0.30:
        out.append(
            "Infrastructure columns are mostly empty, so most rows will not "
            "link on hardware.")

    if prof.get("sellers", 0) < 3:
        out.append(f"Only {prof.get('sellers', 0)} distinct seller(s). "
                   "Cross-seller rings cannot be detected in this file.")

    if prof.get("ratings") is None:
        out.append("No usable rating column — rating-based signals are off.")
    if "account_age_hours" not in mapped:
        out.append("No account_age_hours column — the fresh-account signal is "
                   "unavailable.")
    if prof.get("duplicate_rows"):
        out.append(f"{prof['duplicate_rows']} row(s) repeat an earlier "
                   "text+seller pair. They were kept: duplicated content is "
                   "itself evidence of coordination.")
    if rep.get("rows", {}).get("rejected"):
        out.append(f"{rep['rows']['rejected']} row(s) were skipped before "
                   "scoring — see 'rejected' for the exact rows and reasons.")
    return out