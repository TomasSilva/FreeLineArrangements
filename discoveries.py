"""
discoveries.py

Persistent JSON-based log of discovered free line arrangements.

Each entry records the full line coordinates (exact rational), exponents,
combinatorial invariants, and metadata (timestamp, source command, etc.).
Entries are deduplicated by a canonical key so the same arrangement is
never stored twice even across multiple runs.

File format:  a single JSON object with:
  - "arrangements": list of discovery records
  - "index": dict mapping canonical keys to list indices (for fast dedup)
"""

import json
import os
import time
from datetime import datetime, timezone


DEFAULT_PATH = "discoveries.json"


# ── Canonical key ─────────────────────────────────────────────────────────────

def _canonical_key(lines_str, exponents):
    """
    Deduplplication key for an arrangement.

    Uses the sorted line representations + exponents so that two
    arrangements that are identical up to line ordering are considered
    the same.
    """
    sorted_lines = tuple(sorted(lines_str))
    return f"{sorted_lines}|{exponents}"


# ── Load / save ───────────────────────────────────────────────────────────────

def _load(path):
    if not os.path.exists(path):
        return {"arrangements": [], "index": {}}
    with open(path, "r") as f:
        data = json.load(f)
    # Rebuild index if missing (backward compat)
    if "index" not in data:
        data["index"] = {}
        for i, rec in enumerate(data["arrangements"]):
            key = _canonical_key(rec["lines"], rec.get("exponents"))
            data["index"][key] = i
    return data


def _save(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ── Public API ────────────────────────────────────────────────────────────────

def log_discovery(
    lines,
    exponents,
    b2=None,
    n=None,
    max_mult=None,
    mult_profile=None,
    n_pts=None,
    source=None,
    target_exponents=None,
    path=DEFAULT_PATH,
):
    """
    Save a single free arrangement to the discovery log.

    Args:
        lines: list of str representations of the lines.
        exponents: tuple like (1, d1, d2).
        b2: second Betti number / t2 value.
        n: number of lines (derived from lines if None).
        max_mult: maximum intersection multiplicity.
        mult_profile: sorted list of multiplicities.
        n_pts: number of intersection points.
        source: string tag for how it was found (e.g. "train", "explore").
        path: file path for the JSON log.

    Returns:
        True if this is a new discovery, False if it was already known.
    """
    lines_str = [str(l) for l in lines] if not isinstance(lines[0], str) else list(lines)
    exponents = tuple(exponents) if exponents else None

    key = _canonical_key(lines_str, exponents)
    data = _load(path)

    if key in data["index"]:
        return False  # already known

    record = {
        "n": n or len(lines_str),
        "exponents": list(exponents) if exponents else None,
        "b2": b2,
        "max_mult": max_mult,
        "mult_profile": mult_profile,
        "n_pts": n_pts,
        "lines": lines_str,
        "source": source,
        "target_exponents": list(target_exponents) if target_exponents else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    idx = len(data["arrangements"])
    data["arrangements"].append(record)
    data["index"][key] = idx
    _save(data, path)
    return True


def log_discoveries(records, source=None, path=DEFAULT_PATH):
    """
    Batch-save multiple arrangements.  Returns count of new discoveries.
    Each record is a dict with at least 'lines' and 'exponents' keys.
    """
    data = _load(path)
    n_new = 0

    for rec in records:
        lines_str = (
            [str(l) for l in rec["lines"]]
            if not isinstance(rec["lines"][0], str)
            else list(rec["lines"])
        )
        exponents = tuple(rec["exponents"]) if rec.get("exponents") else None
        key = _canonical_key(lines_str, exponents)

        if key in data["index"]:
            continue

        entry = {
            "n": rec.get("n") or len(lines_str),
            "exponents": list(exponents) if exponents else None,
            "b2": rec.get("b2") or rec.get("t2"),
            "max_mult": rec.get("max_mult"),
            "mult_profile": rec.get("mult_profile"),
            "n_pts": rec.get("n_pts"),
            "lines": lines_str,
            "source": source or rec.get("source"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        idx = len(data["arrangements"])
        data["arrangements"].append(entry)
        data["index"][key] = idx
        n_new += 1

    if n_new > 0:
        _save(data, path)
    return n_new


def load_discoveries(path=DEFAULT_PATH):
    """Load all saved discoveries.  Returns list of record dicts."""
    data = _load(path)
    return data["arrangements"]


def summary(path=DEFAULT_PATH):
    """Print a summary of all saved discoveries."""
    records = load_discoveries(path)
    if not records:
        print(f"No discoveries in {path}")
        return

    from collections import Counter
    by_n = Counter(r["n"] for r in records)
    by_exp = Counter(str(r["exponents"]) for r in records)
    by_src = Counter(r.get("source", "unknown") for r in records)

    print(f"Discoveries in {path}: {len(records)} total")
    print(f"  By n:          {dict(sorted(by_n.items()))}")
    print(f"  By exponents:  {dict(sorted(by_exp.items()))}")
    print(f"  By source:     {dict(sorted(by_src.items()))}")
