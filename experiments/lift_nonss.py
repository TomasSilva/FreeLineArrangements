"""
Non-supersolvable lift seeding: bootstrap novelty upward.

Takes certified NON-SUPERSOLVABLE discoveries at level n (from a triage
report and/or cells/*/certified.jsonl), lifts each by one line into every
admissible (n+1) cell via the Δb2-targeted extension
(saito.extend_arrangement_targeted), exactly certifies each lift, tags it
with the supersolvability screen, and writes seed files

    swap_lift_seeds/n<N>_d<D1>_<D2>.json

which run_swap_campaign.build_seeds auto-loads (committed to git so HPC
units pick them up after a pull).  Non-SS lifts are the gold seeds: swap
chains started there explore non-supersolvable basins instead of
rediscovering the double-pencil one at every level.

Usage:
  python experiments/lift_nonss.py \
      --cells-dir results_penalized_saito/2026-08-17/swap/cells \
      [--source-level 13 14] [--nonss-only] [--out swap_lift_seeds]
"""

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arrangement import LineArrangement
from novelty import (parse_line_str, is_supersolvable_rank3, lattice_wl_hash,
                     coordinate_height)
from certificates import find_exact_saito_certificate, certificate_to_json
from saito import extend_arrangement_targeted


def load_nonss_records(cells_dir, levels, nonss_only=True):
    """Distinct-lattice certified records at the given n levels, preferring
    low height; non-SS only by default."""
    best = {}
    for path in sorted(glob.glob(os.path.join(cells_dir, "*",
                                              "certified.jsonl"))):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["n"] not in levels:
                    continue
                if nonss_only and rec.get("supersolvable", True):
                    continue
                key = rec["lattice_hash"]
                if key not in best or rec["height"] < best[key]["height"]:
                    best[key] = rec
    return list(best.values())


def admissible_next_pairs(n):
    """Nontrivial (d1, d2) at level n+1."""
    m = n + 1
    return [(d1, m - 1 - d1) for d1 in range(2, (m - 1) // 2 + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-dir", required=True)
    ap.add_argument("--source-level", type=int, nargs="+", default=[13, 14])
    ap.add_argument("--nonss-only", action="store_true", default=True)
    ap.add_argument("--include-ss", dest="nonss_only", action="store_false")
    ap.add_argument("--out", default="swap_lift_seeds")
    ap.add_argument("--coord-range", type=int, default=4)
    ap.add_argument("--max-parents", type=int, default=8)
    ap.add_argument("--max-lifts-per-cell", type=int, default=6)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    parents = load_nonss_records(args.cells_dir, set(args.source_level),
                                 args.nonss_only)
    parents.sort(key=lambda r: (r["m_max"], r["height"]))
    parents = parents[:args.max_parents]
    print(f"lifting {len(parents)} parent(s) from levels "
          f"{sorted(args.source_level)}", flush=True)

    per_cell = {}
    for rec in parents:
        arr = LineArrangement([parse_line_str(s) for s in rec["lines"]])
        n = len(arr)
        for (d1, d2) in admissible_next_pairs(n):
            t0 = time.time()
            try:
                successes = extend_arrangement_targeted(
                    arr, target_exponents=(d1, d2),
                    coord_range=args.coord_range, verbose=False)
            except Exception as e:
                print(f"  parent {rec['lattice_hash'][:8]} -> "
                      f"({n+1},{d1},{d2}): extension error {e}", flush=True)
                continue
            for s in successes:
                lifted = s["arrangement"]
                cert = find_exact_saito_certificate(
                    lifted, target_exponents=(d1, d2))
                if cert is None:
                    continue      # never seed on an uncertified lift
                ss = is_supersolvable_rank3(lifted)
                entry = {
                    "lines": [str(l) for l in lifted.lines],
                    "n": n + 1, "d1": d1, "d2": d2,
                    "supersolvable": ss,
                    "m_max": lifted.max_multiplicity(),
                    "height": coordinate_height(lifted),
                    "lattice_hash": lattice_wl_hash(lifted),
                    "parent_lattice_hash": rec["lattice_hash"],
                    "parent_n": n,
                    "lift_line": str(s["new_line"]),
                    "certificate": certificate_to_json(cert),
                    "provenance": "lift_nonss_v1",
                }
                per_cell.setdefault((n + 1, d1, d2), []).append(entry)
            k = len(successes)
            if k:
                print(f"  parent {rec['lattice_hash'][:8]} "
                      f"(n={n}, m_max={rec['m_max']}) -> "
                      f"({n+1},{d1},{d2}): {k} certified lift(s) "
                      f"({time.time()-t0:.0f}s)", flush=True)

    total = 0
    for (n, d1, d2), entries in sorted(per_cell.items()):
        # order: non-SS first, then low m_max, low height; distinct lattices
        entries.sort(key=lambda e: (e["supersolvable"], e["m_max"],
                                    e["height"]))
        seen, keep = set(), []
        for e in entries:
            if e["lattice_hash"] in seen:
                continue
            seen.add(e["lattice_hash"])
            keep.append(e)
            if len(keep) >= args.max_lifts_per_cell:
                break
        path = os.path.join(args.out, f"n{n}_d{d1}_{d2}.json")
        existing = []
        if os.path.exists(path):
            existing = json.load(open(path)).get("seeds", [])
            known = {e["lattice_hash"] for e in existing}
            keep = [e for e in keep if e["lattice_hash"] not in known]
        with open(path, "w") as f:
            json.dump({"cell": [n, d1, d2], "seeds": existing + keep,
                       "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  time.gmtime())},
                      f, indent=1)
        nss = sum(1 for e in existing + keep if not e["supersolvable"])
        total += len(keep)
        print(f"  cell ({n},{d1},{d2}): +{len(keep)} seeds "
              f"({nss} non-SS total) -> {path}", flush=True)
    print(f"done: {total} new lift seeds written to {args.out}/")


if __name__ == "__main__":
    main()
