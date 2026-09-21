"""
Mint swap-campaign lift-seed files for EXISTING cells directly from the
shareable database (certified_free_arrangements.jsonl.gz): for each target
cell (n, d1, d2), export the lowest-(m_max, height) non-supersolvable
certified representatives in the schema that run_swap_campaign.
load_lift_seeds reads (swap_lift_seeds/n<N>_d<D1>_<D2>.json).

Lines are rebuilt through certificate_from_json (the verified parsing
path) and serialized with str(ProjectiveLine), whose format parse_line_str
round-trips exactly — a round-trip assertion guards every seed.

Every seed is an exactly certified discovery — but as always it is only a
START STATE for the chains; no claim rests on it.  Merges with existing
seed files (dedup by lattice hash).

Usage:
  python experiments/seed_cells_from_db.py --cells 25,12,12 26,12,13 \
      [--db certified_free_arrangements.jsonl.gz] [--per-cell 6] \
      [--max-m 7] [--out swap_lift_seeds]

  # specific lattices (hash prefixes), each written to its own cell file:
  python experiments/seed_cells_from_db.py --hashes 088d4d6891aa ... \
      --out swap_nondiv_seeds
"""

import argparse
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arrangement import LineArrangement, ProjectiveLine
from certificates import certificate_from_json
from novelty import parse_line_str
from quadfield import QuadraticField


def args_cells_set(args):
    return {tuple(int(x) for x in c.split(",")) for c in args.cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=[],
                    help="target cells as N,D1,D2")
    ap.add_argument("--hashes", nargs="+", default=[],
                    help="lattice-hash prefixes: export exactly these "
                         "entries (supersolvable or not) into their cells")
    ap.add_argument("--db", default="certified_free_arrangements.jsonl.gz")
    ap.add_argument("--per-cell", type=int, default=6)
    ap.add_argument("--max-m", type=int, default=None,
                    help="only export seeds with m_max <= this")
    ap.add_argument("--out", default="swap_lift_seeds")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    assert args.cells or args.hashes, "give --cells and/or --hashes"
    cells = [tuple(int(x) for x in c.split(",")) for c in args.cells]
    pool = {c: [] for c in cells}
    found = set()
    with gzip.open(args.db, "rt") as f:
        next(f)                      # _meta line
        for line in f:
            e = json.loads(line)
            key = (e["n"], e["exponents"][1], e["exponents"][2])
            hit = [h for h in args.hashes if e["lattice_hash"].startswith(h)]
            if hit:
                found.update(hit)
                pool.setdefault(key, []).append(e)
                if key not in cells:
                    cells.append(key)
                continue
            if key not in args_cells_set(args) or e["supersolvable"]:
                continue
            if args.max_m is not None and e["m_max"] > args.max_m:
                continue
            pool[key].append(e)
    missing = set(args.hashes) - found
    assert not missing, f"hashes not found in the database: {missing}"

    for (n, d1, d2) in cells:
        entries = sorted(pool[(n, d1, d2)],
                         key=lambda e: (e["m_max"], e["height"] or 0))
        seeds = []
        for e in entries[:args.per_cell]:
            cert = certificate_from_json(e["certificate"])
            lines = [ProjectiveLine(*c) for c in cert["lines"]]
            arr = LineArrangement(lines)
            assert len(arr) == n
            field = arr.coefficient_field()
            line_strs = [str(l) for l in lines]
            # round-trip guard: the campaign must reparse these exactly
            for s, l in zip(line_strs, lines):
                assert parse_line_str(s, field=field) == l, s
            seeds.append({
                "lines": line_strs,
                "n": n, "d1": d1, "d2": d2,
                "supersolvable": bool(e["supersolvable"]),
                "m_max": e["m_max"],
                "height": e["height"],
                "lattice_hash": e["lattice_hash"],
                "coefficient_field": ("QQ" if field is None
                                      else field.to_json()),
                "provenance": "seed_cells_from_db_v1",
            })
        path = os.path.join(args.out, f"n{n}_d{d1}_{d2}.json")
        existing = []
        if os.path.exists(path):
            existing = json.load(open(path)).get("seeds", [])
            known = {s.get("lattice_hash") for s in existing}
            seeds = [s for s in seeds if s["lattice_hash"] not in known]
        with open(path, "w") as f:
            json.dump({"cell": [n, d1, d2], "seeds": existing + seeds,
                       "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                  time.gmtime())},
                      f, indent=1)
        print(f"cell ({n},{d1},{d2}): +{len(seeds)} seeds "
              f"(total {len(existing) + len(seeds)}, "
              f"m_max {[s['m_max'] for s in existing + seeds]}) -> {path}")


if __name__ == "__main__":
    main()
