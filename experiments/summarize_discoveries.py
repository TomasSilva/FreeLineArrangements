"""
One-command discovery summary across result trees: distinct certified
lattices, per-n / per-field tables, epsilon = d1 - m_max histogram and the
DKP rare list (eps >= 2, d1 < d - m), with reference-novelty labels.

Usage:
  python experiments/summarize_discoveries.py \
      [--roots results_from_HPC results_local] \
      [--out results_local/discovery_summary]
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_reference():
    ref = set()
    p = "results_penalized_saito/2026-08-17/swap/reference_hashes/headline.json"
    if os.path.exists(p):
        for cell, info in json.load(open(p)).items():
            ref.update(info.get("hashes", {}).keys())
    if os.path.exists("reference_hashes_k_fixtures.json"):
        ref.update(json.load(open("reference_hashes_k_fixtures.json")).keys())
    for q in glob.glob(
            "results_penalized_saito/2026-08-17/swap/cells/*/certified.jsonl"):
        for line in open(q):
            try:
                ref.add(json.loads(line)["lattice_hash"])
            except Exception:
                pass
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+",
                    default=["results_from_HPC", "results_local"])
    ap.add_argument("--out", default="results_local/discovery_summary")
    args = ap.parse_args()
    ref = load_reference()

    best = {}
    for root in args.roots:
        for p in glob.glob(f"{root}/**/certified.jsonl", recursive=True):
            for line in open(p):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                h = r.get("lattice_hash")
                if not h or "d1" not in r or "m_max" not in r:
                    continue
                cf = r.get("coefficient_field", "QQ")
                fld = cf if isinstance(cf, str) else cf.get("name", "?")
                if h not in best or r.get("height", 1e18) < \
                        best[h][0].get("height", 1e18):
                    best[h] = (r, fld, p)

    eps_hist = defaultdict(int)
    by_n = defaultdict(lambda: [0, 0, 0])
    fields = defaultdict(int)
    rare = []
    for h, (r, fld, p) in best.items():
        d, d1, m = r["n"], r["d1"], r["m_max"]
        eps = d1 - m
        eps_hist[eps] += 1
        nss = not r.get("supersolvable", True)
        row = by_n[d]
        row[0] += 1
        if nss:
            row[1] += 1
            if h not in ref:
                row[2] += 1
        fields[fld] += 1
        if eps >= 2 and d1 < d - m:
            rare.append({"eps": eps, "n": d, "pair": [d1, r["d2"]],
                         "m": m, "hash": h, "height": r.get("height"),
                         "field": fld, "nonss": nss,
                         "new_vs_reference": h not in ref, "source": p})

    print(f"distinct certified lattices: {len(best)}")
    print(f"by field: {dict(sorted(fields.items()))}")
    print(f"epsilon histogram: {dict(sorted(eps_hist.items()))}")
    print(f"RARE (eps>=2, d1<d-m): {len(rare)}")
    er = defaultdict(int)
    for t in rare:
        er[(t["eps"], t["n"])] += 1
    print(f"{'eps':>4} {'n':>3}  count")
    for (e, d) in sorted(er):
        print(f"{e:>4} {d:>3}  {er[(e, d)]}")
    print(f"\n{'n':>3} {'distinct':>9} {'nonSS':>7} {'nonSS&ref-new':>14}")
    for d in sorted(by_n):
        r_ = by_n[d]
        print(f"{d:>3} {r_[0]:>9} {r_[1]:>7} {r_[2]:>14}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rare.sort(key=lambda t: (-t["eps"], t["n"], t["height"] or 0))
    json.dump(rare, open(args.out + "_rare.json", "w"), indent=1)
    json.dump({"eps_hist": {str(k): v for k, v in eps_hist.items()},
               "by_n": {str(k): v for k, v in by_n.items()},
               "fields": dict(fields), "total": len(best)},
              open(args.out + "_tables.json", "w"), indent=1)
    print(f"\nsaved {args.out}_rare.json / _tables.json")


if __name__ == "__main__":
    main()
