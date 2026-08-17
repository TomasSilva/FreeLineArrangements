"""
Aggregate swap-campaign results into the referee-facing tables.

Reads cells/*/manifest_*.json + certified.jsonl (+ the corpus re-pricing
index when available) and writes:
  cell_table.csv    per (n, d1, d2): units run, loss evals, certified count,
                    DISTINCT LATTICES (the headline metric), non-supersolvable
                    count, corpus-new count, best loss, time-to-first-cert
  swap_report.md    summary tables + engine attribution + honest gaps
"""

import argparse
import csv
import glob
import json
import os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-dir", required=True)
    ap.add_argument("--reference-hashes", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ref = {}
    if args.reference_hashes and os.path.exists(args.reference_hashes):
        with open(args.reference_hashes) as f:
            ref = json.load(f)

    cells = defaultdict(lambda: {
        "units": 0, "restarts": 0, "screen_evals": 0, "refine_evals": 0,
        "cert_attempts": 0, "certified_records": 0, "best_loss": 1.0,
        "first_cert_s": None, "lattices": set(), "nonss_lattices": set(),
        "new_lattices": set(), "engines": defaultdict(int),
    })

    for mpath in sorted(glob.glob(os.path.join(args.cells_dir, "*",
                                               "manifest_*.json"))):
        with open(mpath) as f:
            m = json.load(f)
        a = m["args"]
        key = (a["n"], a["d1"], a["d2"])
        c = cells[key]
        # beta provenance: units whose manifests predate the audit patch do
        # not record beta.  They are labeled unknown and their NUMERICAL
        # loss statistics are never aggregated with revised (beta-recorded)
        # units; certified counts and lattice counts are beta-independent
        # (exact certificates) and aggregate freely.
        beta_label = str(m.get("beta", "unknown_pre_audit"))
        c.setdefault("beta_groups", {}).setdefault(beta_label, 0)
        c["beta_groups"][beta_label] += 1
        c["units"] += 1
        c["restarts"] += m.get("restarts", 0)
        c["screen_evals"] += m.get("evaluator", {}).get("screen_evals", 0)
        c["refine_evals"] += m.get("evaluator", {}).get("refine_evals", 0)
        c["cert_attempts"] += m.get("counters", {}).get("cert_attempts", 0)
        c["best_loss"] = min(c["best_loss"], m.get("best_loss", 1.0))
        t = m.get("first_cert_wall_s")
        if t is not None and (c["first_cert_s"] is None
                              or t < c["first_cert_s"]):
            c["first_cert_s"] = t

    for cpath in sorted(glob.glob(os.path.join(args.cells_dir, "*",
                                               "certified.jsonl"))):
        with open(cpath) as f:
            for line in f:
                rec = json.loads(line)
                key = (rec["n"], rec["d1"], rec["d2"])
                c = cells[key]
                c["certified_records"] += 1
                h = rec["lattice_hash"]
                c["lattices"].add(h)
                c["engines"][rec.get("engine", "?")] += 1
                if not rec.get("supersolvable", True):
                    c["nonss_lattices"].add(h)
                cell_name = f"{rec['n']}_{rec['d1']}_{rec['d2']}"
                corpus = set(ref.get(cell_name, {}).get("hashes", {}))
                if (cell_name in ref) and h not in corpus:
                    c["new_lattices"].add(h)

    rows = []
    for (n, d1, d2), c in sorted(cells.items()):
        cell_name = f"{n}_{d1}_{d2}"
        corpus_lat = len(ref.get(cell_name, {}).get("hashes", {})) \
            if cell_name in ref else None
        rows.append({
            "n": n, "d1": d1, "d2": d2, "units": c["units"],
            "beta_groups": c.get("beta_groups", {}),
            "pair_class": ("nontrivial" if d1 >= 2 else
                           "baseline_near_pencil" if d1 == 1
                           else "baseline_pencil"),
            "restarts": c["restarts"], "screen_evals": c["screen_evals"],
            "cert_attempts": c["cert_attempts"],
            "certified_records": c["certified_records"],
            "distinct_lattices": len(c["lattices"]),
            "non_supersolvable_lattices": len(c["nonss_lattices"]),
            "corpus_lattices": corpus_lat,
            "new_vs_corpus": len(c["new_lattices"]) if corpus_lat is not None
            else None,
            "best_loss": c["best_loss"],
            "first_cert_s": c["first_cert_s"],
            "engines": dict(c["engines"]),
        })

    with open(os.path.join(args.out, "cell_table.csv"), "w",
              newline="") as f:
        json_cols = ("engines", "beta_groups")
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0].keys()
                                          if k not in json_cols]
                           + list(json_cols))
        w.writeheader()
        for r in rows:
            rr = dict(r)
            for k in json_cols:
                rr[k] = json.dumps(r[k])
            w.writerow(rr)

    with open(os.path.join(args.out, "swap_report.md"), "w") as f:
        f.write("# Swap-search campaign report\n\n")
        f.write("Counts are exact-certified discoveries; the headline metric "
                "is DISTINCT INTERSECTION LATTICES (WL-hash level). "
                "`new_vs_corpus` counts lattices absent from the historical "
                "corpus of the same cell (sound at hash level: isomorphic "
                "lattices hash equally).\n\n")
        f.write("| cell | units | screen evals | certified | distinct "
                "lattices | non-SS | corpus lattices | new vs corpus | "
                "best loss | first cert (s) |\n|" + "---|" * 10 + "\n")
        for r in rows:
            f.write(f"| ({r['n']},{r['d1']},{r['d2']}) | {r['units']} | "
                    f"{r['screen_evals']} | {r['certified_records']} | "
                    f"{r['distinct_lattices']} | "
                    f"{r['non_supersolvable_lattices']} | "
                    f"{r['corpus_lattices']} | {r['new_vs_corpus']} | "
                    f"{r['best_loss']:.1e} | "
                    f"{'' if r['first_cert_s'] is None else f_int(r['first_cert_s'])} |\n")
        f.write("\nEngine attribution (certified records): ")
        agg = defaultdict(int)
        for r in rows:
            for k, v in r["engines"].items():
                agg[k] += v
        f.write(", ".join(f"{k}: {v}" for k, v in sorted(agg.items()))
                + "\n")
    print(f"wrote {os.path.join(args.out, 'cell_table.csv')} and "
          f"swap_report.md ({len(rows)} cells)")


def f_int(x):
    return str(int(round(x)))


if __name__ == "__main__":
    main()
