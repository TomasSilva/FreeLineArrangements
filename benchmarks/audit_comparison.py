"""
Audit comparison: previous penalized implementation (beta = 0.5 defaults)
vs revised (beta = 0.75 production default) on the validation suite, against
exact freeness labels.

Usage:
  python benchmarks/audit_comparison.py --out <dir>
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from penalized_saito import PenalizedSaitoEvaluator
from bench_common import build_suite, arr_from


def evaluate(arr, pair, beta, seed):
    ev = PenalizedSaitoEvaluator(arr, *pair)
    t0 = time.perf_counter()
    res = ev.maximize(beta=beta, n_restarts=8, n_iters=80, seed=seed)
    res["wall_s"] = time.perf_counter() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    items, _ = build_suite(verbose=False, certify=False)
    rows = []
    for it in items:
        arr = arr_from([tuple(v for v in c) for c in it["lines"]])
        pair = tuple(it["pair"])
        row = {"name": it["name"], "label": it["label"], "n": it["n"],
               "pair": list(pair)}
        for tag, beta in (("prev_b050", 0.5), ("rev_b075", 0.75)):
            seeds = [evaluate(arr, pair, beta, s) for s in (0, 1, 2)]
            losses = [r["loss"] for r in seeds]
            row[tag] = {
                "loss_best": min(losses),
                "loss_seed_spread": max(losses) - min(losses),
                "gamma_spread_mean": float(np.mean(
                    [r["gamma_spread"] for r in seeds])),
                "stop_reasons": sorted({s["stop"] for r in seeds
                                        for s in r["restarts"]}),
                "wall_s_mean": float(np.mean([r["wall_s"] for r in seeds])),
                "field": seeds[0]["optimization_field"],
            }
        # wrong-pair check for free items (target-pair correctness)
        if it["label"] == "free" and pair[0] + 1 <= pair[1] - 1:
            wrong = (pair[0] + 1, pair[1] - 1)
            row["wrong_pair"] = list(wrong)
            row["wrong_pair_loss_b075"] = evaluate(arr, wrong, 0.75,
                                                   0)["loss"]
        rows.append(row)
        print(f"{it['name']:26s} [{it['label']:7s}] "
              f"b0.50={row['prev_b050']['loss_best']:.3e} "
              f"b0.75={row['rev_b075']['loss_best']:.3e}", flush=True)

    frees = [r for r in rows if r["label"] == "free"]
    nonfrees = [r for r in rows if r["label"] == "nonfree"]

    def dist(rs, tag):
        v = [r[tag]["loss_best"] for r in rs]
        return {"min": min(v), "max": max(v),
                "median": float(np.median(v))}

    summary = {
        "free_loss_dist": {t: dist(frees, t)
                           for t in ("prev_b050", "rev_b075")},
        "nonfree_loss_dist": {t: dist(nonfrees, t)
                              for t in ("prev_b050", "rev_b075")},
        "free_all_below_1e-8": {
            t: all(r[t]["loss_best"] < 1e-8 for r in frees)
            for t in ("prev_b050", "rev_b075")},
        "nonfree_all_interior": {
            t: all(1e-6 < r[t]["loss_best"] < 1 - 1e-6 for r in nonfrees)
            for t in ("prev_b050", "rev_b075")},
        "label_disagreements_at_1e-6": [
            r["name"] for r in rows
            if (r["prev_b050"]["loss_best"] < 1e-6)
            != (r["rev_b075"]["loss_best"] < 1e-6)],
        "wrong_pair_all_positive_b075": all(
            r.get("wrong_pair_loss_b075", 1.0) > 1e-3 for r in frees
            if "wrong_pair_loss_b075" in r),
        "seed_spread_max": {
            t: max(r[t]["loss_seed_spread"] for r in rows)
            for t in ("prev_b050", "rev_b075")},
        "runtime_mean_s": {
            t: float(np.mean([r[t]["wall_s_mean"] for r in rows]))
            for t in ("prev_b050", "rev_b075")},
        "optimization_field": "real",
    }
    with open(os.path.join(args.out, "audit_comparison.json"), "w") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
