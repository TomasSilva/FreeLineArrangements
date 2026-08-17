"""
Extension-prefilter experiment for the penalized Saito loss (§8).

Reruns the bootstrap-extension pipeline that previously used the old angular
score with a hard-coded 0.05 threshold:

  1. Build labeled data: for free seed arrangements, enumerate one-line
     extension candidates that pass the combinatorial filter, score each with
     (a) the new penalized loss and (b) the legacy score, and label by exact
     symbolic is_free().
  2. Refit the pre-filter threshold on a VALIDATION split (seeds disjoint
     from the test split); report precision / recall / precision-at-k /
     exact-certification rate on the TEST split, for both scores.
  3. Save exact certificates for every discovered free extension.

Usage:
    python experiments/run_extension_experiment.py --out <dir> [--quick]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arrangement import LineArrangement
from saito import (construct_supersolvable, legacy_invalid_angular_score,
                   _enumerate_extension_candidates, saito_loss)
from certificates import find_exact_saito_certificate, certificate_to_json

OLD_DEFAULT_THRESHOLD = 0.05     # the historical legacy-score cutoff


def collect_labeled(seed_arr, seed_name, coord_range=3, max_candidates=None,
                    verbose=True):
    """Score every combinatorially-admissible one-line extension."""
    rows = []
    candidates = _enumerate_extension_candidates(seed_arr,
                                                 coord_range=coord_range)
    if max_candidates is not None and len(candidates) > max_candidates:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(candidates), size=max_candidates, replace=False)
        candidates = [candidates[i] for i in sorted(idx)]
    n_seed = len(seed_arr)
    for line in candidates:
        new_arr = LineArrangement(list(seed_arr.lines) + [line])
        ce = new_arr.candidate_exponents()
        if ce is None:
            continue
        t0 = time.perf_counter()
        new_loss = saito_loss(new_arr, target_exponents=ce, profile='search')
        t_new = time.perf_counter() - t0
        t0 = time.perf_counter()
        leg = legacy_invalid_angular_score(new_arr, target_exponents=ce)
        t_leg = time.perf_counter() - t0
        t0 = time.perf_counter()
        is_free, exps = new_arr.is_free()
        t_exact = time.perf_counter() - t0
        rows.append({
            "seed": seed_name, "n_new": n_seed + 1,
            "line": tuple(str(v) for v in line.coords),
            "cand_exps": list(ce), "is_free": bool(is_free),
            "new_loss": float(new_loss), "legacy_score": float(leg),
            "t_new_loss": t_new, "t_legacy": t_leg, "t_exact": t_exact,
            "lines": [tuple(str(v) for v in l.coords)
                      for l in new_arr.lines],
        })
    if verbose:
        nf = sum(r["is_free"] for r in rows)
        print(f"  seed {seed_name}: {len(rows)} admissible candidates, "
              f"{nf} exactly free")
    return rows


def refit_threshold(rows, score_key, safety=2.0):
    """Smallest threshold keeping every free candidate, times a safety
    factor (the filter must not sacrifice recall — exact verification
    downstream removes false positives)."""
    free_scores = [r[score_key] for r in rows if r["is_free"]]
    if not free_scores:
        return None
    return float(max(free_scores) * safety)


def filter_metrics(rows, score_key, threshold):
    """Precision/recall of `score <= threshold` against exact freeness."""
    tp = sum(1 for r in rows if r[score_key] <= threshold and r["is_free"])
    fp = sum(1 for r in rows if r[score_key] <= threshold and not r["is_free"])
    fn = sum(1 for r in rows if r[score_key] > threshold and r["is_free"])
    kept = tp + fp
    total_free = tp + fn
    # precision-at-k: rank by score ascending, k = number of true frees
    ranked = sorted(rows, key=lambda r: r[score_key])
    k = max(1, total_free)
    p_at_k = sum(1 for r in ranked[:k] if r["is_free"]) / k
    return {
        "threshold": threshold, "kept": kept, "true_free": total_free,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / kept if kept else None,
        "recall": tp / total_free if total_free else None,
        "precision_at_k": p_at_k,
        "exact_checks_saved_frac": 1.0 - kept / len(rows) if rows else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--coord-range", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # validation seeds (threshold refit) and test seeds are disjoint.
    # Mixed seed types: supersolvable seeds yield (almost) only free
    # extensions; braid / nonfree seeds contribute the nonfree class the
    # filter must reject.
    from arrangement import ProjectiveLine as PL

    def _arr(coords):
        return LineArrangement([PL(*c) for c in coords])

    BRAID = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (1, 0, -1),
             (0, 1, -1)]
    NF7A = [(0, 1, -1), (1, 1, -1), (1, 1, 1), (1, 1, 2), (1, 1, 0),
            (1, 1, -2), (1, 0, -2)]
    NF7B = [(1, 0, -1), (1, 0, -2), (2, 0, 1), (0, 0, 1), (1, -2, 0),
            (0, 1, 2), (1, 0, 0)]
    val_seeds = [("braid_A3", _arr(BRAID)),
                 ("nonfree7_a", _arr(NF7A)),
                 ("supersolvable_9_3", construct_supersolvable(9, 3)),
                 ("supersolvable_10_4", construct_supersolvable(10, 4))]
    test_seeds = [("nonfree7_b", _arr(NF7B)),
                  ("supersolvable_11_4", construct_supersolvable(11, 4)),
                  ("supersolvable_12_5", construct_supersolvable(12, 5))]
    if args.quick:
        val_seeds, test_seeds = val_seeds[:2], test_seeds[:1]

    print("== collecting validation split ==")
    val_rows = []
    for name, arr in val_seeds:
        val_rows += collect_labeled(arr, name, coord_range=args.coord_range)
    print("== collecting test split ==")
    test_rows = []
    for name, arr in test_seeds:
        test_rows += collect_labeled(arr, name, coord_range=args.coord_range)

    # One-line extensions of these seeds with integer candidate exponents
    # turn out to be free essentially always (addition-theorem territory), so
    # the extension data alone has no negative class.  The filter, however,
    # must also reject nonfree arrangements met during cascades over
    # arbitrary seeds.  Augment both splits with random integer arrangements
    # that HAVE candidate exponents, labeled by the exact check.
    print("== collecting random labeled arrangements (negative class) ==")
    import random as _random
    from arrangement import ProjectiveLine as _PL

    def random_labeled(seed, count, ns=(7, 8, 9, 10)):
        rng = _random.Random(seed)
        pool, seen = [], set()
        for a in range(-2, 3):
            for b in range(-2, 3):
                for c in range(-2, 3):
                    if (a, b, c) == (0, 0, 0):
                        continue
                    L = _PL(a, b, c)
                    if L.coords not in seen:
                        seen.add(L.coords)
                        pool.append(L)
        rows = []
        trials = 0
        while len(rows) < count and trials < 200000:
            trials += 1
            n = rng.choice(ns)
            arr = LineArrangement(rng.sample(pool, n))
            ce = arr.candidate_exponents()
            if ce is None or ce[0] == 0:
                continue
            t0 = time.perf_counter()
            new_loss = saito_loss(arr, target_exponents=ce, profile='search')
            t_new = time.perf_counter() - t0
            t0 = time.perf_counter()
            leg = legacy_invalid_angular_score(arr, target_exponents=ce)
            t_leg = time.perf_counter() - t0
            t0 = time.perf_counter()
            is_free, _ = arr.is_free()
            t_exact = time.perf_counter() - t0
            rows.append({
                "seed": f"random{seed}", "n_new": n,
                "line": None, "cand_exps": list(ce),
                "is_free": bool(is_free), "new_loss": float(new_loss),
                "legacy_score": float(leg), "t_new_loss": t_new,
                "t_legacy": t_leg, "t_exact": t_exact,
                "lines": [tuple(str(v) for v in l.coords)
                          for l in arr.lines],
            })
        return rows

    n_rand = 20 if args.quick else 40
    val_rand = random_labeled(101, n_rand)
    test_rand = random_labeled(202, n_rand)
    print(f"  val random: {len(val_rand)} "
          f"({sum(r['is_free'] for r in val_rand)} free), "
          f"test random: {len(test_rand)} "
          f"({sum(r['is_free'] for r in test_rand)} free)")
    val_rows += val_rand
    test_rows += test_rand

    tau_new = refit_threshold(val_rows, "new_loss")
    tau_leg = refit_threshold(val_rows, "legacy_score")
    print(f"refit thresholds: new={tau_new}, legacy={tau_leg}")

    report = {
        "old_default_threshold": OLD_DEFAULT_THRESHOLD,
        "refit_threshold_new": tau_new,
        "refit_threshold_legacy": tau_leg,
        "validation": {
            "n_rows": len(val_rows),
            "new_loss@refit": filter_metrics(val_rows, "new_loss", tau_new),
            "legacy@old0.05": filter_metrics(val_rows, "legacy_score",
                                             OLD_DEFAULT_THRESHOLD),
        },
        "test": {
            "n_rows": len(test_rows),
            "new_loss@refit": filter_metrics(test_rows, "new_loss", tau_new),
            "new_loss@old0.05": filter_metrics(test_rows, "new_loss",
                                               OLD_DEFAULT_THRESHOLD),
            "legacy@old0.05": filter_metrics(test_rows, "legacy_score",
                                             OLD_DEFAULT_THRESHOLD),
            "legacy@refit": filter_metrics(test_rows, "legacy_score",
                                           tau_leg),
        },
        "timing": {
            "mean_t_new_loss": float(np.mean([r["t_new_loss"]
                                              for r in test_rows])),
            "mean_t_legacy": float(np.mean([r["t_legacy"]
                                            for r in test_rows])),
            "mean_t_exact": float(np.mean([r["t_exact"]
                                           for r in test_rows])),
        },
    }

    # exact certificates for every free extension found (test + val)
    print("== certifying free extensions ==")
    certs = {}
    cert_rate_num = cert_rate_den = 0
    for row_idx, r in enumerate(val_rows + test_rows):
        if not r["is_free"]:
            continue
        cert_rate_den += 1
        arr = LineArrangement.__new__(LineArrangement)
        from arrangement import ProjectiveLine
        arr.lines = [ProjectiveLine(*c) for c in r["lines"]]
        arr._cache = None
        cert = find_exact_saito_certificate(arr,
                                            target_exponents=tuple(
                                                r["cand_exps"]))
        if cert is not None:
            cert_rate_num += 1
            key = f"{row_idx:04d}_{r['seed']}+{r['line']}"
            certs[key] = certificate_to_json(cert)
    report["exact_certification_rate"] = (
        cert_rate_num / cert_rate_den if cert_rate_den else None)
    print(f"  certified {cert_rate_num}/{cert_rate_den}")

    with open(os.path.join(args.out, "extension_rows.json"), "w") as f:
        json.dump({"validation": val_rows, "test": test_rows}, f, indent=1)
    with open(os.path.join(args.out, "extension_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(args.out, "extension_certificates.json"),
              "w") as f:
        json.dump(certs, f, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
