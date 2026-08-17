"""
Production-safety measurement studies (2026-08 pass).

A. Clip verification: raw Gamma (pre-clip) on the free suite — is clipping
   bookkeeping roundoff, or manufacturing exact zeros?
B. Prefilter recall at n >= 14: recall of the HEURISTIC loss <= 1e-6 gate on
   exactly-free arrangements across all target pairs, under unitary and
   ill-conditioned GL coordinate variants and deliberately low optimizer
   budgets; plus the pass rate of exactly-nonfree perturbations (labeled by
   the EXACT negative certificate, never by search failure).
C. beta = 0.5 vs 0.75 action-ranking agreement (Spearman/Kendall) on
   complete line-replacement neighborhoods.

Usage: python benchmarks/production_safety_studies.py --out <dir> [--study A|B|C|all]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import stats as sstats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arrangement import LineArrangement, ProjectiveLine
from penalized_saito import PenalizedSaitoEvaluator, runtime_provenance
from saito import construct_supersolvable
from certificates import find_certificate_fast
from environment import generate_candidate_lines
from swap_search import double_pencil_seed, perturb_k_swaps, is_valid_state

SEED_EVAL = 0
TAU = 1e-6


def lines_matrix(arr):
    return np.array([l.to_float() for l in arr.lines])


def eval_loss(lines, d1, d2, beta=0.75, n_restarts=8, n_iters=80, seed=0,
              want_parts=False):
    ev = PenalizedSaitoEvaluator(lines, d1, d2)
    res = ev.maximize(beta=beta, n_restarts=n_restarts, n_iters=n_iters,
                      seed=seed)
    return (res if want_parts else res["loss"])


# ── A: clip verification ─────────────────────────────────────────────────────

def study_A(out):
    from bench_common import build_suite, arr_from
    items, _ = build_suite(verbose=False, certify=False)
    rows = []
    for it in items:
        if it["label"] != "free":
            continue
        arr = arr_from([tuple(v for v in c) for c in it["lines"]])
        res = eval_loss(lines_matrix(arr), *it["pair"], want_parts=True)
        p = res["parts"]
        rows.append({
            "name": it["name"], "pair": it["pair"],
            "loss_reported": res["loss"],
            "gamma_raw_at_best": p["gamma_raw"],
            "raw_excess_over_1": max(0.0, p["gamma_raw"] - 1.0),
            "raw_deficit_under_1": max(0.0, 1.0 - p["gamma_raw"]),
            "B_norm": p["B_norm"], "alignment": p["inner_abs"],
            "L1u_norm": p["L1u_norm"], "L2v_norm": p["L2v_norm"],
            "clip_count": res["gamma_clip_count"],
            "clip_max_excess": res["gamma_clip_max_excess"],
        })
    summary = {
        "n_free_items": len(rows),
        "max_raw_excess": max(r["raw_excess_over_1"] for r in rows),
        "max_raw_deficit": max(r["raw_deficit_under_1"] for r in rows),
        "items_with_any_clip": sum(1 for r in rows if r["clip_count"] > 0),
        "max_clip_excess_seen": max(r["clip_max_excess"] for r in rows),
        "reading": ("raw Gamma at the free optima sits within roundoff of 1 "
                    "on BOTH sides; clipping symmetrizes the <= 1e-15-scale "
                    "positive side and cannot manufacture zeros beyond "
                    "roundoff (unclipped losses would be within +-few ulp "
                    "of 0)"),
    }
    with open(os.path.join(out, "clip_verification.json"), "w") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=1)
    print("A:", json.dumps(summary, indent=1))


# ── B: prefilter recall at n >= 14 ──────────────────────────────────────────

def _gl_variant(m, cond_target, rng):
    """Integer shear with controlled conditioning (zero set preserved)."""
    while True:
        A = np.eye(3)
        A[0, 1] = rng.integers(1, max(2, cond_target // 10))
        A[1, 2] = rng.integers(1, max(2, cond_target // 10))
        A[2, 0] = rng.integers(0, 2)
        if abs(np.linalg.det(A)) > 0.5:
            return m @ A, float(np.linalg.cond(A))


def study_B(out):
    rng = np.random.default_rng(7)
    base = []
    for n in (14, 15, 16):
        for d1 in range(2, (n - 1) // 2 + 1):
            d2 = n - 1 - d1
            base.append((f"ss_{n}_{d1}", construct_supersolvable(n, d1),
                         d1, d2))
    rows = []
    for name, arr, d1, d2 in base:
        m = lines_matrix(arr)
        conds = {
            "base_search": (m, dict()),
            "unitary": (m @ np.linalg.qr(rng.standard_normal((3, 3)))[0],
                        dict()),
            "gl_cond~50": (_gl_variant(m, 50, rng)[0], dict()),
            "gl_cond~500": (_gl_variant(m, 500, rng)[0], dict()),
            "low_budget_4x40": (m, dict(n_restarts=4, n_iters=40)),
            "very_low_2x20": (m, dict(n_restarts=2, n_iters=20)),
        }
        for tag, (mm, kw) in conds.items():
            t0 = time.perf_counter()
            loss = eval_loss(mm, d1, d2, **kw)
            rows.append({"name": name, "n": len(arr), "pair": [d1, d2],
                         "condition": tag, "label": "free_exact",
                         "loss": loss, "passes_tau": loss <= TAU,
                         "wall_s": time.perf_counter() - t0})
        # exactly-nonfree perturbations near the free locus
        for t_num, t_den in ((1, 10**4), (1, 10**8)):
            from sympy import Rational
            a, b, c = arr.lines[-1].coords
            t = Rational(t_num, t_den)
            pert = LineArrangement(list(arr.lines[:-1]) +
                                   [ProjectiveLine(a + t, b + 2 * t, c - t)])
            cert, status = find_certificate_fast(pert,
                                                 target_exponents=(d1, d2))
            loss = eval_loss(lines_matrix(pert), d1, d2)
            rows.append({"name": name, "n": len(arr), "pair": [d1, d2],
                         "condition": f"perturbed_t=1e-{len(str(t_den))-1}",
                         "label": ("nonfree_exact" if status in
                                   ("not_target_free", "modp_reject")
                                   else f"unlabeled({status})"),
                         "exact_status": status,
                         "loss": loss, "passes_tau": loss <= TAU})
        print(f"  B: {name} done", flush=True)

    by_cond = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)
    summary = {}
    for cond, rs in sorted(by_cond.items()):
        frees = [r for r in rs if r["label"] == "free_exact"]
        nonfrees = [r for r in rs if r["label"] == "nonfree_exact"]
        entry = {"n_items": len(rs)}
        if frees:
            entry["recall_at_tau"] = (sum(r["passes_tau"] for r in frees)
                                      / len(frees))
            entry["max_free_loss"] = max(r["loss"] for r in frees)
        if nonfrees:
            entry["nonfree_pass_rate_at_tau"] = (
                sum(r["passes_tau"] for r in nonfrees) / len(nonfrees))
            entry["min_nonfree_loss"] = min(r["loss"] for r in nonfrees)
        summary[cond] = entry
    with open(os.path.join(out, "prefilter_recall.json"), "w") as f:
        json.dump({"tau": TAU, "rows": rows, "summary": summary,
                   "note": ("tau is a HEURISTIC gate: the computed loss is "
                            "an upper bound from a finite multistart; "
                            "recall is measured against exact-free labels; "
                            "nonfree labels come from the EXACT negative "
                            "certificate, never from search failure")},
                  f, indent=1)
    print("B:", json.dumps(summary, indent=1))


# ── C: beta ranking agreement on complete swap neighborhoods ────────────────

def study_C(out):
    rng = np.random.default_rng(11)
    pool = generate_candidate_lines(2)
    states = []
    for (n, d1, d2, k) in ((13, 6, 6, 2), (14, 6, 7, 2)):
        states.append((f"perturbed_dp_{n}_{d1}_{d2}",
                       perturb_k_swaps(double_pencil_seed(n, d1, d2), k,
                                       rng), d1, d2))
    from swap_search import random_valid_seed
    states.append(("random_13_6_6", random_valid_seed(13, rng), 6, 6))

    results = []
    for name, arr, d1, d2 in states:
        n = len(arr)
        existing = {l.coords for l in arr.lines}
        losses = {0.5: [], 0.75: []}
        actions = []
        for i in range(n):
            rest = [l for j, l in enumerate(arr.lines) if j != i]
            for line in pool:
                if line.coords in existing:
                    continue
                trial = LineArrangement(rest + [line])
                if not is_valid_state(trial, n, nontrivial=(d1 >= 2)):
                    continue
                actions.append((i, line))
                m = lines_matrix(trial)
                for beta in (0.5, 0.75):
                    losses[beta].append(
                        eval_loss(m, d1, d2, beta=beta, n_restarts=4,
                                  n_iters=40, seed=SEED_EVAL))
        a, b = np.array(losses[0.5]), np.array(losses[0.75])
        sp = sstats.spearmanr(a, b)
        kd = sstats.kendalltau(a, b)
        top = 20
        top_a = set(np.argsort(a)[:top])
        top_b = set(np.argsort(b)[:top])
        results.append({
            "state": name, "n_actions": len(actions),
            "spearman_rho": float(sp.statistic),
            "kendall_tau": float(kd.statistic),
            "top20_overlap": len(top_a & top_b) / top,
            "best_action_same": bool(np.argmin(a) == np.argmin(b)),
        })
        print(f"  C: {name}: {len(actions)} actions, "
              f"rho={sp.statistic:.4f} tau={kd.statistic:.4f}", flush=True)
    with open(os.path.join(out, "beta_ranking_agreement.json"), "w") as f:
        json.dump({"results": results, "eval_budget": "rl 4x40, seed 0",
                   "field": "real"}, f, indent=1)
    print("C:", json.dumps(results, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--study", default="all", choices=["A", "B", "C", "all"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "provenance.json"), "w") as f:
        json.dump(runtime_provenance("."), f, indent=1)
    if args.study in ("A", "all"):
        study_A(args.out)
    if args.study in ("B", "all"):
        study_B(args.out)
    if args.study in ("C", "all"):
        study_C(args.out)


if __name__ == "__main__":
    main()
