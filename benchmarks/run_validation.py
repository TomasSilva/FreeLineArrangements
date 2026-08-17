"""
Numerical validation study for the penalized Saito functional (§7).

Usage:
    python benchmarks/run_validation.py --out results_penalized_saito/<dir>/benchmark

Produces machine-readable JSON/CSV results:
    suite.json                the benchmark arrangements + exact certificates
    main_table.json/.csv      per-item loss, diagnostics, legacy comparison,
                              kernel conditioning, timings
    lambda_sweep.json         log sweep of lambda
    beta_sweep.json           beta in {0.5, 0.75, 0.9}
    restart_study.json        restart-count study with per-restart records
    iteration_study.json      iteration-budget study
    coordinate_sensitivity.json  orthogonal vs projective changes
    rescale_permutation.json  line rescaling / permutation drift
    perturbation.json         degeneration path near a free arrangement
    reference_check.json      float64 vs exact/mpmath reference values
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arrangement import LineArrangement, ProjectiveLine
from penalized_saito import (PenalizedSaitoEvaluator, penalized_saito_loss,
                             kernel_diagnostics, PROFILES)
from saito import legacy_invalid_angular_score
from bench_common import build_suite, perturbation_family, arr_from

DEFAULT_LAM = 1.0
DEFAULT_BETA = 0.5
LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4]
BETAS = [0.5, 0.75, 0.9]
RESTARTS = [1, 2, 4, 8, 16, 32]
ITERS = [10, 20, 40, 80, 160]

# representative subset for the expensive studies
SUBSET = ["braid_A3", "supersolvable_9_3", "supersolvable_11_5",
          "nonfree7_a", "nonfree9_0", "generic_5"]


def item_arr(item):
    return arr_from([tuple(v for v in coords) for coords in item["lines"]])


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()
                if k not in ("u", "v")}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return str(obj)
    return obj


def evaluate(arr, pair, lam=DEFAULT_LAM, beta=DEFAULT_BETA,
             profile="benchmark", n_restarts=None, n_iters=None, seed=0):
    prof = PROFILES[profile]
    n_restarts = prof["n_restarts"] if n_restarts is None else n_restarts
    n_iters = prof["n_iters"] if n_iters is None else n_iters
    ev = PenalizedSaitoEvaluator(arr, *pair)
    t0 = time.perf_counter()
    res = ev.maximize(lam=lam, beta=beta, n_restarts=n_restarts,
                      n_iters=n_iters, seed=seed)
    res["wall_time_s"] = time.perf_counter() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true",
                    help="smaller sweeps (smoke)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lambdas = LAMBDAS if not args.quick else [1e-2, 1.0, 1e2]
    betas = BETAS
    restarts = RESTARTS if not args.quick else [1, 4, 16]
    iters = ITERS if not args.quick else [20, 80]

    print("== building suite (with exact certificates for free items) ==")
    t0 = time.time()
    items, certs = build_suite()
    print(f"   {len(items)} items, {len(certs)} certificates "
          f"({time.time()-t0:.1f}s)")
    with open(os.path.join(args.out, "suite.json"), "w") as f:
        json.dump({"items": items, "certificates": certs}, f, indent=1)

    arrs = {it["name"]: item_arr(it) for it in items}

    # ── main table ──────────────────────────────────────────────────────────
    print("== main table ==")
    main_rows = []
    for it in items:
        arr = arrs[it["name"]]
        pair = tuple(it["pair"])
        res = evaluate(arr, pair)
        # legacy comparisons (regression only)
        t0 = time.perf_counter()
        leg = legacy_invalid_angular_score(arr, target_exponents=pair)
        leg_t = time.perf_counter() - t0
        leg8 = legacy_invalid_angular_score(arr, target_exponents=pair,
                                            min_extra=8)
        kd1 = kernel_diagnostics(arr, pair[0])
        kd2 = kernel_diagnostics(arr, pair[1])
        row = {
            "name": it["name"], "family": it["family"], "label": it["label"],
            "n": it["n"], "pair": list(pair), "cand_exps": it["cand_exps"],
            "loss": res["loss"], "gamma": res["gamma"],
            "gamma_median": res["gamma_median"],
            "gamma_spread": res["gamma_spread"],
            "proj_grad_norm": res.get("proj_grad_norm"),
            "parts": res["parts"],
            "stop_reasons": [r["stop"] for r in res["restarts"]],
            "wall_time_s": res["wall_time_s"],
            "legacy_score": leg, "legacy_score_minextra8": leg8,
            "legacy_time_s": leg_t,
            "kernel_d1": kd1, "kernel_d2": kd2,
        }
        main_rows.append(row)
        print(f"  {it['name']:26s} [{it['label']:7s}] loss={res['loss']:.3e} "
              f"legacy={leg:.3e} t={res['wall_time_s']:.2f}s")
    with open(os.path.join(args.out, "main_table.json"), "w") as f:
        json.dump(sanitize(main_rows), f, indent=1)
    with open(os.path.join(args.out, "main_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "family", "label", "n", "d1", "d2", "loss",
                    "gamma", "gamma_spread", "B_norm", "inner_abs",
                    "L1u_norm", "L2v_norm", "denominator", "legacy_score",
                    "legacy_minextra8", "sigma_min_pos_d1", "cond_d1",
                    "wall_time_s", "legacy_time_s"])
        for r in main_rows:
            w.writerow([r["name"], r["family"], r["label"], r["n"],
                        r["pair"][0], r["pair"][1], r["loss"], r["gamma"],
                        r["gamma_spread"], r["parts"]["B_norm"],
                        r["parts"]["inner_abs"], r["parts"]["L1u_norm"],
                        r["parts"]["L2v_norm"], r["parts"]["denominator"],
                        r["legacy_score"], r["legacy_score_minextra8"],
                        r["kernel_d1"]["sigma_min_pos"],
                        r["kernel_d1"]["cond_estimate"],
                        r["wall_time_s"], r["legacy_time_s"]])

    # ── lambda sweep ────────────────────────────────────────────────────────
    print("== lambda sweep ==")
    sweep = []
    for it in items:
        arr = arrs[it["name"]]
        pair = tuple(it["pair"])
        for lam in lambdas:
            res = evaluate(arr, pair, lam=lam, profile="search")
            sweep.append({"name": it["name"], "label": it["label"],
                          "lambda": lam, "loss": res["loss"],
                          "parts": res["parts"],
                          "wall_time_s": res["wall_time_s"]})
        print(f"  {it['name']}")
    with open(os.path.join(args.out, "lambda_sweep.json"), "w") as f:
        json.dump(sanitize(sweep), f, indent=1)

    # ── beta sweep ──────────────────────────────────────────────────────────
    print("== beta sweep ==")
    bsweep = []
    for it in items:
        arr = arrs[it["name"]]
        pair = tuple(it["pair"])
        for beta in betas:
            res = evaluate(arr, pair, beta=beta, profile="search")
            bsweep.append({"name": it["name"], "label": it["label"],
                           "beta": beta, "loss": res["loss"],
                           "parts": res["parts"],
                           "stop_reasons": [r["stop"] for r in
                                            res["restarts"]],
                           "wall_time_s": res["wall_time_s"]})
    with open(os.path.join(args.out, "beta_sweep.json"), "w") as f:
        json.dump(sanitize(bsweep), f, indent=1)

    # ── restart-count study ─────────────────────────────────────────────────
    print("== restart study ==")
    rstudy = []
    for name in SUBSET:
        if name not in arrs:
            continue
        it = next(i for i in items if i["name"] == name)
        arr = arrs[name]
        pair = tuple(it["pair"])
        for nr in restarts:
            for seed in range(3):
                res = evaluate(arr, pair, profile="search", n_restarts=nr,
                               seed=seed)
                rstudy.append({"name": name, "label": it["label"],
                               "n_restarts": nr, "seed": seed,
                               "loss": res["loss"],
                               "gamma_median": res["gamma_median"],
                               "gamma_spread": res["gamma_spread"],
                               "restart_gammas": [r["gamma"] for r in
                                                  res["restarts"]],
                               "restart_iters": [r["iters"] for r in
                                                 res["restarts"]],
                               "stop_reasons": [r["stop"] for r in
                                                res["restarts"]],
                               "wall_time_s": res["wall_time_s"]})
    with open(os.path.join(args.out, "restart_study.json"), "w") as f:
        json.dump(sanitize(rstudy), f, indent=1)

    # ── iteration-budget study ──────────────────────────────────────────────
    print("== iteration study ==")
    istudy = []
    for name in SUBSET:
        if name not in arrs:
            continue
        it = next(i for i in items if i["name"] == name)
        arr = arrs[name]
        pair = tuple(it["pair"])
        for ni in iters:
            res = evaluate(arr, pair, profile="search", n_iters=ni)
            istudy.append({"name": name, "label": it["label"],
                           "n_iters": ni, "loss": res["loss"],
                           "stop_reasons": [r["stop"] for r in
                                            res["restarts"]],
                           "wall_time_s": res["wall_time_s"]})
    with open(os.path.join(args.out, "iteration_study.json"), "w") as f:
        json.dump(sanitize(istudy), f, indent=1)

    # ── coordinate-change sensitivity ───────────────────────────────────────
    print("== coordinate sensitivity ==")
    rng = np.random.default_rng(99)
    csens = []
    for name in SUBSET:
        if name not in arrs:
            continue
        it = next(i for i in items if i["name"] == name)
        m = np.array([l.to_float() for l in arrs[name].lines])
        pair = tuple(it["pair"])
        base = penalized_saito_loss(m, *pair, profile="search", seed=0)
        for kind in ("orthogonal", "projective"):
            for k in range(5):
                A = rng.standard_normal((3, 3))
                if kind == "orthogonal":
                    T, _ = np.linalg.qr(A)
                else:
                    T = A + 3.0 * np.eye(3)   # well-conditioned shear
                val = penalized_saito_loss(m @ T, *pair, profile="search",
                                           seed=0)
                csens.append({"name": name, "label": it["label"],
                              "kind": kind, "draw": k, "base_loss": base,
                              "transformed_loss": val,
                              "abs_change": abs(val - base)})
    with open(os.path.join(args.out, "coordinate_sensitivity.json"), "w") as f:
        json.dump(sanitize(csens), f, indent=1)

    # ── rescaling / permutation drift ───────────────────────────────────────
    print("== rescale / permutation ==")
    rp = []
    for name in SUBSET:
        if name not in arrs:
            continue
        it = next(i for i in items if i["name"] == name)
        m = np.array([l.to_float() for l in arrs[name].lines])
        pair = tuple(it["pair"])
        base = penalized_saito_loss(m, *pair, profile="search", seed=0)
        for k in range(5):
            scales = rng.uniform(0.2, 5.0, size=m.shape[0]) * \
                rng.choice([-1.0, 1.0], size=m.shape[0])
            val_s = penalized_saito_loss(m * scales[:, None], *pair,
                                         profile="search", seed=0)
            perm = rng.permutation(m.shape[0])
            val_p = penalized_saito_loss(m[perm], *pair, profile="search",
                                         seed=0)
            rp.append({"name": name, "draw": k, "base_loss": base,
                       "rescaled_loss": val_s, "permuted_loss": val_p,
                       "rescale_drift": abs(val_s - base),
                       "permute_drift": abs(val_p - base)})
    with open(os.path.join(args.out, "rescale_permutation.json"), "w") as f:
        json.dump(sanitize(rp), f, indent=1)

    # ── perturbation / degeneration path ────────────────────────────────────
    print("== perturbation path ==")
    pert = []
    for tag, arr, pair in perturbation_family():
        res = evaluate(arr, pair, profile="benchmark")
        leg = legacy_invalid_angular_score(arr, target_exponents=pair)
        pert.append({"tag": tag, "pair": list(pair),
                     "cand_exps": (list(arr.candidate_exponents())
                                   if arr.candidate_exponents() else None),
                     "loss": res["loss"], "legacy_score": leg,
                     "parts": res["parts"]})
        print(f"  {tag:10s} loss={res['loss']:.3e} legacy={leg:.3e}")
    with open(os.path.join(args.out, "perturbation.json"), "w") as f:
        json.dump(sanitize(pert), f, indent=1)

    # ── float64 vs exact/mpmath reference ───────────────────────────────────
    print("== reference check ==")
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "tests"))
    from reference_impl import gamma_reference
    from sympy import Rational
    from penalized_saito import _monoms, _bw_sqrt_weights
    refchecks = []
    for name, pair in (("A2xA1", (1, 2)), ("nonfree7_a", (3, 3)),
                       ("generic_4", (1, 2))):
        arr = arrs[name]
        d1, d2 = pair
        N1, N2 = len(_monoms(d1)), len(_monoms(d2))
        u_mono = [Rational((7 * k) % 11 - 5, 3) for k in range(3 * N1)]
        v_mono = [Rational((5 * k) % 13 - 6, 4) for k in range(3 * N2)]
        ref = float(gamma_reference(arr, d1, d2, u_mono, v_mono,
                                    lam=1.0, beta=0.5))
        sw1, sw2 = _bw_sqrt_weights(d1), _bw_sqrt_weights(d2)

        def to_bw(mono, N, sw):
            c = np.array([float(t) for t in mono])
            w = np.concatenate([c[:N] / sw, c[N:2 * N] / sw, c[2 * N:] / sw])
            return w / np.linalg.norm(w)

        ev = PenalizedSaitoEvaluator(arr, d1, d2)
        g = ev.gamma(to_bw(u_mono, N1, sw1), to_bw(v_mono, N2, sw2))
        refchecks.append({"name": name, "pair": list(pair),
                          "gamma_float64": g, "gamma_reference": ref,
                          "abs_diff": abs(g - ref)})
        print(f"  {name:12s} float64={g:.15e} ref={ref:.15e} "
              f"diff={abs(g-ref):.2e}")
    with open(os.path.join(args.out, "reference_check.json"), "w") as f:
        json.dump(sanitize(refchecks), f, indent=1)

    print("== done ==")


if __name__ == "__main__":
    main()
