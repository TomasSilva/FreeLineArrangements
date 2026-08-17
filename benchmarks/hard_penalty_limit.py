"""
Hard-penalty-limit verification for the penalized Saito functional.

Validates, on an exactly-labeled suite, that S_{lambda,beta} -> 1 as
lambda -> infinity for every NON-TARGET-FREE (arrangement, pair) — including
free arrangements evaluated at a wrong admissible pair — and stays 0 for
every lambda on target-free controls (seeded with their exact Saito pairs).
Mathematical statement and proof: docs/hard_penalty_limit_proof.md.  This
script validates the IMPLEMENTATION; it does not prove the theorem.

Two evaluations per (arrangement, pair, beta, lambda):
  native  — independent production multistart (upper bound S_hat >= S);
  pool    — max Gamma over a COMMON candidate pool (all initializations +
            every lambda's final iterate + exact Saito pairs for controls),
            re-evaluated with the exact objective at every lambda.  Since
            each fixed candidate's Gamma is nonincreasing in lambda, the
            pool loss is exactly nondecreasing — separating true
            lambda-monotonicity from optimizer noise.

Usage:
  python benchmarks/hard_penalty_limit.py --out results_penalized_saito/<date>/hard_penalty
"""

import argparse
import csv
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arrangement import LineArrangement, ProjectiveLine
from penalized_saito import (PenalizedSaitoEvaluator, runtime_provenance,
                             kernel_diagnostics_from_operator)
from saito import construct_supersolvable
from certificates import (find_exact_saito_certificate, find_certificate_fast,
                          certificate_to_bw_vectors)

LAMBDAS = [10.0 ** k for k in range(-3, 9)]      # 1e-3 .. 1e8
LAMBDA_MAX_ADAPTIVE = 1e12
BETAS = [0.5, 0.75, 0.9]
TOL_ASYMPT = 1e-3
SEED = 0
BLUE, ORANGE, INK2 = "#2a78d6", "#eb6834", "#52514e"


def arr_from(coords):
    return LineArrangement([ProjectiveLine(*c) for c in coords])


def lines_matrix(arr):
    return np.array([l.to_float() for l in arr.lines])


NONFREE7 = [(0, 1, -1), (1, 1, -1), (1, 1, 1), (1, 1, 2), (1, 1, 0),
            (1, 1, -2), (1, 0, -2)]
GENERIC5 = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, 3), (3, -1, 2)]
GENERIC6 = GENERIC5 + [(2, 5, -1)]
BRAID = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (1, 0, -1), (0, 1, -1)]


def perturb(arr, num, den):
    from sympy import Rational
    a, b, c = arr.lines[-1].coords
    t = Rational(num, den)
    return LineArrangement(list(arr.lines[:-1]) +
                           [ProjectiveLine(a + t, b + 2 * t, c - t)])


def gl_shear(m, s1=9, s2=7):
    A = np.eye(3)
    A[0, 1], A[1, 2], A[2, 0] = s1, s2, 1.0
    return m @ A


def build_suite():
    """(name, lines_matrix, d1, d2, cls, exact_method, warm) — every label
    established by a COMPLETE exact criterion, recorded per item."""
    suite = []

    def free_ctrl(name, arr, d1, d2):
        cert = find_exact_saito_certificate(arr, target_exponents=(d1, d2))
        assert cert is not None
        warm = [certificate_to_bw_vectors(cert)]
        suite.append((name, lines_matrix(arr), d1, d2, "target_free",
                      "exact_saito_certificate", warm))
        return arr

    def nonfree_item(name, arr_or_m, d1, d2, cls, method):
        m = arr_or_m if isinstance(arr_or_m, np.ndarray) \
            else lines_matrix(arr_or_m)
        suite.append((name, m, d1, d2, cls, method, None))

    # 1. exactly certified nonfree (complete basis-pair scan; see
    #    certificates.py docstring for why the negative is exact)
    nf7 = arr_from(NONFREE7)
    assert nf7.is_free()[0] is False
    nonfree_item("nonfree7", nf7, 3, 3, "nonfree_exact",
                 "exact_basis_pair_scan(is_free)")
    from bench_common import _search_nonfree_with_exponents
    nf9 = _search_nonfree_with_exponents(9, 1, seed=11)[0]
    nonfree_item("nonfree9", nf9, *nf9.candidate_exponents(),
                 "nonfree_exact", "exact_basis_pair_scan(is_free)")

    # 2. generic, only double points: chi(A,t) has no integer candidate
    #    exponents -> Terao factorization obstruction (complete criterion)
    g5 = arr_from(GENERIC5)
    assert g5.candidate_exponents() is None
    nonfree_item("generic5", g5, 1, 3, "nonfree_exact",
                 "terao_factorization_obstruction")
    g6 = arr_from(GENERIC6)
    assert g6.candidate_exponents() is None
    nonfree_item("generic6", g6, 1, 4, "nonfree_exact",
                 "terao_factorization_obstruction")

    # 3. nonfree numerically close to free (exact negative via the complete
    #    point-evaluation pair scan)
    p9 = perturb(construct_supersolvable(9, 3), 1, 10**4)
    _, st = find_certificate_fast(p9, target_exponents=(3, 5))
    assert st in ("not_target_free", "modp_reject")
    nonfree_item("perturbed_ss9_t1e-4", p9, 3, 5, "near_free_nonfree_exact",
                 f"exact_point_eval_pair_scan({st})")
    p14 = perturb(construct_supersolvable(14, 6), 1, 10**6)
    _, st14 = find_certificate_fast(p14, target_exponents=(6, 7))
    assert st14 in ("not_target_free", "modp_reject")
    nonfree_item("perturbed_ss14_t1e-6", p14, 6, 7,
                 "near_free_nonfree_exact",
                 f"exact_point_eval_pair_scan({st14})")

    # 4. free at the correct pair (controls, warm-seeded with exact pairs)
    braid = free_ctrl("braid_correct", arr_from(BRAID), 2, 3)
    ss12 = free_ctrl("ss12_correct", construct_supersolvable(12, 5), 5, 6)
    ss14 = free_ctrl("ss14_correct", construct_supersolvable(14, 6), 6, 7)
    free_ctrl("ss15_correct", construct_supersolvable(15, 5), 5, 9)

    # 5. free at a DIFFERENT admissible pair (non-target-free by exponent
    #    uniqueness; certificate at the true pair recorded as the method)
    nonfree_item("braid_wrongpair", braid, 1, 4, "free_wrong_pair",
                 "free_with_exponents_(2,3)_certified")
    nonfree_item("ss12_wrongpair", ss12, 4, 7, "free_wrong_pair",
                 "free_with_exponents_(5,6)_certified")
    # 6. n >= 14 wrong pair
    nonfree_item("ss14_wrongpair", ss14, 5, 8, "free_wrong_pair",
                 "free_with_exponents_(6,7)_certified")

    # 7. ill-conditioned coordinate images (freeness/nonfreeness is
    #    projectively invariant; labels inherited from the exact originals)
    nonfree_item("nonfree7_glshear", gl_shear(lines_matrix(nf7)), 3, 3,
                 "nonfree_exact_illcond",
                 "gl_image_of_exact_basis_pair_scan")
    nonfree_item("braid_wrongpair_glshear", gl_shear(lines_matrix(braid)),
                 1, 4, "free_wrong_pair_illcond",
                 "gl_image_of_free_(2,3)_certified")
    return suite


def sweep_item(name, m, d1, d2, cls, method, warm, out_rows):
    ev = PenalizedSaitoEvaluator(m, d1, d2)
    kd = {"d1": kernel_diagnostics_from_operator(ev.L1),
          "d2": kernel_diagnostics_from_operator(ev.L2)}
    results = {}
    for beta in BETAS:
        pool = []
        if warm:
            pool.extend(warm)
        inits = ev._initial_points(np.random.default_rng(SEED), 8, warm,
                                   True, lam=1.0, beta=beta)
        pool.extend([(u, v) for (u, v, _) in inits])
        lambdas = list(LAMBDAS)
        native = {}
        i = 0
        while i < len(lambdas):
            lam = lambdas[i]
            t0 = time.perf_counter()
            res = ev.maximize(lam=lam, beta=beta, n_restarts=8, n_iters=80,
                              seed=SEED, warm_starts=warm)
            res["wall_s"] = time.perf_counter() - t0
            native[lam] = res
            pool.append((res["u"], res["v"]))
            # adaptive extension on the last grid point
            if i == len(lambdas) - 1 and cls != "target_free":
                s_now = res["loss"]
                if (1 - s_now) > TOL_ASYMPT and lam < LAMBDA_MAX_ADAPTIVE:
                    lambdas.append(lam * 10)
            i += 1
        # common-pool envelope: exact objective on the fixed pool
        pool_S = {}
        pool_best_parts = {}
        for lam in lambdas:
            best = (0.0, None)
            for (u, v) in pool:
                g, parts = ev.gamma(u, v, lam=lam, beta=beta,
                                    return_parts=True)
                if g > best[0]:
                    best = (g, parts)
            pool_S[lam] = 1.0 - best[0]
            pool_best_parts[lam] = best[1]
        for lam in lambdas:
            res = native[lam]
            p = res["parts"]
            out_rows.append({
                "name": name, "class": cls, "exact_method": method,
                "n": int(m.shape[0]), "d1": d1, "d2": d2, "beta": beta,
                "lambda": lam,
                "S_hat_native": res["loss"],
                "gamma_raw_native": p["gamma_raw"],
                "S_hat_pool": pool_S[lam],
                "B_norm": p["B_norm"], "inner_abs": p["inner_abs"],
                "L1u_norm": p["L1u_norm"], "L2v_norm": p["L2v_norm"],
                "R": p["residual_R"], "denominator": p["denominator"],
                "n_restarts": res["n_restarts"],
                "iters": [r["iters"] for r in res["restarts"]],
                "stop_reasons": sorted({r["stop"] for r in res["restarts"]}),
                "init_kinds": sorted({r["init"] for r in res["restarts"]}),
                "proj_grad_norm": res.get("proj_grad_norm"),
                "gamma_spread": res["gamma_spread"],
                "clip_count": res["gamma_clip_count"],
                "clip_max_excess": res["gamma_clip_max_excess"],
                "seed": SEED, "field": res["optimization_field"],
                "wall_s": res["wall_s"],
                "sigma_min_pos_L1": kd["d1"]["sigma_min_pos"],
                "sigma_min_pos_L2": kd["d2"]["sigma_min_pos"],
                "cond_L1": kd["d1"]["cond_estimate"],
            })
        results[beta] = (lambdas, native, pool_S)
    return ev, results


def run_checks(rows):
    """Part-5 checks per (name, beta); returns checks dict + failure list."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["name"], r["beta"])].append(r)
    checks, failures = {}, []
    for (name, beta), rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["lambda"])
        cls = rs[0]["class"]
        lam = np.array([r["lambda"] for r in rs])
        s_pool = np.array([r["S_hat_pool"] for r in rs])
        s_nat = np.array([r["S_hat_native"] for r in rs])
        entry = {"class": cls, "n_lambdas": len(rs)}
        entry["bounds_ok"] = bool(np.all((s_nat >= 0) & (s_nat <= 1)
                                         & (s_pool >= -1e-15)
                                         & (s_pool <= 1 + 1e-15)))
        mono_viol = float(np.max(np.maximum(0.0, s_pool[:-1] - s_pool[1:])))
        entry["pool_monotone_max_violation"] = mono_viol
        entry["pool_monotone_ok"] = mono_viol <= 1e-12
        entry["native_monotone_violations"] = int(np.sum(
            s_nat[:-1] > s_nat[1:] + 1e-12))
        entry["clip_max_excess"] = max(r["clip_max_excess"] for r in rs)
        if cls == "target_free":
            entry["free_control_max_S"] = float(np.max(s_pool))
            # At the exact Saito pair R is float dust (~1e-32, exact value
            # 0); the penalty lam * R^beta amplifies it at huge lam, most at
            # beta = 0.5 (sqrt).  Scale the tolerance accordingly: dust
            # bound ~ lam_max * (1e-30)^beta relative, floored at 1e-8.
            lam_max = float(np.max(lam))
            dust_tol = max(1e-8, 30.0 * lam_max * (1e-30) ** beta)
            entry["free_control_tol"] = dust_tol
            entry["free_control_ok"] = bool(np.max(s_pool) < dust_tol)
            if not entry["free_control_ok"]:
                failures.append(f"{name}/b{beta}: free control S "
                                f"{np.max(s_pool):.2e} > tol {dust_tol:.1e}")
        else:
            one_minus = np.maximum(1.0 - s_pool, 1e-300)
            entry["final_one_minus_S"] = float(one_minus[-1])
            tail_ok = bool(len(one_minus) >= 2
                           and np.all(one_minus[-2:] < TOL_ASYMPT))
            entry["asymptotic_tol_ok"] = tail_ok
            # log-log slope of the tail of 1 - S (where meaningful)
            sel = (one_minus > 1e-13) & (one_minus < 0.5)
            if np.sum(sel) >= 3:
                slope = np.polyfit(np.log10(lam[sel]),
                                   np.log10(one_minus[sel]), 1)[0]
                entry["loglog_slope"] = float(slope)
                entry["slope_O_1_over_lambda_ok"] = bool(-1.35 <= slope
                                                         <= -0.65)
            lts = lam * one_minus
            entry["lambda_times_one_minus_S_tail"] = [float(x)
                                                      for x in lts[-3:]]
            entry["lts_bounded_ok"] = bool(lts[-1] <= 3.0 * max(lts[-2],
                                                                1e-300))
            # Adaptive convergence disposition (spec: no arbitrary fixed
            # terminal value unless lambda is chosen relative to C_A):
            #   'tolerance_reached'  1-S < TOL_ASYMPT on the last two lams;
            #   'rate_confirmed'     slope ~ -1 AND lambda*(1-S) plateaued —
            #                        the O(1/lambda) limit is confirmed, the
            #                        plateau IS the empirical C_A*2^(1-beta),
            #                        merely larger than lam_max * TOL;
            #   'not_converged'      neither.
            plateaued = bool(len(lts) >= 3 and
                             max(lts[-3:]) <= 3.0 * max(min(lts[-3:]),
                                                        1e-300))
            if tail_ok:
                entry["asymptotic_disposition"] = "tolerance_reached"
            elif entry.get("slope_O_1_over_lambda_ok", False) and plateaued:
                entry["asymptotic_disposition"] = "rate_confirmed"
                entry["empirical_CA_2_1mb"] = float(lts[-1])
            else:
                entry["asymptotic_disposition"] = "not_converged"
            for key in ("pool_monotone_ok", "lts_bounded_ok"):
                if not entry.get(key, True):
                    failures.append(f"{name}/b{beta}: {key} failed")
            if entry["asymptotic_disposition"] == "not_converged":
                failures.append(f"{name}/b{beta}: not converged "
                                f"(slope {entry.get('loglog_slope')})")
            if not entry.get("slope_O_1_over_lambda_ok", True):
                failures.append(f"{name}/b{beta}: slope "
                                f"{entry.get('loglog_slope')}")
        if not entry["bounds_ok"]:
            failures.append(f"{name}/b{beta}: bounds violated")
        checks[f"{name}|beta={beta}"] = entry
    return checks, failures


def make_plots(rows, out):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        if r["beta"] == 0.75:
            groups[r["name"]].append(r)
    figs = {
        "S_vs_lambda.png": ("S_hat (pool)", False, lambda l, s: (l, s)),
        "one_minus_S_loglog.png": ("1 - S_hat (pool)", True,
                                   lambda l, s: (l, np.maximum(1 - s,
                                                               1e-16))),
        "lambda_times_gap.png": ("lambda * (1 - S_hat)", False,
                                 lambda l, s: (l, l * np.maximum(1 - s,
                                                                 0.0))),
    }
    for fname, (ylabel, loglog, xform) in figs.items():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, rs in sorted(groups.items()):
            rs.sort(key=lambda r: r["lambda"])
            lam = np.array([r["lambda"] for r in rs])
            s = np.array([r["S_hat_pool"] for r in rs])
            free = rs[0]["class"] == "target_free"
            x, y = xform(lam, s)
            ax.plot(x, y, color=BLUE if free else ORANGE, alpha=0.6,
                    marker="o", markersize=3)
        if fname == "one_minus_S_loglog.png":
            lam_g = np.array([1e-3, 1e8])
            ax.plot(lam_g, 1.0 / lam_g, "--", color=INK2, linewidth=1,
                    label="1/lambda guide")
            ax.set_yscale("log")
            ax.legend()
        ax.set_xscale("log")
        if loglog:
            ax.set_yscale("log")
        ax.set_xlabel("lambda")
        ax.set_ylabel(ylabel + "   (beta = 0.75)")
        ax.plot([], [], color=BLUE, label="target-free controls")
        ax.plot([], [], color=ORANGE, label="non-target-free")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), dpi=150)
        plt.close(fig)


def diagnostic_CA(ev, rng, n_samples=200):
    """Diagnostic (NOT certified) constant C_A = 3 M_B^2 (c1^2+c2^2+c1^2c2^2)
    from float singular values and sampled ||B||."""
    def c_of(L):
        s = np.linalg.svd(L, compute_uv=False)
        pos = s[s > 1e-8 * max(1.0, s[0])]
        return 1.0 / pos[-1] if len(pos) else np.inf
    c1, c2 = c_of(ev.L1), c_of(ev.L2)
    mb = 0.0
    for _ in range(n_samples):
        u = rng.standard_normal(ev.dim_u)
        v = rng.standard_normal(ev.dim_v)
        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)
        mb = max(mb, float(np.linalg.norm(ev.B_bw(u, v))))
    return {"c1": float(c1), "c2": float(c2), "M_B_sampled": mb,
            "C_A_diag": float(3 * mb ** 2 * (c1 ** 2 + c2 ** 2
                                             + c1 ** 2 * c2 ** 2)),
            "note": "float diagnostic, not a certified bound"}


def precision_check():
    """float64 Gamma vs the exact-rational/mpmath reference at fixed
    rational points, several lambdas (beta = 0.5, the reference's beta)."""
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", "tests"))
    from reference_impl import gamma_reference
    from sympy import Rational
    from penalized_saito import _monoms, _bw_sqrt_weights
    out = []
    for coords, d1, d2 in ((NONFREE7, 3, 3), (GENERIC5, 1, 3)):
        arr = arr_from(coords)
        N1, N2 = len(_monoms(d1)), len(_monoms(d2))
        u_mono = [Rational((7 * k) % 11 - 5, 3) for k in range(3 * N1)]
        v_mono = [Rational((5 * k) % 13 - 6, 4) for k in range(3 * N2)]
        sw1, sw2 = _bw_sqrt_weights(d1), _bw_sqrt_weights(d2)

        def to_bw(mono, N, sw):
            c = np.array([float(t) for t in mono])
            w = np.concatenate([c[:N] / sw, c[N:2 * N] / sw,
                                c[2 * N:] / sw])
            return w / np.linalg.norm(w)

        ev = PenalizedSaitoEvaluator(lines_matrix(arr), d1, d2)
        for lam in (1e-3, 1.0, 1e6):
            ref = float(gamma_reference(arr, d1, d2, u_mono, v_mono,
                                        lam=lam, beta=0.5))
            g = ev.gamma(to_bw(u_mono, N1, sw1), to_bw(v_mono, N2, sw2),
                         lam=lam, beta=0.5)
            out.append({"case": f"n{len(arr)}_d{d1}{d2}", "lambda": lam,
                        "gamma_float64": g, "gamma_reference": ref,
                        "abs_diff": abs(g - ref)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "plots"), exist_ok=True)

    print("== building exactly-labeled suite ==", flush=True)
    suite = build_suite()
    rows = []
    ca_diags = {}
    rng = np.random.default_rng(99)
    for (name, m, d1, d2, cls, method, warm) in suite:
        t0 = time.time()
        ev, _ = sweep_item(name, m, d1, d2, cls, method, warm, rows)
        if name in ("nonfree7", "braid_wrongpair", "perturbed_ss9_t1e-4"):
            ca_diags[name] = diagnostic_CA(ev, rng)
        print(f"  {name:26s} [{cls:24s}] done ({time.time()-t0:.0f}s)",
              flush=True)

    checks, failures = run_checks(rows)
    prec = precision_check()
    make_plots(rows, os.path.join(args.out, "plots"))

    with open(os.path.join(args.out, "sweep.json"), "w") as f:
        json.dump({"rows": rows, "checks": checks, "failures": failures,
                   "C_A_diagnostics": ca_diags,
                   "precision_check": prec,
                   "provenance": runtime_provenance(".")}, f, indent=1)
    with open(os.path.join(args.out, "sweep.csv"), "w", newline="") as f:
        cols = [k for k in rows[0] if k not in ("iters", "stop_reasons",
                                                "init_kinds")]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("== checks ==")
    n_ok = sum(1 for c in checks.values()
               if all(v for k, v in c.items() if k.endswith("_ok")))
    print(f"groups: {len(checks)}, all-ok: {n_ok}, failures: {len(failures)}")
    for fmsg in failures:
        print("  FAIL:", fmsg)
    print("precision:", max(p["abs_diff"] for p in prec))


if __name__ == "__main__":
    main()
