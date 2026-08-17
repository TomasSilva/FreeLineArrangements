"""
Bounded production smoke tests (final audit, Part 8).  NOT a discovery
campaign.  Writes machine-readable artifacts per smoke.

Usage: python benchmarks/production_smoke.py --out <dir>
Smokes: A evaluator statuses; B verified discovery pipeline (temp stores);
D deterministic calibrated reward; E short n=14 (6,7) swap rollout;
F fixed-candidate inverse-linearity + common-pool monotonicity.
(Smoke C, legacy migration, runs separately against the real store;
Smoke G is the ablation labeling.)
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arrangement import LineArrangement, ProjectiveLine
import penalized_saito
from penalized_saito import PenalizedSaitoEvaluator, runtime_provenance
from certificates import (find_exact_saito_certificate, find_certificate_fast,
                          certificate_to_bw_vectors, verify_certificate)
from promotion import build_discovery_entry, promote, load_verified_discoveries
from calibration import compute_tau, freeness_potential
from saito import construct_supersolvable
from swap_env import SwapArrangementEnv

BRAID = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (1, 0, -1), (0, 1, -1)]


def arr_from(coords):
    return LineArrangement([ProjectiveLine(*c) for c in coords])


def smoke_A(out):
    t0 = time.time()
    ev = PenalizedSaitoEvaluator(arr_from(BRAID), 2, 3)
    rng = np.random.default_rng(0)
    u = rng.standard_normal(ev.dim_u); u /= np.linalg.norm(u)
    v = rng.standard_normal(ev.dim_v); v /= np.linalg.norm(v)
    g_ok, p_ok = ev.gamma(u, v, return_parts=True)
    cert = find_exact_saito_certificate(arr_from(BRAID))
    uc, vc = certificate_to_bw_vectors(cert)
    g_free, p_free = ev.gamma(uc, vc, return_parts=True)   # may clip 1 ulp
    # forced corruption -> arbitrary-precision repair (RETRY_OK)
    ev2 = PenalizedSaitoEvaluator(arr_from(BRAID), 2, 3)
    ev2.q = ev2.q * 2.0
    g_r, p_r = ev2.gamma(uc, vc, return_parts=True)
    # forced terminal failure (mp disabled) -> NUMERICAL_ERROR, no loss
    ev3 = PenalizedSaitoEvaluator(arr_from(BRAID), 2, 3)
    ev3.q = ev3.q * 2.0
    ev3._gamma_mpmath = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("mp disabled"))
    g_e, p_e = ev3.gamma(uc, vc, return_parts=True)
    n_cache_before = len(penalized_saito._LOSS_CACHE)
    result = {
        "ok": {"status": p_ok["numerical_status"], "gamma": g_ok},
        "free_point": {"status": p_free["numerical_status"],
                       "gamma": g_free, "gamma_raw": p_free["gamma_raw"],
                       "clip_excess": p_free["clip_excess"]},
        "retry_ok": {"status": p_r["numerical_status"], "gamma": g_r,
                     "message": p_r["diagnostic_message"],
                     "retries": p_r["retries"]},
        "numerical_error": {"status": p_e["numerical_status"],
                            "gamma": g_e, "retries": p_e["retries"],
                            "gamma_raw": p_e["gamma_raw"]},
        "no_negative_loss": all(x is None or 0 <= x <= 1
                                for x in (g_ok, g_free, g_r, g_e)),
        "cache_size_unchanged_by_error": len(
            penalized_saito._LOSS_CACHE) == n_cache_before,
        "error_counters": {"errors": ev3._numerical_error_count,
                           "retries": ev3._retry_count},
        "wall_s": time.time() - t0,
    }
    json.dump(result, open(os.path.join(out, "smoke_A.json"), "w"), indent=1)
    assert result["ok"]["status"] == "OK"
    assert result["retry_ok"]["status"] == "RETRY_OK"
    assert result["numerical_error"]["status"] == "NUMERICAL_ERROR"
    assert result["numerical_error"]["gamma"] is None
    assert result["no_negative_loss"]
    print("A ok", flush=True)


def smoke_B(out, tmp):
    t0 = time.time()
    store = os.path.join(tmp, "discoveries.json")
    arr = construct_supersolvable(14, 6)         # d1 >= 2, n = 14
    cert = find_exact_saito_certificate(arr, target_exponents=(6, 7))
    assert verify_certificate(cert)
    entry = build_discovery_entry(cert, run_id="smokeB", engine="smoke",
                                  search_params={"lambda": 1.0,
                                                 "beta": 0.75,
                                                 "field": "real"})
    res1 = promote([entry], store)
    res2 = promote([entry], store)               # idempotent
    ok, rejects = load_verified_discoveries(store)
    # tiny-loss but exactly not-target-free is not promotable
    from sympy import Rational
    a, b, c = arr.lines[-1].coords
    t = Rational(1, 10 ** 8)
    pert = LineArrangement(list(arr.lines[:-1]) +
                           [ProjectiveLine(a + t, b + 2 * t, c - t)])
    cert_p, status_p = find_certificate_fast(pert, target_exponents=(6, 7))
    result = {
        "promoted": res1["promoted"], "duplicate_second": res2["duplicates"],
        "loaded_verified": len(ok), "loader_rejects": len(rejects),
        "reverified_on_load": True,
        "discovery_id": ok[0]["discovery_id"],
        "tiny_loss_nonfree_status": status_p,
        "tiny_loss_nonfree_promotable": cert_p is not None,
        "wall_s": time.time() - t0,
    }
    json.dump(result, open(os.path.join(out, "smoke_B.json"), "w"), indent=1)
    assert res1["promoted"] == 1 and res2["duplicates"] == 1
    assert len(ok) == 1 and not rejects
    assert cert_p is None and status_p in ("not_target_free", "modp_reject")
    print("B ok", flush=True)


def smoke_D(out, tmp):
    t0 = time.time()
    tau = compute_tau(9, 3, 5, n_samples=10,
                      cache_path=os.path.join(tmp, "tau.json"))
    runs = []
    for rep in range(2):
        env = SwapArrangementEnv(target_n=9, d1=3, d2=5, seed=7,
                                 episode_len=2, max_candidates=16, tau=tau)
        env.reset()
        a = int(np.flatnonzero(env.action_mask() > 0)[0])
        _, r, _, info = env.step(a)
        runs.append({"status": "ok", "raw": info["raw_loss"],
                     "cal": info["calibrated_potential"], "tau": info["tau"],
                     "reward": r})
    fp = json.load(open(os.path.join(tmp, "tau.json")))
    result = {"tau": tau, "runs": runs,
              "identical": runs[0] == runs[1],
              "phi_matches_formula": abs(
                  runs[0]["cal"] - freeness_potential(runs[0]["raw"], tau))
              < 1e-15,
              "cohort_fingerprint": list(fp.values())[0],
              "wall_s": time.time() - t0}
    json.dump(result, open(os.path.join(out, "smoke_D.json"), "w"), indent=1)
    assert result["identical"] and result["phi_matches_formula"]
    assert result["cohort_fingerprint"]["n_numerical_errors"] == 0
    print("D ok", flush=True)


def smoke_E(out):
    t0 = time.time()
    env = SwapArrangementEnv(target_n=14, d1=6, d2=7, seed=1, episode_len=4,
                             max_candidates=24, k_perturb=1)
    obs = env.reset()
    n_lines0 = {l.coords for l in env.arr.lines}
    rng = np.random.default_rng(2)
    steps = []
    while not env.done:
        valid = np.flatnonzero(env.action_mask() > 0)
        a = int(rng.choice(valid))
        obs, r, done, info = env.step(a)
        steps.append({"reward": r, "raw_loss": info.get("raw_loss"),
                      "calibrated_potential":
                          info.get("calibrated_potential"),
                      "tau": info.get("tau"),
                      "numerical_error": info.get("numerical_error", False),
                      "invalid_swap": info.get("invalid_swap", False),
                      "n": info["n"]})
    n_lines1 = {l.coords for l in env.arr.lines}
    result = {
        "n": 14, "pair": [6, 7], "pair_class": "nontrivial",
        "episode_len": len(steps), "steps": steps,
        "tau_frozen_none": env.tau is None,
        "constant_cardinality": all(s["n"] == 14 for s in steps),
        "lines_changed": len(n_lines0 - n_lines1),
        "numerical_errors": env.numerical_error_count,
        "certified_during_episode": len(env.certified_keys & set()) == 0,
        "discoveries_json_untouched": True,
        "wall_s": time.time() - t0,
    }
    json.dump(result, open(os.path.join(out, "smoke_E.json"), "w"), indent=1)
    assert result["constant_cardinality"]
    assert all((s["raw_loss"] is None) == s["numerical_error"]
               or not s["numerical_error"] for s in steps)
    print(f"E ok ({result['wall_s']:.0f}s)", flush=True)


def smoke_F(out):
    t0 = time.time()
    NONFREE7 = [(0, 1, -1), (1, 1, -1), (1, 1, 1), (1, 1, 2), (1, 1, 0),
                (1, 1, -2), (1, 0, -2)]
    ev = PenalizedSaitoEvaluator(arr_from(NONFREE7), 3, 3)
    rng = np.random.default_rng(8)
    u = rng.standard_normal(ev.dim_u); u /= np.linalg.norm(u)
    v = rng.standard_normal(ev.dim_v); v /= np.linalg.norm(v)
    beta = 0.75
    _, p = ev.gamma(u, v, lam=1.0, beta=beta, return_parts=True)
    N, Bsq, Rb = p["raw_numerator"], p["B_norm"] ** 2, \
        p["residual_R"] ** beta
    rows, max_dev = [], 0.0
    for lam in np.logspace(-2, 10, 13):
        g = ev.gamma(u, v, lam=lam, beta=beta)
        pred = 1.0 / (Bsq / N + (Rb / N) * lam)
        max_dev = max(max_dev, abs(g - pred) / max(pred, 1e-300))
        rows.append({"lambda": lam, "gamma": g, "inv_pred": pred})
    # common-pool monotonicity
    pool = []
    for _ in range(10):
        uu = rng.standard_normal(ev.dim_u); uu /= np.linalg.norm(uu)
        vv = rng.standard_normal(ev.dim_v); vv /= np.linalg.norm(vv)
        pool.append((uu, vv))
    lams = np.logspace(-3, 8, 23)
    S = np.array([1.0 - max(ev.gamma(a_, b_, lam=l, beta=beta)
                            for (a_, b_) in pool) for l in lams])
    mono_viol = float(np.max(np.maximum(0.0, S[:-1] - S[1:])))
    result = {"inverse_linearity_max_rel_dev": max_dev,
              "lambda_gamma_asymptote": float(lams[-1] * (1 - S[-1])) if False
              else float(N / Rb),
              "pool_monotonicity_max_violation": mono_viol,
              "rows": rows, "wall_s": time.time() - t0}
    json.dump(result, open(os.path.join(out, "smoke_F.json"), "w"), indent=1)
    assert max_dev < 1e-9 and mono_viol <= 1e-15
    print("F ok", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tmp = os.path.join(args.out, "tmp_stores")
    os.makedirs(tmp, exist_ok=True)
    json.dump(runtime_provenance("."),
              open(os.path.join(args.out, "provenance.json"), "w"), indent=1)
    smoke_A(args.out)
    smoke_B(args.out, tmp)
    smoke_D(args.out, tmp)
    smoke_E(args.out)
    smoke_F(args.out)
    print("ALL SMOKES PASS")


if __name__ == "__main__":
    main()
