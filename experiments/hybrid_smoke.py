"""
Bounded smoke experiment for the EXPERIMENTAL hybrid geometric refinement
(hybrid_refine.py).  Isolated: its own output directory, per-run cache
clearing, never touches production stores or running campaign outputs.

Arms (identical seeds, identical initial states, equal evaluation budgets):
  baseline : deterministic greedy swap descent (production components)
  refine   : same driver + one-line gradient refinement of the swapped-in
             line after each accepted move (--refine flag)

Budget unit = production Saito evaluations (screen losses + every maximize
performed inside refinement re-scoring).  Wall-clock is reported alongside;
an arm can never win merely by spending more evaluations.

Usage:
  python experiments/hybrid_smoke.py --out results_local/hybrid_gradient_experiment \
      [--n 14] [--budget 180] [--seeds 0 1 2] [--pairs 2,11 4,9 6,7]
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arrangement import LineArrangement
import penalized_saito
from penalized_saito import runtime_provenance, PenalizedSaitoEvaluator
from swap_search import (ChainEvaluator, propose_swaps, certify_state,
                         is_valid_state, canonical_lineset_key)
from novelty import lattice_wl_hash, is_supersolvable_rank3
from run_swap_campaign import build_seeds
from hybrid_refine import refine_line, OK


def novelty_labels(wl_hash, run_hashes, local_db_hashes):
    return {
        "new_in_this_run": wl_hash not in run_hashes,
        "new_relative_to_local_discovery_database":
            wl_hash not in local_db_hashes,
        "new_intersection_lattice_signature":
            wl_hash not in run_hashes and wl_hash not in local_db_hashes,
        "literature_novelty_unchecked": True,
    }


def load_local_db_hashes():
    """Every lattice hash this project has ever certified/indexed locally."""
    import glob
    hashes = set()
    ref = "results_penalized_saito/2026-08-17/swap/reference_hashes/headline.json"
    if os.path.exists(ref):
        for cell, info in json.load(open(ref)).items():
            hashes.update(info.get("hashes", {}).keys())
    if os.path.exists("reference_hashes_k_fixtures.json"):
        hashes.update(json.load(
            open("reference_hashes_k_fixtures.json")).keys())
    for path in glob.glob("results_*/**/certified.jsonl", recursive=True):
        try:
            for line in open(path):
                try:
                    hashes.add(json.loads(line)["lattice_hash"])
                except Exception:
                    continue
        except OSError:
            continue
    return hashes


def run_arm(arm, n, d1, d2, seed, budget, out_dir, local_db_hashes):
    """One (arm, cell, seed) run under an evaluation budget."""
    penalized_saito.clear_cache()          # no cross-arm cache subsidies
    rng = np.random.default_rng(1000003 * seed + 101 * n + d1)
    ev = ChainEvaluator(n, d1, d2, seed=seed)
    seeds = build_seeds(n, d1, d2, rng, 3, mode="mixed")
    state = seeds[0]
    metrics = {
        "arm": arm, "n": n, "d1": d1, "d2": d2, "seed": seed,
        "budget": budget, "evals": 0, "wall_s": 0.0,
        "best_raw_loss": None, "numerical_failures": 0,
        "certificates": 0, "unique_exact": set(), "unique_lattices": set(),
        "nonss_certified": 0, "time_to_first_cert": None,
        "refine_calls": 0, "refine_accepted": 0, "refine_statuses": {},
        "collisions": 0, "gamma_exceeds_one": 0,
        "best_gamma": None, "best_alignment": None,
        "best_residuals": None, "best_grad_norm": None,
        "restart_best_losses": [],
    }
    run_hashes = set()
    candidates = []
    t0 = time.time()

    def evals():
        return ev.n_screen + metrics["refine_calls_evals"] \
            if False else metrics["evals"]

    def spend(k=1):
        metrics["evals"] += k

    cur_loss = ev.screen_loss_or_none(state)
    spend()
    if cur_loss is None:
        metrics["numerical_failures"] += 1
        return metrics, candidates
    best = (state, cur_loss)
    restart_idx = 0
    tabu = {canonical_lineset_key(state)}
    _pending_initial = True     # initial/restart states go through consider

    def consider(arr, loss):
        nonlocal best
        if loss < best[1]:
            best = (arr, loss)
        if loss < 1e-6:
            cert = certify_state(arr, d1, d2)
            if cert is not None:
                wl = lattice_wl_hash(arr)
                ss = is_supersolvable_rank3(arr)
                labels = novelty_labels(wl, run_hashes, local_db_hashes)
                run_hashes.add(wl)
                metrics["certificates"] += 1
                metrics["unique_exact"].add(canonical_lineset_key(arr))
                metrics["unique_lattices"].add(wl)
                if not ss:
                    metrics["nonss_certified"] += 1
                if metrics["time_to_first_cert"] is None:
                    metrics["time_to_first_cert"] = time.time() - t0
                from certificates import certificate_to_json
                candidates.append({
                    "arm": arm, "seed": seed, "n": n, "d1": d1, "d2": d2,
                    "lines": [str(l) for l in arr.lines],
                    "loss": loss, "lattice_hash": wl,
                    "supersolvable": bool(ss),
                    "m_max": arr.max_multiplicity(),
                    "novelty": labels,
                    "certificate": certificate_to_json(cert),
                })

    consider(state, cur_loss)
    while metrics["evals"] < budget:
        props = propose_swaps(state, d1, d2, rng, n_remove=6, tabu=tabu)
        if not props:
            restart_idx += 1
            metrics["restart_best_losses"].append(best[1])
            state = seeds[restart_idx % len(seeds)]
            cur_loss = ev.screen_loss_or_none(state)
            spend()
            if cur_loss is None:
                metrics["numerical_failures"] += 1
                break
            consider(state, cur_loss)
            continue
        scored = []
        for (i, line, trial) in props:
            if metrics["evals"] >= budget:
                break
            l = ev.screen_loss_or_none(trial)
            spend()
            if l is None:
                metrics["numerical_failures"] += 1
                continue
            scored.append((l, i, trial))
        if not scored:
            break
        scored.sort(key=lambda t: t[0])
        l_new, i_new, arr_new = scored[0]
        if l_new >= cur_loss:
            restart_idx += 1
            metrics["restart_best_losses"].append(best[1])
            state = seeds[restart_idx % len(seeds)]
            cur_loss = ev.screen_loss_or_none(state)
            spend()
            if cur_loss is None:
                metrics["numerical_failures"] += 1
                break
            consider(state, cur_loss)
            continue
        state, cur_loss = arr_new, l_new
        tabu.add(canonical_lineset_key(state))
        consider(state, cur_loss)

        if arm == "refine" and metrics["evals"] < budget:
            metrics["refine_calls"] += 1
            new_idx = len(state) - 1        # swapped-in line is appended
            refined, rep = refine_line(state, new_idx, d1, d2,
                                       steps=10, seed=seed)
            spend(rep.get("eval_calls", 0))
            st = rep["status"]
            metrics["refine_statuses"][st] = \
                metrics["refine_statuses"].get(st, 0) + 1
            metrics["gamma_exceeds_one"] += rep.get("gamma_exceeds_one", 0)
            if st == "line_collision":
                metrics["collisions"] += 1
            if refined is not None and st == OK:
                metrics["refine_accepted"] += 1
                l_ref = ev.screen_loss_or_none(refined)
                spend()
                if l_ref is not None and l_ref < cur_loss:
                    state, cur_loss = refined, l_ref
                    tabu.add(canonical_lineset_key(state))
                    consider(state, cur_loss)
            if rep.get("grad_norms"):
                metrics["best_grad_norm"] = rep["grad_norms"][-1]

    metrics["wall_s"] = round(time.time() - t0, 1)
    metrics["best_raw_loss"] = best[1]
    # raw Gamma diagnostics at the best state (production evaluator)
    try:
        ev_b = PenalizedSaitoEvaluator(best[0], d1, d2)
        res_b = ev_b.maximize(n_restarts=4, n_iters=60, seed=seed)
        g, parts = ev_b.gamma(res_b["u"], res_b["v"], return_parts=True)
        metrics["best_gamma"] = g
        metrics["best_alignment"] = (
            abs(parts["inner_abs"]) /
            max(parts["B_norm"], 1e-300))
        metrics["best_residuals"] = [parts["L1u_norm"] ** 2,
                                     parts["L2v_norm"] ** 2]
    except Exception as e:
        metrics.setdefault("warnings", []).append(f"diag:{e}")
    metrics["unique_exact"] = len(metrics["unique_exact"])
    metrics["unique_lattices"] = len(metrics["unique_lattices"])
    return metrics, candidates


def recovery_test(out_dir):
    """Perturb one line of a verified free (14,6,7) discovery; check the
    refinement lowers the raw loss and returns to a certifiable state."""
    from sympy import Rational
    from arrangement import ProjectiveLine
    from novelty import parse_line_str
    from certificates import find_certificate_fast
    from penalized_saito import penalized_saito_loss
    path = "swap_lift_seeds/n14_d6_7.json"
    rec = json.load(open(path))["seeds"][0]
    arr = LineArrangement([parse_line_str(s) for s in rec["lines"]])
    a, b, c = arr.lines[-1].coords
    t = Rational(1, 3)
    pert = LineArrangement(list(arr.lines[:-1]) +
                           [ProjectiveLine(a + t, b - t, c + 2 * t)])
    loss_p = penalized_saito_loss(pert, 6, 7, profile="search", seed=0)
    refined, rep = refine_line(pert, len(pert) - 1, 6, 7, steps=20, seed=0)
    out = {"seed_lattice": rec["lattice_hash"],
           "perturbed_loss": loss_p, "status": rep["status"],
           "loss_after": rep.get("raw_loss_after"),
           "gamma_trace_len": len(rep.get("gamma_trace", [])),
           "recovered_certificate": False}
    if refined is not None:
        cert, status = find_certificate_fast(refined,
                                             target_exponents=(6, 7))
        out["recovered_certificate"] = status == "certified"
        out["recovered_lattice"] = lattice_wl_hash(refined)
        out["lattice_matches_seed"] = \
            out["recovered_lattice"] == rec["lattice_hash"]
    json.dump(out, open(os.path.join(out_dir, "recovery_test.json"), "w"),
              indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_local/hybrid_gradient_experiment")
    ap.add_argument("--n", type=int, default=14)
    ap.add_argument("--budget", type=int, default=180)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pairs", nargs="+", default=["2,11", "4,9", "6,7"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    manifest = {"args": vars(args), "provenance": runtime_provenance("."),
                "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"),
              indent=1)
    local_db = load_local_db_hashes()
    print(f"local discovery DB: {len(local_db)} lattice hashes", flush=True)

    rows, all_cands = [], []
    for pair in args.pairs:
        d1, d2 = map(int, pair.split(","))
        for seed in args.seeds:
            for arm in ("baseline", "refine"):
                m, cands = run_arm(arm, args.n, d1, d2, seed, args.budget,
                                   args.out, local_db)
                rows.append(m)
                all_cands.extend(cands)
                print(f"[{arm:8s}] ({args.n},{d1},{d2}) s{seed}: "
                      f"best={m['best_raw_loss']:.2e} evals={m['evals']} "
                      f"wall={m['wall_s']}s certs={m['certificates']} "
                      f"nonSS={m['nonss_certified']}", flush=True)
                json.dump(rows, open(os.path.join(args.out, "metrics.json"),
                                     "w"), indent=1, default=str)
    with open(os.path.join(args.out, "candidates.jsonl"), "w") as f:
        for c in all_cands:
            f.write(json.dumps(c) + "\n")
    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as f:
        keys = [k for k in rows[0] if not isinstance(rows[0][k],
                                                     (dict, list))]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

    rec = recovery_test(args.out)
    print("recovery test:", rec, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
