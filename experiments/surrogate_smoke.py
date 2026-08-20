"""
Equal-budget benchmark: surrogate-ranked proposals vs production proposals,
with the epsilon-directed machinery (m_max ceilings) active in both arms.

Isolated (own out dir, per-run cache clearing); identical seeds and initial
states; budget = TRUE Saito evaluations (surrogate ranking is charged zero
because it never calls the evaluator — that asymmetry is the point being
measured: better spend of the same true-evaluation budget).

Arms:
  baseline  : greedy swap descent, production propose_swaps
  surrogate : same driver, WIDE proposals ranked by the trained model

Usage:
  python experiments/surrogate_smoke.py \
      --model results_local/surrogate_experiment/model.pt \
      --out results_local/surrogate_experiment/smoke \
      [--budget 250] [--seeds 0 1 2] \
      [--cells 18,8,9,5 20,9,10,6 20,9,10,5]     # n,d1,d2,max_mult
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

import penalized_saito
from penalized_saito import runtime_provenance
from swap_search import (ChainEvaluator, propose_swaps, certify_state,
                         is_valid_state, canonical_lineset_key,
                         perturb_k_swaps, min_feasible_m)
from novelty import lattice_wl_hash, is_supersolvable_rank3
from run_swap_campaign import build_seeds
from surrogate import SurrogateRanker


def smoke_seeds(n, d1, d2, max_mult, rng):
    """Identical-across-arms starts: certified cell records from past runs
    (lowest height, distinct lattices, ceiling-compatible), perturbed TWO
    swaps away — so the benchmark measures how efficiently each arm's
    proposal selection walks back to (and around) the free locus."""
    import glob
    from novelty import arrangement_from_record
    recs, seen = [], set()
    for p in sorted(glob.glob("results_from_HPC/**/certified.jsonl",
                              recursive=True)
                    + glob.glob("results_local/**/certified.jsonl",
                                recursive=True)):
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if (r.get("n"), r.get("d1"), r.get("d2")) != (n, d1, d2):
                continue
            if r.get("coefficient_field", "QQ") != "QQ":
                continue
            if max_mult is not None and r.get("m_max", 99) > max_mult:
                continue
            if r["lattice_hash"] in seen:
                continue
            seen.add(r["lattice_hash"])
            recs.append(r)
    recs.sort(key=lambda r: r.get("height", 1 << 30))
    out = []
    for r in recs[:3]:
        try:
            base = arrangement_from_record(r)
        except Exception:
            continue
        pert = perturb_k_swaps(base, 2, rng)
        if is_valid_state(pert, n, nontrivial=True):
            out.append(pert)
    return out


def run_arm(arm, n, d1, d2, max_mult, seed, budget, ranker, local_db):
    penalized_saito.clear_cache()
    rng = np.random.default_rng(7000003 * seed + 131 * n + d1)
    ev = ChainEvaluator(n, d1, d2, seed=seed, m_target=max_mult)
    seeds = smoke_seeds(n, d1, d2, max_mult, rng)
    if not seeds:
        seeds = build_seeds(n, d1, d2, rng, 3, mode="mixed")
    state = seeds[0]
    m = {"arm": arm, "n": n, "d1": d1, "d2": d2, "max_mult": max_mult,
         "seed": seed, "budget": budget, "evals": 0, "wall_s": 0.0,
         "best_raw_loss": None, "numerical_failures": 0,
         "certificates": 0, "unique_lattices": set(), "nonss": 0,
         "eps_best": None, "time_to_first_cert": None,
         "evals_to_first_cert": None, "proposal_rounds": 0,
         "empty_rounds": 0}
    run_hashes = set()
    cands = []
    t0 = time.time()
    pk = {"max_mult": max_mult}
    if arm == "surrogate":
        pk["ranker"] = ranker

    def spend(k=1):
        m["evals"] += k

    def consider(arr, loss):
        nonlocal best
        if loss < best[1]:
            best = (arr, loss)
        if loss < 1e-6:
            cert = certify_state(arr, d1, d2)
            if cert is not None:
                wl = lattice_wl_hash(arr)
                new = wl not in run_hashes
                run_hashes.add(wl)
                if not new:
                    return
                ss = is_supersolvable_rank3(arr)
                eps = d1 - arr.max_multiplicity()
                m["certificates"] += 1
                m["unique_lattices"].add(wl)
                if not ss:
                    m["nonss"] += 1
                m["eps_best"] = (eps if m["eps_best"] is None
                                 else max(m["eps_best"], eps))
                if m["time_to_first_cert"] is None:
                    m["time_to_first_cert"] = time.time() - t0
                    m["evals_to_first_cert"] = m["evals"]
                from certificates import certificate_to_json
                cands.append({"arm": arm, "seed": seed, "n": n,
                              "d1": d1, "d2": d2, "eps": eps,
                              "m_max": arr.max_multiplicity(),
                              "supersolvable": bool(ss),
                              "lattice_hash": wl,
                              "new_relative_to_local_db": wl not in local_db,
                              "literature_novelty_unchecked": True,
                              "lines": [str(l) for l in arr.lines],
                              "certificate": certificate_to_json(cert)})

    cur = ev.screen_loss_or_none(state)
    spend()
    if cur is None:
        return m, cands
    best = (state, cur)
    consider(state, cur)
    tabu = {canonical_lineset_key(state)}
    restart = 0
    while m["evals"] < budget:
        m["proposal_rounds"] += 1
        props = propose_swaps(state, d1, d2, rng, n_remove=6, tabu=tabu,
                              **pk)
        if not props:
            m["empty_rounds"] += 1
            restart += 1
            state = seeds[restart % len(seeds)]
            if restart % len(seeds) == 0:
                state = perturb_k_swaps(seeds[0],
                                        min(2 + restart // len(seeds), 6),
                                        rng)
            cur = ev.screen_loss_or_none(state)
            spend()
            if cur is None:
                m["numerical_failures"] += 1
                break
            consider(state, cur)
            continue
        scored = []
        for (i, line, trial) in props:
            if m["evals"] >= budget:
                break
            l = ev.screen_loss_or_none(trial)
            spend()
            if l is None:
                m["numerical_failures"] += 1
                continue
            scored.append((ev.energy(trial, l), l, trial))
        if not scored:
            break
        scored.sort(key=lambda t: t[0])
        e_new, l_new, arr_new = scored[0]
        if e_new >= ev.energy(state, cur):
            restart += 1
            state = seeds[restart % len(seeds)]
            cur = ev.screen_loss_or_none(state)
            spend()
            if cur is None:
                m["numerical_failures"] += 1
                break
            consider(state, cur)
            continue
        state, cur = arr_new, l_new
        tabu.add(canonical_lineset_key(state))
        consider(state, cur)
    m["wall_s"] = round(time.time() - t0, 1)
    m["best_raw_loss"] = best[1]
    m["unique_lattices"] = len(m["unique_lattices"])
    return m, cands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=int, default=250)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--cells", nargs="+",
                    default=["18,8,9,5", "20,9,10,6", "20,9,10,5"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ranker = SurrogateRanker.load(args.model)
    json.dump({"args": vars(args), "provenance": runtime_provenance("."),
               "ranker_provenance": ranker.provenance},
              open(os.path.join(args.out, "manifest.json"), "w"),
              indent=1, default=str)
    # local DB for honest novelty labels
    from hybrid_smoke import load_local_db_hashes
    local_db = load_local_db_hashes()

    rows, allc = [], []
    for cell in args.cells:
        n, d1, d2, mm = map(int, cell.split(","))
        for seed in args.seeds:
            for arm in ("baseline", "surrogate"):
                met, cands = run_arm(arm, n, d1, d2, mm, seed,
                                     args.budget, ranker, local_db)
                rows.append(met)
                allc.extend(cands)
                print(f"[{arm:9s}] ({n},{d1},{d2}) m<={mm} s{seed}: "
                      f"best={met['best_raw_loss']:.2e} "
                      f"certs={met['certificates']} nonSS={met['nonss']} "
                      f"eps={met['eps_best']} "
                      f"e2c={met['evals_to_first_cert']} "
                      f"wall={met['wall_s']}s", flush=True)
                json.dump(rows, open(os.path.join(args.out, "metrics.json"),
                                     "w"), indent=1, default=str)
    with open(os.path.join(args.out, "candidates.jsonl"), "w") as f:
        for c in allc:
            f.write(json.dumps(c) + "\n")
    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as f:
        keys = [k for k in rows[0] if not isinstance(rows[0][k],
                                                     (dict, list, set))]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
