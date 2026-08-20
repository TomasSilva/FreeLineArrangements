"""
Cascading K-field swap campaign: one unit owns (field_d, engine, seed) and
CLIMBS levels.

Loop until the wall:
  1. run an engine slice at level n (cells layout under --out, shared with
     run_swap_campaign; certification/dedup via CampaignIO);
  2. harvest this unit's certified, distinct-lattice, non-supersolvable
     arrangements at level n;
  3. lift each (exact-only Delta-b2-targeted extension over K, every lift
     exactly certified) into admissible nontrivial n+1 cells;
  4. if lifts exist: advance to n+1 with the lifts (+ k-swap perturbations)
     as seeds; else keep mining level n with escalating perturbations.

Levels start from a validated fixture/basin (see known_arrangements.
FIELD_SEED_REGISTRY) or from lifted/QQ seeds under the forced-field pools.

Usage:
  python experiments/run_cascade_campaign.py --start-n 13 --field-d 3 \
      --engine me --seed 0 --wall-minutes 1380 --out results_from_HPC/x/cells
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arrangement import LineArrangement
from quadfield import QuadraticField
from novelty import (lattice_wl_hash, is_supersolvable_rank3,
                     canonical_lineset_key, parse_line_str)
from swap_search import (ChainEvaluator, map_elites, simulated_annealing,
                         greedy_search, perturb_k_swaps, is_valid_state)
from saito import extend_arrangement_targeted
from run_swap_campaign import CampaignIO, build_field_seeds, _git_rev


def balanced_pairs(n):
    """Admissible nontrivial (d1, d2) at level n, most balanced first."""
    pairs = [(d1, n - 1 - d1) for d1 in range(2, (n - 1) // 2 + 1)]
    pairs.sort(key=lambda p: p[1] - p[0])
    return pairs


def harvest_nonss(cells_dir, n, run_tag):
    """Distinct-lattice certified non-SS arrangements at level n from this
    run's cells (any pair)."""
    out, seen = [], set()
    for cell in sorted(os.listdir(cells_dir)) if os.path.isdir(cells_dir) \
            else []:
        path = os.path.join(cells_dir, cell, "certified.jsonl")
        if not cell.startswith(f"n{n}_") or not os.path.exists(path):
            continue
        for line in open(path):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("n") != n or rec.get("supersolvable", True):
                continue
            h = rec.get("lattice_hash")
            if h in seen:
                continue
            seen.add(h)
            try:
                field = None
                cf = rec.get("coefficient_field")
                if cf and cf != "QQ":
                    field = QuadraticField.from_json(cf)
                arr = LineArrangement([parse_line_str(s, field=field)
                                       for s in rec["lines"]])
            except Exception:
                continue
            out.append((h, arr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-n", type=int, required=True)
    ap.add_argument("--field-d", type=int, required=True,
                    choices=[2, 3, 5, -1, -3])
    ap.add_argument("--engine", default="me", choices=["me", "anneal",
                                                       "greedy"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wall-minutes", type=float, default=1380.0)
    ap.add_argument("--slice-minutes", type=float, default=90.0)
    ap.add_argument("--max-n", type=int, default=24)
    ap.add_argument("--max-parents-per-level", type=int, default=3)
    ap.add_argument("--coord-range", type=int, default=1)
    ap.add_argument("--max-mult", type=int, default=None)
    ap.add_argument("--surrogate", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    K = QuadraticField(args.field_d)
    rng = np.random.default_rng(args.seed * 900001 + args.start_n * 31
                                + args.field_d)
    deadline = time.time() + args.wall_minutes * 60
    run_tag = f"cascade_d{args.field_d}_{args.engine}_s{args.seed}"
    manifest_path = os.path.join(
        args.out, f"cascade_manifest_d{args.field_d}_{args.engine}"
                  f"_s{args.seed}.json")
    from penalized_saito import runtime_provenance
    manifest = {"args": vars(args), "git_rev": _git_rev("."),
                "coefficient_field": K.to_json(), "levels": [],
                "provenance": runtime_provenance(".")}

    n = args.start_n
    seeds = None      # level seeds; None -> build_field_seeds
    last_seeds = []   # previous level's working seeds (fallback)
    while time.time() < deadline and n <= args.max_n:
        pairs = balanced_pairs(n)
        d1, d2 = pairs[(args.seed + len(manifest["levels"])) % max(
            1, min(2, len(pairs)))] if pairs else (None, None)
        if d1 is None:
            break
        cell_dir = os.path.join(args.out, f"n{n}_d{d1}_{d2}")
        os.makedirs(cell_dir, exist_ok=True)
        if seeds is None:
            try:
                seeds = build_field_seeds(n, d1, d2, rng, args.coord_range,
                                          args.field_d)
            except SystemExit as e:
                if last_seeds:
                    # no registry fixture at this level: keep climbing from
                    # the previous level's seed set (perturbed for variety)
                    seeds = list(last_seeds)
                    seeds.append(perturb_k_swaps(seeds[0], 2, rng,
                                                 coord_range=args.coord_range,
                                                 field=K))
                else:
                    print(f"[{run_tag}] level {n}: {e}", flush=True)
                    break
        io = CampaignIO(cell_dir, n, d1, d2, f"cascade-{args.engine}",
                        args.seed)
        ev = ChainEvaluator(n, d1, d2, seed=args.seed,
                            m_target=args.max_mult)
        slice_end = min(deadline, time.time() + args.slice_minutes * 60)
        print(f"[{run_tag}] level n={n} cell ({n},{d1},{d2}): "
              f"{len(seeds)} seeds, slice "
              f"{(slice_end - time.time())/60:.0f}min", flush=True)

        last_seeds = list(seeds)
        restarts = 0
        pk = {"field": K, "coord_range": args.coord_range,
              "max_mult": args.max_mult}
        if args.surrogate:
            from surrogate import SurrogateRanker
            pk["ranker"] = SurrogateRanker.load(args.surrogate)
        while time.time() < slice_end:
            seed_arr = seeds[restarts % len(seeds)]
            try:
                if args.engine == "me":
                    def snap(a, gen, _end=slice_end):
                        if time.time() > _end:
                            raise TimeoutError
                    try:
                        map_elites(seeds, d1, d2, ev, rng, generations=2000,
                                   on_candidate=io.on_candidate,
                                   on_snapshot=snap, proposal_kwargs=pk)
                    except TimeoutError:
                        pass
                elif args.engine == "anneal":
                    simulated_annealing(seed_arr, d1, d2, ev, rng,
                                        steps=1200,
                                        on_candidate=io.on_candidate,
                                        proposal_kwargs=pk)
                else:
                    greedy_search(seed_arr, d1, d2, ev, rng, steps=150,
                                  on_candidate=io.on_candidate,
                                  proposal_kwargs=pk)
            except RuntimeError as e:
                print(f"[{run_tag}]   restart {restarts}: {e}", flush=True)
            restarts += 1
            if restarts % max(1, len(seeds)) == 0:
                k = min(2 + restarts // max(1, len(seeds)), 6)
                seeds.append(perturb_k_swaps(seeds[0], k, rng,
                                             coord_range=args.coord_range,
                                             field=K))

        # harvest and lift
        parents = harvest_nonss(args.out, n, run_tag)
        parents = parents[:args.max_parents_per_level]
        lifts = []
        for h, arr in parents:
            if time.time() > deadline:
                break
            for (e1, e2) in balanced_pairs(n + 1)[:3]:
                try:
                    succ = extend_arrangement_targeted(
                        arr, target_exponents=(e1, e2),
                        coord_range=2, verbose=False)
                except Exception as ex:
                    print(f"[{run_tag}]   lift error {h[:8]}: {ex}",
                          flush=True)
                    continue
                for s in succ:
                    lifts.append(s["arrangement"])
        level_info = {"n": n, "cell": [n, d1, d2], "restarts": restarts,
                      "nonss_parents": len(parents),
                      "certified_lifts": len(lifts),
                      "t": time.time()}
        manifest["levels"].append(level_info)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=1)
        print(f"[{run_tag}] level {n} done: {len(parents)} non-SS parents, "
              f"{len(lifts)} certified lifts", flush=True)

        if lifts:
            # dedup + perturb; ADVANCE
            seen, next_seeds = set(), []
            for a in lifts:
                key = canonical_lineset_key(a)
                if key in seen:
                    continue
                seen.add(key)
                if is_valid_state(a, n + 1, nontrivial=True):
                    next_seeds.append(a)
            for base in list(next_seeds)[:2]:
                next_seeds.append(perturb_k_swaps(base, 1, rng,
                                                  coord_range=args.coord_range,
                                                  field=K))
            if next_seeds:
                n += 1
                seeds = next_seeds
                continue
        # no lifts: KEEP the current seeds (never drop to None mid-level —
        # a level without a registry fixture would kill the unit) and add an
        # escalating perturbation so repeated slices diversify
        seeds.append(perturb_k_swaps(
            seeds[0], min(2 + len(manifest["levels"]) % 5, 6), rng,
            coord_range=args.coord_range, field=K))

    print(f"[{run_tag}] DONE at level {n}, "
          f"{len(manifest['levels'])} level-slices", flush=True)


if __name__ == "__main__":
    main()
