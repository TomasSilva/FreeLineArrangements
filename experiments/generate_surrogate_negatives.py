"""
Self-labeled negatives for the surrogate: perturb known evaluated states by
k = 1..4 swaps and record the TRUE rl-profile penalized Saito loss.  This
matches the runtime distribution (swap proposals around good states), which
candidates.jsonl cannot provide (it only records sub-gate winners).

Records go to <out>/negatives.jsonl with the same schema as candidate
records (the dataset builder picks them up via the negatives.jsonl glob).

Usage:
  python experiments/generate_surrogate_negatives.py \
      --out results_local/surrogate_experiment [--n-evals 25000] [--seed 0]
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novelty import arrangement_from_record
from swap_search import (perturb_k_swaps, is_valid_state, _record,
                         canonical_lineset_key)
from penalized_saito import cached_penalized_loss, GammaNumericalError


def load_bases(max_per_cell=40):
    """QQ certified records grouped by cell (n = 13..20)."""
    cells = {}
    for p in sorted(glob.glob("results_from_HPC/**/certified.jsonl",
                              recursive=True)
                    + glob.glob("results_local/**/certified.jsonl",
                                recursive=True)):
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("coefficient_field", "QQ") != "QQ":
                continue
            key = (r.get("n"), r.get("d1"), r.get("d2"))
            if not (13 <= (key[0] or 0) <= 20):
                continue
            bucket = cells.setdefault(key, {})
            if r["lattice_hash"] not in bucket and \
                    len(bucket) < max_per_cell:
                bucket[r["lattice_hash"]] = r
    return {k: list(v.values()) for k, v in cells.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-evals", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "negatives.jsonl")
    rng = np.random.default_rng(args.seed)
    bases = load_bases()
    cell_list = sorted(bases.keys())
    print(f"bases: {sum(len(v) for v in bases.values())} records over "
          f"{len(cell_list)} cells", flush=True)
    seen = set()
    n_done = n_err = 0
    t0 = time.time()
    with open(out_path, "a") as f:
        while n_done < args.n_evals:
            cell = cell_list[int(rng.integers(len(cell_list)))]
            n, d1, d2 = cell
            base_rec = bases[cell][int(rng.integers(len(bases[cell])))]
            try:
                base = arrangement_from_record(base_rec)
            except Exception:
                continue
            k = int(rng.integers(1, 5))
            trial = perturb_k_swaps(base, k, rng)
            if not is_valid_state(trial, n, nontrivial=True):
                continue
            ck = canonical_lineset_key(trial)
            if ck in seen:
                continue
            seen.add(ck)
            try:
                loss = cached_penalized_loss(trial, d1=d1, d2=d2,
                                             profile="rl", seed=0)
            except GammaNumericalError:
                n_err += 1
                continue
            rec = _record(trial, d1, d2, loss, f"negatives_k{k}", n_done)
            f.write(json.dumps(rec) + "\n")
            n_done += 1
            if n_done % 2000 == 0:
                f.flush()
                print(f"  {n_done}/{args.n_evals} "
                      f"({(time.time()-t0)/60:.0f} min, {n_err} numerical "
                      f"errors)", flush=True)
    print(f"DONE: {n_done} negatives in {(time.time()-t0)/60:.0f} min "
          f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
