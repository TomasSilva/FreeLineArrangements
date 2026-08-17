"""
Swap-search campaign unit: one (n, d1, d2) cell, one engine, one seed.

Wall-clock-bounded; deterministic for fixed args; append-only outputs.
Certified discoveries carry exact symbolic certificates (verified twice:
inside certify_state and again before persisting).  Nothing here touches the
repo-root discoveries.json.

Usage:
  python experiments/run_swap_campaign.py --n 15 --d1 2 --d2 12 \
      --engine anneal --seed 0 --wall-minutes 60 \
      --out results_penalized_saito/<date>/swap/cells/n15_d2_12
Outputs (in --out):
  manifest_<engine>_s<seed>.json     args, git rev, start/end, counters
  candidates.jsonl                   appended numerically-promising states
  certified.jsonl                    appended exact-certified discoveries
  certificates/cert_<hash>_<k>.json  exact certificates
  archive_<engine>_s<seed>.json      MAP-Elites archive (me engine)
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arrangement import LineArrangement
from swap_search import (ChainEvaluator, double_pencil_seed, random_valid_seed,
                         corpus_seeds, perturb_k_swaps, simulated_annealing,
                         greedy_search, random_walk, map_elites, certify_state,
                         is_valid_state, LOSS_CANDIDATE_THRESHOLD)
from novelty import (lattice_wl_hash, is_supersolvable_rank3,
                     coordinate_height, canonical_lineset_key)
from certificates import certificate_to_json, verify_certificate


def _git_rev(repo):
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unknown"


class CampaignIO:
    """Append-only candidate/certified streams with lattice-level dedup of
    certification work (each lattice hash is certified at most once per
    campaign invocation; repeats are recorded, not re-proved)."""

    def __init__(self, out_dir, n, d1, d2, engine, seed):
        self.out = out_dir
        os.makedirs(os.path.join(out_dir, "certificates"), exist_ok=True)
        self.n, self.d1, self.d2 = n, d1, d2
        self.engine, self.seed = engine, seed
        self.cand_path = os.path.join(out_dir, "candidates.jsonl")
        self.cert_path = os.path.join(out_dir, "certified.jsonl")
        self.tried_hashes = {}       # lattice_hash -> status
        self.certified_hashes = set()
        self.coord_seen = set()
        self.counters = {"candidates": 0, "cert_attempts": 0,
                         "certified": 0, "cert_repeat_lattice": 0,
                         "cert_failed": 0}
        self.first_cert_time = None
        self.t0 = time.time()

    def on_candidate(self, rec):
        """Called by engines when loss < threshold.  Certifies new lattices
        inline (fast path; seconds at n <= 16)."""
        key = str(sorted(rec["lines"]))
        if key in self.coord_seen:
            return
        self.coord_seen.add(key)
        self.counters["candidates"] += 1
        rec["engine_seed"] = self.seed
        with open(self.cand_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        h = rec["lattice_hash"]
        if h in self.tried_hashes:
            if self.tried_hashes[h] == "certified":
                self.counters["cert_repeat_lattice"] += 1
            return
        from novelty import parse_line_str
        arr = LineArrangement([parse_line_str(s) for s in rec["lines"]])
        self.counters["cert_attempts"] += 1
        cert = certify_state(arr, self.d1, self.d2)
        if cert is None:
            self.tried_hashes[h] = "not_free_or_failed"
            self.counters["cert_failed"] += 1
            return
        self.tried_hashes[h] = "certified"
        self.certified_hashes.add(h)
        self.counters["certified"] += 1
        if self.first_cert_time is None:
            self.first_cert_time = time.time() - self.t0
        cj = certificate_to_json(cert)
        assert verify_certificate(cert)
        k = len(self.certified_hashes)
        cert_file = os.path.join(self.out, "certificates",
                                 f"cert_{h[:16]}_{k}.json")
        with open(cert_file, "w") as f:
            json.dump(cj, f, indent=1)
        entry = dict(rec)
        entry.update({
            "certificate_file": os.path.relpath(cert_file, self.out),
            "exponents": [1, self.d1, self.d2],
            "supersolvable": is_supersolvable_rank3(arr),
            "wall_s": time.time() - self.t0,
        })
        with open(self.cert_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"  [CERTIFIED #{self.counters['certified']}] "
              f"lattice={h[:12]} m_max={rec['m_max']} "
              f"ss={entry['supersolvable']} t={entry['wall_s']:.0f}s",
              flush=True)


def build_seeds(n, d1, d2, rng, coord_range, mode="mixed", n_seeds=6,
                repo_root="."):
    seeds = []
    base = double_pencil_seed(n, d1, d2)
    if mode in ("supersolvable", "mixed"):
        seeds.append(base)
    if mode in ("perturbed", "mixed"):
        for k in (1, 2, 3, 5):
            seeds.append(perturb_k_swaps(base, k, rng,
                                         coord_range=coord_range))
    if mode in ("corpus", "mixed"):
        seeds.extend(corpus_seeds(n, d1, d2, limit=3, repo_root=repo_root))
    if mode in ("random", "mixed"):
        while len(seeds) < n_seeds + 4:
            try:
                seeds.append(random_valid_seed(n, rng,
                                               coord_range=coord_range))
            except RuntimeError:
                break
    # dedup by canonical key, keep order
    seen, out = set(), []
    for s in seeds:
        k = canonical_lineset_key(s)
        if k not in seen and is_valid_state(s, n, nontrivial=(d1 >= 2)):
            seen.add(k)
            out.append(s)
    return out[:max(n_seeds, 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--d1", type=int, required=True)
    ap.add_argument("--d2", type=int, required=True)
    ap.add_argument("--engine", required=True,
                    choices=["anneal", "me", "greedy", "walk"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wall-minutes", type=float, default=60.0)
    ap.add_argument("--coord-range", type=int, default=3)
    ap.add_argument("--seed-mode", default="mixed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume-archive", default=None)
    args = ap.parse_args()

    n, d1, d2 = args.n, args.d1, args.d2
    assert d1 + d2 == n - 1 and 1 <= d1 <= d2
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed * 100003 + n * 101 + d1)
    deadline = time.time() + args.wall_minutes * 60

    from penalized_saito import DEFAULT_LAMBDA, DEFAULT_BETA
    manifest = {
        "args": vars(args), "git_rev": _git_rev("."),
        "start": time.time(), "b2_star": (n - 1) + d1 * d2,
        "lambda": DEFAULT_LAMBDA, "beta": DEFAULT_BETA,
        "optimization_field": "real",
    }
    io = CampaignIO(args.out, n, d1, d2, args.engine, args.seed)
    ev = ChainEvaluator(n, d1, d2, seed=args.seed)
    seeds = build_seeds(n, d1, d2, rng, args.coord_range,
                        mode=args.seed_mode)
    print(f"cell ({n},{d1},{d2}) engine={args.engine} seed={args.seed}: "
          f"{len(seeds)} seeds, wall={args.wall_minutes:.0f}min", flush=True)

    restarts = 0
    best_loss_overall = 1.0
    if args.engine == "me":
        archive = None
        if args.resume_archive and os.path.exists(args.resume_archive):
            with open(args.resume_archive) as f:
                archive = json.load(f)
            print(f"  resumed archive with {len(archive)} cells", flush=True)
        arch_path = os.path.join(args.out,
                                 f"archive_{args.engine}_s{args.seed}.json")

        def snapshot(a, gen):
            with open(arch_path, "w") as f:
                json.dump(a, f)
            if time.time() > deadline:
                raise TimeoutError

        try:
            while time.time() < deadline:
                archive, _, bl = map_elites(
                    seeds, d1, d2, ev, rng, generations=2000,
                    on_candidate=io.on_candidate, archive=archive,
                    on_snapshot=snapshot)
                best_loss_overall = min(best_loss_overall, bl)
                restarts += 1
        except TimeoutError:
            pass
        with open(arch_path, "w") as f:
            json.dump(archive, f)
    else:
        engine_fn = {"anneal": simulated_annealing, "greedy": greedy_search,
                     "walk": random_walk}[args.engine]
        while time.time() < deadline:
            seed_arr = seeds[restarts % len(seeds)]
            if args.engine == "anneal":
                budget = {"steps": 1200}
            elif args.engine == "greedy":
                budget = {"steps": 150}
            else:
                budget = {"steps": 1500}
            _, bl, _ = engine_fn(seed_arr, d1, d2, ev, rng,
                                 on_candidate=io.on_candidate, **budget)
            best_loss_overall = min(best_loss_overall, bl)
            restarts += 1
            # escalate perturbation as restarts accumulate (leave the basin)
            if restarts % len(seeds) == 0:
                k = min(2 + restarts // len(seeds), 8)
                seeds.append(perturb_k_swaps(seeds[0], k, rng,
                                             coord_range=args.coord_range))

    manifest.update({
        "end": time.time(), "restarts": restarts,
        "best_loss": best_loss_overall,
        "counters": io.counters,
        "evaluator": ev.stats(),
        "first_cert_wall_s": io.first_cert_time,
        "distinct_certified_lattices": len(io.certified_hashes),
    })
    mpath = os.path.join(args.out,
                         f"manifest_{args.engine}_s{args.seed}.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"done: restarts={restarts} best_loss={best_loss_overall:.2e} "
          f"certified={io.counters['certified']} "
          f"(distinct lattices {len(io.certified_hashes)})", flush=True)


if __name__ == "__main__":
    main()
