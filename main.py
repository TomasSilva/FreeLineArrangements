"""
main.py

Entry point for exploring free line arrangements in CP^2 using RL.

Examples:
    # Verify known free arrangements (sanity check)
    python main.py verify

    # Random search for n=6, coord_range=3
    python main.py search --n 6 --coord-range 3 --max-check 10000

    # Train for fixed n=6
    python main.py train --n 6

    # Train with curriculum: n from 6 to 12
    python main.py train --n-min 6 --n-max 12 --total-steps 1000000

    # Train for large n (skips exact Saito check during training)
    python main.py train --n-min 6 --n-max 16 --coord-range 3 --skip-exact-above 12

    # Greedy exploration from trained model
    python main.py explore --n 10 --model model_final.pt --episodes 1000

    # Post-hoc exact verification of model-found candidates (for large n)
    python main.py verify-found --n 15 --model model_final.pt --episodes 2000
"""

import argparse
import sys
import torch
from arrangement import LineArrangement, ProjectiveLine
from environment import FreeArrangementEnv
from model import TransformerActorCritic
from train import train, evaluate_greedy, get_parser
from saito import verify_arrangement
from discoveries import log_discoveries, summary as discoveries_summary


# ─────────────────────────────────────────────────────────────────────────────
# Known free arrangements for verification
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_FREE = [
    # --- Boolean arrangement (3 coordinate lines): exponents (1,1,1) ---
    {
        'name': 'Boolean A3 (coordinate lines)',
        'lines': [(1,0,0), (0,1,0), (0,0,1)],
        'expected_exps': (1, 1, 1),
    },
    # --- A2 x A1: x=0, y=0, x-y=0, z=0 — exponents (1,1,2) ---
    {
        'name': 'A2 x A1 (n=4)',
        'lines': [(1,0,0), (0,1,0), (1,-1,0), (0,0,1)],
        'expected_exps': (1, 1, 2),
    },
    # --- Braid arrangement B3: 6 lines, exponents (1,2,3) ---
    {
        'name': 'Braid B3 (n=6)',
        'lines': [(1,0,0), (0,1,0), (0,0,1), (1,-1,0), (1,0,-1), (0,1,-1)],
        'expected_exps': (1, 2, 3),
    },
    # --- Generic (non-free): 4 lines in general position ---
    {
        'name': 'Generic 4 lines (NOT free)',
        'lines': [(1,0,0), (0,1,0), (0,0,1), (1,1,1)],
        'expected_exps': None,
        'expect_free': False,
    },
]


def run_verify():
    """Verify the freeness checker on known examples."""
    print("=" * 60)
    print("Verifying known free arrangements")
    print("=" * 60)
    all_pass = True
    for ex in KNOWN_FREE:
        lines = [ProjectiveLine(*abc) for abc in ex['lines']]
        arr = LineArrangement(lines)
        s = arr.summary()
        is_free, exps = arr.is_free()
        expected_free = ex.get('expect_free', ex.get('expected_exps') is not None)

        status = "PASS" if is_free == expected_free else "FAIL"
        if is_free and ex.get('expected_exps'):
            exp_match = (exps == ex['expected_exps'])
            if not exp_match:
                status = "FAIL (exps mismatch)"
        all_pass = all_pass and (status == "PASS")

        print(f"\n[{status}] {ex['name']}")
        print(f"  n={s['n']}, b2={s['b2']}, candidate_exps={s['candidate_exponents']}")
        print(f"  is_free={is_free}, exponents={exps}")
        print(f"  multiplicity profile: {sorted(s['multiplicity_profile'], reverse=True)}")

    print("\n" + "=" * 60)
    print("All tests passed!" if all_pass else "SOME TESTS FAILED.")
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# Exploration: sample free arrangements from trained model
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(model_path, n_max, args):
    model = TransformerActorCritic(
        max_n=n_max,
        d_model=getattr(args, 'd_model', 128),
        n_heads=getattr(args, 'n_heads', 4),
        n_layers=getattr(args, 'n_layers', 3),
        scalar_dim=17,
    )
    if model_path and model_path != '':
        ckpt = torch.load(model_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        print(f"Loaded model from {model_path}")
    else:
        print("No model provided, using random policy")
    return model


def run_explore(args):
    n_max = getattr(args, 'n_max', None) or args.n
    singularity_aware = getattr(args, 'singularity_aware', False)
    max_candidates = getattr(args, 'max_candidates', 200)
    env = FreeArrangementEnv(
        target_n=args.n,
        coord_range=args.coord_range,
        max_n=n_max,
        max_candidates=max_candidates,
        singularity_aware=singularity_aware,
    )
    model = _load_model(args.model, n_max, args)

    target_exponents = tuple(args.target_exponents) if getattr(args, 'target_exponents', None) else None
    if target_exponents:
        print(f"\nExploring {args.episodes} episodes for n={args.n}, target exponents={target_exponents}...")
        env.target_exponents = target_exponents
    else:
        print(f"\nExploring {args.episodes} episodes for n={args.n}...")
    found = evaluate_greedy(model, env, n_episodes=args.episodes, target_n=args.n)

    # Deduplicate by exponents + b2
    unique = {}
    for f in found:
        key = (str(f['exponents']), f['t2'])
        if key not in unique:
            unique[key] = f

    print(f"\nFound {len(found)} free non-pencil arrangements")
    print(f"Unique (by exponents+b2): {len(unique)}")
    for i, f in enumerate(unique.values()):
        print(f"\n--- Arrangement {i+1} ---")
        print(f"  exponents: {f['exponents']}, b2: {f['t2']}")
        for lstr in f['lines']:
            print(f"  {lstr}")

    n_new = log_discoveries(list(unique.values()), source="explore")
    print(f"\nSaved {n_new} new discoveries to discoveries.json")

    return found


# ─────────────────────────────────────────────────────────────────────────────
# Post-hoc exact verification for large n
# ─────────────────────────────────────────────────────────────────────────────

def run_verify_found(args):
    """
    Run the model in greedy mode, collect arrangements with candidate exponents,
    and run exact Saito verification on each.
    Useful after training with skip_exact_above > 0 (i.e., large n).
    """
    n_max = getattr(args, 'n_max', None) or args.n
    singularity_aware = getattr(args, 'singularity_aware', False)
    max_candidates = getattr(args, 'max_candidates', 200)
    env = FreeArrangementEnv(
        target_n=args.n,
        coord_range=args.coord_range,
        max_n=n_max,
        max_candidates=max_candidates,
        singularity_aware=singularity_aware,
        skip_exact_above=999,  # always give algebraic reward, exploration mode
    )
    model = _load_model(args.model, n_max, args)
    model.eval()

    print(f"\nRunning {args.episodes} greedy episodes for n={args.n}...")
    print("Collecting candidates with valid combinatorial structure...")

    candidates = []
    with torch.no_grad():
        for ep in range(args.episodes):
            obs = env.reset(target_n=args.n)
            done = False
            while not done:
                # Alternate between greedy and stochastic for diversity
                action, _, _ = model.act(obs, deterministic=(ep % 2 == 0))
                obs, _, done, info = env.step(action)

            if not info.get('is_pencil') and info.get('candidate_exponents') is not None:
                candidates.append(env.arr.copy())

            if (ep + 1) % 200 == 0:
                print(f"  {ep+1}/{args.episodes} episodes, {len(candidates)} candidates so far")

    print(f"\nCollected {len(candidates)} candidate arrangements.")
    print("Running exact Saito verification (may take a while for large n)...")

    free_found = []
    for i, arr in enumerate(candidates):
        is_free, exps = verify_arrangement(arr)
        if is_free:
            s = arr.summary()
            free_found.append({
                'lines': [str(l) for l in arr.lines],
                'exponents': exps,
                'b2': s['b2'],
                'max_mult': arr.max_multiplicity(),
                'mult_profile': sorted(s['multiplicity_profile'], reverse=True),
            })
        if (i + 1) % 20 == 0:
            print(f"  Verified {i+1}/{len(candidates)}, free so far: {len(free_found)}")

    # Deduplicate by combinatorial type
    unique = {}
    for f in free_found:
        key = (str(f['exponents']), f['b2'], str(f['mult_profile']))
        unique[key] = f

    print(f"\n{'='*60}")
    print(f"Total free arrangements found (exact): {len(free_found)}")
    print(f"Unique (by exponents+b2+profile): {len(unique)}")
    for i, f in enumerate(unique.values()):
        print(f"\n--- Free arrangement {i+1} ---")
        print(f"  exponents: {f['exponents']}, b2: {f['b2']}, max_mult: {f['max_mult']}")
        print(f"  multiplicity profile: {f['mult_profile']}")
        for lstr in f['lines']:
            print(f"  {lstr}")

    n_new = log_discoveries(list(unique.values()), source="verify-found")
    print(f"\nSaved {n_new} new discoveries to discoveries.json")

    return list(unique.values())


# ─────────────────────────────────────────────────────────────────────────────
# Extend mode: bootstrap from known free arrangements at n_from to find n_from+1
# ─────────────────────────────────────────────────────────────────────────────

def _parse_line_str(s):
    """Parse a line string '(ax+by+cz=0)' into a ProjectiveLine with exact rationals."""
    import re
    from fractions import Fraction
    from sympy import Rational
    s = s.strip().strip('(').rstrip(')').replace('=0', '').replace(' ', '').replace('+-', '-')
    m = re.match(r'([+-]?[\d/]*)x([+-][\d/]*)y([+-][\d/]*)z', s)
    if not m:
        raise ValueError(f"Cannot parse line: {s}")
    def to_rat(c):
        c = c.strip('+')
        if c in ('', '+'):
            return 1
        if c == '-':
            return -1
        return Fraction(c)
    return ProjectiveLine(
        Rational(to_rat(m.group(1))),
        Rational(to_rat(m.group(2))),
        Rational(to_rat(m.group(3))),
    )


def run_extend(args):
    """
    Bootstrap-extend mode: load known free arrangements at n_from, attempt to
    extend each by one line, and save successful n_from+1 free arrangements.

    For each seed arrangement, enumerates candidate lines from:
      - Lines through pairs of existing intersection points (singularity-driven)
      - Small-integer pool (a, b, c) in [-coord_range, coord_range]
      - Optionally rational lines through existing multiple points

    Each candidate is pre-filtered by the penalized Saito loss (saito_loss)
    before the expensive exact `is_free()` check.
    """
    import json
    import time
    from saito import extend_arrangement, extend_arrangement_targeted
    from arrangement import all_exponent_types

    # Load seeds
    print(f"Loading seeds from {args.seeds_file}...")
    with open(args.seeds_file) as f:
        data = json.load(f)
    arrs = data.get('arrangements', data) if isinstance(data, dict) else data
    seeds = [a for a in arrs if a.get('n') == args.n_from]
    print(f"Found {len(seeds)} seeds at n={args.n_from}")

    if args.target_exponents:
        d1_t, d2_t = args.target_exponents
        seeds = [s for s in seeds if tuple(s.get('exponents', [None,None,None])[1:]) == (d1_t, d2_t)]
        print(f"Filtered to {len(seeds)} seeds with exponents (1, {d1_t}, {d2_t})")

    if args.max_seeds is not None:
        seeds = seeds[:args.max_seeds]
        print(f"Using first {len(seeds)} seeds")

    if not seeds:
        print("No seeds — aborting.")
        return

    # Determine target list for the new arrangements
    n_new = args.n_from + 1
    if args.all_targets:
        target_list = all_exponent_types(n_new)
        print(f"--all-targets: enumerating {len(target_list)} target exponent types: {target_list}")
    elif args.target_new_exponents:
        target_list = [tuple(args.target_new_exponents)]
        print(f"--target-new-exponents: targeting {target_list[0]} for n={n_new}")
    else:
        target_list = None  # use unfiltered extend_arrangement

    all_extensions = []
    t_total = time.perf_counter()
    for i, rec in enumerate(seeds):
        try:
            seed_lines = [_parse_line_str(s) for s in rec['lines']]
        except Exception as e:
            print(f"  Seed {i+1}: parse error: {e}")
            continue
        seed_arr = LineArrangement(seed_lines)

        t0 = time.perf_counter()
        if target_list is None:
            # Unfiltered: original extend_arrangement
            results = extend_arrangement(
                seed_arr,
                coord_range=args.coord_range,
                loss_threshold=args.loss_threshold,
                n_restarts=args.n_restarts,
                max_denominator=args.max_denominator,
                verbose=False,
            )
        else:
            # Targeted: loop over each target
            results = []
            for target in target_list:
                sub_results = extend_arrangement_targeted(
                    seed_arr,
                    target_exponents=target,
                    coord_range=args.coord_range,
                    loss_threshold=args.loss_threshold,
                    n_restarts=args.n_restarts,
                    max_denominator=args.max_denominator,
                    verbose=False,
                )
                results.extend(sub_results)
        elapsed = time.perf_counter() - t0
        print(f"Seed {i+1}/{len(seeds)} (exps={rec.get('exponents')}): "
              f"{len(results)} extensions in {elapsed:.1f}s")
        all_extensions.extend(results)

    print(f"\n{'='*60}")
    print(f"Total: {len(all_extensions)} free n={args.n_from + 1} arrangements found")
    print(f"Total time: {time.perf_counter() - t_total:.1f}s")

    if not all_extensions:
        return

    # Convert to discovery records
    records = []
    for r in all_extensions:
        new_arr = r['arrangement']
        s = new_arr.summary()
        records.append({
            'lines': [str(l) for l in new_arr.lines],
            'exponents': r['exponents'],
            'b2': s['b2'],
            'n': len(new_arr),
            'max_mult': new_arr.max_multiplicity(),
            'mult_profile': sorted(s['multiplicity_profile'], reverse=True),
        })

    # Group by exponents for printing
    from collections import Counter
    by_exps = Counter(tuple(r['exponents']) for r in records)
    print(f"By exponents: {dict(by_exps)}")

    n_new = log_discoveries(records, source="extend",
                            path=args.output if args.output else "discoveries.json")
    print(f"Saved {n_new} new discoveries to {args.output or 'discoveries.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# Construct: direct construction of known free arrangement families
# ─────────────────────────────────────────────────────────────────────────────

def run_construct(args):
    """
    Direct construction of known free arrangement families. No search, no RL —
    just emit a closed-form free arrangement and save it to discoveries.

    Supports:
      - --family near-pencil: (1, 1, n-2) for any n >= 3
      - --family supersolvable --d1 K: (1, K, n-1-K) for any n, 1 <= K <= (n-1)//2
      - --family all-supersolvable: emit one supersolvable for each valid (d1, d2) at level n
    """
    from saito import construct_near_pencil, construct_supersolvable
    from arrangement import all_exponent_types

    n = args.n
    family = args.family
    output = args.output or "discoveries.json"

    records = []

    if family == 'near-pencil':
        if n < 3:
            print(f"near-pencil requires n >= 3, got {n}")
            return
        arr = construct_near_pencil(n)
        is_free, exps = arr.is_free()
        if not is_free:
            print(f"ERROR: constructed near-pencil is not free!")
            return
        s = arr.summary()
        records.append({
            'lines': [str(l) for l in arr.lines],
            'exponents': list(exps),
            'b2': s['b2'],
            'n': n,
            'max_mult': arr.max_multiplicity(),
            'mult_profile': sorted(s['multiplicity_profile'], reverse=True),
        })
        print(f"Constructed near-pencil for n={n}: exps={exps}")

    elif family == 'supersolvable':
        if args.d1 is None:
            print("--d1 is required for --family supersolvable")
            return
        d1 = args.d1
        if d1 < 1 or d1 > (n - 1) // 2:
            print(f"--d1 must be in [1, {(n-1)//2}], got {d1}")
            return
        arr = construct_supersolvable(n, d1)
        is_free, exps = arr.is_free()
        if not is_free:
            print(f"ERROR: constructed supersolvable is not free!")
            return
        s = arr.summary()
        records.append({
            'lines': [str(l) for l in arr.lines],
            'exponents': list(exps),
            'b2': s['b2'],
            'n': n,
            'max_mult': arr.max_multiplicity(),
            'mult_profile': sorted(s['multiplicity_profile'], reverse=True),
        })
        print(f"Constructed supersolvable for n={n}, d1={d1}: exps={exps}")

    elif family == 'all-supersolvable':
        if n < 3:
            print(f"all-supersolvable requires n >= 3, got {n}")
            return
        for d1, d2 in all_exponent_types(n):
            arr = construct_supersolvable(n, d1)
            is_free, exps = arr.is_free()
            if not is_free:
                print(f"  d1={d1}: ERROR — not free!")
                continue
            s = arr.summary()
            records.append({
                'lines': [str(l) for l in arr.lines],
                'exponents': list(exps),
                'b2': s['b2'],
                'n': n,
                'max_mult': arr.max_multiplicity(),
                'mult_profile': sorted(s['multiplicity_profile'], reverse=True),
            })
            print(f"  d1={d1}: exps={exps}, profile={sorted(s['multiplicity_profile'], reverse=True)}")
    else:
        print(f"Unknown family: {family}")
        return

    n_new = log_discoveries(records, source=f"construct-{family}", path=output)
    print(f"\nSaved {n_new} new discoveries to {output}")


# ─────────────────────────────────────────────────────────────────────────────
# Manual search: exhaustive/random search for small n
# ─────────────────────────────────────────────────────────────────────────────

def run_search(args):
    """
    Random/exhaustive search for free non-pencil arrangements of n lines.
    Does not use RL; useful as a baseline and for generating training examples.
    """
    from itertools import combinations as combs
    from environment import generate_candidate_lines
    import random

    pool = generate_candidate_lines(args.coord_range)
    n = args.n
    print(f"Searching over {len(pool)} candidate lines for n={n}...")

    found = []
    checked = 0
    rng = random.Random(42)

    if args.exhaustive and len(pool) <= 30:
        iterator = combs(pool, n)
    else:
        def random_iter():
            while True:
                yield rng.sample(pool, n)
        iterator = random_iter()

    max_check = args.max_check if not args.exhaustive else float('inf')
    max_mult_filter = getattr(args, 'max_mult', None)

    for selection in iterator:
        if checked >= max_check:
            break
        arr = LineArrangement(list(selection))
        if arr.is_pencil():
            checked += 1
            continue
        if max_mult_filter is not None and arr.max_multiplicity() > max_mult_filter:
            checked += 1
            continue
        exps = arr.candidate_exponents()
        if exps is None:
            checked += 1
            continue
        is_free, exponents = arr.is_free()
        if is_free:
            s = arr.summary()
            found.append({
                'lines': [str(l) for l in arr.lines],
                'exponents': exponents,
                'b2': s['b2'],
                'max_mult': arr.max_multiplicity(),
                'n_pts': arr.n_intersection_points(),
                'mult_profile': sorted(s['multiplicity_profile'], reverse=True),
            })
            print(f"  Found! exps={exponents}, b2={s['b2']}, "
                  f"max_mult={arr.max_multiplicity()}, "
                  f"profile={sorted(s['multiplicity_profile'], reverse=True)}")
        checked += 1
        if checked % 1000 == 0:
            print(f"  Checked {checked}, found {len(found)}")

    print(f"\nTotal checked: {checked}, free found: {len(found)}")

    n_new = log_discoveries(found, source="search")
    print(f"Saved {n_new} new discoveries to discoveries.json")

    return found


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Free line arrangements in CP^2 via RL")
    subparsers = parser.add_subparsers(dest='command')

    # train
    subparsers.add_parser('train', parents=[get_parser()], add_help=False)

    # verify / test
    subparsers.add_parser('verify')
    subparsers.add_parser('test')

    # explore
    exp_parser = subparsers.add_parser('explore')
    exp_parser.add_argument('--n', type=int, default=6)
    exp_parser.add_argument('--n-max', type=int, default=None)
    exp_parser.add_argument('--coord-range', type=int, default=3)
    exp_parser.add_argument('--model', type=str, default='')
    exp_parser.add_argument('--episodes', type=int, default=500)
    exp_parser.add_argument('--d-model', type=int, default=128)
    exp_parser.add_argument('--n-heads', type=int, default=4)
    exp_parser.add_argument('--n-layers', type=int, default=3)
    exp_parser.add_argument('--singularity-aware', action='store_true',
                            help='Use singularity-driven candidate generation')
    exp_parser.add_argument('--max-candidates', type=int, default=200)
    exp_parser.add_argument('--target-exponents', type=int, nargs=2, default=None,
                            metavar=('D1', 'D2'),
                            help='Target exponent pair (d1 d2) for exploration')

    # verify-found (post-hoc exact check for large n)
    vf_parser = subparsers.add_parser('verify-found')
    vf_parser.add_argument('--n', type=int, default=15)
    vf_parser.add_argument('--n-max', type=int, default=None)
    vf_parser.add_argument('--coord-range', type=int, default=3)
    vf_parser.add_argument('--model', type=str, default='')
    vf_parser.add_argument('--episodes', type=int, default=500)
    vf_parser.add_argument('--d-model', type=int, default=128)
    vf_parser.add_argument('--n-heads', type=int, default=4)
    vf_parser.add_argument('--n-layers', type=int, default=3)
    vf_parser.add_argument('--singularity-aware', action='store_true',
                            help='Use singularity-driven candidate generation')
    vf_parser.add_argument('--max-candidates', type=int, default=200)
    vf_parser.add_argument('--target-exponents', type=int, nargs=2, default=None,
                            metavar=('D1', 'D2'),
                            help='Target exponent pair (d1 d2) for verification')

    # discoveries
    subparsers.add_parser('discoveries', help='Show summary of saved discoveries')

    # extend (bootstrap from known free arrangements)
    ext_parser = subparsers.add_parser('extend', help='Extend known free arrangements by one line')
    ext_parser.add_argument('--n-from', type=int, required=True,
                            help='Source n: load free arrangements with this n as seeds')
    ext_parser.add_argument('--seeds-file', type=str, default='discoveries.json',
                            help='JSON file with seed arrangements (default: discoveries.json)')
    ext_parser.add_argument('--output', type=str, default=None,
                            help='Output file for new discoveries (default: same as seeds-file)')
    ext_parser.add_argument('--coord-range', type=int, default=5,
                            help='Integer pool coordinate range for new lines')
    ext_parser.add_argument('--max-denominator', type=int, default=1,
                            help='If >1, also generate rational lines through existing multiple points')
    ext_parser.add_argument('--loss-threshold', type=float, default=1e-6,
                            help='Pre-filter: skip exact check if penalized '
                                 'Saito loss above this (refit on the '
                                 'validation benchmark)')
    ext_parser.add_argument('--n-restarts', type=int, default=10,
                            help='Optimizer restarts in the penalized loss '
                                 'pre-filter')
    ext_parser.add_argument('--max-seeds', type=int, default=None,
                            help='Limit number of seeds to process (for testing)')
    ext_parser.add_argument('--target-exponents', type=int, nargs=2, default=None,
                            metavar=('D1', 'D2'),
                            help='Only use seeds with these exponents')
    ext_parser.add_argument('--target-new-exponents', type=int, nargs=2, default=None,
                            metavar=('D1', 'D2'),
                            help='Target this exponent type for the n+1 result. Uses Δb2 pre-filter for efficiency.')
    ext_parser.add_argument('--all-targets', action='store_true',
                            help='Target ALL valid exponent types for the n+1 result (overrides --target-new-exponents).')

    # construct (direct construction of known free families)
    con_parser = subparsers.add_parser('construct', help='Construct a known free arrangement family directly')
    con_parser.add_argument('--family', choices=['near-pencil', 'supersolvable', 'all-supersolvable'],
                            required=True,
                            help='Which family to construct')
    con_parser.add_argument('--n', type=int, required=True, help='Number of lines')
    con_parser.add_argument('--d1', type=int, default=None,
                            help='Smaller exponent (only for --family supersolvable)')
    con_parser.add_argument('--output', type=str, default=None,
                            help='Output JSON file (default: discoveries.json)')

    # search
    search_parser = subparsers.add_parser('search')
    search_parser.add_argument('--n', type=int, default=6)
    search_parser.add_argument('--coord-range', type=int, default=3)
    search_parser.add_argument('--exhaustive', action='store_true')
    search_parser.add_argument('--max-check', type=int, default=10000)
    search_parser.add_argument('--max-mult', type=int, default=None)

    args = parser.parse_args()

    if args.command == 'train':
        if args.n_min is None:
            args.n_min = args.n
        if args.n_max is None:
            args.n_max = args.n
        train(args)
    elif args.command in ('verify', 'test'):
        success = run_verify()
        sys.exit(0 if success else 1)
    elif args.command == 'explore':
        run_explore(args)
    elif args.command == 'verify-found':
        run_verify_found(args)
    elif args.command == 'extend':
        run_extend(args)
    elif args.command == 'construct':
        run_construct(args)
    elif args.command == 'search':
        run_search(args)
    elif args.command == 'discoveries':
        discoveries_summary()
    else:
        print("Running quick verification...")
        run_verify()
        print("\nCommands:")
        print("  python main.py verify")
        print("  python main.py search --n 6 --coord-range 3")
        print("  python main.py train --n 6")
        print("  python main.py train --n-min 6 --n-max 15 --total-steps 1000000")
        print("  python main.py explore --n 6 --model model_final.pt")
        print("  python main.py verify-found --n 15 --model model_final.pt --episodes 2000")


if __name__ == "__main__":
    main()
