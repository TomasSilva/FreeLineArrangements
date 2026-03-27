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
        scalar_dim=14,
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

    # discoveries
    subparsers.add_parser('discoveries', help='Show summary of saved discoveries')

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
