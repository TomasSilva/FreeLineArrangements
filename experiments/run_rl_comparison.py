"""
RL reward-arm comparison under an equal environment-step budget (§8).

Arms:
  penalized      composite reward with the corrected penalized loss (default)
  potential      potential-based shaping gamma*Phi' - Phi, Phi = Gamma_hat,
                 plus exact terminal certification bonus
  combinatorial  composite reward with w_alg = 0 (no algebraic signal)
  terminal       binary exact-freeness terminal reward only
  random         zero reward (unguided baseline)
  legacy         composite with the explicitly-invalid old angular score
                 (regression baseline only)

Each (arm, seed) run executes `main.py train` in its own working directory,
so discoveries.json / model files never touch the repo's live files.  Runs
are exactly verified: every discovery logged by training is re-verified here
with an exact symbolic Saito certificate.

Usage:
    python experiments/run_rl_comparison.py --out <dir> --n 8 \
        --total-steps 30000 --seeds 5 [--arms penalized,terminal,...]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

UPDATE_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(-?[\d.]+)\s*\|\s*(\d+)\s*\|")


def run_arm(arm, seed, out_root, n, total_steps, coord_range, extra):
    run_dir = os.path.join(out_root, f"{arm}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    cmd = [PY, os.path.join(REPO, "main.py"), "train",
           "--n", str(n), "--coord-range", str(coord_range),
           "--total-steps", str(total_steps),
           "--reward-mode", arm, "--seed", str(seed),
           "--save", "model_final.pt", "--log-every", "1"] + extra
    t0 = time.time()
    with open(os.path.join(run_dir, "stdout.log"), "w") as logf:
        proc = subprocess.run(cmd, cwd=run_dir, stdout=logf,
                              stderr=subprocess.STDOUT,
                              env={**os.environ, "PYTHONPATH": REPO,
                                   "OMP_NUM_THREADS": "1"})
    wall = time.time() - t0

    # parse per-update log lines: update | steps | mean_reward | free | ...
    curve = []
    with open(os.path.join(run_dir, "stdout.log")) as f:
        for line in f:
            m = UPDATE_RE.match(line)
            if m:
                curve.append({"update": int(m.group(1)),
                              "steps": int(m.group(2)),
                              "mean_reward": float(m.group(3)),
                              "free_found": int(m.group(4))})

    # discoveries written by the run (cwd-local file)
    disc_path = os.path.join(run_dir, "discoveries.json")
    discoveries = []
    if os.path.exists(disc_path):
        with open(disc_path) as f:
            discoveries = json.load(f).get("arrangements", [])

    return {"arm": arm, "seed": seed, "returncode": proc.returncode,
            "wall_time_s": wall, "total_steps": total_steps,
            "curve": curve, "n_discoveries_logged": len(discoveries),
            "discoveries": discoveries, "run_dir": run_dir}


def verify_discoveries(result):
    """Exactly re-verify every logged discovery; attach certificates."""
    from main import _parse_line_str
    from arrangement import LineArrangement
    from certificates import find_exact_saito_certificate, certificate_to_json

    verified, certs, lattices, exponent_types = 0, {}, set(), set()
    for i, rec in enumerate(result["discoveries"]):
        try:
            lines = [_parse_line_str(s) for s in rec["lines"]]
            arr = LineArrangement(lines)
            cert = find_exact_saito_certificate(arr)
            if cert is not None:
                verified += 1
                certs[f"disc_{i}"] = certificate_to_json(cert)
                lattices.add(tuple(sorted(arr.multiplicities(),
                                          reverse=True)))
                exps = rec.get("exponents")
                if exps:
                    exponent_types.add(tuple(exps))
        except Exception as e:
            print(f"  verify error on discovery {i}: {e}")
    result["n_exactly_certified"] = verified
    result["n_distinct_lattices"] = len(lattices)
    result["exponent_types"] = sorted(str(t) for t in exponent_types)
    result["certificates"] = certs

    # time to first certified free: first curve point with free_found > 0
    first = next((c for c in result["curve"] if c["free_found"] > 0), None)
    result["steps_to_first_free"] = first["steps"] if first else None
    if result["curve"]:
        frac = (first["steps"] / result["curve"][-1]["steps"]
                if first else None)
        result["time_to_first_free_s"] = (frac * result["wall_time_s"]
                                          if frac else None)
    # optimizer cost per environment step
    result["seconds_per_env_step"] = (result["wall_time_s"]
                                      / max(1, result["total_steps"]))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--total-steps", type=int, default=30000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--coord-range", type=int, default=2)
    ap.add_argument("--arms", type=str,
                    default="penalized,potential,combinatorial,terminal,"
                            "random,legacy")
    ap.add_argument("--extra", type=str, default="",
                    help="extra args passed to main.py train")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    arms = args.arms.split(",")
    extra = args.extra.split() if args.extra else []

    results = []
    for arm in arms:
        for seed in range(args.seeds):
            print(f"== arm={arm} seed={seed} ==", flush=True)
            r = run_arm(arm, seed, args.out, args.n, args.total_steps,
                        args.coord_range, extra)
            r = verify_discoveries(r)
            print(f"   certified={r['n_exactly_certified']} "
                  f"lattices={r['n_distinct_lattices']} "
                  f"first_free_steps={r['steps_to_first_free']} "
                  f"wall={r['wall_time_s']:.0f}s", flush=True)
            results.append(r)
            # persist incrementally (discoveries omitted from summary)
            slim = [{k: v for k, v in x.items()
                     if k not in ("discoveries", "certificates")}
                    for x in results]
            with open(os.path.join(args.out, "rl_comparison.json"),
                      "w") as f:
                json.dump(slim, f, indent=1)
            with open(os.path.join(args.out,
                                   f"certs_{arm}_seed{seed}.json"),
                      "w") as f:
                json.dump(r["certificates"], f, indent=1)

    # aggregate per arm
    agg = {}
    for arm in arms:
        rs = [r for r in results if r["arm"] == arm]
        cert_counts = [r["n_exactly_certified"] for r in rs]
        firsts = [r["steps_to_first_free"] for r in rs
                  if r["steps_to_first_free"] is not None]
        agg[arm] = {
            "seeds": len(rs),
            "certified_mean": sum(cert_counts) / max(1, len(cert_counts)),
            "certified_all": cert_counts,
            "hit_rate": sum(1 for c in cert_counts if c > 0) / max(1, len(rs)),
            "steps_to_first_free": firsts,
            "distinct_lattices": [r["n_distinct_lattices"] for r in rs],
            "exponent_types": sorted({t for r in rs
                                      for t in r["exponent_types"]}),
            "mean_seconds_per_env_step": (sum(r["seconds_per_env_step"]
                                              for r in rs) / max(1, len(rs))),
        }
    with open(os.path.join(args.out, "rl_comparison_summary.json"), "w") as f:
        json.dump(agg, f, indent=1)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
