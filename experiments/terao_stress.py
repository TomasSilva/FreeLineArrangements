"""
Terao-conjecture stress campaign: re-realize certified lattices, test
freeness of every accepted realization with exact certificates on BOTH
sides.

Per target lattice (from the shareable database):
  1. load the certified reference realization (proven certificate path);
  2. special-position profile + moduli/tangent dimensions;
  3. sample K lattice-preserving re-realizations (exact WL+VF2 gate),
     plus optional conic-forcing solves;
  4. per realization: numeric loss ROUTES the work (never evidence) ->
     exact Saito certification (freeness persists) or exact non-freeness
     witness (counterexample candidate).
  5. any 'nonfree_certified' triggers the counterexample protocol:
     self-contained bundle + fresh-subprocess re-verification before it
     is even labeled a candidate.  Nothing is auto-published.

Outputs (report.jsonl, summary.json, bundles) live strictly under
results_local/terao_stress/.

Usage:
  python experiments/terao_stress.py --targets flagship --per-lattice 10
  python experiments/terao_stress.py --targets braid,akn13 --per-lattice 5
  python experiments/terao_stress.py --targets 'eps>=6' --per-lattice 8
"""

import argparse
import gzip
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arrangement import LineArrangement, ProjectiveLine
from certificates import (certificate_from_json, find_certificate_fast,
                          nonfreeness_certificate, certificate_to_json,
                          verify_certificate)
from novelty import canonical_lineset_key, lattice_wl_hash
from realization import (incidence_structure, expected_moduli_dim,
                         tangent_space_dim, sample_realization,
                         force_conic_realization, lattice_isomorphism_map)
from geometry import special_position_profile

OUT_PREFIX = os.path.join("results_local", "terao_stress")

REVERIFY_SRC = '''\
"""Standalone re-verification of a counterexample candidate bundle.
Run from the repo root:  python <bundle>/reverify.py"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from certificates import (certificate_from_json, verify_certificate,
                          verify_nonfreeness_certificate)
from arrangement import LineArrangement, ProjectiveLine
from novelty import parse_line_str, lattices_isomorphic
from quadfield import QuadraticField

ref = json.load(open(os.path.join(HERE, "reference_entry.json")))
cert = certificate_from_json(ref["certificate"])
assert verify_certificate(cert), "reference Saito certificate FAILED"
ref_arr = LineArrangement([ProjectiveLine(*c) for c in cert["lines"]])

w = json.load(open(os.path.join(HERE, "witness.json")))
assert verify_nonfreeness_certificate(w), "non-freeness witness FAILED"
K = QuadraticField.from_json(w.get("field", "QQ"))
new_arr = LineArrangement([parse_line_str(s, field=K) for s in w["lines"]])

assert lattices_isomorphic(ref_arr, new_arr), "lattices NOT isomorphic"
print("REVERIFY OK: same lattice, reference free, new realization nonfree")
'''


def _fixture_targets(names):
    """Fixture pseudo-targets for dry runs: {name: (arr, (d1, d2))}."""
    from tests.conftest import BRAID_A3
    out = {}
    for name in names:
        if name == "braid":
            arr = LineArrangement([ProjectiveLine(*c) for c in BRAID_A3])
            out["fixture:braid"] = (arr, (2, 3))
        elif name == "akn13":
            from known_arrangements import akn13, validate_akn13_lattice
            arr = akn13()
            assert validate_akn13_lattice(arr)
            out["fixture:akn13"] = (arr, (6, 6))
        else:
            raise SystemExit(f"unknown fixture target: {name}")
    return out


def load_targets(db_path, spec):
    """{label: (reference_arr, (d1, d2))} plus the raw DB entry."""
    fixtures = [t for t in spec if t in ("braid", "akn13")]
    targets = {}
    entries = {}
    for label, (arr, pair) in _fixture_targets(fixtures).items():
        targets[label] = (arr, pair)
        entries[label] = None
    want_flagship = "flagship" in spec
    eps_min = None
    hashes = set()
    for t in spec:
        if t.startswith("eps>="):
            eps_min = int(t[5:])
        elif t not in ("braid", "akn13", "flagship"):
            hashes.add(t)
    if want_flagship or eps_min is not None or hashes:
        with gzip.open(db_path, "rt") as f:
            next(f)
            for line in f:
                e = json.loads(line)
                take = (e["lattice_hash"] in hashes
                        or (eps_min is not None and e["epsilon"] >= eps_min)
                        or (want_flagship and e["epsilon"] >= 7))
                if not take:
                    continue
                cert = certificate_from_json(e["certificate"])
                arr = LineArrangement([ProjectiveLine(*c)
                                       for c in cert["lines"]])
                label = e["lattice_hash"]
                targets[label] = (arr, (e["exponents"][1],
                                        e["exponents"][2]))
                entries[label] = e
    if want_flagship and "fixture:akn13" not in targets:
        for label, (arr, pair) in _fixture_targets(["akn13"]).items():
            targets[label] = (arr, pair)
            entries[label] = None
    return targets, entries


def counterexample_bundle(out_dir, label, k, entry, reference, arr,
                          witness, prof_ref, prof_new):
    bdir = os.path.join(out_dir, f"counterexample_candidate_{label}_{k}")
    os.makedirs(bdir, exist_ok=True)
    if entry is not None:
        ref_entry = entry
    else:
        cert, status = find_certificate_fast(reference)
        assert status == "certified"
        ref_entry = {"certificate": certificate_to_json(cert)}
    json.dump(ref_entry, open(os.path.join(bdir, "reference_entry.json"),
                              "w"), indent=1)
    json.dump(witness, open(os.path.join(bdir, "witness.json"), "w"),
              indent=1)
    iso = lattice_isomorphism_map(reference, arr)
    json.dump(iso, open(os.path.join(bdir, "isomorphism.json"), "w"),
              indent=1)
    json.dump({"reference": prof_ref, "new": prof_new},
              open(os.path.join(bdir, "profiles.json"), "w"), indent=1)
    with open(os.path.join(bdir, "reverify.py"), "w") as f:
        f.write(REVERIFY_SRC)
    r = subprocess.run([sys.executable, os.path.join(bdir, "reverify.py")],
                       capture_output=True, text=True, timeout=3600)
    ok = (r.returncode == 0)
    json.dump({"returncode": r.returncode, "stdout": r.stdout,
               "stderr": r.stderr},
              open(os.path.join(bdir, "reverify_result.json"), "w"),
              indent=1)
    return bdir, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="certified_free_arrangements.jsonl.gz")
    ap.add_argument("--out", default=os.path.join(
        OUT_PREFIX, time.strftime("%Y-%m-%d")))
    ap.add_argument("--targets", nargs="+", required=True,
                    help="'flagship' | 'eps>=E' | lattice hashes | "
                         "fixtures braid/akn13")
    ap.add_argument("--per-lattice", type=int, default=10)
    ap.add_argument("--height-bound", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", nargs="+",
                    default=["random", "small_height", "force_conic"])
    ap.add_argument("--max-targets", type=int, default=None)
    ap.add_argument("--loss-route-threshold", type=float, default=1e-6)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    assert os.path.abspath(OUT_PREFIX) in out_dir, \
        f"outputs must stay under {OUT_PREFIX}"
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    targets, entries = load_targets(args.db, args.targets)
    labels = list(targets)
    if args.max_targets:
        labels = labels[:args.max_targets]
    print(f"terao_stress: {len(labels)} target lattices -> {out_dir}",
          flush=True)

    report_path = os.path.join(out_dir, "report.jsonl")
    summary = {"args": vars(args), "targets": {}, "counterexamples": []}
    rep = open(report_path, "a")

    for label in labels:
        reference, (d1, d2) = targets[label]
        entry = entries.get(label)
        t0 = time.time()
        struct = incidence_structure(reference)
        prof_ref = special_position_profile(reference, rng=rng)
        dims = {"expected_moduli_dim": expected_moduli_dim(struct),
                "tangent_dim": tangent_space_dim(reference, struct)}
        tstats = {"accepted": 0, "free_recertified": 0, "unresolved": 0,
                  "nonfree_certified": 0, "sampler_failures": 0,
                  "zero_information": 0, "dims": dims,
                  "reference_conics": [c for c in prof_ref["conics"]
                                       if not c.get("implied")]}
        ref_key = canonical_lineset_key(reference)
        print(f"[{label[:16]}] n={struct['n']} pair=({d1},{d2}) "
              f"dims={dims} ref_conics={len(tstats['reference_conics'])}",
              flush=True)

        for k in range(args.per_lattice):
            mode = args.modes[k % len(args.modes)]
            arr = None
            if mode == "force_conic":
                T = len(struct["points"])
                if T >= 6 and not tstats["reference_conics"]:
                    ids = sorted(int(i) for i in
                                 rng.choice(T, size=6, replace=False))
                    arr, _flog = force_conic_realization(
                        reference, ids, rng, struct=struct, tries=10,
                        height_bound=args.height_bound)
                if arr is None:
                    mode = "random"
            if arr is None:
                hb = (3 if mode == "small_height" else args.height_bound)
                arr = sample_realization(reference, rng, struct=struct,
                                         height_bound=hb)
            if arr is None:
                tstats["sampler_failures"] += 1
                continue
            tstats["accepted"] += 1
            rec = {"target": label, "k": k, "mode": mode,
                   "lines": [str(l) for l in arr.lines],
                   "wl": lattice_wl_hash(arr)}
            if canonical_lineset_key(arr) == ref_key:
                tstats["zero_information"] += 1
                rec["zero_information"] = True
            prof_new = special_position_profile(arr, rng=rng)
            rec["conics"] = [c for c in prof_new["conics"]
                             if not c.get("implied")]
            # numeric routing (never evidence)
            loss = None
            try:
                from penalized_saito import (cached_penalized_loss,
                                             GammaNumericalError)
                loss = float(cached_penalized_loss(arr, d1=d1, d2=d2,
                                                   profile="rl",
                                                   seed=args.seed))
            except Exception:         # noqa: BLE001
                pass
            rec["loss"] = loss

            status = None
            if loss is not None and loss < args.loss_route_threshold:
                cert, cstatus = find_certificate_fast(
                    arr, target_exponents=(d1, d2))
                if cstatus == "certified":
                    status = "free_recertified"
            if status is None:
                w, wstatus = nonfreeness_certificate(arr)
                if wstatus == "nonfree_certified":
                    prof_json = {"reference": prof_ref, "new": prof_new}
                    bdir, ok = counterexample_bundle(
                        out_dir, label, k, entry, reference, arr, w,
                        prof_ref, prof_new)
                    status = ("COUNTEREXAMPLE_CANDIDATE" if ok
                              else "counterexample_reverify_FAILED")
                    rec["bundle"] = bdir
                    tstats["nonfree_certified"] += 1
                    if ok:
                        summary["counterexamples"].append(bdir)
                        print(f"  !!! COUNTEREXAMPLE CANDIDATE: {bdir}",
                              flush=True)
                else:
                    # possibly free with a numerically missed zero: one
                    # full certification attempt resolves it
                    cert, cstatus = find_certificate_fast(
                        arr, target_exponents=(d1, d2))
                    status = ("free_recertified" if cstatus == "certified"
                              else "unresolved")
            if status == "free_recertified":
                tstats["free_recertified"] += 1
            elif status == "unresolved":
                tstats["unresolved"] += 1
            rec["status"] = status
            rep.write(json.dumps(rec) + "\n")
            rep.flush()

        tstats["seconds"] = round(time.time() - t0, 1)
        summary["targets"][label] = tstats
        json.dump(summary, open(os.path.join(out_dir, "summary.json"),
                                "w"), indent=1)
        print(f"[{label[:16]}] done: {tstats}", flush=True)

    rep.close()
    n_ce = len(summary["counterexamples"])
    print(f"DONE: {len(labels)} targets; counterexample candidates: {n_ce}",
          flush=True)
    if n_ce:
        print("REVIEW THE BUNDLES BEFORE ANY CLAIM.", flush=True)


if __name__ == "__main__":
    main()
