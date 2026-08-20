"""
Build the surrogate training dataset from this project's own evaluated
candidates: every candidates.jsonl / certified.jsonl record carries the
exact lines, the target pair and the TRUE raw penalized Saito loss.

Stratified by (cell, loss decade) with a per-stratum cap so certified-heavy
files don't dominate; whole cells are HELD OUT for the generalization
go/no-go.  Output: npz (X, y_log, y_cls, holdout, cell tags) + a JSON
manifest with source hashes, commit and stratum counts.

Usage:
  python experiments/build_surrogate_dataset.py \
      --out results_local/surrogate_experiment/dataset \
      [--roots results_from_HPC results_local results_penalized_saito] \
      [--cap 3000] [--holdout-cells 16,7,8 18,7,10]
"""

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novelty import arrangement_from_record
from surrogate import extract_features, FEATURE_SCHEMA_VERSION, GATE


def loss_decade(loss):
    return int(math.log10(max(loss, 1e-16)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", nargs="+",
                    default=["results_from_HPC", "results_local",
                             "results_penalized_saito"])
    ap.add_argument("--cap", type=int, default=3000,
                    help="max records per (cell, loss-decade) stratum")
    ap.add_argument("--holdout-cells", nargs="+",
                    default=["16,7,8", "18,7,10"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    holdout = {tuple(map(int, c.split(","))) for c in args.holdout_cells}
    rng = np.random.default_rng(args.seed)

    files = []
    for root in args.roots:
        files += glob.glob(f"{root}/**/candidates.jsonl", recursive=True)
        files += glob.glob(f"{root}/**/certified.jsonl", recursive=True)
    files = sorted(set(files))

    strata = {}                     # (cell, decade) -> list of records
    seen_lines = set()
    n_read = n_dupe = n_bad = 0
    for path in files:
        try:
            fh = open(path)
        except OSError:
            continue
        for line in fh:
            try:
                r = json.loads(line)
                loss = float(r["loss"])
                key = (r["n"], r["d1"], r["d2"])
            except Exception:
                n_bad += 1
                continue
            n_read += 1
            lk = hash((tuple(sorted(r["lines"])), key))
            if lk in seen_lines:
                n_dupe += 1
                continue
            seen_lines.add(lk)
            s = (key, loss_decade(loss))
            bucket = strata.setdefault(s, [])
            if len(bucket) < 4 * args.cap:     # reservoir headroom
                bucket.append(r)

    rows, meta = [], []
    t0 = time.time()
    for (cell, dec), bucket in sorted(strata.items(), key=str):
        if len(bucket) > args.cap:
            idx = rng.choice(len(bucket), size=args.cap, replace=False)
            bucket = [bucket[int(i)] for i in idx]
        for r in bucket:
            try:
                arr = arrangement_from_record(r)
                x = extract_features(arr, r["d1"], r["d2"],
                                     height=r.get("height"))
            except Exception:
                n_bad += 1
                continue
            loss = float(r["loss"])
            rows.append(x)
            meta.append((cell, math.log10(max(loss, 1e-16)),
                         1.0 if loss < GATE else 0.0))
    X = np.stack(rows)
    cells = [m[0] for m in meta]
    y_log = np.array([m[1] for m in meta])
    y_cls = np.array([m[2] for m in meta])
    ho = np.array([tuple(c) in holdout for c in cells])

    manifest = {
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "n_examples": int(len(X)),
        "n_read": n_read, "n_dupe": n_dupe, "n_bad": n_bad,
        "n_holdout": int(ho.sum()),
        "pos_rate": float(y_cls.mean()),
        "holdout_cells": sorted(map(list, holdout)),
        "strata": {f"{k}": len(v) for k, v in
                   sorted(strata.items(), key=str)},
        "n_files": len(files),
        "cap": args.cap, "seed": args.seed,
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seconds": round(time.time() - t0, 1),
    }
    mh = hashlib.sha256(json.dumps(manifest, sort_keys=True)
                        .encode()).hexdigest()[:16]
    manifest["manifest_hash"] = mh
    np.savez_compressed(args.out + ".npz", X=X, y_log=y_log, y_cls=y_cls,
                        holdout=ho,
                        cells=np.array([f"{c[0]},{c[1]},{c[2]}"
                                        for c in cells]),
                        manifest_hash=mh)
    json.dump(manifest, open(args.out + "_manifest.json", "w"), indent=1)
    print(f"dataset: {len(X)} examples ({int(ho.sum())} held out), "
          f"pos rate {y_cls.mean():.4f}, "
          f"{manifest['seconds']}s -> {args.out}.npz")


if __name__ == "__main__":
    main()
