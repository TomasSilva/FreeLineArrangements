"""
Re-verify EVERY distinct certified discovery from scratch.

For each distinct lattice hash across the given roots:
  1. locate its stored certificate JSON (next to the certified.jsonl that
     recorded it; falls back to any other run that certified the same
     lattice);
  2. parse with certificate_from_json — field-aware: QQ certificates parse
     as exact rationals, quadratic-field certificates parse their bracket
     tokens in the declared Q(sqrt d) with its recorded embedding;
  3. verify_certificate: exactness of all scalars, field consistency,
     distinct lines, degree bookkeeping, logarithmicity of both thetas,
     and det M(theta_E, theta1, theta2) = c*Q with c != 0 — all exact,
     over the certificate's own field;
  4. cross-check certificate <-> record: the arrangement rebuilt from the
     CERTIFICATE's lines must have the record's WL lattice hash.

Parallel across worker processes.  Output: JSONL of per-lattice results +
a summary with failures listed explicitly (never smoothed over).

Usage:
  python experiments/recheck_all_certificates.py \
      --out results_local/full_recheck [--roots ...] [--workers 7]
"""

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def collect(roots):
    """hash -> (record, source_path) keeping the lowest-height record."""
    best = {}
    for root in roots:
        for p in glob.glob(f"{root}/**/certified.jsonl", recursive=True):
            for line in open(p):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                h = r.get("lattice_hash")
                if not h or "d1" not in r:
                    continue
                if h not in best or r.get("height", 1e18) < \
                        best[h][0].get("height", 1e18):
                    best[h] = (r, p)
    return best


def cert_paths_for(h, source, all_cert_dirs):
    """Certificate file candidates: same-run first, then any run."""
    out = []
    d = os.path.join(os.path.dirname(source), "certificates")
    out += sorted(glob.glob(os.path.join(d, f"cert_{h[:16]}_*.json")))
    for cd in all_cert_dirs:
        if cd != d:
            out += sorted(glob.glob(os.path.join(cd,
                                                 f"cert_{h[:16]}_*.json")))
    return out


def check_one(task):
    h, rec, source, cert_files = task
    from certificates import verify_certificate, certificate_from_json
    from novelty import lattice_wl_hash
    from arrangement import LineArrangement, ProjectiveLine
    t0 = time.time()
    res = {"hash": h, "n": rec.get("n"), "d1": rec.get("d1"),
           "d2": rec.get("d2"), "m_max": rec.get("m_max"),
           "field": (rec.get("coefficient_field", "QQ")
                     if isinstance(rec.get("coefficient_field", "QQ"), str)
                     else rec["coefficient_field"].get("name")),
           "source": source, "status": None, "seconds": None}
    if not cert_files:
        res["status"] = "MISSING_CERT_FILE"
        return res
    try:
        cert = certificate_from_json(json.load(open(cert_files[0])))
    except Exception as e:
        res["status"] = f"PARSE_ERROR({e})"
        return res
    try:
        ok = verify_certificate(cert)
    except Exception as e:
        res["status"] = f"VERIFY_EXCEPTION({e})"
        return res
    if not ok:
        res["status"] = "VERIFY_FAILED"
        return res
    # certificate <-> record lattice cross-check
    try:
        arr = LineArrangement([ProjectiveLine(*c) for c in cert["lines"]])
        if lattice_wl_hash(arr) != h:
            res["status"] = "LATTICE_MISMATCH"
            return res
    except Exception as e:
        res["status"] = f"REBUILD_ERROR({e})"
        return res
    res["status"] = "VERIFIED"
    res["seconds"] = round(time.time() - t0, 2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--roots", nargs="+",
                    default=["results_from_HPC", "results_local"])
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    best = collect(args.roots)
    all_cert_dirs = sorted({os.path.join(os.path.dirname(p), "certificates")
                            for _, p in best.values()})
    tasks = [(h, rec, p, cert_paths_for(h, p, all_cert_dirs))
             for h, (rec, p) in sorted(best.items())]
    print(f"re-checking {len(tasks)} distinct certified lattices "
          f"({args.workers} workers)", flush=True)

    t0 = time.time()
    tallies = {}
    out_path = os.path.join(args.out, "recheck.jsonl")
    with open(out_path, "w") as f, Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(check_one, tasks,
                                                    chunksize=8)):
            f.write(json.dumps(res) + "\n")
            tallies[res["status"]] = tallies.get(res["status"], 0) + 1
            if res["status"] != "VERIFIED":
                print(f"  !! {res['status']}: {res['hash'][:12]} "
                      f"n={res['n']} ({res['d1']},{res['d2']}) "
                      f"field={res['field']} src={res['source']}",
                      flush=True)
            if (i + 1) % 250 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(tasks)} ({el/60:.0f} min, "
                      f"{tallies})", flush=True)
    summary = {"total": len(tasks), "tallies": tallies,
               "minutes": round((time.time() - t0) / 60, 1),
               "roots": args.roots}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"),
              indent=1)
    print(f"DONE: {tallies} in {summary['minutes']} min", flush=True)


if __name__ == "__main__":
    main()
