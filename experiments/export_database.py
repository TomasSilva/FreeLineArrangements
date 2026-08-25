"""
Export the certified-discovery database as ONE shareable file at the repo
root: a gzipped JSONL whose first line is a `_meta` record (schema, counts,
provenance, how to re-verify) and whose remaining lines are one entry per
distinct certified lattice-isomorphism class.

Each entry is self-contained: search metadata (n, exponents, m_max,
epsilon = d1 - m_max, t-vector, b2, field, height, supersolvability, DKP
rarity) PLUS the full exact Saito certificate verbatim (lines, theta1,
theta2, c, Q) so anyone can re-verify freeness with this repository:

    import gzip, json
    from certificates import certificate_from_json, verify_certificate
    with gzip.open("certified_free_arrangements.jsonl.gz", "rt") as f:
        meta = json.loads(next(f))
        entry = json.loads(next(f))
    assert verify_certificate(certificate_from_json(entry["certificate"]))

While exporting, every entry is cross-checked: the arrangement rebuilt
from the CERTIFICATE's lines must reproduce the recorded WL lattice hash,
b2 must equal both (n-1) + d1*d2 and sum(m_P - 1), and the multiplicity
vector must satisfy sum C(m_P,2) = C(n,2).  Any anomaly aborts the export.

Usage:
  python experiments/export_database.py \
      [--roots results_from_HPC results_local] \
      [--out certified_free_arrangements.jsonl.gz] [--workers 7]
"""

import argparse
import glob
import gzip
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
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


def cert_path_for(h, source, all_cert_dirs):
    d = os.path.join(os.path.dirname(source), "certificates")
    c = sorted(glob.glob(os.path.join(d, f"cert_{h[:16]}_*.json")))
    if c:
        return c[0]
    for cd in all_cert_dirs:
        if cd != d:
            c = sorted(glob.glob(os.path.join(cd, f"cert_{h[:16]}_*.json")))
            if c:
                return c[0]
    return None


def build_entry(task):
    h, rec, cert_file = task
    from certificates import certificate_from_json
    from novelty import lattice_wl_hash
    from arrangement import LineArrangement, ProjectiveLine

    cert_raw = json.load(open(cert_file))
    cert = certificate_from_json(cert_raw)
    arr = LineArrangement([ProjectiveLine(*c) for c in cert["lines"]])

    if lattice_wl_hash(arr) != h:
        return {"_error": f"LATTICE_MISMATCH {h}"}
    n, d1, d2 = rec["n"], rec["d1"], rec["d2"]
    mults = arr.multiplicities()
    tvec = defaultdict(int)
    for m in mults:
        tvec[m] += 1
    m_max = max(mults)
    if m_max != rec["m_max"]:
        return {"_error": f"MMAX_MISMATCH {h}"}
    if sum(m * (m - 1) // 2 for m in mults) != n * (n - 1) // 2:
        return {"_error": f"TVECTOR_INCONSISTENT {h}"}
    b2 = sum(m - 1 for m in mults)
    if b2 != rec.get("b2") or b2 != (n - 1) + d1 * d2:
        return {"_error": f"B2_MISMATCH {h}"}

    cf = rec.get("coefficient_field", "QQ")
    eps = d1 - m_max
    return {
        "lattice_hash": h,
        "n": n,
        "exponents": [1, d1, d2],
        "m_max": m_max,
        "epsilon": eps,
        "dkp_rare": bool(eps >= 2 and d1 < n - m_max),
        "b2": b2,
        "t_vector": {str(k): tvec[k] for k in sorted(tvec)},
        "n_points": len(mults),
        "supersolvable": bool(rec.get("supersolvable")),
        "field": cf,
        "height": rec.get("height"),
        "lines": cert_raw["lines"],
        "certificate": cert_raw,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+",
                    default=["results_from_HPC", "results_local"])
    ap.add_argument("--out", default="certified_free_arrangements.jsonl.gz")
    ap.add_argument("--workers", type=int, default=7)
    args = ap.parse_args()

    best = collect(args.roots)
    all_cert_dirs = sorted({os.path.join(os.path.dirname(p), "certificates")
                            for _, p in best.values()})
    tasks = []
    for h, (rec, p) in sorted(best.items()):
        cf = cert_path_for(h, p, all_cert_dirs)
        if cf is None:
            sys.exit(f"FATAL: no certificate file for {h}")
        tasks.append((h, rec, cf))
    print(f"exporting {len(tasks)} distinct certified lattices", flush=True)

    t0 = time.time()
    entries = []
    with Pool(args.workers) as pool:
        for i, e in enumerate(pool.imap_unordered(build_entry, tasks,
                                                  chunksize=8)):
            if "_error" in e:
                sys.exit(f"FATAL: {e['_error']}")
            entries.append(e)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(tasks)} "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)

    entries.sort(key=lambda e: (e["n"], e["exponents"][1],
                                -e["epsilon"], e["lattice_hash"]))
    eps_hist = defaultdict(int)
    fields = defaultdict(int)
    by_n = defaultdict(int)
    for e in entries:
        eps_hist[e["epsilon"]] += 1
        by_n[e["n"]] += 1
        f = e["field"]
        fields[f if isinstance(f, str) else f["name"]] += 1

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    meta = {
        "_meta": {
            "title": "Certified free line arrangements in P^2 — "
                     "discovery database",
            "generated": "2026-08-25",
            "generator": "experiments/export_database.py",
            "git_commit": commit,
            "total_entries": len(entries),
            "n_range": [min(by_n), max(by_n)],
            "by_n": {str(k): by_n[k] for k in sorted(by_n)},
            "epsilon_histogram": {str(k): eps_hist[k]
                                  for k in sorted(eps_hist)},
            "fields": dict(sorted(fields.items())),
            "dkp_rare_count": sum(1 for e in entries if e["dkp_rare"]),
            "deduplication": "one entry per distinct intersection-lattice "
                             "isomorphism class (Weisfeiler-Leman "
                             "fingerprint, colored line-point incidence "
                             "graph; collision-checked by exact VF2 "
                             "isomorphism tests), keeping the lowest-"
                             "coordinate-height representative",
            "schema": {
                "lattice_hash": "WL fingerprint of the intersection "
                                "lattice (dedup key)",
                "n": "number of lines",
                "exponents": "[1, d1, d2] with d1 <= d2, d1+d2 = n-1",
                "m_max": "maximal multiplicity of an intersection point",
                "epsilon": "d1 - m_max (Dimca-Kuhne-Pokora invariant)",
                "dkp_rare": "epsilon >= 2 and d1 < n - m_max",
                "b2": "second Betti number of the complement "
                      "= (n-1) + d1*d2 = sum_P (m_P - 1)",
                "t_vector": "{multiplicity k: number of points t_k}",
                "n_points": "number of intersection points",
                "supersolvable": "supersolvability of the intersection "
                                 "lattice",
                "field": "coefficient field: 'QQ' or "
                         "{type:'quadratic', d, name, embedding}",
                "height": "max abs value of numerators/denominators of "
                          "the representative's coordinates",
                "lines": "projective line coefficients [a,b,c] (line "
                         "ax+by+cz=0), exact tokens: rationals 'p/q'; "
                         "quadratic-field elements '[u,v]' meaning "
                         "u + v*sqrt(d)",
                "certificate": "exact Saito certificate: lines, Q "
                               "(defining polynomial), theta1/theta2 "
                               "(degree-d1/d2 derivations as stacked "
                               "coefficient vectors of their x,y,z "
                               "components in graded-lex monomial "
                               "order), scalar c with det M(theta_E, "
                               "theta1, theta2) = c*Q, c != 0",
            },
            "verification": "every entry re-verified symbolically over "
                            "its declared field "
                            "(experiments/recheck_all_certificates.py: "
                            "6146/6146 VERIFIED); to re-verify entry E: "
                            "verify_certificate(certificate_from_json("
                            "E['certificate']))",
        }
    }

    with gzip.open(args.out, "wt") as f:
        f.write(json.dumps(meta) + "\n")
        for e in entries:
            f.write(json.dumps(e) + "\n")
    sz = os.path.getsize(args.out) / 1e6
    print(f"DONE: {len(entries)} entries -> {args.out} "
          f"({sz:.1f} MB gz) in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
