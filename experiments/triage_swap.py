"""
Novelty triage of swap-campaign discoveries.

Reads every cells/*/certified.jsonl, dedups across cells by lattice hash,
screens (supersolvable, near-pencil, reference corpus), ranks by
interestingness, and writes dossiers for the top candidates:

  triage_report.json    all certified discoveries with screen verdicts
  dossiers/<hash>.md    evidence package per headline candidate

Novelty wording is strictly `not_found_in_reference_corpus`: the WL hash is
an isomorphism invariant, so a hash absent from the corpus hashes of the
same cell proves non-isomorphism to every corpus record of that cell
(isomorphic lattices have equal hashes).  "Previously unknown" is never
claimed; a literature check is a human TODO recorded in each dossier.

Optional deep check: bounded inductive-freeness disproof (n <= --if-max-n)
via memoized deletion recursion with the Orlik-Terao addition-deletion
exponent condition (any two of (a) A free {1,d1,d2}, (b) A' free with one
exponent decremented, (c) A'' free {1, t-1} imply the third; for a line
ℓ with t intersection points on it, a deletion step is admissible iff
t - 1 ∈ {d1, d2}).  Outcomes: 'inductively_free' (chain found),
'not_inductively_free' (exhaustive), 'timeout' (inconclusive).
"""

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arrangement import LineArrangement
from novelty import (parse_line_str, lattice_wl_hash, is_supersolvable_rank3,
                     is_near_pencil, coordinate_height, modular_points)
from certificates import find_certificate_fast

EMPTY_CELLS = {(15, 2, 12), (15, 3, 11), (15, 4, 10), (15, 5, 9),
               (16, 2, 13), (16, 3, 12), (16, 4, 11), (16, 5, 10),
               (17, 2, 14), (17, 3, 13), (17, 4, 12), (17, 5, 11), (17, 6, 10),
               (18, 2, 15), (18, 3, 14), (18, 4, 13), (18, 5, 12), (18, 6, 11),
               (19, 2, 16), (19, 3, 15), (19, 4, 14), (19, 5, 13), (19, 6, 12),
               (20, 2, 17), (20, 3, 16), (20, 4, 15), (20, 5, 14), (20, 6, 13),
               (20, 7, 12)}


def _points_on_line(arr, idx):
    return sum(1 for lines in arr.intersection_points().values()
               if idx in lines)


def inductive_freeness_status(arr, d1, d2, deadline, memo=None):
    """('inductively_free' | 'not_inductively_free' | 'timeout', explored)."""
    memo = {} if memo is None else memo
    n = len(arr)
    if n <= 2 or d1 == 0:
        return "inductively_free", 0
    if min(d1, d2) == 1:
        # exponents (1,1,n-2): near-pencil; near-pencils are inductively free
        return "inductively_free", 0
    key = (lattice_wl_hash(arr), d1, d2)
    if key in memo:
        return memo[key], 0
    explored = 0
    any_timeout = False
    for i in range(n):
        if time.time() > deadline:
            memo[key] = "timeout"
            return "timeout", explored
        t = _points_on_line(arr, i)
        if t - 1 == d1:
            sub_exps = tuple(sorted((d1, d2 - 1)))
        elif t - 1 == d2:
            sub_exps = tuple(sorted((d1 - 1, d2)))
        else:
            continue        # addition-deletion exponent condition fails
        rest = LineArrangement([l for j, l in enumerate(arr.lines) if j != i])
        cert, status = find_certificate_fast(rest, target_exponents=sub_exps)
        explored += 1
        if cert is None:
            continue        # deletion not free with the required exponents
        verdict, sub_explored = inductive_freeness_status(
            rest, sub_exps[0], sub_exps[1], deadline, memo)
        explored += sub_explored
        if verdict == "inductively_free":
            memo[key] = "inductively_free"
            return "inductively_free", explored
        if verdict == "timeout":
            any_timeout = True
    verdict = "timeout" if any_timeout else "not_inductively_free"
    memo[key] = verdict
    return verdict, explored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells-dir", required=True)
    ap.add_argument("--reference-hashes", default=None,
                    help="headline.json from the corpus re-pricing")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--if-max-n", type=int, default=14)
    ap.add_argument("--if-timeout", type=float, default=1800.0)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "dossiers"), exist_ok=True)

    ref = {}
    if args.reference_hashes and os.path.exists(args.reference_hashes):
        with open(args.reference_hashes) as f:
            ref = json.load(f)

    # collect and cross-cell dedup by lattice hash
    seen = {}
    for path in sorted(glob.glob(os.path.join(args.cells_dir, "*",
                                              "certified.jsonl"))):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                h = rec["lattice_hash"]
                cellkey = (rec["n"], rec["d1"], rec["d2"])
                k = (cellkey, h)
                if k not in seen or rec["height"] < seen[k]["height"]:
                    rec["cell_dir"] = os.path.dirname(path)
                    seen[k] = rec

    rows = []
    for ((n, d1, d2), h), rec in sorted(seen.items()):
        cell_name = f"{n}_{d1}_{d2}"
        corpus_hashes = set(ref.get(cell_name, {}).get("hashes", {}))
        row = {
            "n": n, "d1": d1, "d2": d2, "lattice_hash": h,
            "lines": rec["lines"], "m_max": rec["m_max"],
            "height": rec["height"], "loss": rec["loss"],
            "engine": rec.get("engine"),
            "supersolvable": rec.get("supersolvable"),
            "near_pencil": rec["m_max"] >= n - 1,
            "empty_cell": (n, d1, d2) in EMPTY_CELLS,
            "in_reference_corpus": h in corpus_hashes,
            "corpus_screen": ("found_in_reference_corpus"
                              if h in corpus_hashes else
                              ("not_found_in_reference_corpus"
                               if corpus_hashes or cell_name in ref
                               else "no_reference_index")),
            "certificate_file": os.path.join(rec["cell_dir"],
                                             rec["certificate_file"]),
        }
        row["interest_rank"] = (
            0 if not row["supersolvable"] else 1,
            0 if row["corpus_screen"] == "not_found_in_reference_corpus" else 1,
            0 if row["empty_cell"] else 1,
            row["m_max"], row["height"],
        )
        rows.append(row)
    rows.sort(key=lambda r: r["interest_rank"])

    # deep checks on the top candidates
    for rec in rows[:args.top]:
        arr = LineArrangement([parse_line_str(s) for s in rec["lines"]])
        rec["modular_points"] = [int(m) for _, m in modular_points(arr)]
        if rec["n"] <= args.if_max_n and not rec["supersolvable"]:
            deadline = time.time() + args.if_timeout
            verdict, explored = inductive_freeness_status(
                arr, rec["d1"], rec["d2"], deadline)
            rec["inductive_freeness"] = verdict
            rec["if_nodes_explored"] = explored
        dossier = os.path.join(args.out, "dossiers",
                               f"{rec['lattice_hash'][:16]}.md")
        with open(dossier, "w") as f:
            f.write(
                f"# Candidate {rec['lattice_hash'][:16]} — "
                f"cell ({rec['n']}, {rec['d1']}, {rec['d2']})\n\n"
                f"* exactly certified free, exponents (1, {rec['d1']}, "
                f"{rec['d2']}); certificate: `{rec['certificate_file']}`\n"
                f"* m_max = {rec['m_max']}; coordinate height = "
                f"{rec['height']}; found by `{rec['engine']}`\n"
                f"* supersolvable (modular point): {rec['supersolvable']} "
                f"(modular multiplicities: {rec.get('modular_points', [])})\n"
                f"* near-pencil: {rec['near_pencil']}; empty-cell: "
                f"{rec['empty_cell']}\n"
                f"* corpus screen: **{rec['corpus_screen']}** (WL-hash level; "
                f"equal lattices have equal hashes, so an absent hash proves "
                f"non-isomorphism to every indexed corpus record of this "
                f"cell)\n"
                f"* inductive freeness: "
                f"{rec.get('inductive_freeness', 'not_checked')}\n"
                f"* TODO (human): literature check against known families "
                f"(Grünbaum/simplicial, reflection, Ziegler-type) before any "
                f"novelty wording beyond the corpus screen.\n\n"
                f"Lines:\n\n"
                + "\n".join(f"    {s}" for s in rec["lines"]) + "\n")

    with open(os.path.join(args.out, "triage_report.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "interest_rank"}
                   for r in rows], f, indent=1)

    n_ss = sum(1 for r in rows if r["supersolvable"])
    n_new = sum(1 for r in rows
                if r["corpus_screen"] == "not_found_in_reference_corpus")
    print(f"triage: {len(rows)} distinct (cell, lattice) discoveries; "
          f"{len(rows) - n_ss} non-supersolvable; "
          f"{n_new} not_found_in_reference_corpus; "
          f"dossiers for top {min(args.top, len(rows))}")


if __name__ == "__main__":
    main()
