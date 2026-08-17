"""
Generate PBS job lists for the swap campaign (one line per unit:
`n d1 d2 engine seed outdir`).

Tiers:
  A: the 29 empty cells + 5 stragglers (novel-example frontier)  — 2 engines
     x 2 seeds each.
  B: volume/re-pricing cells n = 13, 14 (nontrivial) + populated n = 15, 16
     diagonals — 2 engines x 1 seed.

Usage:
  python experiments/gen_swap_jobs.py --out-base results_penalized_saito/<date>/swap/cells
Writes jobs_swap_tierA.txt and jobs_swap_tierB.txt in the repo root.
"""

import argparse

EMPTY_CELLS = (
    [(15, d1, 14 - d1) for d1 in (2, 3, 4, 5)]
    + [(16, d1, 15 - d1) for d1 in (2, 3, 4, 5)]
    + [(17, d1, 16 - d1) for d1 in (2, 3, 4, 5, 6)]
    + [(18, d1, 17 - d1) for d1 in (2, 3, 4, 5, 6)]
    + [(19, d1, 18 - d1) for d1 in (2, 3, 4, 5, 6)]
    + [(20, d1, 19 - d1) for d1 in (2, 3, 4, 5, 6, 7)]
)
STRAGGLERS = [(15, 6, 8), (16, 6, 9), (18, 7, 10), (19, 7, 11), (20, 8, 11)]
VOLUME = ([(13, d1, 12 - d1) for d1 in (2, 3, 4, 5, 6)]
          + [(14, d1, 13 - d1) for d1 in (2, 3, 4, 5, 6)]
          + [(15, 7, 7), (16, 7, 8)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--engines", default="anneal,me")
    ap.add_argument("--tierA-seeds", type=int, default=2)
    ap.add_argument("--tierB-seeds", type=int, default=1)
    args = ap.parse_args()
    engines = args.engines.split(",")

    def lines(cells, seeds):
        out = []
        for (n, d1, d2) in cells:
            cell_dir = f"{args.out_base}/n{n}_d{d1}_{d2}"
            for eng in engines:
                for s in range(seeds):
                    out.append(f"{n} {d1} {d2} {eng} {s} {cell_dir}")
        return out

    a = lines(EMPTY_CELLS + STRAGGLERS, args.tierA_seeds)
    b = lines(VOLUME, args.tierB_seeds)
    with open("jobs_swap_tierA.txt", "w") as f:
        f.write("\n".join(a) + "\n")
    with open("jobs_swap_tierB.txt", "w") as f:
        f.write("\n".join(b) + "\n")
    print(f"tier A: {len(a)} units ({len(EMPTY_CELLS) + len(STRAGGLERS)} cells)")
    print(f"tier B: {len(b)} units ({len(VOLUME)} cells)")
    print("submit: qsub -v JOBS_FILE=jobs_swap_tierA.txt pbs/step6_swap.pbs")


if __name__ == "__main__":
    main()
