"""
Benchmark suite construction for the penalized Saito functional validation
study.  Every "free" item is exactly certified (symbolic Saito certificate
over Q) when it is built; certificates are saved alongside the results.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational

from arrangement import LineArrangement, ProjectiveLine
from saito import construct_near_pencil, construct_supersolvable
from certificates import find_exact_saito_certificate, certificate_to_json


def arr_from(coords):
    return LineArrangement([ProjectiveLine(*c) for c in coords])


NONFREE7_A = [(0, 1, -1), (1, 1, -1), (1, 1, 1), (1, 1, 2), (1, 1, 0),
              (1, 1, -2), (1, 0, -2)]
NONFREE7_B = [(1, 0, -1), (1, 0, -2), (2, 0, 1), (0, 0, 1), (1, -2, 0),
              (0, 1, 2), (1, 0, 0)]
GENERIC = {
    4: [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, 3)],
    5: [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, 3), (3, -1, 2)],
    6: [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, 3), (3, -1, 2), (2, 5, -1)],
}
BRAID_A3 = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (1, 0, -1),
            (0, 1, -1)]


def _search_nonfree_with_exponents(n, count, seed, max_trials=60000,
                                   coord=2):
    """Deterministic search for nonfree integer arrangements with integer
    candidate exponents (exactly verified nonfree)."""
    rng = random.Random(seed)
    pool, seen = [], set()
    for a in range(-coord, coord + 1):
        for b in range(-coord, coord + 1):
            for c in range(-coord, coord + 1):
                if (a, b, c) == (0, 0, 0):
                    continue
                L = ProjectiveLine(a, b, c)
                if L.coords not in seen:
                    seen.add(L.coords)
                    pool.append(L)
    found = []
    for _ in range(max_trials):
        lines = rng.sample(pool, n)
        arr = LineArrangement(lines)
        ce = arr.candidate_exponents()
        if ce is None or ce[0] == 0:
            continue
        free, _ = arr.is_free()
        if not free:
            found.append(arr)
            if len(found) >= count:
                break
    return found


def build_suite(verbose=True, certify=True, max_certify_n=12):
    """Build the benchmark items.

    Returns (items, certificates):
      items: list of dicts {name, family, label ('free'|'nonfree'),
             lines (list of coord tuples), pair (d1, d2), cand_exps, n, notes}
      certificates: dict name -> exact certificate JSON (free items only)
    """
    items = []
    certs = {}

    def add(name, family, arr, pair, label, notes=""):
        item = {
            "name": name,
            "family": family,
            "label": label,
            "lines": [tuple(str(v) for v in l.coords) for l in arr.lines],
            "pair": list(pair),
            "cand_exps": (list(arr.candidate_exponents())
                          if arr.candidate_exponents() else None),
            "n": len(arr),
            "b2": arr.b2(),
            "notes": notes,
        }
        items.append(item)
        if label == "free" and certify and len(arr) <= max_certify_n:
            cert = find_exact_saito_certificate(arr, target_exponents=pair)
            assert cert is not None, f"free item {name} failed certification!"
            certs[name] = certificate_to_json(cert)
        if verbose:
            print(f"  [{label:7s}] {name:26s} n={len(arr):2d} pair={pair}")
        return arr

    # ── free items, several exponent types ──────────────────────────────────
    add("braid_A3", "braid", arr_from(BRAID_A3), (2, 3), "free")
    add("A2xA1", "reflection",
        arr_from([(1, 0, 0), (0, 1, 0), (1, -1, 0), (0, 0, 1)]), (1, 2),
        "free")
    add("pencil_4", "pencil", arr_from([(1, 0, 0), (0, 1, 0), (1, 1, 0),
                                        (1, 2, 0)]), (0, 3), "free",
        "nonessential; exponents (1, 0, 3)")
    for n in (6, 8, 10):
        arr = construct_near_pencil(n)
        add(f"near_pencil_{n}", "near-pencil", arr, (1, n - 2), "free")
    for (n, d1) in ((8, 2), (9, 3), (9, 4), (10, 4), (11, 5), (12, 5)):
        arr = construct_supersolvable(n, d1)
        add(f"supersolvable_{n}_{d1}", "supersolvable", arr,
            (d1, n - 1 - d1), "free")

    # ── nonfree with the same cardinalities and candidate arithmetic ────────
    nf7a = arr_from(NONFREE7_A)
    add("nonfree7_a", "random-int", nf7a, (3, 3), "nonfree",
        "integer candidate exponents (3,3); exactly verified nonfree")
    nf7b = arr_from(NONFREE7_B)
    add("nonfree7_b", "random-int", nf7b, (3, 3), "nonfree",
        "integer candidate exponents (3,3); exactly verified nonfree")
    for n, seed in ((9, 11), (11, 13)):
        hits = _search_nonfree_with_exponents(n, 2, seed)
        for k, arr in enumerate(hits):
            ce = arr.candidate_exponents()
            add(f"nonfree{n}_{k}", "random-int", arr, ce, "nonfree",
                f"integer candidate exponents {ce}; exactly verified nonfree")

    # ── no candidate exponents (chi does not factor) ────────────────────────
    for n, coords in GENERIC.items():
        arr = arr_from(coords)
        assert arr.candidate_exponents() is None or n == 4
        pairs = [(1, n - 2)] if n > 4 else [(1, 2)]
        add(f"generic_{n}", "generic", arr, pairs[0], "nonfree",
        "no integer candidate exponents; generic; D(A) needs > 3 minimal "
        "generators (loss evaluated at the given pair regardless)")

    return items, certs


def perturbation_family(base_n=9, d1=3, denominators=(10, 100, 1000, 10**4,
                                                      10**5, 10**6)):
    """Free supersolvable arrangement with one line perturbed by t = 1/den.

    The perturbed arrangements degenerate the lattice; the pair-specific loss
    at the seed's exponents stays defined and should rise continuously from ~0.
    """
    seed_arr = construct_supersolvable(base_n, d1)
    pair = (d1, base_n - 1 - d1)
    out = [("t=0", seed_arr, pair)]
    lines = list(seed_arr.lines)
    a, b, c = lines[-1].coords
    for den in denominators:
        t = Rational(1, den)
        pert = LineArrangement(lines[:-1] +
                               [ProjectiveLine(a + t, b + 2 * t, c - t)])
        out.append((f"t=1/{den}", pert, pair))
    return out
