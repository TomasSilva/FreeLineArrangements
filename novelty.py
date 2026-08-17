"""
novelty.py

Lattice-level accounting and known-family screens for free line arrangements.

The repo's historical dedup key is the sorted tuple of line-coordinate
strings (discoveries.py), which counts coordinate representatives, not
combinatorial types (120k records collapse to a few hundred intersection
lattices).  This module supplies the missing layer:

  * `lattice_wl_hash(arr)`   — canonical-ish hash of the intersection lattice
    via Weisfeiler-Leman refinement of the line/point incidence bipartite
    graph.  Equal lattices always get equal hashes (it is an isomorphism
    invariant); distinct lattices collide only with WL-blindness probability,
    so headline claims confirm within-bucket with `lattices_isomorphic`
    (exact VF2).
  * `is_supersolvable_rank3(arr)` — exact modular-point criterion: a rank-3
    essential arrangement is supersolvable iff some intersection point p is
    modular, i.e. every other intersection point shares a line of A with p.
    For a modular point of multiplicity m the exponents are (1, m-1, n-m)
    (consistency assertion used by campaigns).
  * near-pencil / pencil / essentiality screens, coordinate height,
  * reference-corpus indexing so novelty claims use the honest wording
    `not_found_in_reference_corpus` (never "previously unknown").

Import cost is kept light (numpy/sympy/networkx only — no torch), so HPC
workers can use it.
"""

import json
import os
import re
from fractions import Fraction

import networkx as nx
import numpy as np
from sympy import Rational

from arrangement import LineArrangement, ProjectiveLine

__all__ = [
    "parse_line_str",
    "arrangement_from_record",
    "incidence_graph",
    "lattice_wl_hash",
    "lattices_isomorphic",
    "is_essential",
    "is_near_pencil",
    "modular_points",
    "is_supersolvable_rank3",
    "supersolvable_exponents",
    "check_supersolvable_consistency",
    "coordinate_height",
    "canonical_lineset_key",
    "iter_corpus_records",
    "build_reference_hashes",
]


# ─────────────────────────────────────────────────────────────────────────────
# Parsing (self-contained copy of main._parse_line_str to avoid torch import)
# ─────────────────────────────────────────────────────────────────────────────

_LINE_RE = re.compile(r'([+-]?[\d/]*)x([+-][\d/]*)y([+-][\d/]*)z')


def parse_line_str(s):
    """Parse '(ax+by+cz=0)' (discoveries.json format) into a ProjectiveLine."""
    s = (s.strip().strip('(').rstrip(')').replace('=0', '')
          .replace(' ', '').replace('+-', '-'))
    m = _LINE_RE.match(s)
    if not m:
        raise ValueError(f"Cannot parse line: {s}")

    def to_rat(c):
        c = c.strip('+')
        if c in ('', '+'):
            return 1
        if c == '-':
            return -1
        return Fraction(c)

    return ProjectiveLine(Rational(to_rat(m.group(1))),
                          Rational(to_rat(m.group(2))),
                          Rational(to_rat(m.group(3))))


def arrangement_from_record(rec):
    """Build a LineArrangement from a discoveries-JSON record."""
    return LineArrangement([parse_line_str(s) for s in rec["lines"]])


# ─────────────────────────────────────────────────────────────────────────────
# Lattice hashing
# ─────────────────────────────────────────────────────────────────────────────

def incidence_graph(arr: LineArrangement) -> nx.Graph:
    """Bipartite line/point incidence graph with multiplicity-colored points.

    Nodes: ('L', i) for lines with label 'line'; ('P', k) for intersection
    points with label 'pt<multiplicity>'.  Exact-arithmetic incidences.
    """
    G = nx.Graph()
    for i in range(len(arr)):
        G.add_node(("L", i), label="line")
    pts = arr.intersection_points()
    for k, (p, lines) in enumerate(sorted(pts.items(),
                                          key=lambda kv: str(kv[0]))):
        G.add_node(("P", k), label=f"pt{len(lines)}")
        for i in lines:
            G.add_edge(("P", k), ("L", i))
    return G


def lattice_wl_hash(arr: LineArrangement, iterations: int = 4) -> str:
    """Weisfeiler-Leman hash of the incidence graph.

    An isomorphism invariant of the intersection lattice: equal lattices give
    equal hashes; use `lattices_isomorphic` to confirm within a hash bucket
    before making a headline claim.
    """
    G = incidence_graph(arr)
    return nx.weisfeiler_lehman_graph_hash(G, node_attr="label",
                                           iterations=iterations)


def lattices_isomorphic(a: LineArrangement, b: LineArrangement) -> bool:
    """Exact incidence-graph isomorphism (VF2 with point-color matching)."""
    if len(a) != len(b):
        return False
    if sorted(a.multiplicities()) != sorted(b.multiplicities()):
        return False
    Ga, Gb = incidence_graph(a), incidence_graph(b)
    nm = nx.algorithms.isomorphism.categorical_node_match("label", "")
    return nx.is_isomorphic(Ga, Gb, node_match=nm)


# ─────────────────────────────────────────────────────────────────────────────
# Family screens
# ─────────────────────────────────────────────────────────────────────────────

def is_essential(arr: LineArrangement) -> bool:
    """Rank-3 normal matrix (the cone is essential in C^3)."""
    m = np.array([l.to_float() for l in arr.lines])
    return len(arr) >= 3 and np.linalg.matrix_rank(m) == 3


def is_near_pencil(arr: LineArrangement) -> bool:
    """n-1 (or all n) lines through one point."""
    n = len(arr)
    return n >= 3 and arr.max_multiplicity() >= n - 1


def modular_points(arr: LineArrangement):
    """Intersection points p such that every other intersection point shares
    a line of A with p (the rank-3 modularity criterion).

    Returns a list of (point, multiplicity).
    """
    pts = arr.intersection_points()
    out = []
    items = list(pts.items())
    for p, lines_p in items:
        if all(q == p or (lines_p & lines_q) for q, lines_q in items):
            out.append((p, len(lines_p)))
    return out


def is_supersolvable_rank3(arr: LineArrangement) -> bool:
    """Supersolvable iff a modular point exists (rank 3; includes pencils and
    near-pencils).  Exact, O(P^2) in the number of intersection points."""
    return len(modular_points(arr)) > 0


def supersolvable_exponents(arr: LineArrangement):
    """Exponents (1, m-1, n-m) from a modular point of multiplicity m, or
    None if no modular point exists."""
    mps = modular_points(arr)
    if not mps:
        return None
    n = len(arr)
    # any modular point gives a valid supersolvable filtration
    m = mps[0][1]
    return (1, min(m - 1, n - m), max(m - 1, n - m))


def check_supersolvable_consistency(arr: LineArrangement, exponents) -> bool:
    """Campaign-time self-check: if a modular point of multiplicity m exists
    on a certified-free arrangement, the exponents must be {1, m-1, n-m}.
    Returns True when consistent (or when no modular point exists)."""
    mps = modular_points(arr)
    if not mps:
        return True
    n = len(arr)
    exp_set = tuple(sorted(exponents))
    return any(tuple(sorted((1, m - 1, n - m))) == exp_set for _, m in mps)


def coordinate_height(arr: LineArrangement) -> int:
    """Max |numerator|/|denominator| over all line coordinates (elite
    tie-breaker: low-height representatives certify faster)."""
    h = 1
    for line in arr.lines:
        for c in line.coords:
            r = Rational(c)
            h = max(h, abs(int(r.p)), abs(int(r.q)))
    return int(h)


def canonical_lineset_key(arr: LineArrangement) -> str:
    """Coordinate-level dedup key (matches discoveries.py convention)."""
    return str(tuple(sorted(str(line.coords) for line in arr.lines)))


# ─────────────────────────────────────────────────────────────────────────────
# Reference corpus
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CORPUS_PATHS = [
    "discoveries.json", "cascade_v2.json", "coverage_test.json",
    "discoveries_final.json", "cascade_discoveries.json",
    "HPC_partil_discoveries.json", "ddddiscoveries.json",
    "discoveries_agora.json", "new_discoveries.json", "old_discoveries.json",
    "parcial_HPC_discoveries.json", "partial_HPC_discoveries_old.json",
]


def iter_corpus_records(paths=None, repo_root="."):
    """Stream (path, record) over every readable corpus file."""
    paths = DEFAULT_CORPUS_PATHS if paths is None else paths
    for p in paths:
        full = os.path.join(repo_root, p)
        if not os.path.exists(full):
            continue
        try:
            with open(full) as f:
                data = json.load(f)
        except Exception:
            continue
        for rec in data.get("arrangements", []):
            yield p, rec


def build_reference_hashes(cells, out_path, paths=None, repo_root=".",
                           verbose=True):
    """WL-hash every corpus record in the given (n, (d1, d2)) cells.

    cells: iterable of (n, d1, d2).  Writes/returns
        {"n_d1_d2": {"records": R, "parse_errors": E,
                     "hashes": {hash: count}, "profiles": {profile: count}}}
    Coordinate-level dedup is applied first (the corpus has no lattice
    dedup); each distinct coordinate class is hashed once.
    """
    want = {(int(n), int(d1), int(d2)) for (n, d1, d2) in cells}
    seen_coords = set()
    out = {f"{n}_{d1}_{d2}": {"records": 0, "parse_errors": 0,
                              "hashes": {}, "profiles": {}}
           for (n, d1, d2) in want}
    for path, rec in iter_corpus_records(paths, repo_root):
        exps = rec.get("exponents")
        n = rec.get("n")
        if not exps or n is None:
            continue
        try:
            d1, d2 = int(exps[1]), int(exps[2])
        except Exception:
            continue
        key3 = (int(n), d1, d2)
        if key3 not in want:
            continue
        cell = out[f"{n}_{d1}_{d2}"]
        cell["records"] += 1
        ck = str(tuple(sorted(rec["lines"])))
        if ck in seen_coords:
            continue
        seen_coords.add(ck)
        try:
            arr = arrangement_from_record(rec)
            h = lattice_wl_hash(arr)
            prof = str(sorted(arr.multiplicities(), reverse=True))
        except Exception:
            cell["parse_errors"] += 1
            continue
        cell["hashes"][h] = cell["hashes"].get(h, 0) + 1
        cell["profiles"][prof] = cell["profiles"].get(prof, 0) + 1
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1)
    if verbose:
        for k, v in sorted(out.items()):
            print(f"  cell {k}: {v['records']} records -> "
                  f"{len(v['hashes'])} distinct lattices "
                  f"({len(v['profiles'])} profiles, "
                  f"{v['parse_errors']} parse errors)")
    return out
