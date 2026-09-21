"""
geometry.py — special-position detection for the Terao-stress pipeline.

Given the multiple points (multiplicity >= 3) of a line arrangement, detect
algebraic coincidences NOT implied by the intersection lattice: k >= 6
points on a conic, k >= 10 points on a cubic (or 9 on a pencil of cubics).
These are the Ziegler-pair-style observables: two realizations of the same
lattice can differ exactly by such coincidences.

All computations are EXACT over the arrangement's field (Q or Q(sqrt d));
ranks over quadratic fields go through quadfield.k_rank (Weil restriction).
This module produces observables and sampling targets only — it makes no
freeness claims.
"""

from itertools import combinations

from sympy import Matrix, Rational

from arrangement import LineArrangement
from quadfield import QuadElem, k_rank, k_nullspace, scalar_field


# ─────────────────────────────────────────────────────────────────────────────
# Exact rank / nullspace over the point field
# ─────────────────────────────────────────────────────────────────────────────

def _field_of(rows):
    """QuadraticField of the entries, or None for rational rows."""
    vals = [v for row in rows for v in row]
    return scalar_field(vals)


def exact_rank(rows):
    K = _field_of(rows)
    if K is None:
        return Matrix([[Rational(v) for v in row] for row in rows]).rank()
    return k_rank(rows, K)


def exact_nullspace(rows):
    K = _field_of(rows)
    if K is None:
        return [list(v) for v in
                Matrix([[Rational(v) for v in row] for row in rows]
                       ).nullspace()], None
    return k_nullspace(rows, K), K


# ─────────────────────────────────────────────────────────────────────────────
# Veronese lifts
# ─────────────────────────────────────────────────────────────────────────────

def veronese2(p):
    """Degree-2 Veronese row (x^2, y^2, z^2, xy, xz, yz)."""
    x, y, z = p
    return [x * x, y * y, z * z, x * y, x * z, y * z]


def veronese3(p):
    """Degree-3 Veronese row (10 monomials, graded-lex in (x, y, z))."""
    x, y, z = p
    return [x**3, x*x*y, x*x*z, x*y*y, x*y*z, x*z*z,
            y**3, y*y*z, y*z*z, z**3]


def _conic_value(conic, p):
    """Evaluate a conic coefficient vector (Veronese-2 basis) at a point."""
    row = veronese2(p)
    return sum(c * r for c, r in zip(conic, row))


def conic_is_degenerate(conic):
    """Exact: det of the symmetric 3x3 matrix of the conic is zero."""
    a, b, c, d, e, f = conic          # x2 y2 z2 xy xz yz
    half = Rational(1, 2)
    M = [[a, half * d, half * e],
         [half * d, b, half * f],
         [half * e, half * f, c]]
    det = (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
           - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
           + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    # QuadElem collapse invariant: a zero result is a plain Rational(0),
    # so `det == 0` is exact over both fields.
    return not isinstance(det, QuadElem) and det == 0


# ─────────────────────────────────────────────────────────────────────────────
# Multiple points and lattice-implied coverage
# ─────────────────────────────────────────────────────────────────────────────

def multiple_points(arr: LineArrangement, min_mult=3):
    """[(exact point, sorted tuple of line indices)], deterministic order.

    The line-index tuple is the lattice-transportable identity of the
    point (usable across realizations via a lattice isomorphism)."""
    pts = [(p, tuple(sorted(idx)))
           for p, idx in arr.intersection_points().items()
           if len(idx) >= min_mult]
    pts.sort(key=lambda t: t[1])
    return pts


def _covered_by_two_arrangement_lines(idx_sets):
    """True if two arrangement lines i, j cover the whole point set (every
    point lies on line i or line j) — then the 'conic' i*j is lattice
    data, not a geometric coincidence."""
    all_lines = sorted({i for s in idx_sets for i in s})
    for i, j in combinations(all_lines, 2):
        if all((i in s) or (j in s) for s in idx_sets):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Coincidence detection
# ─────────────────────────────────────────────────────────────────────────────

def points_on_conic(points, idx_sets=None, min_size=6,
                    max_five_subsets=3000, rng=None):
    """Maximal subsets of >= min_size points lying on a common conic.

    points: exact projective points; idx_sets: optional matching list of
    line-index tuples (enables the lattice-implied filter).  5 points
    always lie on a conic, so only k >= 6 is a coincidence.

    Returns [{'point_ids': sorted positions, 'size', 'degenerate',
              'implied'}].  Enumerates 5-subsets (each determines its
    conic when unique); above `max_five_subsets` combinations a random
    sample of that many subsets is used (rng required) and the result is
    flagged 'exhaustive': False.
    """
    T = len(points)
    out = []
    if T < min_size:
        return out

    # global check first: all points on one conic
    if exact_rank([veronese2(p) for p in points]) <= 5:
        candidates = [tuple(range(T))]
        exhaustive = True
    else:
        from math import comb
        n_combos = comb(T, 5)
        exhaustive = n_combos <= max_five_subsets
        if exhaustive:
            combos = combinations(range(T), 5)
        else:
            assert rng is not None, "rng required for sampled subsets"
            combos = (tuple(sorted(int(i) for i in
                                   rng.choice(T, size=5, replace=False)))
                      for _ in range(max_five_subsets))
        candidates = []
        seen = set()
        for sub in combos:
            if sub in seen:
                continue
            seen.add(sub)
            rows = [veronese2(points[i]) for i in sub]
            null, _K = exact_nullspace(rows)
            if len(null) != 1:
                continue              # degenerate 5-subset: no unique conic
            conic = list(null[0])
            # QuadElem collapse invariant makes `== 0` exact over K too
            incident = tuple(i for i in range(T)
                             if _conic_value(conic, points[i]) == 0)
            if len(incident) >= min_size and incident not in seen:
                seen.add(incident)
                candidates.append(incident)

    # keep maximal sets only
    candidates = [set(c) for c in candidates]
    maximal = [c for c in candidates
               if not any(c < other for other in candidates)]
    reported = set()
    for c in maximal:
        key = tuple(sorted(c))
        if key in reported:
            continue
        reported.add(key)
        rows = [veronese2(points[i]) for i in key]
        null, _K = exact_nullspace(rows)
        conic = list(null[0]) if len(null) == 1 else None
        rec = {'point_ids': list(key), 'size': len(key),
               'exhaustive': exhaustive,
               'degenerate': (conic_is_degenerate(conic)
                              if conic is not None else None)}
        if idx_sets is not None:
            rec['implied'] = _covered_by_two_arrangement_lines(
                [idx_sets[i] for i in key])
        out.append(rec)
    out.sort(key=lambda r: (-r['size'], r['point_ids']))
    return out


def points_on_cubic(points):
    """Global cubic coincidence: k >= 10 points with Veronese-3 rank <= 9,
    or k == 9 with rank <= 8 (9 points ALWAYS lie on some cubic — only a
    pencil of cubics through 9 points is a coincidence).  Subset
    enumeration is out of scope in v1 (documented limitation)."""
    T = len(points)
    if T < 9:
        return None
    rank = exact_rank([veronese3(p) for p in points])
    if (T >= 10 and rank <= 9) or (T == 9 and rank <= 8):
        return {'size': T, 'veronese3_rank': int(rank),
                'cubic_space_dim': int(10 - rank)}
    return None


def special_position_profile(arr: LineArrangement, min_mult=3,
                             max_five_subsets=3000, rng=None):
    """Special-position observables of one realization.

    {'n_multiple_points', 'point_line_sets', 'conics', 'cubic'} — point
    identities are line-index tuples (lattice-transportable)."""
    pts = multiple_points(arr, min_mult=min_mult)
    points = [p for p, _ in pts]
    idx_sets = [s for _, s in pts]
    return {
        'n_multiple_points': len(points),
        'point_line_sets': [list(s) for s in idx_sets],
        'conics': points_on_conic(points, idx_sets=idx_sets,
                                  max_five_subsets=max_five_subsets,
                                  rng=rng),
        'cubic': points_on_cubic(points),
    }
