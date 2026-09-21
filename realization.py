"""
realization.py — lattice-preserving re-realization of line arrangements.

Terao-stress core: given a certified arrangement, produce OTHER exact
realizations of the SAME intersection lattice (verified by WL hash + VF2),
so freeness can be tested across the realization space.  Genericity is
enforced by exact lattice equality: a sampled arrangement with a missing
or EXTRA coincidence fails the lattice gate and is rejected.

All coordinates are exact (Rational or quadfield.QuadElem).  This module
makes no freeness claims.
"""

from fractions import Fraction

from sympy import Rational

from arrangement import LineArrangement, ProjectiveLine
from novelty import lattice_wl_hash, lattices_isomorphic, incidence_graph
from quadfield import QuadElem


# ─────────────────────────────────────────────────────────────────────────────
# Incidence structure (the combinatorial identity being re-realized)
# ─────────────────────────────────────────────────────────────────────────────

def incidence_structure(arr: LineArrangement):
    """{'n', 'points': [sorted line-index tuples of the m>=3 points],
    'wl': lattice hash}.  Purely combinatorial."""
    pts = sorted(tuple(sorted(idx))
                 for idx in arr.intersection_points().values()
                 if len(idx) >= 3)
    return {"n": len(arr), "points": pts, "wl": lattice_wl_hash(arr)}


def expected_moduli_dim(struct):
    """Naive expected dimension of the realization space mod PGL3:
    2n - sum over m>=3 points of (m - 2) - 8.  Heuristic only."""
    return (2 * struct["n"]
            - sum(len(p) - 2 for p in struct["points"]) - 8)


# ─────────────────────────────────────────────────────────────────────────────
# Exact 3x3 helpers (Rational and QuadElem entries)
# ─────────────────────────────────────────────────────────────────────────────

def _cross(u, v):
    return (u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0])


def _det3(a, b, c):
    x = _cross(b, c)
    return a[0] * x[0] + a[1] * x[1] + a[2] * x[2]


def _mat3_inv(M):
    """Exact inverse via adjugate; entries Rational/QuadElem."""
    a, b, c = M
    det = _det3(a, b, c)
    if det == 0 and not isinstance(det, QuadElem):
        raise ValueError("singular matrix")
    # adjugate rows = cross products of columns
    cols = list(zip(*M))
    adj_cols = [_cross(cols[1], cols[2]),
                _cross(cols[2], cols[0]),
                _cross(cols[0], cols[1])]
    inv_det = 1 / det if not isinstance(det, QuadElem) else det.inverse()
    # inverse = adj(M)/det, adj rows are adj_cols as rows of M^{-1}
    return [tuple(v * inv_det for v in row) for row in adj_cols]


def _matvec(M, v):
    return tuple(sum(M[i][j] * v[j] for j in range(3)) for i in range(3))


# ─────────────────────────────────────────────────────────────────────────────
# PGL3 frame normalization
# ─────────────────────────────────────────────────────────────────────────────

def choose_frame(struct):
    """4 line indices, no 3 of them through a common structure point (a
    projective frame in the dual plane, as far as the lattice can tell).
    Deterministic; returns None if none exists (never for essential
    arrangements of interest)."""
    from itertools import combinations
    n = struct["n"]
    pointsets = [set(p) for p in struct["points"]]
    for quad in combinations(range(n), 4):
        if any(len(ps & set(quad)) >= 3 for ps in pointsets):
            continue
        return list(quad)
    return None


def normalize_frame(arr: LineArrangement, frame):
    """Exact projective change of coordinates sending the 4 frame lines to
    x, y, z, x+y+z.  Line coords transform l -> N l; returns (new_arr, N).

    Requires the frame to be a genuine dual frame at the GEOMETRIC level
    (no 3 of the 4 concurrent, pairwise distinct)."""
    l0, l1, l2, l3 = [arr.lines[i].coords for i in frame]
    A = [l0, l1, l2]                 # columns of A in the derivation below
    A_rows = [tuple(A[j][i] for j in range(3)) for i in range(3)]
    Ainv = _mat3_inv(A_rows)
    c = _matvec(Ainv, l3)            # l3 = c0*l0 + c1*l1 + c2*l2
    if any(v == 0 and not isinstance(v, QuadElem) for v in c):
        raise ValueError("frame lines not in general position")
    # N = (A * diag(c))^{-1}: N l_i ~ e_i, N l3 = (1,1,1)
    AD_rows = [tuple(A_rows[i][j] * c[j] for j in range(3)) for i in range(3)]
    N = _mat3_inv(AD_rows)
    new_lines = [ProjectiveLine(*_matvec(N, l.coords)) for l in arr.lines]
    return LineArrangement(new_lines), N


# ─────────────────────────────────────────────────────────────────────────────
# Inductive lattice-preserving sampler
# ─────────────────────────────────────────────────────────────────────────────

FRAME_COORDS = [(Rational(1), Rational(0), Rational(0)),
                (Rational(0), Rational(1), Rational(0)),
                (Rational(0), Rational(0), Rational(1)),
                (Rational(1), Rational(1), Rational(1))]


def _rand_rational(rng, height):
    p = int(rng.integers(-height, height + 1))
    q = int(rng.integers(1, height + 1))
    return Rational(Fraction(p, q))


def _rand_scalar(rng, height, field):
    if field is None:
        return _rand_rational(rng, height)
    a = int(rng.integers(-height, height + 1))
    b = int(rng.integers(-height, height + 1))
    return field.element(a, b)


def _pencil_basis(p):
    """Two independent lines through the exact point p (reuses the kernel
    parametrization of arrangement.py)."""
    u, v = LineArrangement._ker_basis(*p)
    return u, v


def _determined_points(struct, line_idx, placed, point_coords):
    """Structure points containing line_idx whose coords are already
    determined (>= 2 placed lines)."""
    out = []
    for pi, pset in enumerate(struct["points"]):
        if line_idx in pset and point_coords[pi] is not None:
            out.append(pi)
    return out


def _update_determined(struct, placed_lines, point_coords):
    """(Re)compute coords of structure points with >= 2 placed lines."""
    for pi, pset in enumerate(struct["points"]):
        if point_coords[pi] is not None:
            continue
        got = [i for i in pset if placed_lines[i] is not None]
        if len(got) >= 2:
            la = placed_lines[got[0]]
            lb = placed_lines[got[1]]
            pt = la.intersect(lb)
            point_coords[pi] = pt


def sample_realization(reference: LineArrangement, rng, struct=None,
                       field=None, height_bound=12, max_attempts=60,
                       max_resamples=40):
    """One lattice-preserving re-realization of `reference`, or None.

    Inductive placement: the 4 frame lines are pinned to x, y, z, x+y+z
    (killing PGL3); each further line passes through its already-
    determined structure points, with remaining freedom sampled as exact
    bounded-height parameters; lines determined by >= 2 points must pass
    exact `passes_through` checks on the extras (closure).  Acceptance is
    exact lattice equality: WL hash match AND VF2 isomorphism with the
    reference — missing AND extra coincidences are both rejected.
    """
    if struct is None:
        struct = incidence_structure(reference)
    n = struct["n"]
    frame = choose_frame(struct)
    if frame is None:
        return None

    for _attempt in range(max_attempts):
        placed = [None] * n
        point_coords = [None] * len(struct["points"])
        for fi, coords in zip(frame, FRAME_COORDS):
            placed[fi] = ProjectiveLine(*coords)
        _update_determined(struct, placed, point_coords)

        order = [i for i in range(n) if i not in frame]
        ok = True
        while order and ok:
            # greedy: most-determined placeable line first (2 constraints
            # -> rigid; closure extras checked immediately)
            order.sort(key=lambda i: -len(
                _determined_points(struct, i, placed, point_coords)))
            i = order.pop(0)
            dpts = _determined_points(struct, i, placed, point_coords)
            success = False
            for _try in range(max_resamples):
                if len(dpts) >= 2:
                    line = ProjectiveLine.from_two_points(
                        point_coords[dpts[0]], point_coords[dpts[1]])
                    if line is None:
                        break
                elif len(dpts) == 1:
                    u, v = _pencil_basis(point_coords[dpts[0]])
                    t = _rand_scalar(rng, height_bound, field)
                    if rng.integers(2):
                        coords = tuple(u[k] + t * v[k] for k in range(3))
                    else:
                        coords = tuple(v[k] + t * u[k] for k in range(3))
                    if all(c == 0 and not isinstance(c, QuadElem)
                           for c in coords):
                        continue
                    line = ProjectiveLine(*coords)
                else:
                    coords = tuple(_rand_scalar(rng, height_bound, field)
                                   for _ in range(3))
                    if all(c == 0 and not isinstance(c, QuadElem)
                           for c in coords):
                        continue
                    line = ProjectiveLine(*coords)
                # closure: all determined constraint points must be hit
                if any(not line.passes_through(point_coords[pi])
                       for pi in dpts):
                    if len(dpts) >= 2:
                        break         # rigid contradiction: restart attempt
                    continue
                # local genericity: no unintended determined point hit,
                # no duplicate line
                unintended = any(
                    point_coords[pi] is not None and pi not in dpts
                    and line.passes_through(point_coords[pi])
                    for pi in range(len(struct["points"])))
                if unintended or any(
                        pl is not None and pl.coords == line.coords
                        for pl in placed):
                    if len(dpts) >= 2:
                        break         # rigid line violates genericity
                    continue
                placed[i] = line
                _update_determined(struct, placed, point_coords)
                success = True
                break
            if not success:
                ok = False
        if not ok:
            continue

        arr = LineArrangement(list(placed))
        if len({l.coords for l in arr.lines}) != n:
            continue
        if lattice_wl_hash(arr) != struct["wl"]:
            continue
        if not lattices_isomorphic(reference, arr):
            continue
        return arr
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Exact tangent-space dimension at a realization
# ─────────────────────────────────────────────────────────────────────────────

def tangent_space_dim(arr: LineArrangement, struct=None):
    """dim of the tangent space of the realization variety at `arr`,
    modulo per-line scaling and PGL3: 2n - 8 - rank(Jacobian).

    Constraints: for each m>=3 structure point with lines (a, b, c, ...),
    the (m - 2) concurrency determinants det(l_a, l_b, l_x) = 0.  The
    Jacobian w.r.t. all 3n homogeneous line coordinates is evaluated
    exactly at `arr` (grad_a det = l_b x l_x etc.).  Equals the true
    local dimension only at smooth points — a TANGENT BOUND, not a
    certified moduli dimension."""
    from geometry import exact_rank
    if struct is None:
        struct = incidence_structure(arr)
    n = struct["n"]
    zero = Rational(0)
    rows = []
    for pset in struct["points"]:
        idx = sorted(pset)
        a, b = idx[0], idx[1]
        la = arr.lines[a].coords
        lb = arr.lines[b].coords
        for x in idx[2:]:
            lx = arr.lines[x].coords
            row = [zero] * (3 * n)
            ga = _cross(lb, lx)
            gb = _cross(lx, la)
            gx = _cross(la, lb)
            for k in range(3):
                row[3 * a + k] = ga[k]
                row[3 * b + k] = gb[k]
                row[3 * x + k] = gx[k]
            rows.append(row)
    if not rows:
        return 2 * n - 8
    return 2 * n - 8 - int(exact_rank(rows))


# ─────────────────────────────────────────────────────────────────────────────
# Explicit lattice isomorphism (for transport and counterexample bundles)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Exact solving: rigid lattices and coincidence forcing
# ─────────────────────────────────────────────────────────────────────────────

def _classify_algebraic(val):
    """('rational', Rational) | ('quad', (a, b, d)) with val = a + b*sqrt(d),
    d in the supported set | ('other', minimal_poly_str)."""
    import sympy as sp
    val = sp.nsimplify(sp.radsimp(val), rational=False)
    if val.is_rational:
        return 'rational', Rational(val)
    try:
        t = sp.Symbol('_t')
        p = sp.minimal_poly(val, t)
    except Exception:
        return 'other', str(val)
    if p.as_poly(t).degree() != 2:
        return 'other', str(p)
    coeffs = p.as_poly(t).all_coeffs()          # [1, B, C] (monic)
    B, C = Rational(coeffs[1]), Rational(coeffs[2])
    disc = B * B - 4 * C
    if disc == 0:
        return 'rational', -B / 2
    # squarefree kernel of disc: disc = d * s^2 with d squarefree
    fac = sp.factorint(sp.Rational(disc).p * sp.Rational(disc).q)
    d = 1
    for prime, e in fac.items():
        if e % 2:
            d *= prime
    if sp.Rational(disc) < 0:
        d = -d
    if d not in (2, 3, 5, -1, -3):
        return 'other', str(p)
    s2 = sp.sqrt(sp.Rational(disc) / d)
    if not sp.Rational(s2 * s2) == sp.Rational(disc) / d:
        return 'other', str(p)
    b_abs = Rational(sp.nsimplify(s2)) / 2
    a = -B / 2
    # sign of b: compare numerically
    import math
    approx = complex(val.evalf())
    for b in (b_abs, -b_abs):
        cand = float(a) + float(b) * math.sqrt(abs(d)) * (1 if d > 0 else 0)
        if d > 0 and abs(cand - approx.real) < 1e-9:
            return 'quad', (a, b, d)
        if d < 0:
            cand_im = float(b) * math.sqrt(-d)
            if (abs(float(a) - approx.real) < 1e-9
                    and abs(cand_im - approx.imag) < 1e-9):
                return 'quad', (a, b, d)
    return 'other', str(p)


def _to_exact(val, field_cache):
    """sympy value -> Rational or QuadElem; raises on unsupported."""
    from quadfield import QuadraticField
    kind, data = _classify_algebraic(val)
    if kind == 'rational':
        return data
    if kind == 'quad':
        a, b, d = data
        K = field_cache.setdefault(d, QuadraticField(d))
        return K.element(a, b)
    raise ValueError(f"unsupported field: {data}")


def force_conic_realization(reference: LineArrangement, point_ids, rng,
                            struct=None, tries=40, height_bound=8):
    """Try to build a re-realization of `reference`'s lattice whose
    structure points `point_ids` (6 indices into struct['points']) lie on
    a common conic — the coincidence-FORCING move (Ziegler direction).

    Strategy 'sample-and-solve-last': run the inductive sampler but leave
    the LAST free parameter symbolic; impose the 6-point Veronese
    determinant; solve the univariate polynomial exactly; keep solutions
    over Q or a supported quadratic field; accept only realizations that
    pass the exact lattice gate.  Returns (arr, log) with arr=None on
    failure; log records higher-degree solutions honestly.
    """
    import sympy as sp
    from geometry import veronese2
    if struct is None:
        struct = incidence_structure(reference)
    n = struct["n"]
    frame = choose_frame(struct)
    if frame is None:
        return None, [{"status": "no_frame"}]
    target = list(point_ids)
    assert len(target) == 6
    log = []
    t_sym = sp.Symbol('t_last')
    field_cache = {}

    for _ in range(tries):
        placed = [None] * n
        point_coords = [None] * len(struct["points"])
        for fi, coords in zip(frame, FRAME_COORDS):
            placed[fi] = ProjectiveLine(*coords)
        _update_determined(struct, placed, point_coords)
        order = [i for i in range(n) if i not in frame]
        free_slots = []
        ok = True
        while order and ok:
            order.sort(key=lambda i: -len(
                _determined_points(struct, i, placed, point_coords)))
            i = order.pop(0)
            dpts = _determined_points(struct, i, placed, point_coords)
            placed_line = None
            for _try in range(30):
                if len(dpts) >= 2:
                    line = ProjectiveLine.from_two_points(
                        point_coords[dpts[0]], point_coords[dpts[1]])
                    if line is None or any(
                            not line.passes_through(point_coords[pi])
                            for pi in dpts):
                        line = None
                        break
                elif len(dpts) == 1:
                    u, v = _pencil_basis(point_coords[dpts[0]])
                    t = _rand_rational(rng, height_bound)
                    line = ProjectiveLine(*[u[k] + t * v[k]
                                            for k in range(3)])
                else:
                    line = ProjectiveLine(
                        *[_rand_rational(rng, height_bound)
                          for _ in range(3)])
                if line is None:
                    continue
                bad = any(
                    point_coords[pi] is not None and pi not in dpts
                    and line.passes_through(point_coords[pi])
                    for pi in range(len(struct["points"]))) or any(
                    pl is not None and pl.coords == line.coords
                    for pl in placed)
                if bad:
                    if len(dpts) >= 2:
                        break
                    continue
                placed_line = line
                if len(dpts) < 2:
                    free_slots.append(i)
                break
            if placed_line is None:
                ok = False
                break
            placed[i] = placed_line
            _update_determined(struct, placed, point_coords)
        if not ok or not free_slots:
            continue

        # symbolic re-do of ONE free line (the last sampled one): replace
        # its parameter with t_sym, recompute the 6 target points
        # symbolically, impose the Veronese det, solve.
        i_free = free_slots[-1]
        dpts = [pi for pi, pset in enumerate(struct["points"])
                if i_free in pset]
        # symbolic line through its (numeric) determined points
        det_dpts = [pi for pi in dpts if point_coords[pi] is not None
                    and all(placed[j] is not None
                            for j in list(struct["points"][pi])[:2])]
        base = placed[i_free].coords
        if len([pi for pi in dpts if point_coords[pi] is not None]) == 1:
            pin = next(pi for pi in dpts if point_coords[pi] is not None)
            u, v = _pencil_basis(point_coords[pin])
            sym_line = tuple(sp.Rational(u[k]) + t_sym * sp.Rational(v[k])
                             for k in range(3))
        else:
            sym_line = (sp.Rational(base[0]), sp.Rational(base[1]) + t_sym,
                        sp.Rational(base[2]))
        # symbolic coords of the 6 target points
        rows = []
        feasible = True
        for pi in target:
            pset = list(struct["points"][pi])
            others = [j for j in pset if j != i_free and placed[j] is not None]
            if i_free in pset:
                if not others:
                    feasible = False
                    break
                lo = placed[others[0]].coords
                pt = tuple(sp.expand(a) for a in (
                    sym_line[1] * sp.Rational(lo[2]) - sym_line[2] * sp.Rational(lo[1]),
                    sym_line[2] * sp.Rational(lo[0]) - sym_line[0] * sp.Rational(lo[2]),
                    sym_line[0] * sp.Rational(lo[1]) - sym_line[1] * sp.Rational(lo[0])))
            else:
                if point_coords[pi] is None:
                    feasible = False
                    break
                pt = tuple(sp.Rational(c) for c in point_coords[pi])
            rows.append([sp.expand(m) for m in veronese2(pt)])
        if not feasible:
            continue
        poly = sp.expand(sp.Matrix(rows).det())
        if poly == 0:
            continue                  # already identically on a conic
        try:
            roots = sp.solve(sp.Eq(poly, 0), t_sym)
        except Exception as e:        # noqa: BLE001
            log.append({"status": f"solve_error({e})"})
            continue
        for root in roots:
            # complex roots are fine iff they land in Q(i)/Q(sqrt-3):
            # _to_exact classifies and raises otherwise
            try:
                t_val = _to_exact(root, field_cache)
            except ValueError as e:
                log.append({"status": "higher_degree",
                            "root": str(root), "detail": str(e)})
                continue
            # rebuild the free line at the exact root
            if len([pi for pi in dpts if point_coords[pi] is not None]) == 1:
                pin = next(pi for pi in dpts
                           if point_coords[pi] is not None)
                u, v = _pencil_basis(point_coords[pin])
                new_line = ProjectiveLine(*[u[k] + t_val * v[k]
                                            for k in range(3)])
            else:
                new_line = ProjectiveLine(base[0], base[1] + t_val, base[2])
            cand = list(placed)
            cand[i_free] = new_line
            arr = LineArrangement(cand)
            if len({l.coords for l in arr.lines}) != n:
                continue
            if lattice_wl_hash(arr) != struct["wl"]:
                continue
            if not lattices_isomorphic(reference, arr):
                continue
            return arr, log
    return None, log


def lattice_isomorphism_map(a: LineArrangement, b: LineArrangement):
    """Explicit isomorphism of intersection lattices as
    {'lines': {i_a: i_b}, 'points': {point_key_a: point_key_b}} or None.
    Uses the same colored bipartite incidence graphs as
    novelty.lattices_isomorphic."""
    import networkx as nx
    from networkx.algorithms.isomorphism import (GraphMatcher,
                                                 categorical_node_match)
    Ga, Gb = incidence_graph(a), incidence_graph(b)
    gm = GraphMatcher(Ga, Gb,
                      node_match=categorical_node_match("label", ""))
    if not gm.is_isomorphic():
        return None
    lines, points = {}, {}
    for na, nb in gm.mapping.items():
        if na[0] == 'L':
            lines[int(na[1])] = int(nb[1])
        else:
            points[str(na[1])] = str(nb[1])
    return {"lines": lines, "points": points}
