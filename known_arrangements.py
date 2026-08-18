"""
known_arrangements.py

Exact fixtures for arrangements that require irrational coefficients —
the known small free-but-not-inductively-free examples.  These are both
test ground truths and campaign seeds.

* `akn13(lam)` — Abe–Kawanoue–Nozawa (arXiv:1406.5820): the smallest line
  arrangement that is free but NOT recursively free.  13 lines over
  Q(sqrt(3)), exponents (1, 6, 6), one rational parameter lam (generic).
  Lattice profile (generic lam): t2=21, t3=3, t4=3, t5=3, 30 points,
  b2 = 48 = (13-1) + 6*6.
* `dual_hesse()` — 9 lines over Q(sqrt(-3)) (the dual Hessian
  configuration): free with exponents (1, 4, 4), 12 triple points, not
  inductively free; not realizable over the reals.
"""

from sympy import Rational

from arrangement import LineArrangement, ProjectiveLine
from quadfield import QuadraticField

AKN13_PROFILE = {2: 21, 3: 3, 4: 3, 5: 3}
AKN13_N_POINTS = 30
AKN13_B2 = 48
DUAL_HESSE_PROFILE = {3: 12}


def akn13(lam=Rational(2, 3)):
    """The AKN 13-line arrangement over Q(sqrt(3)) at parameter `lam`.

    Cone over 12 affine lines plus the line at infinity z = 0
    (projectivized with z; equations as in arXiv:1406.5820).  `lam` must
    be a generic rational; validate with `validate_akn13_lattice`.
    """
    lam = Rational(lam)
    K = QuadraticField(3)
    s = K.sqrt                      # sqrt(3)
    one = Rational(1)
    lam2 = lam * lam
    lines = [
        (-s, -one, lam + 1),
        (0, 2, lam + 1),
        (s, -one, lam + 1),
        (s, -one, lam - 2),
        (-s, -one, lam - 2),
        (0, 2, lam - 2),
        (0, 2, -2 * lam + 1),
        (s, -one, -2 * lam + 1),
        (-s, -one, -2 * lam + 1),
        (s * (1 - lam), lam + 1, -lam2 + lam - 1),
        (s * lam, lam - 2, -lam2 + lam - 1),
        (-s, 1 - 2 * lam, -lam2 + lam - 1),
        (0, 0, 1),
    ]
    return LineArrangement([ProjectiveLine(*c) for c in lines])


def validate_akn13_lattice(arr):
    """True iff `arr` has the generic AKN A(13) combinatorics.

    Gates every lam before an akn13 instance is trusted as a seed or
    ground truth: 13 distinct lines, the exact multiplicity profile,
    b2 = 48 and candidate exponents (6, 6).
    """
    if len(arr) != 13 or len({l.coords for l in arr.lines}) != 13:
        return False
    mults = arr.multiplicities()
    profile = {}
    for m in mults:
        profile[m] = profile.get(m, 0) + 1
    if profile != AKN13_PROFILE:
        return False
    if arr.n_intersection_points() != AKN13_N_POINTS:
        return False
    if arr.b2() != AKN13_B2:
        return False
    return arr.candidate_exponents() == (6, 6)


def dual_hesse():
    """The dual Hesse arrangement: 9 lines over Q(sqrt(-3)).

    Lines x - w^j*y, y - w^j*z, x - w^j*z for j = 0, 1, 2 with w a
    primitive cube root of unity; 12 triple points, exponents (1, 4, 4).
    """
    K = QuadraticField(-3)
    w = K.element(Rational(-1, 2), Rational(1, 2))     # (-1 + sqrt(-3))/2
    powers = [Rational(1), w, w * w]
    lines = []
    for wj in powers:
        lines.append(ProjectiveLine(1, -wj, 0))
    for wj in powers:
        lines.append(ProjectiveLine(0, 1, -wj))
    for wj in powers:
        lines.append(ProjectiveLine(1, 0, -wj))
    return LineArrangement(lines)


def validate_dual_hesse_lattice(arr):
    """True iff `arr` has the dual Hesse combinatorics (12 triple points)."""
    if len(arr) != 9 or len({l.coords for l in arr.lines}) != 9:
        return False
    mults = arr.multiplicities()
    profile = {}
    for m in mults:
        profile[m] = profile.get(m, 0) + 1
    return (profile == DUAL_HESSE_PROFILE and arr.b2() == 24
            and arr.candidate_exponents() == (4, 4))


# ─── Reflection / CEVA family fixtures (overnight K-campaign seeds) ─────────
#
# G(r,r,3):  x - w^k y, y - w^k z, x - w^k z  (k = 0..r-1), w primitive r-th
#            root of unity — 3r lines, exponents (1, r+1, 2r-2).
# G(r,1,3):  the above plus the coordinate lines x, y, z — 3r + 3 lines,
#            exponents (1, r+1, 2r+1).
# Quadratic-field cases: r = 3, 6 need w in Q(sqrt(-3)); r = 4 needs Q(i).
# H3: the icosahedral arrangement, 15 lines over Q(sqrt(5)).

def _profile(arr):
    p = {}
    for m in arr.multiplicities():
        p[m] = p.get(m, 0) + 1
    return p


def _roots_of_unity(r):
    """Powers of a primitive r-th root of unity as exact field elements."""
    if r == 3:
        w = QuadraticField(-3).element(Rational(-1, 2), Rational(1, 2))
    elif r == 4:
        w = QuadraticField(-1).element(0, 1)             # i
    elif r == 6:
        w = QuadraticField(-3).element(Rational(1, 2), Rational(1, 2))
    else:
        raise ValueError(f"no quadratic field contains a primitive "
                         f"{r}-th root of unity")
    pw = [Rational(1)]
    for _ in range(r - 1):
        pw.append(pw[-1] * w)
    return pw


def _monomial_lines(r, full=False):
    pw = _roots_of_unity(r)
    lines = ([ProjectiveLine(1, -p, 0) for p in pw]
             + [ProjectiveLine(0, 1, -p) for p in pw]
             + [ProjectiveLine(1, 0, -p) for p in pw])
    if full:
        lines = [ProjectiveLine(1, 0, 0), ProjectiveLine(0, 1, 0),
                 ProjectiveLine(0, 0, 1)] + lines
    return LineArrangement(lines)


def ceva6():
    """Ceva(6) = G(6,6,3): 18 lines over Q(sqrt(-3)), exponents (1,7,10)."""
    return _monomial_lines(6)


def hesse12():
    """Extended Hesse = G(3,1,3): 12 lines over Q(sqrt(-3)), exps (1,4,7)."""
    return _monomial_lines(3, full=True)


def g443():
    """G(4,4,3): 12 lines over Q(i), exponents (1,5,6)."""
    return _monomial_lines(4)


def g413():
    """G(4,1,3): 15 lines over Q(i), exponents (1,5,9)."""
    return _monomial_lines(4, full=True)


def h3_15():
    """The icosahedral H3 arrangement: 15 lines over Q(sqrt(5)),
    exponents (1,5,9); 3 coordinate lines + cyclic (1, ±phi, ±1/phi)."""
    import itertools
    K = QuadraticField(5)
    phi = K.element(Rational(1, 2), Rational(1, 2))
    ip = phi - 1                                        # 1/phi
    lines = [ProjectiveLine(1, 0, 0), ProjectiveLine(0, 1, 0),
             ProjectiveLine(0, 0, 1)]
    seen = {l.coords for l in lines}
    for (s1, s2) in itertools.product((1, -1), (1, -1)):
        for (a, b, c) in [(1, s1 * phi, s2 * ip),
                          (s1 * phi, s2 * ip, 1),
                          (s2 * ip, 1, s1 * phi)]:
            L = ProjectiveLine(a, b, c)
            if L.coords not in seen:
                seen.add(L.coords)
                lines.append(L)
    return LineArrangement(lines)


# frozen combinatorial gates: (n, profile, b2, exponents) — computed
# exactly on 2026-08-18; all candidate exponents match the reflection-
# arrangement theory values.
FIXTURE_GATES = {
    "ceva6":   (18, {3: 36, 6: 3}, 87, (7, 10)),
    "hesse12": (12, {2: 9, 3: 9, 5: 3}, 39, (4, 7)),
    "g443":    (12, {3: 16, 4: 3}, 41, (5, 6)),
    "g413":    (15, {2: 12, 3: 16, 6: 3}, 59, (5, 9)),
    "h3_15":   (15, {2: 15, 3: 10, 5: 6}, 59, (5, 9)),
}


def validate_fixture(name, arr):
    """Frozen combinatorial gate for a named fixture."""
    n, prof, b2, exps = FIXTURE_GATES[name]
    if len(arr) != n or len({l.coords for l in arr.lines}) != n:
        return False
    return (_profile(arr) == prof and arr.b2() == b2
            and arr.candidate_exponents() == exps)


# (field_d, n) -> list of validated fixture factories, used by the
# K-campaign seed builder
FIELD_SEED_REGISTRY = {
    (3, 13): [akn13],
    (-3, 9): [dual_hesse],
    (-3, 18): [ceva6],
    (-3, 12): [hesse12],
    (-1, 12): [g443],
    (-1, 15): [g413],
    (5, 15): [h3_15],
}
