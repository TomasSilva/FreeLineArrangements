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
