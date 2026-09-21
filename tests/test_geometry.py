"""Special-position detector: conic coincidences detected exactly,
generic point sets rejected, degenerate/implied filtering, quadratic-field
path, and empty profiles on small fixtures."""

import os
import sys

import numpy as np
from sympy import Rational

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry import (points_on_conic, points_on_cubic,     # noqa: E402
                      special_position_profile, conic_is_degenerate,
                      exact_rank, veronese2)
from quadfield import QuadraticField                        # noqa: E402


def conic_pts(ts):
    """Points (1, t, t^2) on the conic xz = y^2."""
    return [(Rational(1), Rational(t), Rational(t) ** 2) for t in ts]


def test_six_points_on_conic_detected():
    pts = conic_pts([0, 1, -1, 2, -2, 3])
    found = points_on_conic(pts)
    assert len(found) == 1
    assert found[0]['size'] == 6
    assert found[0]['degenerate'] is False


def test_seven_points_report_is_maximal():
    pts = conic_pts([0, 1, -1, 2, -2, 3, 5])
    found = points_on_conic(pts)
    assert len(found) == 1
    assert found[0]['size'] == 7


def test_generic_six_points_rejected():
    pts = conic_pts([0, 1, -1, 2, -2]) + [(Rational(1), Rational(3),
                                           Rational(7))]
    # last point off the conic; no other conic through 6 of these
    found = points_on_conic(pts)
    assert found == []


def test_two_lines_degenerate_conic():
    # 3 points on x=0, 3 points on y=0: conic x*y, degenerate
    pts = [(Rational(0), Rational(1), Rational(t)) for t in (0, 1, 2)] + \
          [(Rational(1), Rational(0), Rational(t)) for t in (0, 1, 2)]
    found = points_on_conic(pts)
    assert len(found) == 1
    assert found[0]['degenerate'] is True


def test_implied_filter_flags_lattice_coverage():
    # same two-line configuration, with idx_sets saying the points lie on
    # arrangement lines 0 and 1 respectively
    pts = [(Rational(0), Rational(1), Rational(t)) for t in (0, 1, 2)] + \
          [(Rational(1), Rational(0), Rational(t)) for t in (0, 1, 2)]
    idx = [(0, 2, 3), (0, 4, 5), (0, 6, 7),
           (1, 2, 4), (1, 3, 6), (1, 5, 7)]
    found = points_on_conic(pts, idx_sets=idx)
    assert len(found) == 1
    assert found[0]['implied'] is True


def test_quadelem_conic_rank():
    K = QuadraticField(2)
    s = K.sqrt
    ts = [K.element(0), K.element(1), K.element(-1), s, -s, K.element(2)]
    pts = [(K.element(1), t, t * t) for t in ts]
    assert exact_rank([veronese2(p) for p in pts]) <= 5
    found = points_on_conic(pts)
    assert len(found) == 1
    assert found[0]['size'] == 6


def test_cubic_nine_points_no_false_positive():
    # 9 generic-ish points: always on SOME cubic; only a 2-dim cubic space
    # would be a coincidence
    rng = np.random.default_rng(7)
    pts = [(Rational(1), Rational(int(a)), Rational(int(b)))
           for a, b in rng.integers(-20, 20, size=(9, 2))]
    res = points_on_cubic(pts)
    assert res is None or res['cubic_space_dim'] >= 2


def test_profile_on_braid_empty(braid):
    prof = special_position_profile(braid)
    assert prof['n_multiple_points'] == 4        # braid A3: 4 triple points
    assert prof['conics'] == []                  # fewer than 6 points
    assert prof['cubic'] is None


def test_conic_degeneracy_predicate():
    # xz - y^2: smooth
    assert not conic_is_degenerate([Rational(0), Rational(-1), Rational(0),
                                    Rational(0), Rational(1), Rational(0)])
    # xy: two lines
    assert conic_is_degenerate([Rational(0), Rational(0), Rational(0),
                                Rational(1), Rational(0), Rational(0)])
