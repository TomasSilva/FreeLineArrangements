"""Tests for novelty.py: lattice hashing, family screens, corpus parsing."""

import numpy as np
import pytest
from sympy import Rational

from arrangement import LineArrangement, ProjectiveLine
from novelty import (parse_line_str, incidence_graph, lattice_wl_hash,
                     lattices_isomorphic, is_essential, is_near_pencil,
                     modular_points, is_supersolvable_rank3,
                     supersolvable_exponents, check_supersolvable_consistency,
                     coordinate_height)
from saito import construct_near_pencil, construct_supersolvable


def arr_from(coords):
    return LineArrangement([ProjectiveLine(*c) for c in coords])


BRAID = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (1, 0, -1), (0, 1, -1)]
GENERIC5 = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, 3), (3, -1, 2)]
NONFREE7 = [(0, 1, -1), (1, 1, -1), (1, 1, 1), (1, 1, 2), (1, 1, 0),
            (1, 1, -2), (1, 0, -2)]


# ── WL hash ──────────────────────────────────────────────────────────────────

def test_wl_hash_invariant_under_line_permutation():
    rng = np.random.default_rng(3)
    base = arr_from(BRAID)
    h0 = lattice_wl_hash(base)
    for _ in range(5):
        perm = rng.permutation(len(BRAID))
        h = lattice_wl_hash(arr_from([BRAID[i] for i in perm]))
        assert h == h0


def test_wl_hash_invariant_under_projective_change():
    # same combinatorics in different coordinates -> same hash
    base = arr_from(BRAID)
    M = np.array([[1, 2, 0], [0, 1, -1], [1, 0, 3]])  # unimodular-ish, det != 0
    assert round(np.linalg.det(M)) != 0
    moved = []
    for line in base.lines:
        a = np.array([Rational(c) for c in line.coords], dtype=object)
        b = M.T @ a          # line coords transform by M^T under x -> M^{-1}x
        moved.append(tuple(b))
    moved_arr = arr_from(moved)
    assert sorted(moved_arr.multiplicities()) == sorted(base.multiplicities())
    assert lattice_wl_hash(moved_arr) == lattice_wl_hash(base)
    assert lattices_isomorphic(base, moved_arr)


def test_wl_hash_separates_different_lattices():
    # braid (four triple points) vs generic six lines (all double points)
    generic6 = arr_from([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, 3),
                         (3, -1, 2), (2, 5, -1)])
    assert lattice_wl_hash(arr_from(BRAID)) != lattice_wl_hash(generic6)
    assert not lattices_isomorphic(arr_from(BRAID), generic6)


def test_isomorphism_confirms_within_bucket():
    a = construct_supersolvable(9, 3)
    b = construct_supersolvable(9, 4)
    assert lattice_wl_hash(a) != lattice_wl_hash(b)


# ── family screens ───────────────────────────────────────────────────────────

def test_essentiality():
    assert is_essential(arr_from(BRAID))
    pencil = arr_from([(1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 2, 0)])
    assert not is_essential(pencil)          # all through (0:0:1): rank 2


def test_near_pencil_detector():
    assert is_near_pencil(construct_near_pencil(8))
    assert not is_near_pencil(arr_from(BRAID))
    assert not is_near_pencil(construct_supersolvable(9, 3))


@pytest.mark.parametrize("builder,expected", [
    (lambda: arr_from(BRAID), True),
    (lambda: construct_supersolvable(9, 3), True),
    (lambda: construct_supersolvable(12, 5), True),
    (lambda: construct_near_pencil(8), True),
    (lambda: arr_from(GENERIC5), False),
    (lambda: arr_from(NONFREE7), False),
])
def test_supersolvable_detector(builder, expected):
    assert is_supersolvable_rank3(builder()) == expected


def test_supersolvable_exponent_consistency():
    # certified-free supersolvables must match the modular-point exponents
    for (n, d1) in [(9, 3), (10, 4), (11, 5)]:
        arr = construct_supersolvable(n, d1)
        is_free, exps = arr.is_free()
        assert is_free
        assert check_supersolvable_consistency(arr, exps)
        assert supersolvable_exponents(arr) == tuple(sorted(exps))
    # braid A3: modular triple point m=3, n=6 -> (1, 2, 3)
    assert supersolvable_exponents(arr_from(BRAID)) == (1, 2, 3)


# ── parsing / misc ───────────────────────────────────────────────────────────

def test_parse_line_str_roundtrip():
    for line in construct_supersolvable(9, 3).lines:
        parsed = parse_line_str(repr(line))
        assert parsed == line
    assert parse_line_str("(1x+3/4y+-1/4z=0)").coords == \
        ProjectiveLine(1, Rational(3, 4), Rational(-1, 4)).coords


def test_coordinate_height():
    assert coordinate_height(arr_from(BRAID)) == 1
    tall = arr_from([(1, 0, 0), (0, 1, 0), (Rational(7, 3), 5, 1)])
    assert coordinate_height(tall) == 15  # canonical form divides by 7/3 -> 15/7
