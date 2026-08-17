"""Property tests for quadfield.QuadElem against the sympy oracle."""

import os
import random
import sys

import pytest
import sympy as sp
from sympy import Rational

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quadfield import (QuadraticField, QuadElem, parse_quad_token,
                       block_matrix, k_rank, k_nullspace, split_parts,
                       SUPPORTED_DISCRIMINANTS)

RNG = random.Random(20260817)


def rand_rat():
    return Rational(RNG.randint(-9, 9), RNG.randint(1, 7))


def rand_elem(K):
    return K.element(rand_rat(), rand_rat())


def to_expr(v):
    return v.to_sympy() if isinstance(v, QuadElem) else sp.sympify(v)


def agree(ours, oracle):
    return sp.simplify(to_expr(ours) - oracle) == 0


@pytest.mark.parametrize("d", SUPPORTED_DISCRIMINANTS)
def test_arithmetic_matches_sympy_oracle(d):
    K = QuadraticField(d)
    for _ in range(400):
        u, v = rand_elem(K), rand_elem(K)
        eu, ev = to_expr(u), to_expr(v)
        assert agree(u + v, eu + ev)
        assert agree(u - v, eu - ev)
        assert agree(u * v, sp.expand(eu * ev))
        if not (isinstance(v, QuadElem) and v.norm() == 0) and v != 0:
            assert agree(u / v, sp.radsimp(eu / ev))
        r = rand_rat()
        assert agree(u * r, eu * r)
        assert agree(r * u, eu * r)
        assert agree(u + r, eu + r)
        assert agree(r - u, r - eu)
        if u != 0:
            assert agree(r / u, sp.radsimp(r / eu))


@pytest.mark.parametrize("d", SUPPORTED_DISCRIMINANTS)
def test_pow_norm_conj(d):
    K = QuadraticField(d)
    for _ in range(60):
        u = rand_elem(K)
        eu = to_expr(u)
        for n in (0, 1, 2, 3, 5):
            assert agree(u ** n, sp.expand(eu ** n))
        if isinstance(u, QuadElem):
            conj_oracle = u.a - u.b * sp.sqrt(d)
            assert agree(u.conjugate(), conj_oracle)
            assert u.norm() == sp.expand(eu * conj_oracle)
            assert agree(u * u.inverse(), sp.Integer(1))


def test_collapse_invariant_and_identity():
    K = QuadraticField(5)
    s = K.sqrt
    assert isinstance(s * s, Rational) and s * s == 5
    assert isinstance(s - s, Rational) and (s - s) == 0
    phi = K.element(Rational(1, 2), Rational(1, 2))       # golden ratio
    psi = K.element(Rational(-1, 2), Rational(1, 2))      # 1/phi
    assert phi * psi == 1                                  # collapses exactly
    assert phi.inverse() == psi
    assert hash(phi) == hash(K.element(Rational(1, 2), Rational(1, 2)))
    assert phi != Rational(1, 2)
    assert {phi: 1}[phi * psi * phi] == 1                  # canonical dict key
    # equal numbers from different routes are the same key: 2/(sqrt5-1) = phi
    assert 2 / (K.sqrt - 1) == phi


def test_mixed_field_rejected():
    a = QuadraticField(3).sqrt
    b = QuadraticField(5).sqrt
    with pytest.raises(ValueError):
        _ = a + b
    with pytest.raises(ValueError):
        _ = a * b
    assert a != b


@pytest.mark.parametrize("d", [2, 3, 5])
def test_real_sign_and_float(d):
    K = QuadraticField(d)
    import math
    for _ in range(200):
        u = rand_elem(K)
        if isinstance(u, QuadElem):
            f = float(u)
            assert abs(f - (float(u.a) + float(u.b) * math.sqrt(d))) < 1e-12
            if abs(f) > 1e-9:
                assert u.sign() == (1 if f > 0 else -1)


def test_complex_embedding():
    K = QuadraticField(-3)
    omega = K.element(Rational(-1, 2), Rational(1, 2))    # primitive cube root
    with pytest.raises(TypeError):
        float(omega)
    w = complex(omega)
    assert abs(w - complex(-0.5, 3 ** 0.5 / 2)) < 1e-12
    assert omega ** 3 == 1
    assert isinstance(omega ** 3, Rational)


@pytest.mark.parametrize("d", SUPPORTED_DISCRIMINANTS)
def test_repr_parse_roundtrip(d):
    K = QuadraticField(d)
    for _ in range(200):
        u = rand_elem(K)
        if not isinstance(u, QuadElem):
            continue
        token = repr(u)
        assert token.startswith("[") and token.endswith("]")
        back = parse_quad_token(token[1:-1], K)
        assert back == u
    with pytest.raises(ValueError):
        parse_quad_token("1+1s", None)


def test_block_nullspace_matches_domain_matrix_oracle():
    from sympy.polys.matrices import DomainMatrix
    for d in (3, 5, -1):
        K = QuadraticField(d)
        dom = sp.QQ.algebraic_field(sp.sqrt(d))
        for trial in range(6):
            nrows, ncols = RNG.randint(2, 4), RNG.randint(2, 5)
            rows = [[rand_elem(K) if RNG.random() < 0.7 else rand_rat()
                     for _ in range(ncols)] for _ in range(nrows)]
            # oracle: DomainMatrix over QQ(sqrt(d))
            dm = DomainMatrix([[dom.from_sympy(to_expr(v)) for v in row]
                               for row in rows], (nrows, ncols), dom)
            oracle_rank = dm.rank()
            assert k_rank(rows, K) == oracle_rank
            ns = k_nullspace(rows, K)
            assert len(ns) == ncols - oracle_rank
            # every returned vector annihilates M over K, exactly
            for vec in ns:
                for row in rows:
                    acc = sum((a * b for a, b in zip(row, vec)),
                              Rational(0))
                    assert acc == 0
            # K-linear independence of the basis
            if ns:
                assert k_rank(ns, K) == len(ns)


def test_split_parts():
    K = QuadraticField(3)
    assert split_parts(Rational(7, 2)) == (Rational(7, 2), 0)
    a, b = split_parts(K.element(1, 2))
    assert (a, b) == (1, 2)
