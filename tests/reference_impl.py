"""
Independent reference implementation of the penalized Saito functional.

Used by the test suite to cross-validate penalized_saito.py:

  * the residual here is computed from the DEFINITION — the BW-orthogonal
    projector onto alpha * S_{d-1} built from an exact rational Gram matrix —
    whereas the production code uses the restriction identity;
  * all quantities except the final beta-power and square roots are exact
    rationals (line normalization enters only through even powers), so the
    reference value is exact up to one mpmath high-precision power.

This doubles as the "higher precision" comparator for the float64 agreement
test (§6.14).
"""

import sympy as sp
from sympy import Rational, Matrix, symbols
import mpmath

from arrangement import LineArrangement

x, y, z = symbols('x y z')


def _monoms(d):
    return LineArrangement._monoms(d)


def _multinom(d, m):
    return Rational(sp.factorial(d) // (sp.factorial(m[0]) * sp.factorial(m[1])
                                        * sp.factorial(m[2])))


def bw_inner_poly(p1, p2, d):
    """Exact BW inner product of two degree-d sympy polynomials (real)."""
    P1 = sp.Poly(p1, x, y, z)
    P2 = sp.Poly(p2, x, y, z)
    total = Rational(0)
    for m in _monoms(d):
        c1 = P1.coeff_monomial(x**m[0] * y**m[1] * z**m[2])
        c2 = P2.coeff_monomial(x**m[0] * y**m[1] * z**m[2])
        total += Rational(c1) * Rational(c2) / _multinom(d, m)
    return total


def residual_sq_projector(arr, d, theta_coeff_vectors):
    """||L_{A,d} u||^2 by the projector definition, exactly over Q.

    theta_coeff_vectors: stacked rational monomial coefficient vector of
    u = (f, g, h) in the _monoms(d) order.  The lines are NOT pre-normalized;
    normalization enters as the exact factor 1/||alpha_i||^2 per line.

    Returns an exact Rational.
    """
    n = len(arr)
    monoms = _monoms(d)
    N = len(monoms)
    vec = [Rational(v) for v in theta_coeff_vectors]
    f = sum(vec[i] * x**m[0] * y**m[1] * z**m[2] for i, m in enumerate(monoms))
    g = sum(vec[N + i] * x**m[0] * y**m[1] * z**m[2]
            for i, m in enumerate(monoms))
    h = sum(vec[2 * N + i] * x**m[0] * y**m[1] * z**m[2]
            for i, m in enumerate(monoms))

    total = Rational(0)
    for line in arr.lines:
        a, b, c = line.coords
        alpha = a * x + b * y + c * z
        norm_sq = a * a + b * b + c * c
        val = sp.expand(a * f + b * g + c * h)     # theta(alpha), degree d
        # ||rho||^2 = ||val||^2 - ||Pi val||^2, projector onto alpha*S_{d-1}
        val_sq = bw_inner_poly(val, val, d)
        if d >= 1:
            basis = [sp.expand(alpha * x**m[0] * y**m[1] * z**m[2])
                     for m in _monoms(d - 1)]
            k = len(basis)
            G = Matrix(k, k, lambda i, j: bw_inner_poly(basis[i], basis[j], d))
            rhs = Matrix(k, 1, lambda i, _: bw_inner_poly(basis[i], val, d))
            sol = G.LUsolve(rhs)
            proj_sq = (rhs.T * sol)[0, 0]
        else:
            proj_sq = Rational(0)
        total += (val_sq - proj_sq) / norm_sq      # 1/||alpha||^2 from unit
    return sp.nsimplify(total / n)


def gamma_reference(arr, d1, d2, u_mono, v_mono, lam, beta, dps=60):
    """Reference Gamma at rational monomial-coefficient vectors (u, v).

    u_mono, v_mono: stacked rational monomial coefficients of the two
    derivations (NOT BW coordinates, NOT necessarily unit — Gamma is evaluated
    on the normalized vectors; the E_d normalization uses exact BW norms).

    Returns an mpmath.mpf computed at `dps` decimal digits.
    """
    n = len(arr)
    monoms1, monoms2 = _monoms(d1), _monoms(d2)
    N1, N2 = len(monoms1), len(monoms2)

    u = [Rational(t) for t in u_mono]
    v = [Rational(t) for t in v_mono]

    def comps(vec, monoms, N):
        f = sum(vec[i] * x**m[0] * y**m[1] * z**m[2]
                for i, m in enumerate(monoms))
        g = sum(vec[N + i] * x**m[0] * y**m[1] * z**m[2]
                for i, m in enumerate(monoms))
        h = sum(vec[2 * N + i] * x**m[0] * y**m[1] * z**m[2]
                for i, m in enumerate(monoms))
        return f, g, h

    f1, g1, h1 = comps(u, monoms1, N1)
    f2, g2, h2 = comps(v, monoms2, N2)

    # exact E_d norms (for unit normalization; enters through even powers)
    u_sq = (bw_inner_poly(f1, f1, d1) + bw_inner_poly(g1, g1, d1)
            + bw_inner_poly(h1, h1, d1))
    v_sq = (bw_inner_poly(f2, f2, d2) + bw_inner_poly(g2, g2, d2)
            + bw_inner_poly(h2, h2, d2))
    if u_sq == 0 or v_sq == 0:
        raise ValueError("zero derivation vector")

    B = sp.expand(Matrix([[x, f1, f2], [y, g1, g2], [z, h1, h2]]).det())
    Q = sp.expand(sp.prod(line.linear_form() for line in arr.lines))

    B_sq = bw_inner_poly(B, B, n)                  # ||B(u,v)||^2, unnormalized
    Q_sq = bw_inner_poly(Q, Q, n)
    BQ = bw_inner_poly(B, Q, n)

    R_u = residual_sq_projector(arr, d1, u)        # per unit ||alpha||, exact
    R_v = residual_sq_projector(arr, d2, v)

    def _mpf_rat(r):
        r = Rational(sp.nsimplify(r))
        return mpmath.mpf(int(r.p)) / mpmath.mpf(int(r.q))

    # normalize u, v to unit spheres: B is bilinear; residuals quadratic
    with mpmath.workdps(dps):
        u_sq_m = _mpf_rat(u_sq)
        v_sq_m = _mpf_rat(v_sq)
        num = _mpf_rat(BQ ** 2 / Q_sq) / (u_sq_m * v_sq_m)
        B_sq_m = _mpf_rat(B_sq) / (u_sq_m * v_sq_m)
        R = _mpf_rat(R_u) / u_sq_m + _mpf_rat(R_v) / v_sq_m
        den = B_sq_m + mpmath.mpf(lam) * R ** mpmath.mpf(beta)
        if den == 0:
            return mpmath.mpf(0)
        return num / den
