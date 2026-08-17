"""
certificates.py

Exact symbolic Saito certificates over Q.

A freeness claim is only accepted with an exact certificate

    det M(theta_E, theta_1, theta_2) = c * Q(A),   c != 0,

with theta_1, theta_2 exact rational logarithmic derivations of degrees
(d1, d2).  This module extracts such certificates (sympy, exact over Q),
serializes them to JSON, and re-verifies serialized certificates from
scratch.  The numerical penalized loss (penalized_saito.py) is never part of
a certificate.
"""

import numpy as np
import sympy as sp
from sympy import Rational, Matrix, symbols

from arrangement import LineArrangement, ProjectiveLine

x, y, z = symbols('x y z')

__all__ = [
    "find_exact_saito_certificate",
    "find_certificate_fast",
    "free_module_dims",
    "modp_nullity_reject",
    "verify_certificate",
    "certificate_to_bw_vectors",
    "certificate_to_json",
    "certificate_from_json",
]


def _vec_to_components(vec, monoms):
    """Split a stacked coefficient vector into (f, g, h) sympy polynomials."""
    N = len(monoms)
    f = sum(vec[i] * x**ma * y**mb * z**mc
            for i, (ma, mb, mc) in enumerate(monoms))
    g = sum(vec[N + i] * x**ma * y**mb * z**mc
            for i, (ma, mb, mc) in enumerate(monoms))
    h = sum(vec[2 * N + i] * x**ma * y**mb * z**mc
            for i, (ma, mb, mc) in enumerate(monoms))
    return f, g, h


def find_exact_saito_certificate(arr: LineArrangement, target_exponents=None):
    """Find an exact Saito certificate for `arr`, or None if none exists.

    Searches theta_1, theta_2 among exact rational null-space basis vectors of
    the derivation matrices (this suffices: if the arrangement is free with
    exponents (1, d1, d2), some basis pair has det = c*Q with c != 0, because
    det is multilinear and vanishes on the theta_E- and lower-degree
    components of the null spaces).

    Returns a dict:
        {'d1', 'd2', 'c', 'theta1', 'theta2', 'Q', 'lines'}
    with theta* as stacked rational coefficient vectors (strings) in the
    monomial order of LineArrangement._monoms, or None.
    """
    if target_exponents is None:
        exps = arr.candidate_exponents()
        if exps is None:
            return None
        d1, d2 = exps
    else:
        d1, d2 = target_exponents
    n = len(arr)
    if d1 + d2 != n - 1 or d1 < 0:
        return None

    M1 = arr.derivation_matrix(d1)
    null1 = M1.nullspace()
    if not null1:
        return None
    if d1 == d2:
        null2 = null1
    else:
        M2 = arr.derivation_matrix(d2)
        null2 = M2.nullspace()
        if not null2:
            return None

    Q = sp.expand(sp.prod(line.linear_form() for line in arr.lines))
    monoms1 = LineArrangement._monoms(d1)
    monoms2 = LineArrangement._monoms(d2)

    for v1 in null1:
        f1, g1, h1 = _vec_to_components(list(v1), monoms1)
        for v2 in null2:
            if d1 == d2 and v1 == v2:
                continue
            f2, g2, h2 = _vec_to_components(list(v2), monoms2)
            det = sp.expand(Matrix([[x, f1, f2],
                                    [y, g1, g2],
                                    [z, h1, h2]]).det())
            ratio = sp.cancel(det / Q)
            if ratio.is_number and ratio != 0:
                return {
                    'd1': int(d1),
                    'd2': int(d2),
                    'c': ratio,
                    'theta1': [sp.nsimplify(t) for t in list(v1)],
                    'theta2': [sp.nsimplify(t) for t in list(v2)],
                    'Q': Q,
                    'lines': [line.coords for line in arr.lines],
                }
    return None


def _dim_S(k):
    """dim S_k = C(k+2, 2) for k >= 0, else 0."""
    return (k + 2) * (k + 1) // 2 if k >= 0 else 0


def free_module_dims(d, d1, d2):
    """dim D(A)_d for an arrangement free with exponents (1, d1, d2):
    D(A) = S(-1) + S(-d1) + S(-d2)."""
    return _dim_S(d - 1) + _dim_S(d - d1) + _dim_S(d - d2)


def _modp_rank(M, p):
    """Rank of an integer matrix over GF(p) (numpy Gaussian elimination).
    Entries must already be reduced representatives in [0, p)."""
    A = M.copy() % p
    rows, cols = A.shape
    rank = 0
    for c in range(cols):
        piv = None
        for r in range(rank, rows):
            if A[r, c] % p:
                piv = r
                break
        if piv is None:
            continue
        A[[rank, piv]] = A[[piv, rank]]
        inv = pow(int(A[rank, c]), p - 2, p)
        A[rank] = (A[rank] * inv) % p
        for r in range(rows):
            if r != rank and A[r, c]:
                A[r] = (A[r] - A[r, c] * A[rank]) % p
        rank += 1
        if rank == rows:
            break
    return rank


def modp_nullity_reject(arr: LineArrangement, d1: int, d2: int,
                        p: int = 1000003):
    """SOUND negative freeness test via GF(p) nullity.

    rank_p <= rank_Q, hence nullity_p >= nullity_Q.  If the GF(p) nullity in
    degree d is already BELOW the dimension a free module would require,
    the exact nullity is too, so the arrangement is not free with exponents
    (1, d1, d2).  Returns True when that sound rejection fires (checks d1;
    also d2 when different).  False means "no conclusion".
    Rows are scaled to integers first (row scaling preserves the kernel);
    a prime dividing any scaled denominator is skipped conservatively.
    """
    for d in ((d1,) if d1 == d2 else (d1, d2)):
        M = arr.derivation_matrix(d)
        rows, cols = M.shape
        Int = np.zeros((rows, cols), dtype=np.int64)
        ok = True
        for r in range(rows):
            dens = [sp.Rational(M[r, c]).q for c in range(cols)]
            L = int(sp.ilcm(*dens)) if dens else 1
            if L % p == 0:
                ok = False
                break
            for c in range(cols):
                v = sp.Rational(M[r, c]) * L
                Int[r, c] = int(v) % p
        if not ok:
            continue
        nullity_p = cols - _modp_rank(Int, p)
        if nullity_p < free_module_dims(d, d1, d2):
            return True
    return False


def _eval_components(vec, monoms, pt):
    """Evaluate the (f, g, h) components of a stacked coefficient vector at
    an exact rational point pt = (x0, y0, z0).  Returns three Rationals."""
    x0, y0, z0 = pt
    N = len(monoms)
    powers = [x0 ** ma * y0 ** mb * z0 ** mc for (ma, mb, mc) in monoms]
    f = sum(vec[i] * powers[i] for i in range(N))
    g = sum(vec[N + i] * powers[i] for i in range(N))
    h = sum(vec[2 * N + i] * powers[i] for i in range(N))
    return f, g, h


def find_certificate_fast(arr: LineArrangement, target_exponents=None,
                          prescreen_prime=1000003):
    """Fast exact Saito certification.

    Same guarantees as find_exact_saito_certificate — every positive is a
    fully verified symbolic certificate; the negative is EXACT — but the
    pair search is done by point evaluation instead of symbolic expansion:

    Since u, v are exact logarithmic derivations with deg u + deg v = n - 1,
    det M(theta_E, u, v) = c * Q identically.  Evaluating at one exact
    rational point pt with Q(pt) != 0 therefore determines c exactly:
    c = det(pt) / Q(pt).  A pair with c != 0 is then verified symbolically
    (the shipped certificate never rests on the shortcut); if every basis
    pair has c = 0 the arrangement is NOT free with these exponents (exact
    negative: if it were free, some basis pair would have c != 0).

    Returns (cert_dict, status) with status in
    {'certified', 'not_free_exact', 'modp_reject', 'no_exponents'}.
    """
    if target_exponents is None:
        exps = arr.candidate_exponents()
        if exps is None:
            return None, 'no_exponents'
        d1, d2 = exps
    else:
        d1, d2 = target_exponents
    n = len(arr)
    if d1 + d2 != n - 1 or d1 < 0:
        return None, 'no_exponents'

    if prescreen_prime and modp_nullity_reject(arr, d1, d2, prescreen_prime):
        return None, 'modp_reject'

    M1 = arr.derivation_matrix(d1)
    null1 = M1.nullspace()
    if len(null1) < free_module_dims(d1, d1, d2):
        return None, 'not_free_exact'
    if d1 == d2:
        null2 = null1
    else:
        null2 = arr.derivation_matrix(d2).nullspace()
        if len(null2) < free_module_dims(d2, d1, d2):
            return None, 'not_free_exact'

    monoms1 = LineArrangement._monoms(d1)
    monoms2 = LineArrangement._monoms(d2)

    # exact evaluation point with Q(pt) != 0
    Qpoly = sp.prod(line.linear_form() for line in arr.lines)
    pt = None
    for t in (2, 3, 5, 7, 11, 13, 17):
        cand = (Rational(1), Rational(t), Rational(t) ** 2)
        Qv = Qpoly.subs({sp.Symbol('x'): cand[0], sp.Symbol('y'): cand[1],
                         sp.Symbol('z'): cand[2]})
        if Qv != 0:
            pt, Qval = cand, Rational(Qv)
            break
    if pt is None:                      # pathological; fall back to slow path
        cert = find_exact_saito_certificate(arr, target_exponents=(d1, d2))
        return (cert, 'certified') if cert else (None, 'not_free_exact')

    evals1 = [_eval_components(list(v), monoms1, pt) for v in null1]
    evals2 = (evals1 if d1 == d2 else
              [_eval_components(list(v), monoms2, pt) for v in null2])
    ex, ey, ez = pt

    for i1, (f1, g1, h1) in enumerate(evals1):
        for i2, (f2, g2, h2) in enumerate(evals2):
            if d1 == d2 and i1 == i2:
                continue
            det_val = (ex * (g1 * h2 - g2 * h1)
                       - ey * (f1 * h2 - f2 * h1)
                       + ez * (f1 * g2 - f2 * g1))
            if det_val == 0:
                continue
            c = Rational(det_val) / Qval
            # symbolic confirmation of THIS pair only
            v1, v2 = null1[i1], null2[i2]
            ff1, gg1, hh1 = _vec_to_components(list(v1), monoms1)
            ff2, gg2, hh2 = _vec_to_components(list(v2), monoms2)
            det = sp.expand(Matrix([[x, ff1, ff2], [y, gg1, gg2],
                                    [z, hh1, hh2]]).det())
            if sp.simplify(det - c * sp.expand(Qpoly)) != 0:
                continue                 # should not happen; stay sound
            cert = {
                'd1': int(d1), 'd2': int(d2), 'c': c,
                'theta1': [sp.nsimplify(t_) for t_ in list(v1)],
                'theta2': [sp.nsimplify(t_) for t_ in list(v2)],
                'Q': sp.expand(Qpoly),
                'lines': [line.coords for line in arr.lines],
            }
            return cert, 'certified'
    return None, 'not_free_exact'


def verify_certificate(cert) -> bool:
    """Re-verify a certificate from scratch, exactly over Q.

    Checks (1) theta_1, theta_2 are logarithmic (alpha_i | theta(alpha_i) for
    every line), and (2) det M(theta_E, theta_1, theta_2) = c * Q with c != 0.
    """
    lines = [ProjectiveLine(*c) for c in cert['lines']]
    arr = LineArrangement(lines)
    d1, d2 = cert['d1'], cert['d2']
    monoms1 = LineArrangement._monoms(d1)
    monoms2 = LineArrangement._monoms(d2)
    th1 = [sp.nsimplify(t) for t in cert['theta1']]
    th2 = [sp.nsimplify(t) for t in cert['theta2']]
    f1, g1, h1 = _vec_to_components(th1, monoms1)
    f2, g2, h2 = _vec_to_components(th2, monoms2)

    # logarithmicity: alpha_i | theta(alpha_i) for every line and both thetas
    for line in arr.lines:
        a, b, c = line.coords
        alpha = a * x + b * y + c * z
        for (f, g, h) in ((f1, g1, h1), (f2, g2, h2)):
            val = sp.expand(a * f + b * g + c * h)
            if val == 0:
                continue
            _, r = sp.div(val, alpha, x, y, z)
            if sp.expand(r) != 0:
                return False

    Q = sp.expand(sp.prod(line.linear_form() for line in arr.lines))
    det = sp.expand(Matrix([[x, f1, f2], [y, g1, g2], [z, h1, h2]]).det())
    c_expected = sp.nsimplify(cert['c'])
    return sp.simplify(det - c_expected * Q) == 0 and c_expected != 0


def certificate_to_bw_vectors(cert):
    """Float BW-orthonormal unit coordinate vectors (u, v) of the certified
    pair, in the convention of penalized_saito.PenalizedSaitoEvaluator.

    NOTE: the evaluator normalizes each line to unit norm, which rescales Q
    but not the derivations; Gamma is invariant, so evaluating Gamma at these
    vectors on the same arrangement must give ~1.
    """
    from penalized_saito import _bw_sqrt_weights, _monoms

    out = []
    for key, d in (('theta1', cert['d1']), ('theta2', cert['d2'])):
        N = len(_monoms(d))
        c = np.array([float(sp.nsimplify(t)) for t in cert[key]],
                     dtype=np.float64)
        sw = _bw_sqrt_weights(d)
        w = np.concatenate([c[:N] / sw, c[N:2 * N] / sw, c[2 * N:] / sw])
        out.append(w / np.linalg.norm(w))
    return out[0], out[1]


def certificate_to_json(cert):
    """JSON-serializable form (all sympy objects -> strings)."""
    return {
        'd1': cert['d1'],
        'd2': cert['d2'],
        'c': str(cert['c']),
        'theta1': [str(t) for t in cert['theta1']],
        'theta2': [str(t) for t in cert['theta2']],
        'Q': str(cert['Q']),
        'lines': [[str(v) for v in coords] for coords in cert['lines']],
    }


def certificate_from_json(d):
    return {
        'd1': int(d['d1']),
        'd2': int(d['d2']),
        'c': sp.nsimplify(d['c']),
        'theta1': [sp.nsimplify(t) for t in d['theta1']],
        'theta2': [sp.nsimplify(t) for t in d['theta2']],
        'Q': sp.nsimplify(d['Q']),
        'lines': [tuple(Rational(v) for v in coords) for coords in d['lines']],
    }
