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

WHY THE NEGATIVE VERDICT IS ALSO EXACT (completeness of the pair search).
"Not certified" here is a PROOF of non-freeness with the prescribed
exponents, not a mere search failure.  Let {v_i}, {w_j} be exact bases of
the degree-(d1, d2) logarithmic kernels.  For u = sum a_i v_i and
v = sum b_j w_j, every det M(theta_E, u, v) equals c(a, b) * Q by the
divisibility theorem (deg = n forces proportionality), and c(a, b) =
a^T C b is BILINEAR with C_ij = c(v_i, w_j).  Hence: if every basis pair
has c(v_i, w_j) = 0, then C = 0, so EVERY pair of derivations of these
degrees has determinant identically zero, and no Saito basis with these
degrees exists — the arrangement is not free with exponents (1, d1, d2).
Conversely a free arrangement always has some basis pair with c != 0.
The fast path decides each c(v_i, w_j) exactly by evaluating both sides at
one exact rational point where Q does not vanish (the identity det = cQ
holds as polynomials, so a single non-vanishing point determines c).
`arrangement.is_free` implements the same exhaustive basis-pair criterion
symbolically; both negatives are exact.  Additionally, if the candidate-
exponent arithmetic fails (chi(A, t) does not factor with the required
integer roots), Terao's factorization theorem already excludes freeness.
"""

import numpy as np
import sympy as sp
from sympy import Rational, Matrix, symbols

from arrangement import LineArrangement, ProjectiveLine
from quadfield import (QuadElem, QuadraticField, k_nullspace,
                       block_matrix as _qf_block_matrix, parse_quad_token)

x, y, z = symbols('x y z')


def _field_tag(K):
    """Certificate 'field' entry: 'QQ' or the structured quadratic tag."""
    return 'QQ' if K is None else K.to_json()


def _null_basis(arr, d, K):
    """Exact kernel basis of the degree-d derivation matrix over QQ or K.

    QQ: sympy Matrix.nullspace() (unchanged).  K: Weil-restriction
    nullspace; vectors are lists of Rational/QuadElem.
    """
    if K is None:
        return arr.derivation_matrix(d).nullspace()
    return k_nullspace(arr._derivation_rows(d), K)


def _scalar_to_sympy(v):
    return v.to_sympy() if isinstance(v, QuadElem) else v


def _verify_scalar(t):
    """Exact sympy form of a certificate scalar.

    NEVER nsimplify an already-exact value: nsimplify is a heuristic
    float-matcher and can return a DIFFERENT number (e.g. it maps the
    exact Rational 93769/301320 to 5*2**(16/99)*3**(17/198)*5**(7/22)*
    7**(65/66)/224).  QuadElem converts exactly; exact sympy numbers pass
    through; only non-sympy data (legacy strings/ints) goes through
    nsimplify, where sympification of the string is exact.
    """
    if isinstance(t, QuadElem):
        return t.to_sympy()
    if getattr(t, 'is_Number', False):
        return t
    return sp.nsimplify(t)

__all__ = [
    "find_exact_saito_certificate",
    "find_certificate_fast",
    "classify_freeness",
    "free_module_dims",
    "modp_nullity_reject",
    "verify_certificate",
    "certificate_to_bw_vectors",
    "certificate_to_json",
    "certificate_from_json",
    "FREE_TARGET", "NOT_TARGET_FREE", "GLOBALLY_NONFREE", "UNRESOLVED",
    "NUMERICAL_ERROR",
]

# Structured freeness statuses.  None is never used to encode an outcome.
FREE_TARGET = "FREE_TARGET"            # exact certificate for the pair
NOT_TARGET_FREE = "NOT_TARGET_FREE"    # C = 0 for the PRESCRIBED pair only:
#   proves no Saito basis with these degrees exists.  It does NOT prove
#   global nonfreeness — a free arrangement at a wrong pair also has C = 0.
GLOBALLY_NONFREE = "GLOBALLY_NONFREE"  # exact two-branch argument (below)
UNRESOLVED = "UNRESOLVED"              # timeout / incomplete computation
NUMERICAL_ERROR = "NUMERICAL_ERROR"    # evaluation failure; never a proof


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

    K = arr.coefficient_field()
    null1 = _null_basis(arr, d1, K)
    if not null1:
        return None
    if d1 == d2:
        null2 = null1
    else:
        null2 = _null_basis(arr, d2, K)
        if not null2:
            return None

    Q = sp.expand(sp.prod(line.linear_form() for line in arr.lines))
    monoms1 = LineArrangement._monoms(d1)
    monoms2 = LineArrangement._monoms(d2)

    for v1 in null1:
        f1, g1, h1 = _vec_to_components(list(v1), monoms1)
        for v2 in null2:
            if d1 == d2 and list(v1) == list(v2):
                continue
            f2, g2, h2 = _vec_to_components(list(v2), monoms2)
            det = sp.expand(Matrix([[x, f1, f2],
                                    [y, g1, g2],
                                    [z, h1, h2]]).det())
            ratio = sp.cancel(det / Q)
            if ratio.is_number and ratio != 0:
                if K is None:
                    th1 = [_verify_scalar(t) for t in list(v1)]
                    th2 = [_verify_scalar(t) for t in list(v2)]
                else:
                    th1, th2 = list(v1), list(v2)   # exact K scalars
                return {
                    'd1': int(d1),
                    'd2': int(d2),
                    'c': ratio,
                    'theta1': th1,
                    'theta2': th2,
                    'Q': Q,
                    'lines': [line.coords for line in arr.lines],
                    'field': _field_tag(K),
                    'normalization': 'projective_first_nonzero_one',
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


def _modp_nullity(M, p):
    """GF(p) nullity of a RATIONAL sympy matrix, or None on a bad prime.

    Rows are scaled to integers first (row scaling preserves the kernel);
    a prime dividing any scaled denominator skips the computation
    conservatively (None = no conclusion).
    """
    rows, cols = M.shape
    Int = np.zeros((rows, cols), dtype=np.int64)
    for r in range(rows):
        dens = [sp.Rational(M[r, c]).q for c in range(cols)]
        L = int(sp.ilcm(*dens)) if dens else 1
        if L % p == 0:
            return None
        for c in range(cols):
            v = sp.Rational(M[r, c]) * L
            Int[r, c] = int(v) % p
    return cols - _modp_rank(Int, p)


def modp_nullity_reject(arr: LineArrangement, d1: int, d2: int,
                        p: int = 1000003):
    """SOUND negative freeness test via GF(p) nullity.

    rank_p <= rank_Q, hence nullity_p >= nullity_Q.  If the GF(p) nullity in
    degree d is already BELOW the dimension a free module would require,
    the exact nullity is too, so the arrangement is not free with exponents
    (1, d1, d2).  Returns True when that sound rejection fires (checks d1;
    also d2 when different).  False means "no conclusion".

    Quadratic fields: the same rejection runs on the RATIONAL Weil-
    restriction block B = [[M0, d*M1], [M1, M0]] with the doubled bound
    nullity_p(B) < 2 * free_dim, since nullity_Q(B) = 2 * nullity_K(M).
    No split-prime number theory needed; soundness is inherited from the
    rational case.
    """
    K = arr.coefficient_field()
    for d in ((d1,) if d1 == d2 else (d1, d2)):
        if K is None:
            M = arr.derivation_matrix(d)
            bound = free_module_dims(d, d1, d2)
        else:
            M = _qf_block_matrix(arr._derivation_rows(d), K)
            bound = 2 * free_module_dims(d, d1, d2)
        nullity_p = _modp_nullity(M, p)
        if nullity_p is None:
            continue
        if nullity_p < bound:
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
    {'certified', 'not_target_free', 'modp_reject', 'no_exponents'}.
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

    K = arr.coefficient_field()
    null1 = _null_basis(arr, d1, K)
    if len(null1) < free_module_dims(d1, d1, d2):
        return None, 'not_target_free'
    if d1 == d2:
        null2 = null1
    else:
        null2 = _null_basis(arr, d2, K)
        if len(null2) < free_module_dims(d2, d1, d2):
            return None, 'not_target_free'

    monoms1 = LineArrangement._monoms(d1)
    monoms2 = LineArrangement._monoms(d2)

    # exact evaluation point with Q(pt) != 0.  Q(pt) is evaluated by the
    # exact product of the line forms at pt (stays inside K; c = det/Q in K).
    Qpoly = sp.prod(line.linear_form() for line in arr.lines)
    pt = None
    for t in (2, 3, 5, 7, 11, 13, 17):
        cand = (Rational(1), Rational(t), Rational(t) ** 2)
        Qv = Rational(1)
        for line in arr.lines:
            a_, b_, c_ = line.coords
            Qv = Qv * (a_ * cand[0] + b_ * cand[1] + c_ * cand[2])
        if Qv != 0:
            pt, Qval = cand, Qv
            break
    if pt is None:                      # pathological; fall back to slow path
        cert = find_exact_saito_certificate(arr, target_exponents=(d1, d2))
        return (cert, 'certified') if cert else (None, 'not_target_free')

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
            c = det_val / Qval          # exact in K (Rational over QQ)
            # symbolic confirmation of THIS pair only
            v1, v2 = null1[i1], null2[i2]
            ff1, gg1, hh1 = _vec_to_components(list(v1), monoms1)
            ff2, gg2, hh2 = _vec_to_components(list(v2), monoms2)
            det = sp.expand(Matrix([[x, ff1, ff2], [y, gg1, gg2],
                                    [z, hh1, hh2]]).det())
            if sp.simplify(det - _scalar_to_sympy(c)
                           * sp.expand(Qpoly)) != 0:
                continue                 # should not happen; stay sound
            if K is None:
                th1 = [_verify_scalar(t_) for t_ in list(v1)]
                th2 = [_verify_scalar(t_) for t_ in list(v2)]
            else:
                th1, th2 = list(v1), list(v2)   # exact K scalars
            cert = {
                'd1': int(d1), 'd2': int(d2), 'c': c,
                'theta1': th1,
                'theta2': th2,
                'Q': sp.expand(Qpoly),
                'lines': [line.coords for line in arr.lines],
                'field': _field_tag(K),
                'normalization': 'projective_first_nonzero_one',
            }
            return cert, 'certified'
    return None, 'not_target_free'


def classify_freeness(arr: LineArrangement, target_pair=None):
    """Structured exact freeness classification.

    Returns {'status', 'evidence', 'certificate', 'candidate_pair'} with
    status in {FREE_TARGET, NOT_TARGET_FREE, GLOBALLY_NONFREE, UNRESOLVED}.

    With a target pair: FREE_TARGET (exact certificate) or NOT_TARGET_FREE
    (exact C = 0 for that pair — which does NOT prove global nonfreeness).

    Without a target pair, the exact two-branch global argument:
      (1) the characteristic polynomial chi(A, t)/(t-1) is computed exactly
          from the intersection lattice; if it does not factor as
          (t - e1)(t - e2) with admissible nonnegative integers, the
          arrangement is GLOBALLY_NONFREE by the contrapositive of Terao's
          factorization theorem;
      (2) if it factors, the exponents of any free structure are FORCED to
          be the unique unordered candidate pair (e1, e2); the exact
          determinant-pair matrix C for that pair then decides:
          C != 0 -> FREE_TARGET (for the candidate pair);
          C == 0 -> GLOBALLY_NONFREE.
    Any exception/timeout yields UNRESOLVED — never treated as a proof.

    Point-evaluation preconditions (all enforced by construction in
    find_certificate_fast): v_i, w_j are exact null-space vectors, hence
    exactly logarithmic; d1 + d2 = n - 1; the divisibility theorem gives
    B(v_i, w_j) = c Q identically; the evaluation point p is exact rational
    with Q(p) != 0; all arithmetic is exact over the declared field (QQ).
    """
    try:
        if target_pair is not None:
            cert, status = find_certificate_fast(arr,
                                                 target_exponents=target_pair)
            if status == 'certified':
                return {'status': FREE_TARGET, 'certificate': cert,
                        'candidate_pair': tuple(target_pair),
                        'evidence': 'exact_saito_certificate'}
            if status in ('not_target_free', 'modp_reject'):
                return {'status': NOT_TARGET_FREE, 'certificate': None,
                        'candidate_pair': tuple(target_pair),
                        'evidence': f'exact_pair_matrix_zero({status})'}
            return {'status': UNRESOLVED, 'certificate': None,
                    'candidate_pair': tuple(target_pair),
                    'evidence': f'inadmissible_or_incomplete({status})'}
        exps = arr.candidate_exponents()
        if exps is None:
            return {'status': GLOBALLY_NONFREE, 'certificate': None,
                    'candidate_pair': None,
                    'evidence': 'terao_factorization_obstruction'}
        cert, status = find_certificate_fast(arr, target_exponents=exps)
        if status == 'certified':
            return {'status': FREE_TARGET, 'certificate': cert,
                    'candidate_pair': tuple(exps),
                    'evidence': 'exact_saito_certificate'}
        if status in ('not_target_free', 'modp_reject'):
            return {'status': GLOBALLY_NONFREE, 'certificate': None,
                    'candidate_pair': tuple(exps),
                    'evidence': ('exponents_forced_by_terao_and_'
                                 f'pair_matrix_zero({status})')}
        return {'status': UNRESOLVED, 'certificate': None,
                'candidate_pair': tuple(exps),
                'evidence': f'incomplete({status})'}
    except Exception as e:            # noqa: BLE001 — never a nonfree proof
        return {'status': UNRESOLVED, 'certificate': None,
                'candidate_pair': None, 'evidence': f'exception({e})'}


def _is_exact_number(v):
    """Exact data only: int, str, sympy Rational/Integer, Fraction, or a
    quadfield.QuadElem.  Floats (silent rationalization) are rejected."""
    from fractions import Fraction
    if isinstance(v, float) or isinstance(v, np.floating):
        return False
    if isinstance(v, complex) or isinstance(v, np.complexfloating):
        return False
    if isinstance(v, QuadElem):
        return True
    if isinstance(v, (int, str, Fraction)):
        return True
    return getattr(v, 'is_Rational', False) is True


def _cert_field(cert):
    """QuadraticField declared by a certificate, or None for QQ."""
    return QuadraticField.from_json(cert.get('field', 'QQ'))


def _field_consistent(cert, K):
    """All coefficient scalars lie in the declared field."""
    scalars = list(cert['theta1']) + list(cert['theta2']) + [cert['c']]
    for coords in cert['lines']:
        scalars.extend(coords)
    for v in scalars:
        if isinstance(v, QuadElem):
            if K is None or v.field.d != K.d:
                return False
        elif isinstance(v, str) and '[' in v:
            return False       # unparsed quadratic token reached the verifier
    return True


def verify_certificate(cert) -> bool:
    """Re-verify a certificate from scratch, exactly over Q.

    A valid certificate requires ALL of:
      (0) every line triple exact (no floats) and nonzero; exactly n
          pairwise non-proportional projective lines (n recomputed from the
          list); d1 <= d2 nonnegative integers with d1 + d2 = n - 1;
      (1) theta_1, theta_2 given by exact stacked coefficient vectors of the
          exact lengths 3*dim S_{d1} / 3*dim S_{d2} (hence homogeneous of
          exactly the stated degrees when nonzero), each nonzero, and
          logarithmic: alpha_i | theta_j(alpha_i) exactly for every line;
      (2) Q recomputed from the supplied exact lines and, coefficientwise,
          det M(theta_E, theta_1, theta_2) = c * Q with exact c != 0.
    Field/normalization provenance is recorded on new certificates
    ('field': 'QQ'; projective normalization by first nonzero coordinate).
    """
    try:
        K = _cert_field(cert)
        for coords in cert['lines']:
            if len(coords) != 3 or not all(_is_exact_number(v)
                                           for v in coords):
                return False              # floats / malformed line data
        for key in ('theta1', 'theta2'):
            if not all(_is_exact_number(v) for v in cert[key]):
                return False
        if not _is_exact_number(cert['c']):
            return False
        if not _field_consistent(cert, K):
            return False                  # scalars outside the declared field
        lines = [ProjectiveLine(*c) for c in cert['lines']]
    except (AssertionError, TypeError, ValueError, KeyError):
        return False                      # zero line / malformed input
    arr = LineArrangement(lines)
    try:
        arr_K = arr.coefficient_field()
    except ValueError:
        return False                      # mixed fields
    if (arr_K is None) != (K is None) or (K and arr_K and arr_K.d != K.d):
        return False                      # declared field must match lines
    d1, d2 = cert['d1'], cert['d2']
    n = len(arr)
    if len({l.coords for l in arr.lines}) != n:
        return False    # duplicate/proportional lines: not reduced
    if not (isinstance(d1, int) and isinstance(d2, int)):
        return False
    if d1 < 0 or d1 > d2 or d1 + d2 != n - 1:
        return False                      # degree bookkeeping must match n
    N1 = len(LineArrangement._monoms(d1))
    N2 = len(LineArrangement._monoms(d2))
    if len(cert['theta1']) != 3 * N1 or len(cert['theta2']) != 3 * N2:
        return False    # wrong stated degree / nonhomogeneous packing
    if all(_verify_scalar(t) == 0 for t in cert['theta1']):
        return False
    if all(_verify_scalar(t) == 0 for t in cert['theta2']):
        return False
    monoms1 = LineArrangement._monoms(d1)
    monoms2 = LineArrangement._monoms(d2)
    th1 = [_verify_scalar(t) for t in cert['theta1']]
    th2 = [_verify_scalar(t) for t in cert['theta2']]
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
    c_expected = _verify_scalar(cert['c'])
    return sp.simplify(det - c_expected * Q) == 0 and c_expected != 0


def certificate_to_bw_vectors(cert):
    """Float BW-orthonormal unit coordinate vectors (u, v) of the certified
    pair, in the convention of penalized_saito.PenalizedSaitoEvaluator.

    NOTE: the evaluator normalizes each line to unit norm, which rescales Q
    but not the derivations; Gamma is invariant, so evaluating Gamma at these
    vectors on the same arrangement must give ~1.
    """
    from penalized_saito import _bw_sqrt_weights, _monoms

    K = _cert_field(cert)
    is_complex = K is not None and not K.is_real

    def _embed(t):
        if isinstance(t, QuadElem):
            return t.embed()
        if is_complex:
            return complex(_verify_scalar(t))
        return float(_verify_scalar(t))

    dtype = np.complex128 if is_complex else np.float64
    out = []
    for key, d in (('theta1', cert['d1']), ('theta2', cert['d2'])):
        N = len(_monoms(d))
        c = np.array([_embed(t) for t in cert[key]], dtype=dtype)
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
        'field': cert.get('field', 'QQ'),
        'normalization': cert.get('normalization',
                                  'projective_first_nonzero_one'),
        'theta1': [str(t) for t in cert['theta1']],
        'theta2': [str(t) for t in cert['theta2']],
        'Q': str(cert['Q']),
        'lines': [[str(v) for v in coords] for coords in cert['lines']],
    }


def _parse_exact_scalar(s, K):
    """Exact scalar from its serialized string under the declared field.

    Bracket tokens '[a+bs]' (optionally '-[...]') require K.  Everything
    else is parsed EXACTLY: Rational(s) for plain rational strings, exact
    sympification otherwise.  nsimplify is never used here — on rationals
    with large denominators its heuristic constant-matching can return a
    mathematically different number.
    """
    if isinstance(s, str):
        t = s.strip()
        if t.startswith('[') and t.endswith(']'):
            return parse_quad_token(t[1:-1], K)
        if t.startswith('-[') and t.endswith(']'):
            return -parse_quad_token(t[2:-1], K)
        try:
            return Rational(t)
        except (TypeError, ValueError):
            return sp.sympify(t)
    if getattr(s, 'is_Number', False) or isinstance(s, QuadElem):
        return s
    return sp.sympify(s)


def certificate_from_json(d):
    tag = d.get('field', 'QQ')
    K = QuadraticField.from_json(tag)
    return {
        'd1': int(d['d1']),
        'd2': int(d['d2']),
        'c': _parse_exact_scalar(d['c'], K),
        'field': tag,
        'normalization': d.get('normalization',
                               'projective_first_nonzero_one'),
        'theta1': [_parse_exact_scalar(t, K) for t in d['theta1']],
        'theta2': [_parse_exact_scalar(t, K) for t in d['theta2']],
        'Q': sp.sympify(d['Q']),
        'lines': [tuple(_parse_exact_scalar(v, K) for v in coords)
                  for coords in d['lines']],
    }
