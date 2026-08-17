"""
penalized_saito.py

Corrected penalized Saito functional for line arrangements in P^2.

Why this module exists
----------------------
The former "smooth Saito loss" (saito.py: smooth_saito_loss / _als_minimize)
evaluated the angle between Saito determinants and Q(A) over numerically
approximated logarithmic kernels.  That construction is mathematically binary:
for exact logarithmic derivations u, v of degrees d1 + d2 = n - 1, every
defining form alpha_i divides B(u, v) = det M(theta_E, u, v), and
deg B = n = deg Q, hence B = c * Q identically (possibly c = 0).  The exact
angular score is therefore 0 on free arrangements and 1 on nonfree ones;
intermediate floating-point values measured SVD tolerances and rounding, not
proximity to freeness.  The old score survives only as
saito.legacy_invalid_angular_score for regression comparisons.

The corrected functional
------------------------
Everything is defined with Bombieri-Weyl (BW) Hermitian norms:

    ||f||_d^2 = sum_{|gamma| = d} |c_gamma|^2 / multinomial(d; gamma),

for f = sum c_gamma x^gamma in S_d = C[x0, x1, x2]_d, and E_d = S_d^3 with the
sum of component norms.  Each line form alpha is normalized to ||alpha|| = 1.

Canonical logarithmic residual.  For a line alpha and u = (f0, f1, f2) in E_d,

    rho_{alpha, d}(u) = (I - Pi_{alpha, d}) (a0 f0 + a1 f1 + a2 f2),

where Pi_{alpha, d} is the BW-orthogonal projector of S_d onto alpha*S_{d-1}
(Pi_{alpha, 0} = 0).  Stacking over the n lines with weight 1/sqrt(n) gives

    L_{A, d} = (1/sqrt(n)) (+)_i rho_{alpha_i, d},      ker L_{A, d} = D(A)_d.

Implementation note: for unit alpha, the BW-orthogonal complement of
alpha*S_{d-1} in S_d is spanned, after a unitary change of coordinates taking
alpha to x0, by the monomials free of x0.  Concretely, if {u_i, w_i} is a
Hermitian-orthonormal basis of the algebraic kernel {x : alpha(x) = 0}, and

    theta(alpha)(s*u_i + t*w_i) = sum_p lambda_p s^p t^{d - p},

then ||rho_{alpha, d}(u)||^2 = sum_p |lambda_p|^2 / binom(d, p).  This is an
exact identity of the projector, NOT an SVD null-space construction: no rank
threshold or tolerance enters the definition of the functional.

Penalized functional.  With q = Q / ||Q|| (BW-unit), lambda > 0, 0 < beta < 1,
unit u in E_{d1}, unit v in E_{d2}, d1 + d2 = n - 1:

    R(u, v)     = ||L_{A, d1} u||^2 + ||L_{A, d2} v||^2
    Gamma(u, v) = |<B(u, v), q>|^2 / ( ||B(u, v)||^2 + lambda * R(u, v)^beta )

(Gamma := 0 when numerator and denominator both vanish), and the loss is

    S_{lambda, beta}(A; d1, d2) = 1 - sup_{||u|| = ||v|| = 1} Gamma(u, v).

Properties reflected here (and exercised in tests/test_penalized_saito.py
and tests/test_audit_penalized.py):
  1. 0 <= S <= 1 (and Gamma in [0, 1] pointwise).
  2. S = 0 exactly when A is free with exponents exactly (1, d1, d2).
  3. Otherwise 0 < S < 1 strictly — this covers BOTH genuinely nonfree
     arrangements AND free arrangements whose true exponent pair differs
     from the prescribed target pair.
  4. S is upper semicontinuous in the arrangement (no stratification needed:
     the sup is over fixed compact spheres of a jointly lower-semicontinuous
     integrand), and therefore CONTINUOUS at every arrangement where the
     target-pair loss is zero (usc gives limsup <= 0 there; S >= 0 gives
     liminf >= 0).
  5. For fixed A off the target-free locus, S is nondecreasing in lambda and
     -> 1 as lambda -> infinity; on the target-free locus S stays 0 for
     every lambda.
  6. The zero set is invariant under projective coordinate changes; positive
     values are invariant only under line permutation, per-line scalar/phase
     rescaling, and unitary/orthogonal coordinate changes (after the stated
     normalizations).  No PGL(3, C) invariance of positive values is claimed.

Field convention: the default computation optimizes REAL coefficient vectors
(the repo's arrangements are rational); dtype=complex128 switches to complex
spheres.  For a real arrangement S_complex <= S_real with identical zero
loci; results must be labeled with their field (maximize() reports
`optimization_field`) and never compared across fields unlabeled.

Exponent-independent search uses the finite MINIMUM over admissible pairs
(penalized_saito_loss_all_pairs) — never an average, whose zero set would be
the intersection rather than the union of the per-pair free loci.

S is a bounded, coordinate-normalized, upper-semicontinuous loss.  It is NOT a
metric, NOT globally continuous, NOT smooth, and its positive values are NOT
projectively invariant.  The computed value uses a numerical maximizer, so
1 - Gamma_hat >= S is an upper bound on the ideal loss: a search signal, never
a freeness or nonfreeness certificate.  Exact certification is always done
symbolically (arrangement.LineArrangement.is_free / saito.verify_arrangement).

Regularity caveat: upper semicontinuity is a property of the IDEAL loss S
(the true supremum over the compact sphere product).  The finite-multistart
evaluator returns an upper APPROXIMATION S_hat >= S whose value depends on
initialization and budget; S_hat as a function of the arrangement need not
inherit any semicontinuity, and no such regularity is claimed for it.

Optimizer
---------
maximize() runs a multistart MM ("IRLS-style") ascent on the product of unit
spheres.  For fixed v the concave tangent overestimate of R^beta at the
current iterate turns the denominator into a quadratic form u^H D u with

    D = B_v^H B_v + lambda*beta*R0^(beta-1) L1^H L1
        + lambda*((1-beta)*R0^beta + beta*R0^(beta-1)*||L2 v||^2) I,

and the surrogate |q^H B_v u|^2 / (u^H D u) is maximized in closed form by
u* = D^{-1} B_v^H q (scale-invariant, then normalized to the sphere).  Because
the tangent overestimates R^beta (concavity for 0 < beta < 1), each exact
surrogate step cannot decrease Gamma; a safeguarded backtracking line search
enforces monotonicity under floating point.  The only internal regularization
is the linearization floor _MM_R_FLOOR on R0 inside the surrogate weights
(NOT in Gamma itself, which is always evaluated exactly); its effect is tested
as it tends to zero in the test suite.  Initialization uses the leading
singular vectors of the q-contracted bilinear map A_q[i, j] = <B(e_i, e_j), q>
plus Sobol/random sphere points and optional warm starts.
"""

from functools import lru_cache
from math import comb as _comb, factorial as _factorial

import numpy as np

from arrangement import LineArrangement

__all__ = [
    "PenalizedSaitoEvaluator",
    "penalized_saito_loss",
    "penalized_saito_loss_all_pairs",
    "cached_penalized_loss",
    "admissible_degree_pairs",
    "kernel_diagnostics",
    "clear_cache",
    "DEFAULT_LAMBDA",
    "DEFAULT_BETA",
    "PROFILES",
]


DEFAULT_LAMBDA = 1.0
# Production default per the 2026-08 audit: beta = 0.75.  For beta > 1/2 the
# penalty R^beta is differentiable in the residual vectors, including at
# R = 0 (the gradient carries an explicit R == 0 branch; the raw expression
# beta * R^(beta-1) is numerically singular there).  beta = 0.5 remains
# supported but is NONSMOOTH at R = 0; it is optimized here by the MM
# (concave-tangent / IRLS-family) scheme with a safeguarded line search —
# never by plain smooth gradient ascent through R = 0.  beta = 1 is NOT a
# valid reported loss (compact attainment can fail at the 0/0 base locus)
# and is not used anywhere in this module, not even as an initializer.
DEFAULT_BETA = 0.75

# Functional/implementation version, recorded in diagnostics and manifests.
# 2.0.0 = 2026-08-16 migration; 2.1.0 = 2026-08 audit (beta 0.75 default,
# explicit branches); 2.2.0 = production-safety pass (raw-Gamma recording,
# calibrated clipping, provenance).
FUNCTIONAL_VERSION = "2.2.0"

# Clip tolerance for Cauchy-Schwarz violations of Gamma <= 1.  Violations
# up to this size are floating-point roundoff and are clipped (counted and
# reported); anything larger indicates a genuine numerical problem and
# raises a warning while being recorded unclipped in gamma_raw.
GAMMA_CLIP_TOL = 1e-9

# Optimizer effort presets (restarts x MM sweeps).  'rl' is the hot path used
# inside the reward; 'search' for extension pre-filtering; 'benchmark' for the
# validation study.
PROFILES = {
    "rl":        {"n_restarts": 4,  "n_iters": 40},
    "search":    {"n_restarts": 8,  "n_iters": 80},
    "benchmark": {"n_restarts": 20, "n_iters": 150},
}

# Linearization floor for R0 inside the MM surrogate weights only.  Gamma is
# always evaluated exactly; monotonicity is enforced by the safeguard line
# search, so this floor affects step *proposals*, never the objective.
_MM_R_FLOOR = 1e-300


# ─────────────────────────────────────────────────────────────────────────────
# Monomial bookkeeping and Bombieri-Weyl weights
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _monoms(d):
    """Degree-d exponent triples, in the repo-wide order of LineArrangement."""
    return tuple(LineArrangement._monoms(d))


@lru_cache(maxsize=None)
def _monom_index(d):
    return {m: i for i, m in enumerate(_monoms(d))}


@lru_cache(maxsize=None)
def _bw_sqrt_weights(d):
    """sqrt(multinomial(d; gamma)) per monomial.

    BW-orthonormal coordinates w of a polynomial with monomial coefficients c
    satisfy c = sqrt_w * w  (entrywise), so that ||f||_BW = ||w||_2.
    """
    fd = _factorial(d)
    w = [fd / (_factorial(a) * _factorial(b) * _factorial(c))
         for (a, b, c) in _monoms(d)]
    return np.sqrt(np.array(w, dtype=np.float64))


@lru_cache(maxsize=None)
def _mult_table(d_a, d_b):
    """(idx_a, idx_b, idx_out) arrays with monomial products
    x^{gamma_a} * x^{gamma_b} = x^{gamma_out}."""
    idx_out = _monom_index(d_a + d_b)
    ia, ib, io = [], [], []
    for i, (a1, b1, c1) in enumerate(_monoms(d_a)):
        for j, (a2, b2, c2) in enumerate(_monoms(d_b)):
            ia.append(i)
            ib.append(j)
            io.append(idx_out[(a1 + a2, b1 + b2, c1 + c2)])
    return np.array(ia), np.array(ib), np.array(io)


@lru_cache(maxsize=None)
def _xyz_shift(deg_from):
    """Index maps for multiplication by x, y, z: S_{deg_from} -> S_{deg_from+1}."""
    idx_out = _monom_index(deg_from + 1)
    ms = _monoms(deg_from)
    sx = np.array([idx_out[(a + 1, b, c)] for a, b, c in ms])
    sy = np.array([idx_out[(a, b + 1, c)] for a, b, c in ms])
    sz = np.array([idx_out[(a, b, c + 1)] for a, b, c in ms])
    return sx, sy, sz


@lru_cache(maxsize=4096)
def _restriction_expansion(ma, mb, mc, p):
    """(i, j, k, binom) terms for the coefficient of s^p t^{d-p} in
    (s*u0 + t*w0)^ma (s*u1 + t*w1)^mb (s*u2 + t*w2)^mc."""
    out = []
    for i in range(ma + 1):
        for j in range(mb + 1):
            k = p - i - j
            if 0 <= k <= mc:
                out.append((i, j, k,
                            _comb(ma, i) * _comb(mb, j) * _comb(mc, k)))
    return tuple(out)


def _restriction_coeff(u, w, ma, mb, mc, p):
    """Coefficient of s^p t^{d-p} in the restriction of x^{(ma,mb,mc)} to the
    parametrized line x = s*u + t*w.  Supports complex u, w."""
    res = 0.0
    for i, j, k, bc in _restriction_expansion(ma, mb, mc, p):
        res += (bc * (u[0] ** i) * (w[0] ** (ma - i))
                   * (u[1] ** j) * (w[1] ** (mb - j))
                   * (u[2] ** k) * (w[2] ** (mc - k)))
    return res


def _line_kernel_basis(a):
    """Hermitian-orthonormal basis {u, w} of the algebraic kernel
    {x in C^3 : a . x = 0} of the (unit) linear form a.

    The algebraic kernel of a equals the Hermitian orthogonal complement of
    conj(a), so [conj(a) | u | w] is unitary and the restriction identity in
    the module docstring applies.
    """
    _, _, Vh = np.linalg.svd(a.reshape(1, 3))
    u = Vh[1].conj()
    w = Vh[2].conj()
    return u, w


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

def _lines_matrix(arr_or_lines, dtype):
    """Extract an (n, 3) coefficient matrix from a LineArrangement or array."""
    if isinstance(arr_or_lines, LineArrangement):
        rows = [line.to_float() for line in arr_or_lines.lines]
        return np.array(rows, dtype=dtype)
    m = np.asarray(arr_or_lines, dtype=dtype)
    if m.ndim != 2 or m.shape[1] != 3:
        raise ValueError("lines must be an (n, 3) coefficient matrix")
    return m


class PenalizedSaitoEvaluator:
    """Penalized Saito functional for one arrangement and one degree pair.

    Parameters
    ----------
    arr_or_lines : LineArrangement or (n, 3) array (real or complex)
        Line coefficients [a0, a1, a2] for a0*x0 + a1*x1 + a2*x2 = 0.
        Each line is normalized internally to unit Hermitian norm.
    d1, d2 : int
        Target degrees, d1 + d2 = n - 1, d1, d2 >= 0.  d1 = 0 is allowed
        (constant vector fields; relevant for pencils / nonessential
        arrangements).
    """

    def __init__(self, arr_or_lines, d1, d2, dtype=np.float64):
        lines = _lines_matrix(arr_or_lines, dtype)
        n = lines.shape[0]
        if n < 1:
            raise ValueError("empty arrangement")
        if d1 < 0 or d2 < 0 or d1 + d2 != n - 1:
            raise ValueError(
                f"degree pair ({d1}, {d2}) must be nonnegative with "
                f"d1 + d2 = n - 1 = {n - 1}")
        self.n = n
        self.d1, self.d2 = d1, d2
        self.dtype = np.dtype(dtype)
        self.iscomplex = np.issubdtype(self.dtype, np.complexfloating)

        norms = np.linalg.norm(lines, axis=1)
        if np.any(norms < 1e-300):
            raise ValueError("zero line")
        self.lines = lines / norms[:, None]
        self._clip_count = 0
        self._clip_max_excess = 0.0

        self.N1 = len(_monoms(d1))
        self.N2 = len(_monoms(d2))
        self.dim_u = 3 * self.N1
        self.dim_v = 3 * self.N2
        self.N_out = len(_monoms(n))

        self._sw1 = _bw_sqrt_weights(d1)      # BW -> monomial, degree d1
        self._sw2 = _bw_sqrt_weights(d2)
        self._sw_out_inv = 1.0 / _bw_sqrt_weights(n)

        self.L1 = self._build_residual_operator(d1)
        self.L2 = self.L1 if d1 == d2 else self._build_residual_operator(d2)

        self.q = self._build_unit_q()
        # q in monomial-adjoint form used for contractions:
        # <B, q>_BW = sum_out conj(q_bw[out]) * c_out[out] / sw_out[out]
        self._q_tilde = self.q.conj() * self._sw_out_inv

        self._table = _mult_table(d1, d2)
        self._shift = _xyz_shift(n - 1)

        self._Aq = None      # lazy: q-contracted bilinear map (dim_u, dim_v)
        self._L1tL1 = self.L1.conj().T @ self.L1
        self._L2tL2 = (self._L1tL1 if self.L2 is self.L1
                       else self.L2.conj().T @ self.L2)

    # ── construction ────────────────────────────────────────────────────────

    def _build_residual_operator(self, d):
        """L_{A, d} in BW-orthonormal coordinates on both sides.

        Rows: (line i, p) for p = 0..d, scaled by 1/sqrt(binom(d, p)) and the
        global 1/sqrt(n).  Columns: E_d in BW coordinates (f | g | h blocks).
        ker L = D(A)_d exactly; no tolerance enters the construction.
        """
        n, N = self.n, len(_monoms(d))
        sw = _bw_sqrt_weights(d)
        L = np.zeros((n * (d + 1), 3 * N), dtype=self.dtype)
        row_scale = np.array([1.0 / np.sqrt(_comb(d, p)) for p in range(d + 1)])
        for i in range(n):
            a = self.lines[i]
            u, w = _line_kernel_basis(a)
            for p in range(d + 1):
                r = i * (d + 1) + p
                coeffs = np.array(
                    [_restriction_coeff(u, w, ma, mb, mc, p)
                     for (ma, mb, mc) in _monoms(d)], dtype=self.dtype)
                coeffs = coeffs * sw          # domain BW -> monomial
                L[r, :N] = a[0] * coeffs
                L[r, N:2 * N] = a[1] * coeffs
                L[r, 2 * N:] = a[2] * coeffs
                L[r] *= row_scale[p]
        L /= np.sqrt(n)
        return L

    def _build_unit_q(self):
        """BW-unit coefficient vector of Q = prod(normalized lines)."""
        c = np.array([1.0], dtype=self.dtype)   # degree 0
        for deg, i in enumerate(range(self.n)):
            a = self.lines[i]
            ia, ib, io = _mult_table(deg, 1)
            out = np.zeros(len(_monoms(deg + 1)), dtype=self.dtype)
            # degree-1 monomial order is _monoms(1); map coefficients:
            lin = np.zeros(3, dtype=self.dtype)
            for idx, m in enumerate(_monoms(1)):
                lin[idx] = a[{(1, 0, 0): 0, (0, 1, 0): 1, (0, 0, 1): 2}[m]]
            np.add.at(out, io, c[ia] * lin[ib])
            c = out
        q_bw = c * self._sw_out_inv
        nq = np.linalg.norm(q_bw)
        if nq < 1e-300:
            raise ValueError("degenerate defining polynomial")
        return q_bw / nq

    # ── bilinear determinant map ────────────────────────────────────────────

    def _components_mono(self, u_bw, which):
        """Split a BW coordinate vector into monomial-coefficient components."""
        if which == 1:
            N, sw = self.N1, self._sw1
        else:
            N, sw = self.N2, self._sw2
        f = u_bw[:N] * sw
        g = u_bw[N:2 * N] * sw
        h = u_bw[2 * N:] * sw
        return f, g, h

    def B_bw(self, u_bw, v_bw):
        """BW coordinates of det M(theta_E, u, v) in S_n."""
        return self.B_v_matrix(v_bw) @ u_bw

    def B_v_matrix(self, v_bw):
        """Matrix of u |-> B(u, v) : (N_out, dim_u), BW coords both sides."""
        f2, g2, h2 = self._components_mono(v_bw, 2)
        ia, ib, io = self._table
        sx, sy, sz = self._shift
        N1, N_out = self.N1, self.N_out
        M = np.zeros((N_out, 3 * N1), dtype=self.dtype)
        # det = x*(g1 h2 - g2 h1) - y*(f1 h2 - f2 h1) + z*(f1 g2 - f2 g1)
        # columns: f1-block [0:N1], g1-block [N1:2N1], h1-block [2N1:3N1]
        np.add.at(M, (sx[io], N1 + ia), h2[ib])        # +x * g1 h2
        np.add.at(M, (sx[io], 2 * N1 + ia), -g2[ib])   # -x * g2 h1
        np.add.at(M, (sy[io], ia), -h2[ib])            # -y * f1 h2
        np.add.at(M, (sy[io], 2 * N1 + ia), f2[ib])    # +y * f2 h1
        np.add.at(M, (sz[io], ia), g2[ib])             # +z * f1 g2
        np.add.at(M, (sz[io], N1 + ia), -f2[ib])       # -z * f2 g1
        # domain conversion (BW -> monomial) and range conversion (-> BW)
        M *= self._sw_out_inv[:, None]
        M[:, :N1] *= self._sw1
        M[:, N1:2 * N1] *= self._sw1
        M[:, 2 * N1:] *= self._sw1
        return M

    def B_u_matrix(self, u_bw):
        """Matrix of v |-> B(u, v) : (N_out, dim_v), BW coords both sides."""
        f1, g1, h1 = self._components_mono(u_bw, 1)
        ia, ib, io = self._table
        sx, sy, sz = self._shift
        N2, N_out = self.N2, self.N_out
        M = np.zeros((N_out, 3 * N2), dtype=self.dtype)
        np.add.at(M, (sx[io], 2 * N2 + ib), g1[ia])    # +x * g1 h2
        np.add.at(M, (sx[io], N2 + ib), -h1[ia])       # -x * g2 h1
        np.add.at(M, (sy[io], 2 * N2 + ib), -f1[ia])   # -y * f1 h2
        np.add.at(M, (sy[io], ib), h1[ia])             # +y * f2 h1
        np.add.at(M, (sz[io], N2 + ib), f1[ia])        # +z * f1 g2
        np.add.at(M, (sz[io], ib), -g1[ia])            # -z * f2 g1
        M *= self._sw_out_inv[:, None]
        M[:, :N2] *= self._sw2
        M[:, N2:2 * N2] *= self._sw2
        M[:, 2 * N2:] *= self._sw2
        return M

    @property
    def Aq(self):
        """q-contracted bilinear map: Aq[i, j] = <B(e_i, e_j), q>_BW,
        so that <B(u, v), q> = (Aq^H u)^H ... concretely
        <B(u,v), q> = u^T Aq_bilinear v in the real case.  Used for
        initialization (leading singular vectors) and the positivity argument
        (Aq is never the zero matrix: <z*f*g, Q> = 0 for all f, g would force
        dQ/dz = 0)."""
        if self._Aq is None:
            ia, ib, io = self._table
            sx, sy, sz = self._shift
            N1, N2 = self.N1, self.N2
            A = np.zeros((3 * N1, 3 * N2), dtype=self.dtype)
            qx = self._q_tilde[sx[io]]
            qy = self._q_tilde[sy[io]]
            qz = self._q_tilde[sz[io]]
            np.add.at(A, (N1 + ia, 2 * N2 + ib), qx)      # +x g1 h2
            np.add.at(A, (2 * N1 + ia, N2 + ib), -qx)     # -x h1 g2
            np.add.at(A, (ia, 2 * N2 + ib), -qy)          # -y f1 h2
            np.add.at(A, (2 * N1 + ia, ib), qy)           # +y h1 f2
            np.add.at(A, (ia, N2 + ib), qz)               # +z f1 g2
            np.add.at(A, (N1 + ia, ib), -qz)              # -z g1 f2
            A[:N1] *= self._sw1[:, None]
            A[N1:2 * N1] *= self._sw1[:, None]
            A[2 * N1:] *= self._sw1[:, None]
            A[:, :N2] *= self._sw2
            A[:, N2:2 * N2] *= self._sw2
            A[:, 2 * N2:] *= self._sw2
            self._Aq = A
        return self._Aq

    # ── objective ───────────────────────────────────────────────────────────

    def residual(self, u_bw, v_bw):
        """R(u, v) = ||L1 u||^2 + ||L2 v||^2 (tangency residual)."""
        r1 = np.linalg.norm(self.L1 @ u_bw) ** 2
        r2 = np.linalg.norm(self.L2 @ v_bw) ** 2
        return float(r1), float(r2)

    def gamma(self, u_bw, v_bw, lam=DEFAULT_LAMBDA, beta=DEFAULT_BETA,
              return_parts=False):
        """Gamma(u, v) for unit u, v.  Exact formula; no regularization.

        Explicit scale-aware branches (audit spec): the mathematical
        denominator carries NO epsilon; the branches below only make the
        exact case analysis explicit and clip floating-point noise.
        """
        B = self.B_bw(u_bw, v_bw)
        inner = np.vdot(self.q, B)             # <B, q>_BW  (conjugates q)
        num = abs(inner) ** 2
        B_sq = float(np.real(np.vdot(B, B)))
        r1, r2 = self.residual(u_bw, v_bw)
        R = r1 + r2
        if R == 0.0:
            penalty = 0.0                      # R = 0 branch (exact zero)
        else:
            penalty = lam * (R ** beta)        # tiny R: underflow-safe power
        den = B_sq + penalty
        if den == 0.0:
            g_raw = 0.0                        # R = 0 AND B = 0: base locus
            g = 0.0
        else:
            # includes the R = 0, B != 0 free direction (den = ||B||^2)
            g_raw = float(num / den)
            if g_raw > 1.0:
                # Cauchy-Schwarz gives num <= ||B||^2 <= den exactly, so any
                # excess is floating-point rounding.  Clip ONLY roundoff-
                # sized violations; larger ones signal a numerical bug and
                # are surfaced, never silently absorbed.
                excess = g_raw - 1.0
                self._clip_count += 1
                if excess > self._clip_max_excess:
                    self._clip_max_excess = excess
                if excess <= GAMMA_CLIP_TOL:
                    g = 1.0
                else:
                    import warnings
                    warnings.warn(
                        f"Gamma exceeded 1 by {excess:.3e} > GAMMA_CLIP_TOL="
                        f"{GAMMA_CLIP_TOL:.1e}; leaving unclipped — "
                        f"investigate conditioning", RuntimeWarning)
                    g = g_raw
            else:
                g = g_raw
        if not return_parts:
            return g
        return g, {
            "gamma_raw": float(g_raw),
            "B_norm": float(np.sqrt(B_sq)),
            "inner_abs": float(abs(inner)),
            "L1u_norm": float(np.sqrt(r1)),
            "L2v_norm": float(np.sqrt(r2)),
            "residual_R": float(R),
            "denominator": float(den),
            "lambda": float(lam),
            "beta": float(beta),
        }

    def gamma_and_grad(self, u_bw, v_bw, lam=DEFAULT_LAMBDA, beta=DEFAULT_BETA):
        """Gamma and its Euclidean gradient w.r.t. (u, v).  Real dtype only.

        Differentiable wherever the denominator is nonzero and R > 0.
        Riemannian (sphere-projected) gradients are obtained by removing the
        radial component.
        """
        if self.iscomplex:
            raise NotImplementedError("analytic gradient implemented for the "
                                      "real slice only")
        Bv = self.B_v_matrix(v_bw)
        B = Bv @ u_bw
        Bu = self.B_u_matrix(u_bw)
        inner = float(self.q @ B)
        num = inner ** 2
        B_sq = float(B @ B)
        L1u = self.L1 @ u_bw
        L2v = self.L2 @ v_bw
        r1 = float(L1u @ L1u)
        r2 = float(L2v @ L2v)
        R = r1 + r2
        den = B_sq + lam * (R ** beta)
        if den == 0.0:
            z_u = np.zeros_like(u_bw)
            z_v = np.zeros_like(v_bw)
            return 0.0, z_u, z_v
        g = num / den
        dnum_u = 2.0 * inner * (Bv.T @ self.q)
        dnum_v = 2.0 * inner * (Bu.T @ self.q)
        # Explicit branch (audit spec): for beta > 1/2 the penalty term is
        # differentiable at R = 0 with gradient 0, but beta * R^(beta-1) is
        # numerically singular there — branch instead of evaluating it.
        dRbeta = (beta * R ** (beta - 1.0)) if R > 0 else 0.0
        dden_u = 2.0 * (Bv.T @ B) + lam * dRbeta * 2.0 * (self.L1.T @ L1u)
        dden_v = 2.0 * (Bu.T @ B) + lam * dRbeta * 2.0 * (self.L2.T @ L2v)
        grad_u = (dnum_u * den - num * dden_u) / den ** 2
        grad_v = (dnum_v * den - num * dden_v) / den ** 2
        return g, grad_u, grad_v

    def projected_grad_norm(self, u_bw, v_bw, lam=DEFAULT_LAMBDA,
                            beta=DEFAULT_BETA):
        """Norm of the Riemannian gradient on the product of spheres."""
        if self.iscomplex:
            return float("nan")
        _, gu, gv = self.gamma_and_grad(u_bw, v_bw, lam, beta)
        gu = gu - (u_bw @ gu) * u_bw
        gv = gv - (v_bw @ gv) * v_bw
        return float(np.sqrt(gu @ gu + gv @ gv))

    # ── optimizer ───────────────────────────────────────────────────────────

    def _random_unit(self, rng, dim):
        if self.iscomplex:
            x = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        else:
            x = rng.standard_normal(dim)
        x = x.astype(self.dtype, copy=False)
        return x / np.linalg.norm(x)

    def _near_kernel_vectors(self, L, m=2):
        """Right singular vectors of L with the smallest singular values.

        OPTIMIZER INITIALIZATION HEURISTIC ONLY.  The loss is defined without
        any null-space basis or rank threshold; these vectors merely seed the
        ascent (for a free arrangement, a generic pair of kernel vectors
        already attains Gamma = 1, so this makes the free optimum easy to
        find).  Changing or removing this heuristic can only change how tight
        the numerical upper bound is, never the mathematical value.
        """
        try:
            _, _, Vh = np.linalg.svd(L, full_matrices=True)
        except np.linalg.LinAlgError:
            return []
        rows = Vh[-m:] if Vh.shape[0] >= m else Vh
        return [r.conj() for r in rows[::-1]]

    def _kernel_pair_inits(self, lam, beta, m=2):
        ku = self._near_kernel_vectors(self.L1, m)
        kv = (ku if self.L2 is self.L1 else
              self._near_kernel_vectors(self.L2, m))
        scored = []
        for uu in ku:
            for vv in kv:
                if self.L2 is self.L1 and \
                        abs(abs(np.vdot(uu, vv)) - 1.0) < 1e-12:
                    continue   # same vector twice -> B = 0 identically
                scored.append((self.gamma(uu, vv, lam, beta), uu, vv))
        scored.sort(key=lambda t: -t[0])
        return [(u, v, "kernel") for (_, u, v) in scored[:2]]

    def _initial_points(self, rng, n_restarts, warm_starts, sobol,
                        lam=DEFAULT_LAMBDA, beta=DEFAULT_BETA):
        """(u0, v0) list: warm starts, kernel-pair inits, SVD-of-Aq pairs,
        Sobol/random points."""
        inits = []
        for (u0, v0) in (warm_starts or []):
            u0 = np.asarray(u0, dtype=self.dtype)
            v0 = np.asarray(v0, dtype=self.dtype)
            if u0.shape == (self.dim_u,) and v0.shape == (self.dim_v,):
                nu, nv = np.linalg.norm(u0), np.linalg.norm(v0)
                if nu > 0 and nv > 0:
                    inits.append((u0 / nu, v0 / nv, "warm"))
        inits.extend(self._kernel_pair_inits(lam, beta))
        try:
            U, s, Vh = np.linalg.svd(self.Aq)
            for k in range(min(2, len(s))):
                if s[k] > 0:
                    inits.append((U[:, k].conj() if self.iscomplex else U[:, k],
                                  Vh[k].conj().T, "svd_init"))
        except np.linalg.LinAlgError:
            pass
        n_fill = max(0, n_restarts - len(inits))
        sob_pts = []
        if sobol and not self.iscomplex and n_fill > 0:
            try:
                from scipy.stats import qmc, norm
                eng = qmc.Sobol(d=self.dim_u + self.dim_v,
                                scramble=True, seed=rng)
                m = int(np.ceil(np.log2(max(2, n_fill))))
                pts = norm.ppf(np.clip(eng.random(2 ** m)[:n_fill],
                                       1e-12, 1 - 1e-12))
                for row in pts:
                    u0 = row[:self.dim_u]
                    v0 = row[self.dim_u:]
                    nu, nv = np.linalg.norm(u0), np.linalg.norm(v0)
                    if nu > 0 and nv > 0:
                        sob_pts.append((u0 / nu, v0 / nv, "sobol"))
            except Exception:
                sob_pts = []
        inits.extend(sob_pts)
        while len(inits) < n_restarts:
            inits.append((self._random_unit(rng, self.dim_u),
                          self._random_unit(rng, self.dim_v), "random"))
        return inits[:n_restarts]

    def _mm_half_step(self, Bmat, LtL, r_other, x0, r_self, lam, beta):
        """Exact maximizer of the MM surrogate for one sphere variable.

        Returns the proposed unit vector, or None on breakdown.
        """
        R0 = max(r_self + r_other, _MM_R_FLOOR)
        c1 = beta * R0 ** (beta - 1.0)
        sigma = (1.0 - beta) * R0 ** beta + c1 * r_other
        b = Bmat.conj().T @ self.q
        if np.linalg.norm(b) == 0.0:
            return None
        D = Bmat.conj().T @ Bmat + (lam * c1) * LtL
        D[np.diag_indices_from(D)] += lam * sigma
        try:
            xs = np.linalg.solve(D, b)
        except np.linalg.LinAlgError:
            try:
                xs = np.linalg.lstsq(D, b, rcond=None)[0]
            except np.linalg.LinAlgError:
                return None
        nx = np.linalg.norm(xs)
        if nx == 0.0 or not np.isfinite(nx):
            return None
        return xs / nx

    def _safeguard(self, u0, v0, u_new, v_new, g0, lam, beta):
        """Accept the proposal, backtracking toward (u0, v0) if Gamma dropped."""
        t = 1.0
        for _ in range(24):
            u_t = u0 + t * (u_new - u0)
            v_t = v0 + t * (v_new - v0)
            nu, nv = np.linalg.norm(u_t), np.linalg.norm(v_t)
            if nu > 0 and nv > 0:
                u_t, v_t = u_t / nu, v_t / nv
                g_t = self.gamma(u_t, v_t, lam, beta)
                if g_t >= g0 - 1e-15:
                    return u_t, v_t, g_t, t
            t *= 0.5
        return u0, v0, g0, 0.0

    def maximize(self, lam=DEFAULT_LAMBDA, beta=DEFAULT_BETA,
                 n_restarts=8, n_iters=80, tol=1e-13, seed=0,
                 warm_starts=None, sobol=True, full_diagnostics=False):
        """Multistart MM ascent of Gamma over the product of unit spheres.

        Returns a dict with:
          loss           1 - best Gamma (>= the ideal loss; never a certificate)
          gamma          best Gamma found
          gamma_median / gamma_min / gamma_spread   restart statistics
          u, v           best point (BW coordinates, unit)
          parts          objective components at the best point
          restarts       per-restart records (gamma, iters, stop, init kind)
          proj_grad_norm Riemannian gradient norm at the best point (real only)
        """
        rng = np.random.default_rng(seed)
        inits = self._initial_points(rng, n_restarts, warm_starts, sobol,
                                     lam=lam, beta=beta)

        best = (-1.0, None, None)
        records = []
        for u0, v0, kind in inits:
            u, v = u0.copy(), v0.copy()
            g = self.gamma(u, v, lam, beta)
            stop = "max_iters"
            it_done = 0
            for it in range(n_iters):
                it_done = it + 1
                L1u = self.L1 @ u
                L2v = self.L2 @ v
                r1 = float(np.real(np.vdot(L1u, L1u)))
                r2 = float(np.real(np.vdot(L2v, L2v)))
                # u half-step (v fixed)
                Bv = self.B_v_matrix(v)
                u_prop = self._mm_half_step(Bv, self._L1tL1, r2, u, r1,
                                            lam, beta)
                if u_prop is not None:
                    u, v, g, _ = self._safeguard(u, v, u_prop, v, g, lam, beta)
                # v half-step (u fixed)
                L1u = self.L1 @ u
                r1 = float(np.real(np.vdot(L1u, L1u)))
                Bu = self.B_u_matrix(u)
                v_prop = self._mm_half_step(Bu, self._L2tL2, r1, v, r2,
                                            lam, beta)
                g_prev = g
                if v_prop is not None:
                    u, v, g, _ = self._safeguard(u, v, u, v_prop, g, lam, beta)
                if u_prop is None and v_prop is None:
                    stop = "surrogate_breakdown"
                    break
                if g >= 1.0 - 1e-15:
                    stop = "gamma_one"
                    break
                if abs(g - g_prev) < tol * max(1.0, g):
                    stop = "converged"
                    break
            records.append({"gamma": g, "iters": it_done, "stop": stop,
                            "init": kind})
            if g > best[0]:
                best = (g, u.copy(), v.copy())

        g_best, u_best, v_best = best
        gs = np.array([r["gamma"] for r in records])
        g_best = float(max(g_best, 0.0))
        _, parts = self.gamma(u_best, v_best, lam, beta, return_parts=True)
        out = {
            "loss": float(min(max(1.0 - g_best, 0.0), 1.0)),
            "gamma": g_best,
            "gamma_median": float(np.median(gs)),
            "gamma_min": float(np.min(gs)),
            "gamma_spread": float(np.max(gs) - np.min(gs)),
            "u": u_best,
            "v": v_best,
            "parts": parts,
            "restarts": records,
            "lambda": float(lam),
            "beta": float(beta),
            "n_restarts": len(records),
            # conventions and solver internals (audit spec): the value is a
            # finite-multistart LOWER bound on sup Gamma, so the loss is an
            # UPPER bound on the true loss — never a nonfreeness proof.
            "optimization_field": ("complex" if self.iscomplex else "real"),
            "beta_smoothness": ("nonsmooth_at_R0" if beta <= 0.5
                                else "differentiable"),
            "mm_r_floor": _MM_R_FLOOR,
            "functional_version": FUNCTIONAL_VERSION,
            # clipping accounting for THIS evaluator instance (item: verify
            # clipping is roundoff bookkeeping, not value manufacturing)
            "gamma_clip_count": self._clip_count,
            "gamma_clip_max_excess": self._clip_max_excess,
        }
        if not self.iscomplex:
            try:
                out["proj_grad_norm"] = self.projected_grad_norm(
                    u_best, v_best, lam, beta)
            except Exception:
                out["proj_grad_norm"] = float("nan")
        if full_diagnostics:
            out["kernel_diag_d1"] = kernel_diagnostics_from_operator(self.L1)
            out["kernel_diag_d2"] = (out["kernel_diag_d1"]
                                     if self.L2 is self.L1 else
                                     kernel_diagnostics_from_operator(self.L2))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics (explicitly NOT part of the definition of the loss)
# ─────────────────────────────────────────────────────────────────────────────

def kernel_diagnostics_from_operator(L, tol=1e-8):
    """Singular-value diagnostics of a residual operator.

    Reports the numerical kernel dimension at the given tolerance, the
    smallest positive singular value, and a condition estimate.  This is a
    conditioning DIAGNOSTIC for the benchmark; the loss itself never uses a
    rank threshold.
    """
    s = np.linalg.svd(L, compute_uv=False)
    smax = float(s[0]) if len(s) else 0.0
    thresh = tol * max(1.0, smax)
    k_num = int(L.shape[1] - np.sum(s > thresh))
    pos = s[s > thresh]
    s_min_pos = float(pos[-1]) if len(pos) else 0.0
    return {
        "sigma_max": smax,
        "sigma_min_pos": s_min_pos,
        "numerical_kernel_dim": k_num,
        "cond_estimate": float(smax / s_min_pos) if s_min_pos > 0 else np.inf,
        "tol": tol,
    }


def kernel_diagnostics(arr_or_lines, d, tol=1e-8, dtype=np.float64):
    """Diagnostics for L_{A, d} of an arrangement (see above)."""
    lines = _lines_matrix(arr_or_lines, dtype)
    n = lines.shape[0]
    ev = PenalizedSaitoEvaluator.__new__(PenalizedSaitoEvaluator)
    # minimal init to reuse _build_residual_operator
    norms = np.linalg.norm(lines, axis=1)
    ev.lines = lines / norms[:, None]
    ev.n = n
    ev.dtype = np.dtype(dtype)
    ev.iscomplex = np.issubdtype(ev.dtype, np.complexfloating)
    L = ev._build_residual_operator(d)
    return kernel_diagnostics_from_operator(L, tol=tol)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level API
# ─────────────────────────────────────────────────────────────────────────────

def admissible_degree_pairs(n, include_zero=True):
    """All (d1, d2), d1 <= d2, d1 + d2 = n - 1.  d1 = 0 included by default
    (pencils / nonessential arrangements)."""
    lo = 0 if include_zero else 1
    return [(d1, n - 1 - d1) for d1 in range(lo, (n - 1) // 2 + 1)]


def penalized_saito_loss(arr_or_lines, d1=None, d2=None,
                         lam=DEFAULT_LAMBDA, beta=DEFAULT_BETA,
                         profile="search", n_restarts=None, n_iters=None,
                         seed=0, warm_starts=None, dtype=np.float64,
                         return_details=False):
    """Penalized Saito loss S_{lam, beta}(A; d1, d2) (numerical upper bound).

    If (d1, d2) is omitted, candidate exponents from the characteristic
    polynomial are used when they exist, otherwise the all-pairs envelope is
    returned.  Candidate-exponent arithmetic is a convenience default here; it
    never gates the definition — pass any (d1, d2) with d1 + d2 = n - 1.

    Boundary convention: for n < 3 this wrapper returns 1.0 (the search
    pipeline treats such stubs as maximally unfinished).  The evaluator
    itself handles n < 3 degree pairs correctly if constructed directly.
    """
    lines = _lines_matrix(arr_or_lines, dtype)
    n = lines.shape[0]
    if n < 3:
        return (1.0, None) if return_details else 1.0
    prof = PROFILES[profile]
    n_restarts = prof["n_restarts"] if n_restarts is None else n_restarts
    n_iters = prof["n_iters"] if n_iters is None else n_iters

    if d1 is None or d2 is None:
        if isinstance(arr_or_lines, LineArrangement):
            exps = arr_or_lines.candidate_exponents()
        else:
            exps = None
        if exps is None:
            return penalized_saito_loss_all_pairs(
                arr_or_lines, lam=lam, beta=beta, profile=profile,
                n_restarts=n_restarts, n_iters=n_iters, seed=seed,
                dtype=dtype, return_details=return_details)
        d1, d2 = exps

    ev = PenalizedSaitoEvaluator(lines, d1, d2, dtype=dtype)
    res = ev.maximize(lam=lam, beta=beta, n_restarts=n_restarts,
                      n_iters=n_iters, seed=seed, warm_starts=warm_starts)
    res["d1"], res["d2"] = d1, d2
    return (res["loss"], res) if return_details else res["loss"]


def penalized_saito_loss_all_pairs(arr_or_lines, lam=DEFAULT_LAMBDA,
                                   beta=DEFAULT_BETA, profile="search",
                                   n_restarts=None, n_iters=None, seed=0,
                                   include_zero=True, dtype=np.float64,
                                   return_details=False):
    """Exponent-independent envelope: min over all (d1, d2), d1 + d2 = n - 1."""
    lines = _lines_matrix(arr_or_lines, dtype)
    n = lines.shape[0]
    if n < 3:
        return (1.0, None) if return_details else 1.0
    prof = PROFILES[profile]
    n_restarts = prof["n_restarts"] if n_restarts is None else n_restarts
    n_iters = prof["n_iters"] if n_iters is None else n_iters

    best = (1.0, None)
    per_pair = {}
    for (d1, d2) in admissible_degree_pairs(n, include_zero=include_zero):
        ev = PenalizedSaitoEvaluator(lines, d1, d2, dtype=dtype)
        res = ev.maximize(lam=lam, beta=beta, n_restarts=n_restarts,
                          n_iters=n_iters, seed=seed)
        per_pair[(d1, d2)] = res["loss"]
        if res["loss"] < best[0]:
            res["d1"], res["d2"] = d1, d2
            best = (res["loss"], res)
    if return_details:
        details = best[1] or {}
        details["per_pair"] = per_pair
        return best[0], details
    return best[0]


# ─────────────────────────────────────────────────────────────────────────────
# Cached evaluation (keyed by canonical line subset and degree pair)
# ─────────────────────────────────────────────────────────────────────────────

_LOSS_CACHE = {}
_LOSS_CACHE_MAX = 200_000


def _canonical_key(arr: LineArrangement):
    return tuple(sorted(line.coords for line in arr.lines))


def cached_penalized_loss(arr: LineArrangement, d1=None, d2=None,
                          lam=DEFAULT_LAMBDA, beta=DEFAULT_BETA,
                          profile="rl", seed=0):
    """Loss with memoization by (canonical line subset, degree pair, lam, beta).

    Line order does not affect the key; exact Rational coordinates are used,
    so the cache is exact.  Intended for the RL hot path where the same
    partial arrangement is scored several times per step.
    """
    if d1 is None or d2 is None:
        exps = arr.candidate_exponents()
        if exps is None:
            d1 = d2 = None
        else:
            d1, d2 = exps
    key = (_canonical_key(arr), d1, d2, float(lam), float(beta), profile, seed)
    hit = _LOSS_CACHE.get(key)
    if hit is not None:
        return hit
    val = penalized_saito_loss(arr, d1=d1, d2=d2, lam=lam, beta=beta,
                               profile=profile, seed=seed)
    if len(_LOSS_CACHE) < _LOSS_CACHE_MAX:
        _LOSS_CACHE[key] = val
    return val


def clear_cache():
    _LOSS_CACHE.clear()


def runtime_provenance(repo_root="."):
    """Provenance block for manifests/logs: code commit, dirty-tree state,
    functional version, defaults, field convention."""
    import subprocess
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                             capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    cwd=repo_root, capture_output=True,
                                    text=True, timeout=10).stdout.strip())
    except Exception:
        rev, dirty = "unknown", None
    return {
        "functional_version": FUNCTIONAL_VERSION,
        "code_commit": rev,
        "dirty_tree": dirty,
        "default_lambda": DEFAULT_LAMBDA,
        "default_beta": DEFAULT_BETA,
        "optimization_field_default": "real",
        "gamma_clip_tol": GAMMA_CLIP_TOL,
        "mm_r_floor": _MM_R_FLOOR,
        "profiles": PROFILES,
    }
