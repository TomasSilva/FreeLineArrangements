"""
saito.py

Saito loss / reward shaping for the RL agent.

Three levels of signal:
  1. Combinatorial: Does b2(A) give integer candidate exponents?
     disc = (n-1)^2 - 4*(b2-(n-1)) must be >= 0 and a perfect square.

  2. Algebraic (penalized): the corrected penalized Saito functional
     (penalized_saito.py).  For a degree pair (d1, d2) with d1+d2 = n-1 it
     maximizes
         Gamma(u, v) = |<B(u,v), q>|^2 / (||B(u,v)||^2 + lambda*R(u,v)^beta)
     over unit u, v in the FULL coefficient spaces E_d1, E_d2, where
     R(u, v) = ||L_{A,d1} u||^2 + ||L_{A,d2} v||^2 penalizes violation of the
     logarithmic tangency conditions (Bombieri-Weyl norms throughout).
     The loss 1 - Gamma_hat is a bounded, coordinate-normalized, upper-
     semicontinuous search signal: 0 exactly on free arrangements, strictly
     inside (0, 1) on nonfree ones.  It is an upper bound on the ideal loss
     and is never a freeness certificate.

  3. Algebraic (hard): Does a Saito basis exist?
     Exact symbolic verification over Q (arrangement.is_free); the only
     accepted certificate of freeness.

HISTORICAL NOTE (old angular score).  The former "smooth Saito loss"
(ALS angle over SVD null-space bases) is mathematically binary in exact
arithmetic: for exact logarithmic derivations of candidate degrees, the Saito
determinant is always a scalar multiple of Q, so the exact angular score is
0 (free) or 1 (nonfree) and intermediate floating-point values only measured
numerical violation of logarithmicity and SVD tolerances.  It is retained
verbatim as `legacy_invalid_angular_score` strictly for regression
comparisons and MUST NOT be used as a proximity measure or reward.

The reward function R(A) returned to the RL agent is:
  R(A) = w_comb * score_comb(A)
       + w_alg  * score_alg(A)
       - w_pen  * is_pencil(A)
       + w_free * is_free_exact(A)  [terminal bonus]

where each score is in [-1, 1].
"""

import warnings

import numpy as np
import sympy as sp
from sympy import Rational, Matrix
from arrangement import LineArrangement, ProjectiveLine
from penalized_saito import (
    penalized_saito_loss,
    penalized_saito_loss_all_pairs,
    cached_penalized_loss,
    DEFAULT_LAMBDA,
    DEFAULT_BETA,
)


# ─────────────────────────────────────────────────────────────────────────────
# Combinatorial score
# ─────────────────────────────────────────────────────────────────────────────

def combinatorial_score(arr: LineArrangement) -> float:
    """
    Continuous score in [-1, 1] measuring how close b2(A) is to
    yielding integer candidate exponents.

    The discriminant for candidate exponents (1, d1, d2) is:
        disc = (n-1)^2 - 4*(b2 - (n-1))

    Score = 1 if disc >= 0 and a perfect square (candidate exponents exist).
    Smooth interpolation otherwise.
    """
    n = len(arr)
    if n < 2:
        return 0.0

    # Fast path: candidate_exponents already encodes this check
    if arr.candidate_exponents() is not None:
        return 1.0

    b2 = arr.b2()
    product = b2 - (n - 1)
    if product < 0:
        return max(-1.0, product / float((n - 1) + 1))

    disc = (n - 1) ** 2 - 4 * product
    if disc < 0:
        return max(-1.0, disc / float((n - 1) ** 2 + 1))

    # disc >= 0 but not a perfect square
    sq = int(disc ** 0.5 + 0.5)
    nearest_sq = sq * sq
    error = abs(disc - nearest_sq)
    return 1.0 - error / (disc + 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Algebraic (soft) score via singular values
# ─────────────────────────────────────────────────────────────────────────────

def _float_derivation_matrix(arr: LineArrangement, d: int) -> np.ndarray:
    """
    Build the derivation matrix over R (float64).
    Much faster than sympy's exact version; used for the soft algebraic score.

    For each line alpha_i = a_i x + b_i y + c_i z, and for each exponent
    pair (p, q=d-p), the row encodes the condition alpha_i | theta(alpha_i)
    evaluated at the parameterization of ker(alpha_i).
    """
    lines_float = [line.to_float() for line in arr.lines]
    monoms = arr._monoms(d)
    N = len(monoms)

    rows = []
    for abc in lines_float:
        a, b, c = abc
        norm = np.sqrt(a**2 + b**2 + c**2)
        if norm < 1e-12:
            continue
        # Kernel basis (float)
        if abs(a) > 1e-10:
            u = np.array([-b, a, 0.0]) / abs(a)
            v = np.array([-c, 0.0, a]) / abs(a)
        elif abs(b) > 1e-10:
            u = np.array([1.0, 0.0, 0.0])
            v = np.array([0.0, -c, b]) / abs(b)
        else:
            u = np.array([1.0, 0.0, 0.0])
            v = np.array([0.0, 1.0, 0.0])

        for p in range(d + 1):
            q = d - p
            row = np.zeros(3 * N)
            for idx, (ma, mb, mc) in enumerate(monoms):
                coeff = _mono_param_float(u, v, ma, mb, mc, p, q)
                row[idx]       += a * coeff
                row[N + idx]   += b * coeff
                row[2*N + idx] += c * coeff
            rows.append(row)

    return np.array(rows, dtype=np.float64) if rows else np.zeros((1, 3*N))


from functools import lru_cache
from math import comb as _comb


@lru_cache(maxsize=4096)
def _mono_param_indices(ma, mb, mc, p):
    """Precompute (i, j, k, binom_coeff) tuples for given monomial and p."""
    result = []
    for i in range(ma + 1):
        for j in range(mb + 1):
            k = p - i - j
            if 0 <= k <= mc:
                result.append((i, j, k, _comb(ma, i) * _comb(mb, j) * _comb(mc, k)))
    return tuple(result)


def _mono_param_float(u, v, ma, mb, mc, p, q):
    """Float version of monomial parameterization coefficient."""
    res = 0.0
    for i, j, k, binom_c in _mono_param_indices(ma, mb, mc, p):
        res += (binom_c
                * (u[0]**i) * (v[0]**(ma-i))
                * (u[1]**j) * (v[1]**(mb-j))
                * (u[2]**k) * (v[2]**(mc-k)))
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Smooth Saito loss: polynomial arithmetic in coefficient space
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=64)
def _monomial_index_map(d):
    """Cached dict mapping (a,b,c) -> index for degree-d monomials."""
    monoms = LineArrangement._monoms(d)
    return {m: i for i, m in enumerate(monoms)}, monoms


@lru_cache(maxsize=256)
def _poly_mult_table(d_a, d_b):
    """Precompute sparse multiplication table for degree-d_a * degree-d_b.

    Returns arrays (idx_a, idx_b, idx_out) such that:
        out[idx_out[k]] += coeffs_a[idx_a[k]] * coeffs_b[idx_b[k]]
    """
    idx_map_a, monoms_a = _monomial_index_map(d_a)
    idx_map_b, monoms_b = _monomial_index_map(d_b)
    idx_map_out, _ = _monomial_index_map(d_a + d_b)

    ia_list, ib_list, io_list = [], [], []
    for i, (a1, b1, c1) in enumerate(monoms_a):
        for j, (a2, b2, c2) in enumerate(monoms_b):
            out_key = (a1 + a2, b1 + b2, c1 + c2)
            ia_list.append(i)
            ib_list.append(j)
            io_list.append(idx_map_out[out_key])
    return np.array(ia_list), np.array(ib_list), np.array(io_list)


@lru_cache(maxsize=256)
def _poly_mult_table_grouped(d_a, d_b):
    """Multiplication table grouped by output index, sorted for batched matmul.

    Returns:
        ia_sorted: (L,) int — input-a indices, sorted by output index
        ib_sorted: (L,) int — input-b indices, sorted by output index
        offsets: (N_out + 1,) int — start positions in ia_sorted/ib_sorted for each output index
                 Block for output o is ia_sorted[offsets[o]:offsets[o+1]].

    Used by `_cross_term_tensor` to compute T_term[o] = Va[ia_block].T @ Vb[ib_block]
    via batched matmul, avoiding the slow `np.add.at` scatter.
    """
    ia, ib, io = _poly_mult_table(d_a, d_b)
    sort_idx = np.argsort(io, kind='stable')
    ia_sorted = ia[sort_idx]
    ib_sorted = ib[sort_idx]
    io_sorted = io[sort_idx]

    _, monoms_out = _monomial_index_map(d_a + d_b)
    N_out = len(monoms_out)
    # offsets[o] = first index in io_sorted where io == o (or where it would be)
    offsets = np.zeros(N_out + 1, dtype=np.int64)
    counts = np.bincount(io_sorted, minlength=N_out)
    offsets[1:] = np.cumsum(counts)
    return ia_sorted, ib_sorted, offsets


def _poly_multiply_coeffs(coeffs_a, d_a, coeffs_b, d_b):
    """Multiply two polynomials in coefficient space. Returns output coefficients."""
    _, monoms_out = _monomial_index_map(d_a + d_b)
    out = np.zeros(len(monoms_out), dtype=np.float64)
    ia, ib, io = _poly_mult_table(d_a, d_b)
    np.add.at(out, io, coeffs_a[ia] * coeffs_b[ib])
    return out


def _compute_Q_coefficients(arr):
    """Compute coefficient vector of Q = prod(linear forms) in degree-n monomial basis."""
    n = len(arr)
    if n == 0:
        return np.ones(1)

    # Start with first linear form (degree 1)
    idx_map_1, _ = _monomial_index_map(1)
    a, b, c = [float(v) for v in arr.lines[0].coords]
    q = np.zeros(3, dtype=np.float64)
    q[idx_map_1[(1, 0, 0)]] = a
    q[idx_map_1[(0, 1, 0)]] = b
    q[idx_map_1[(0, 0, 1)]] = c

    current_deg = 1
    for i in range(1, n):
        a, b, c = [float(v) for v in arr.lines[i].coords]
        lin = np.zeros(3, dtype=np.float64)
        lin[idx_map_1[(1, 0, 0)]] = a
        lin[idx_map_1[(0, 1, 0)]] = b
        lin[idx_map_1[(0, 0, 1)]] = c
        q = _poly_multiply_coeffs(q, current_deg, lin, 1)
        current_deg += 1

    return q


# ─────────────────────────────────────────────────────────────────────────────
# Smooth Saito loss: null space and bilinear tensor
# ─────────────────────────────────────────────────────────────────────────────

def _null_space_basis(M, tol=1e-10, min_extra=0):
    """LEGACY (used only by legacy_invalid_angular_score).
    Compute orthonormal basis for ker(M) via SVD.

    Args:
        M: (rows, cols) matrix
        tol: relative tolerance for determining null vectors
        min_extra: extra near-null right singular vectors to include beyond the
            strict null space. Default 0 (use strict null space). NOTE: setting
            this > 0 breaks the strict Saito derivation condition — the augmented
            vectors are not true derivations. Useful only as a continuation
            heuristic when polishing very close to a free arrangement.

    Returns:
        V: (cols, k) matrix where columns are basis vectors, or None
        k: dimension of returned subspace
    """
    if M.size == 0:
        return None, 0
    try:
        _, s, Vt = np.linalg.svd(M, full_matrices=True)
    except np.linalg.LinAlgError:
        return None, 0

    rows, cols = M.shape
    if len(s) == 0:
        return Vt.T, cols

    threshold = max(tol, s[0] * tol)
    n_significant = int(np.sum(s > threshold))
    k_null = cols - n_significant
    k = min(cols, k_null + min_extra)
    if k == 0:
        return None, 0

    V = Vt[cols - k:].T  # shape (cols, k)
    return V, k


def _build_det_tensor(V2, V3, d1, d2, n):
    """LEGACY (used only by legacy_invalid_angular_score).
    Precompute bilinear tensor T mapping (alpha2, alpha3) -> det coefficients.

    det(Euler, theta1, theta2) = x*(g2*h3 - g3*h2) - y*(f2*h3 - f3*h2) + z*(f2*g3 - f3*g2)

    where theta1 = V2 @ alpha2 = (f2, g2, h2) and theta2 = V3 @ alpha3 = (f3, g3, h3).

    Each cross-term is a bilinear product of degree-d1 and degree-d2 polynomials (degree d1+d2=n-1),
    then multiplied by x, y, or z to give degree n.

    Returns T of shape (N_out, k2, k3) where N_out = C(n+2, 2).
    """
    _, monoms_d1 = _monomial_index_map(d1)
    _, monoms_d2 = _monomial_index_map(d2)
    idx_map_out, monoms_out = _monomial_index_map(n)

    N1 = len(monoms_d1)
    N2 = len(monoms_d2)
    N_out = len(monoms_out)
    k2 = V2.shape[1]
    k3 = V3.shape[1]

    # Extract component sub-matrices from V2 and V3
    # V2 has shape (3*N1, k2): rows [0:N1] = f2, [N1:2*N1] = g2, [2*N1:3*N1] = h2
    V2_f = V2[:N1]       # (N1, k2)
    V2_g = V2[N1:2*N1]   # (N1, k2)
    V2_h = V2[2*N1:]     # (N1, k2)
    V3_f = V3[:N2]       # (N2, k3)
    V3_g = V3[N2:2*N2]   # (N2, k3)
    V3_h = V3[2*N2:]     # (N2, k3)

    # Precompute the multiplication table for d1 * d2 (grouped by output index)
    ia_sorted, ib_sorted, offsets = _poly_mult_table_grouped(d1, d2)
    _, monoms_nm1 = _monomial_index_map(d1 + d2)  # degree n-1
    N_nm1 = len(monoms_nm1)

    # Shift indices for multiplication by x, y, z (degree-1 monomials)
    # x = monomial (1,0,0), y = (0,1,0), z = (0,0,1)
    shift_x = np.array([idx_map_out[(a+1, b, c)] for a, b, c in monoms_nm1])
    shift_y = np.array([idx_map_out[(a, b+1, c)] for a, b, c in monoms_nm1])
    shift_z = np.array([idx_map_out[(a, b, c+1)] for a, b, c in monoms_nm1])

    def _cross_term_tensor(Va, Vb):
        """Compute T_term[o, a, b] = sum_{(i,j): mult table maps to o} Va[i,a] * Vb[j,b].

        Implementation: for each output index o, T_term[o] = Va[ia_block].T @ Vb[ib_block]
        where ia_block, ib_block are the input indices that map to o. Each block becomes
        a single BLAS matmul. For free / near-free arrangements (small null spaces) this is
        ~the same speed as per-entry outer products; for large null spaces it's much faster
        because the per-call Python overhead is amortized over a larger matmul.
        """
        ka, kb = Va.shape[1], Vb.shape[1]
        T_term = np.zeros((N_nm1, ka, kb), dtype=np.float64)
        for o in range(N_nm1):
            start, end = offsets[o], offsets[o + 1]
            if start == end:
                continue
            T_term[o] = Va[ia_sorted[start:end]].T @ Vb[ib_sorted[start:end]]
        return T_term

    # 6 cross terms:
    # det = x*(g2*h3 - g3*h2) - y*(f2*h3 - f3*h2) + z*(f2*g3 - f3*g2)
    T_g2h3 = _cross_term_tensor(V2_g, V3_h)  # (N_nm1, k2, k3)
    T_g3h2 = _cross_term_tensor(V2_h, V3_g)  # note: g3=V3_g, h2=V2_h → this is h2*g3
    T_f2h3 = _cross_term_tensor(V2_f, V3_h)
    T_f3h2 = _cross_term_tensor(V2_h, V3_f)  # h2*f3
    T_f2g3 = _cross_term_tensor(V2_f, V3_g)
    T_f3g2 = _cross_term_tensor(V2_g, V3_f)  # g2*f3

    # Combine: x*(g2h3 - g3h2) - y*(f2h3 - f3h2) + z*(f2g3 - f3g2)
    # Note on sign: "g3*h2" means V3_g @ alpha3 * V2_h @ alpha2, which is
    # bilinear with V2_h on alpha2 side and V3_g on alpha3 side → T_g3h2 above
    T_nm1 = (T_g2h3 - T_g3h2)  # coefficient of x*(...), shape (N_nm1, k2, k3)
    T_nm1_y = -(T_f2h3 - T_f3h2)  # coefficient of -y*(...)
    T_nm1_z = (T_f2g3 - T_f3g2)   # coefficient of z*(...)

    # Shift to degree-n monomials. The shift indices may have collisions across
    # x/y/z (different shifts can land on the same output monomial), so accumulate
    # via Python loop (N_nm1 is small, ~hundreds).
    T = np.zeros((N_out, k2, k3), dtype=np.float64)
    for j in range(N_nm1):
        T[shift_x[j]] += T_nm1[j]
        T[shift_y[j]] += T_nm1_y[j]
        T[shift_z[j]] += T_nm1_z[j]

    return T


# ─────────────────────────────────────────────────────────────────────────────
# Smooth Saito loss: ALS optimization
# ─────────────────────────────────────────────────────────────────────────────

def _als_minimize(T, q, n_iters=10, n_restarts=3, rng=None):
    """LEGACY (used only by legacy_invalid_angular_score).
    Minimize ||D(alpha2, alpha3) - c*q||^2 / ||q||^2 via ALS.

    D_j = sum_{ik} T[j,i,k] * alpha2[i] * alpha3[k]  (bilinear)

    The loss = 1 - cos^2(angle between D and q) = squared sine of angle.

    Returns:
        loss: float in [0, 1], 0 = free arrangement
        best_alpha2, best_alpha3: optimal parameters
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N_out, k2, k3 = T.shape
    q_norm_sq = np.dot(q, q)
    if q_norm_sq < 1e-30:
        return 1.0, None, None

    q_hat = q / np.sqrt(q_norm_sq)  # unit vector

    best_loss = 1.0
    best_a2 = None
    best_a3 = None

    for restart in range(n_restarts):
        # Initialize alpha3 randomly
        alpha3 = rng.standard_normal(k3)
        alpha3 /= np.linalg.norm(alpha3) + 1e-14

        for it in range(n_iters):
            # Fix alpha3, solve for alpha2
            # A[j, i] = sum_k T[j, i, k] * alpha3[k]
            A = np.tensordot(T, alpha3, axes=([2], [0]))  # (N_out, k2)

            # Solve min_{alpha2, c} ||A @ alpha2 - c * q||^2
            # Augmented system: [A | -q] @ [alpha2; c] ≈ 0
            A_aug = np.column_stack([A, -q])  # (N_out, k2 + 1)
            try:
                _, s_aug, Vt_aug = np.linalg.svd(A_aug, full_matrices=True)
            except np.linalg.LinAlgError:
                break
            # Solution = last row of Vt (smallest singular value)
            sol = Vt_aug[-1]
            alpha2 = sol[:k2]
            a2_norm = np.linalg.norm(alpha2)
            if a2_norm < 1e-14:
                break
            alpha2 /= a2_norm

            # Fix alpha2, solve for alpha3
            # B[j, k] = sum_i T[j, i, k] * alpha2[i]
            B = np.tensordot(T, alpha2, axes=([1], [0]))  # (N_out, k3)

            B_aug = np.column_stack([B, -q])  # (N_out, k3 + 1)
            try:
                _, s_aug, Vt_aug = np.linalg.svd(B_aug, full_matrices=True)
            except np.linalg.LinAlgError:
                break
            sol = Vt_aug[-1]
            alpha3 = sol[:k3]
            a3_norm = np.linalg.norm(alpha3)
            if a3_norm < 1e-14:
                break
            alpha3 /= a3_norm

            # Compute current loss: 1 - cos^2(D, q)
            D = np.tensordot(T, alpha2, axes=([1], [0]))  # (N_out, k3)
            D = D @ alpha3  # (N_out,)
            D_norm_sq = np.dot(D, D)
            if D_norm_sq < 1e-30:
                continue
            cos_sq = (np.dot(D, q_hat) ** 2) / D_norm_sq
            loss = 1.0 - min(1.0, cos_sq)

            if loss < best_loss:
                best_loss = loss
                best_a2 = alpha2.copy()
                best_a3 = alpha3.copy()

            if loss < 1e-12:
                return best_loss, best_a2, best_a3

    return best_loss, best_a2, best_a3


def saito_loss(arr, target_exponents=None, lam=DEFAULT_LAMBDA,
               beta=DEFAULT_BETA, profile='search', n_restarts=None,
               seed=0, cached=False):
    """Production Saito loss: the corrected penalized functional.

    Thin wrapper around penalized_saito.penalized_saito_loss.  If
    target_exponents is None, candidate exponents are used when they exist and
    the all-pairs envelope otherwise (the loss is defined for EVERY degree
    pair with d1 + d2 = n - 1; candidate arithmetic never gates it).

    Returns a loss in [0, 1]: ~0 on free arrangements (for the correct
    exponents), strictly inside (0, 1) on nonfree ones.  The value is a
    numerical upper bound on the ideal loss — a search signal, never a
    freeness or nonfreeness certificate.  Use verify_arrangement / is_free
    for exact certification.
    """
    if target_exponents is not None:
        d1, d2 = target_exponents
        n = len(arr)
        if d1 + d2 != n - 1:
            # Degrees only make sense at the target size; during growth the
            # callers use combinatorial signals instead.
            return 1.0
    else:
        d1 = d2 = None
    if cached and isinstance(arr, LineArrangement):
        return cached_penalized_loss(arr, d1=d1, d2=d2, lam=lam, beta=beta,
                                     profile=profile, seed=seed)
    return penalized_saito_loss(arr, d1=d1, d2=d2, lam=lam, beta=beta,
                                profile=profile, n_restarts=n_restarts,
                                seed=seed)


def smooth_saito_loss(arr, target_exponents=None, n_restarts=None, n_iters=None,
                      min_extra=None):
    """DEPRECATED alias: now computes the corrected penalized Saito loss.

    The old ALS angular score this name used to compute is mathematically
    binary in exact arithmetic (see module docstring) and survives only as
    `legacy_invalid_angular_score` for regression comparisons.  The
    n_iters / min_extra arguments of the old implementation are ignored.
    """
    warnings.warn(
        "smooth_saito_loss is deprecated: it now computes the corrected "
        "penalized Saito loss (penalized_saito.py). The old ALS angular "
        "score is available as legacy_invalid_angular_score and is invalid "
        "as a proximity measure.", DeprecationWarning, stacklevel=2)
    return saito_loss(arr, target_exponents=target_exponents,
                      n_restarts=n_restarts)


def legacy_invalid_angular_score(arr, target_exponents=None, n_restarts=10,
                                 n_iters=10, min_extra=0):
    """LEGACY, MATHEMATICALLY INVALID as a proximity measure — regression only.

    This is the old "smooth Saito loss": the ALS angle between Saito
    determinants and Q evaluated over SVD null-space bases of the derivation
    matrices.  In exact arithmetic this quantity is BINARY: for exact
    logarithmic derivations u, v with deg u + deg v = n - 1, every alpha_i
    divides det M(theta_E, u, v) and the degrees match deg Q, so the
    determinant is c*Q (c = 0 unless the arrangement is free).  The exact
    score is therefore 0 on free and 1 on nonfree arrangements; the
    intermediate values this float implementation returns measure SVD
    tolerances, conditioning, and rounding — not proximity to freeness.
    Kept verbatim (including its tolerance-dependent behavior) so that old
    experiments can be reproduced and compared.  Do not use in production.

    Args:
        arr: LineArrangement
        target_exponents: optional (d1, d2) tuple. If provided, use these
            exponents instead of deriving from the arrangement's b2.
        n_restarts: number of random restarts for ALS (default 10).
        n_iters: number of ALS iterations per restart (default 10).
        min_extra: number of "near-null" singular vectors to include beyond the
            strict null space (NOT true derivations when > 0).

    Returns:
        float in [0, 1]; exact-arithmetic value would be binary {0, 1}.
    """
    n = len(arr)
    if n < 3:
        return 1.0

    if target_exponents is not None:
        d1, d2 = target_exponents
    else:
        exps = arr.candidate_exponents()
        if exps is None:
            return 1.0
        d1, d2 = exps

    # Build float derivation matrices
    M_d1 = _float_derivation_matrix(arr, d1)
    M_d2 = M_d1 if d1 == d2 else _float_derivation_matrix(arr, d2)

    # Extract null space bases (with optional augmentation for polish continuity)
    V2, k2 = _null_space_basis(M_d1, min_extra=min_extra)
    if V2 is None or k2 == 0:
        return 1.0

    if d1 == d2:
        V3, k3 = V2, k2
        if k3 < 2:
            return 1.0  # need 2 independent derivations from same space
    else:
        V3, k3 = _null_space_basis(M_d2, min_extra=min_extra)
        if V3 is None or k3 == 0:
            return 1.0

    # Compute Q coefficient vector
    q = _compute_Q_coefficients(arr)
    if np.dot(q, q) < 1e-30:
        return 1.0

    # Build bilinear tensor
    T = _build_det_tensor(V2, V3, d1, d2, n)

    # Optimize via ALS
    loss, _, _ = _als_minimize(T, q, n_iters=n_iters, n_restarts=n_restarts)
    return float(np.clip(loss, 0.0, 1.0))


def algebraic_score(arr: LineArrangement, target_exponents=None,
                    use_legacy: bool = False) -> float:
    """
    Score in [-1, 1] measuring progress toward freeness.

    Two-tier design:
      Tier 1 ([-1, 0]): discriminant proximity — how close is b2 to producing
        the target exponents?  Cheap arithmetic, always computable.
      Tier 2 ([0, 1]): 1 - penalized Saito loss = Gamma_hat, the corrected
        bounded upper-semicontinuous signal (penalized_saito.py).

    Args:
        arr: LineArrangement
        target_exponents: optional (d1, d2) tuple. If provided, Tier 1 measures
            distance from current b2 to target_b2 = (n-1) + d1*d2.
        use_legacy: if True, Tier 2 uses legacy_invalid_angular_score instead
            of the corrected loss.  ONLY for the explicitly-invalid regression
            baseline; never in production.

    Score landscape:
      [-1.0]       product < 0 (b2 too small for any exponents)
      [-1.0, -0.5] disc < 0 (b2 too large, no real roots)
      [-0.5, 0.0)  disc >= 0 but not a perfect square
      [0.0, 1.0]   exponents exist; 1.0 = exactly free
    """
    n = len(arr)
    if n < 3:
        return -1.0

    b2 = arr.b2()

    if target_exponents is not None:
        d1_t, d2_t = target_exponents
        target_n = d1_t + d2_t + 1
        target_b2 = (target_n - 1) + d1_t * d2_t
        # Tier 1: distance from b2 to target_b2
        # Scale by target_b2 itself (not max_b2) for sharper signal on high-b2 targets
        scale = max(1, target_b2)
        distance = abs(b2 - target_b2) / scale
        if distance > 1.0:
            return -1.0
        if distance > 0:
            return -distance  # in [-1, 0)
        # b2 matches target — Tier 2 only when arrangement is at target size
        if n == target_n:
            if use_legacy:
                loss = legacy_invalid_angular_score(
                    arr, target_exponents=target_exponents)
            else:
                loss = saito_loss(arr, target_exponents=target_exponents,
                                  profile='rl', cached=True)
            return 1.0 - loss
        else:
            # Dense positive signal during growth: reward progress toward target size
            return 0.3 * (n / target_n)  # in (0, 0.3) — grows as arrangement builds
    else:
        product = b2 - (n - 1)       # = d1 * d2
        disc = (n - 1) ** 2 - 4 * product

        # ── Tier 1: discriminant proximity → [-1, 0] ────────────────────────

        if product < 0:
            return -1.0

        if disc < 0:
            proximity = max(0.0, 1.0 - abs(disc) / max(1, (n - 1) ** 2))
            return -1.0 + proximity * 0.5  # in [-1.0, -0.5]

        sqrt_disc = disc ** 0.5
        nearest_int = round(sqrt_disc)

        if int(nearest_int) * int(nearest_int) != disc:
            frac = abs(sqrt_disc - nearest_int)  # in (0, 0.5]
            return -frac  # in [-0.5, 0)

        sq = int(nearest_int)
        if (n - 1 - sq) % 2 != 0 or (n - 1 - sq) < 0:
            return -0.01

        # ── Tier 2: penalized Saito loss → [0, 1] ───────────────────────────

        if use_legacy:
            loss = legacy_invalid_angular_score(arr)
        else:
            loss = saito_loss(arr, profile='rl', cached=True)
        return 1.0 - loss  # 0 = far from free, 1 = exactly free


# ─────────────────────────────────────────────────────────────────────────────
# Full reward function
# ─────────────────────────────────────────────────────────────────────────────

def interestingness_score(arr: LineArrangement) -> float:
    """
    Score in [0, 1] measuring how "interesting" the combinatorial structure is.

    Rewards:
      - Multiplicity diversity: varied mult profile > uniform.
      - Multiple high-multiplicity points: several triple+ points are more
        interesting than a single high-mult point.
      - Spread of singularities: not all high-mult points collinear.
      - Non-generic: arrangements where most intersection points are NOT
        simple double points.

    These criteria mirror what algebraic geometers look for in "interesting"
    free arrangements — rich lattice structure, non-trivial combinatorics,
    and singularities in non-degenerate relative position.
    """
    n = len(arr)
    if n < 3:
        return 0.0

    mults = arr.multiplicities()
    if not mults:
        return 0.0

    from collections import Counter

    total_pts = len(mults)

    # 1. Multiplicity diversity (normalized entropy)
    mult_counts = Counter(mults)
    total = sum(mult_counts.values())
    entropy = -sum(
        (c / total) * np.log(c / total + 1e-10)
        for c in mult_counts.values()
    )
    max_entropy = np.log(max(2, len(mult_counts)))
    norm_entropy = float(entropy / (max_entropy + 1e-10)) if max_entropy > 0 else 0.0

    # 2. Multiple high-multiplicity points (triple+)
    n_high = sum(1 for m in mults if m >= 3)
    # Scale: having ~n/3 triple+ points is ideal for free arrangements
    high_target = max(1, n // 3)
    high_ratio = min(1.0, n_high / high_target)

    # 3. Non-generic ratio: fraction of points that are NOT simple doubles
    n_non_double = sum(1 for m in mults if m != 2)
    non_double_ratio = n_non_double / max(1, total_pts)

    # 4. Spread of singularities: check that high-mult points are not
    #    all collinear (i.e., not all on one line of the arrangement).
    #    We measure this as: max fraction of high-mult points on any
    #    single line.  Lower = more spread out = better.
    spread = 1.0
    if n_high >= 2:
        pts = arr._structure()
        high_pts = [pt for pt, lines in pts.items() if len(lines) >= 3]
        if high_pts:
            max_on_line = 0
            for line in arr.lines:
                count = sum(1 for pt in high_pts if line.passes_through(pt))
                max_on_line = max(max_on_line, count)
            spread = 1.0 - max_on_line / max(1, len(high_pts))

    return float(
        0.20 * norm_entropy
        + 0.35 * high_ratio
        + 0.20 * non_double_ratio
        + 0.25 * spread
    )


def multiplicity_penalty(arr: LineArrangement, target_n: int) -> float:
    """
    Penalize near-pencil configurations by measuring how much any single
    intersection point dominates the arrangement.

    For n lines, a "pencil at a point" has multiplicity n (all lines concurrent).
    We penalize multiplicity above the threshold floor(n/2), scaled linearly.

    The penalty is normalized so that a full pencil would give -1.0.
    """
    n = len(arr)
    if n < 3:
        return 0.0
    max_mult = arr.max_multiplicity()
    # Threshold: above n//2 we start penalizing
    threshold = max(2, target_n // 2)
    if max_mult <= threshold:
        return 0.0
    # Linear penalty: reaches -1.0 when max_mult == target_n (full pencil)
    return -(max_mult - threshold) / max(1, target_n - threshold)


def b2_trajectory_bonus(arr: LineArrangement, target_b2: int, target_n: int) -> float:
    """Reward for b2 moving toward target_b2. Returns float in [0, 1].

    Scales distance relative to target_b2 itself (not max_b2), so high-b2
    targets get sharper gradient signal.
    """
    n = len(arr)
    if n < 2:
        return 0.0
    b2 = arr.b2()
    # Scale by target_b2, not max_b2 — being 10 off from b2=100 is
    # much more on-track than 10 off from b2=20
    scale = max(1, target_b2)
    distance = abs(b2 - target_b2) / scale
    return max(0.0, 1.0 - distance)


def saito_reward(
    arr: LineArrangement,
    target_n: int,
    prev_arr=None,
    w_comb: float = 0.3,
    w_alg: float = 0.5,
    w_pencil: float = 5.0,
    w_free: float = 10.0,
    w_mult: float = 2.0,
    w_interest: float = 1.0,
    w_feasibility: float = 0.5,
    w_mult_growth: float = 0.3,
    w_b2_traj: float = 1.5,
    terminal_only_free_bonus: bool = True,
    skip_exact_above: int = 12,
    target_exponents=None,
    use_legacy: bool = False,
    terminal_alg_bonus: bool = True,
) -> float:
    """
    Compute shaped reward for the RL agent.

    Args:
        arr: Current line arrangement.
        target_n: Target number of lines.
        prev_arr: Arrangement before the last line was added (for per-step shaping).
        w_comb: Weight for combinatorial score.
        w_alg: Weight for algebraic (soft) score.
        w_pencil: Penalty for pencil arrangements.
        w_free: Bonus for verified free arrangement.
        w_mult: Penalty weight for near-pencil (high-multiplicity) configurations.
        w_interest: Weight for interestingness bonus (combinatorial richness).
        w_feasibility: Bonus when candidate exponents become feasible.
        w_mult_growth: Bonus when a point's multiplicity grows to 3+.
        w_b2_traj: Weight for b2 trajectory bonus toward target exponents.
        terminal_only_free_bonus: Only give w_free at terminal step (len==target_n).
        skip_exact_above: Skip costly sympy exact check for n > this value during training.
                          Instead give a partial bonus based on algebraic score.
        target_exponents: optional (d1, d2) tuple for exponent-targeted training.

    Returns:
        float reward
    """
    reward = 0.0

    # Pencil penalty (all lines concurrent)
    if arr.is_pencil():
        reward -= w_pencil
        return reward

    # Near-pencil penalty (one point dominates)
    reward += w_mult * multiplicity_penalty(arr, target_n)

    # Combinatorial score
    reward += w_comb * combinatorial_score(arr)

    # Algebraic score (skip entirely at zero weight: avoids paying for the
    # loss evaluation and keeps score-free reward arms from warming the
    # shared loss cache)
    n = len(arr)
    if n >= 3 and w_alg != 0.0:
        reward += w_alg * algebraic_score(arr, target_exponents=target_exponents,
                                          use_legacy=use_legacy)

    # Interestingness bonus (rich singularity structure)
    if n >= 3:
        reward += w_interest * interestingness_score(arr)

    # ── Per-step shaping (dense signal) ──────────────────────────────────────
    if n >= 3:
        # Candidate exponent feasibility: reward staying on a viable path
        has_cand = arr.candidate_exponents() is not None
        if has_cand:
            reward += w_feasibility

        # b2 trajectory bonus: guide toward target b2
        if target_exponents is not None:
            d1_t, d2_t = target_exponents
            target_b2 = (target_n - 1) + d1_t * d2_t
            reward += w_b2_traj * b2_trajectory_bonus(arr, target_b2, target_n)

        # Multiplicity growth: reward creating new triple+ points
        if prev_arr is not None and len(prev_arr) >= 2:
            prev_mults = prev_arr.multiplicities()
            curr_mults = arr.multiplicities()
            prev_high = sum(1 for m in prev_mults if m >= 3)
            curr_high = sum(1 for m in curr_mults if m >= 3)
            if curr_high > prev_high:
                reward += w_mult_growth * (curr_high - prev_high)

    # Terminal freeness bonus
    is_terminal = (n == target_n)
    if is_terminal or not terminal_only_free_bonus:
        if n >= 3 and arr.candidate_exponents() is not None:
            if n <= skip_exact_above:
                # Exact Saito check (tractable for small n)
                is_free, _ = arr.is_free()
                if is_free:
                    reward += w_free
            elif terminal_alg_bonus:
                # Large n: give stronger partial bonus based on algebraic
                # score.  Disabled (terminal_alg_bonus=False) in score-free
                # reward arms so no algebraic signal leaks through the
                # terminal branch.
                alg = algebraic_score(arr, target_exponents=target_exponents,
                                      use_legacy=use_legacy)
                if alg > 0.95:
                    reward += w_free * 0.8 * ((alg - 0.95) / 0.05)
                elif alg > 0.80:
                    reward += w_free * 0.4 * ((alg - 0.80) / 0.15)

    return reward


# ─────────────────────────────────────────────────────────────────────────────
# Post-hoc exact verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_arrangement(arr: LineArrangement):
    """
    Exact freeness check using Saito's criterion (sympy, exact over Q).
    Use this post-hoc on candidates produced by the RL agent.

    Returns:
        (is_free: bool, exponents: tuple or None)
    """
    exps = arr.candidate_exponents()
    if exps is None:
        return False, None
    return arr.is_free()


# ─────────────────────────────────────────────────────────────────────────────
# Continuous polish: L-BFGS-B in line-coefficient space
# ─────────────────────────────────────────────────────────────────────────────

def polish_arrangement(
    arr: LineArrangement,
    target_exponents=None,
    fixed_indices=None,
    max_iter: int = 100,
    tol: float = 1e-12,
    n_restarts_loss: int = 30,
    min_extra: int = 8,
    method: str = 'L-BFGS-B',
    verbose: bool = False,
):
    """
    Continuous polish: starting from `arr`, run L-BFGS-B in coefficient space
    to drive `smooth_saito_loss` to zero, then run the exact `is_free()` check.

    The idea: the RL agent finds arrangements with the right combinatorial type
    (correct b2, valid candidate exponents, low Saito loss). Polish converts
    these "near-misses" into actually-free arrangements via gradient descent
    in the continuous parameter space.

    Args:
        arr: starting LineArrangement (must have valid candidate_exponents
            unless `target_exponents` is provided).
        target_exponents: optional (d1, d2) override.
        fixed_indices: list of line indices to keep fixed during optimization.
            Useful for "extend by one line" mode where you want to lock all
            seed lines and only optimize the new line. Default: fix line 0
            (to break the global gauge).
        max_iter: max L-BFGS iterations.
        tol: stop when smooth loss is below this.
        n_restarts_loss: ALS restarts inside `smooth_saito_loss` (use 30+ for
            polish to get a reliable signal).
        min_extra: extra near-null singular vectors to include in the search
            subspace, making the loss continuous near the free point. Default 8.
        method: scipy.optimize method ('L-BFGS-B' for gradient-based,
            'Nelder-Mead' for gradient-free).
        verbose: print iteration progress.

    Returns:
        dict with keys:
            'success': bool — True if exact `is_free()` returns True after polish.
            'arrangement': polished LineArrangement (with float coords cast to Rational).
            'final_loss': final smooth_saito_loss value.
            'exponents': exact exponents tuple if free, else None.
            'n_iter': number of L-BFGS iterations performed.
    """
    from scipy.optimize import minimize

    n = len(arr)
    if n < 3:
        return {'success': False, 'arrangement': arr, 'final_loss': 1.0,
                'exponents': None, 'n_iter': 0}

    # Determine target exponents
    if target_exponents is None:
        exps = arr.candidate_exponents()
        if exps is None:
            return {'success': False, 'arrangement': arr, 'final_loss': 1.0,
                    'exponents': None, 'n_iter': 0}
        target_exponents = exps

    # Default: fix line 0 to break global scaling/gauge
    if fixed_indices is None:
        fixed_indices = [0]
    fixed_set = set(fixed_indices)
    free_indices = [i for i in range(n) if i not in fixed_set]
    n_free = len(free_indices)

    if n_free == 0:
        # Nothing to optimize — just verify the input directly
        is_free, exps = arr.is_free()
        return {'success': bool(is_free), 'arrangement': arr,
                'final_loss': 0.0 if is_free else 1.0,
                'exponents': exps if is_free else None, 'n_iter': 0}

    # Initial parameters: float coords of free lines, flattened (3 per line)
    fixed_lines = [arr.lines[i] for i in fixed_indices]
    x0 = np.array([
        [float(c) for c in arr.lines[i].coords]
        for i in free_indices
    ], dtype=np.float64).flatten()

    def _build(params):
        """Build LineArrangement from flat parameter vector + fixed lines."""
        free_lines = []
        for k, i in enumerate(free_indices):
            a, b, c = params[3*k], params[3*k+1], params[3*k+2]
            try:
                free_lines.append(ProjectiveLine(
                    Rational(float(a)).limit_denominator(10**12),
                    Rational(float(b)).limit_denominator(10**12),
                    Rational(float(c)).limit_denominator(10**12),
                ))
            except (AssertionError, ValueError):
                # Zero line — return None to signal failure
                return None
        # Reassemble in original order
        all_lines = [None] * n
        for k, i in enumerate(free_indices):
            all_lines[i] = free_lines[k]
        for k, i in enumerate(fixed_indices):
            all_lines[i] = fixed_lines[k]
        return LineArrangement(all_lines)

    def _objective(params):
        candidate = _build(params)
        if candidate is None:
            return 1.0
        try:
            # Penalized loss is defined off the free stratum (no null-space
            # dimension jumps), so the old min_extra continuity hack is
            # unnecessary; min_extra is accepted for signature compatibility
            # and ignored.
            return saito_loss(candidate, target_exponents=target_exponents,
                              profile='search', n_restarts=n_restarts_loss)
        except Exception:
            return 1.0

    iter_count = [0]
    best_loss = [_objective(x0)]
    if verbose:
        print(f"  polish init loss: {best_loss[0]:.6e}")

    def _callback(xk):
        iter_count[0] += 1
        loss = _objective(xk)
        if loss < best_loss[0]:
            best_loss[0] = loss
        if verbose and iter_count[0] % 5 == 0:
            print(f"  polish iter {iter_count[0]}: loss={loss:.6e}")

    minimize_kwargs = {
        'method': method,
        'options': {'maxiter': max_iter},
        'callback': _callback,
    }
    if method == 'L-BFGS-B':
        minimize_kwargs['jac'] = '2-point'
        minimize_kwargs['options'].update({'ftol': tol, 'gtol': tol, 'eps': 1e-6})
    elif method == 'Nelder-Mead':
        minimize_kwargs['options'].update({'xatol': 1e-8, 'fatol': tol, 'adaptive': True})

    result = minimize(_objective, x0, **minimize_kwargs)

    final_arr = _build(result.x)
    if final_arr is None:
        return {'success': False, 'arrangement': arr, 'final_loss': 1.0,
                'exponents': None, 'n_iter': iter_count[0],
                'has_cand_exps': False, 'b2': None, 'rationalized': False}

    final_loss = float(result.fun)
    if verbose:
        print(f"  polish final loss: {final_loss:.6e}")

    # Rationalization sweep: float polish gives a numerically near-free arrangement,
    # but the exact b2 may be wrong because float coordinates don't preserve the exact
    # triple-point coincidences. Try rounding free-line coordinates to nearby rationals
    # at increasing denominator bounds. The smallest denominator that yields an exactly
    # free arrangement (b2 correct + sympy is_free) is the target.
    rationalized = False
    best_arr = final_arr
    best_loss = final_loss
    is_free = False
    exps = None

    if final_loss < 0.01:  # only try rationalization for near-free results
        for max_denom in [3, 5, 10, 30, 100, 300, 1000, 3000, 10000]:
            try:
                rounded_lines = []
                for k, i in enumerate(free_indices):
                    a, b, c = result.x[3*k], result.x[3*k+1], result.x[3*k+2]
                    rounded_lines.append(ProjectiveLine(
                        Rational(float(a)).limit_denominator(max_denom),
                        Rational(float(b)).limit_denominator(max_denom),
                        Rational(float(c)).limit_denominator(max_denom),
                    ))
                all_lines = [None] * n
                for k, i in enumerate(free_indices):
                    all_lines[i] = rounded_lines[k]
                for k, i in enumerate(fixed_indices):
                    all_lines[i] = fixed_lines[k]
                test_arr = LineArrangement(all_lines)
            except Exception:
                continue
            if test_arr.candidate_exponents() is None:
                continue
            try:
                free_check, exps_check = test_arr.is_free()
            except Exception:
                continue
            if free_check:
                is_free = True
                exps = exps_check
                best_arr = test_arr
                best_loss = 0.0
                rationalized = True
                if verbose:
                    print(f"  rationalized at denom={max_denom}: free!")
                break

    # If no rationalization worked, still try the exact final_arr (already rationalized at 1e12)
    if not is_free and final_arr.candidate_exponents() is not None:
        try:
            is_free_orig, exps_orig = final_arr.is_free()
            if is_free_orig:
                is_free = True
                exps = exps_orig
                best_arr = final_arr
                best_loss = 0.0
        except Exception:
            pass

    try:
        b2 = best_arr.b2()
    except Exception:
        b2 = None
    has_cand_exps = best_arr.candidate_exponents() is not None

    return {
        'success': bool(is_free),
        'arrangement': best_arr,
        'final_loss': best_loss,
        'exponents': exps,
        'n_iter': iter_count[0],
        'has_cand_exps': has_cand_exps,
        'b2': b2,
        'rationalized': rationalized,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Extend by one line: enumerate candidates and verify
# ─────────────────────────────────────────────────────────────────────────────

def _enumerate_extension_candidates(arr, coord_range=5, include_singularity=True,
                                      max_denominator=1):
    """Generate candidate lines for extending `arr` by one line.

    Combines three strategies:
    1. Lines through pairs of existing intersection points (singularity-driven).
       These are the most "structural" choices because they preserve intersection
       coincidences.
    2. Lines from the small-integer pool (a, b, c) in [-coord_range, coord_range].
    3. Lines through one existing intersection point with a rational direction
       (a/d, b/d, c/d) for d up to `max_denominator`. This adds rational lines
       with controlled denominators that pass through existing structure.

    Args:
        arr: seed arrangement
        coord_range: integer pool range
        include_singularity: enable strategy 1
        max_denominator: enable strategy 3 if > 1; cap on denominator

    Returns:
        List of ProjectiveLine objects, deduplicated and excluding lines already in arr.
    """
    existing = set(l.coords for l in arr.lines)
    seen = set()
    result = []

    if include_singularity:
        from environment import _singularity_candidates
        sing = _singularity_candidates(arr)
        for score, line in sing:
            if line.coords in existing or line.coords in seen:
                continue
            seen.add(line.coords)
            result.append(line)

    # Pool: small integer coordinates
    r = coord_range
    for a in range(-r, r + 1):
        for b in range(-r, r + 1):
            for c in range(-r, r + 1):
                if a == 0 and b == 0 and c == 0:
                    continue
                try:
                    line = ProjectiveLine(a, b, c)
                except AssertionError:
                    continue
                if line.coords in existing or line.coords in seen:
                    continue
                seen.add(line.coords)
                result.append(line)

    # Lines through ONE existing intersection point, with integer direction
    # Each such line has form: through point P with direction D = (dx, dy, dz)
    # Parametrize: line normal = P × D (cross product). The new line passes
    # through P (an existing multiple point) and has small-integer normal coords.
    if max_denominator > 1:
        from arrangement import ProjectiveLine as PL
        pts = arr._structure() if len(arr) >= 2 else {}
        # Use only multiple points (multiplicity >= 2)
        mult_pts = [pt for pt, lines_through in pts.items() if len(lines_through) >= 2]
        for pt in mult_pts:
            px, py, pz = [float(c) for c in pt]
            for dx in range(-coord_range, coord_range + 1):
                for dy in range(-coord_range, coord_range + 1):
                    for dz in range(-coord_range, coord_range + 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        # New line has normal = pt × direction
                        nx = py * dz - pz * dy
                        ny = pz * dx - px * dz
                        nz = px * dy - py * dx
                        if abs(nx) < 1e-12 and abs(ny) < 1e-12 and abs(nz) < 1e-12:
                            continue
                        try:
                            from sympy import Rational
                            new_line = PL(
                                Rational(float(nx)).limit_denominator(max_denominator * 100),
                                Rational(float(ny)).limit_denominator(max_denominator * 100),
                                Rational(float(nz)).limit_denominator(max_denominator * 100),
                            )
                        except (AssertionError, ValueError):
                            continue
                        if new_line.coords in existing or new_line.coords in seen:
                            continue
                        seen.add(new_line.coords)
                        result.append(new_line)

    return result


def extend_arrangement(
    seed_arr: LineArrangement,
    coord_range: int = 5,
    loss_threshold: float = 1e-6,
    n_restarts: int = 10,
    max_denominator: int = 1,
    verbose: bool = False,
    target_exponents=None,
):
    """Extend a free arrangement by one line, returning all valid extensions.

    For each candidate line L (from singularity + small-integer pool):
        1. Build new_arr = seed_arr + [L]
        2. Quick reject: skip if new_arr.candidate_exponents() is None
        3. Pre-filter: skip if the penalized Saito loss > loss_threshold
        4. Verify exactly via new_arr.is_free()
        5. If free, record it.

    Args:
        seed_arr: known free arrangement to extend.
        coord_range: integer pool range for the new line.
        loss_threshold: HEURISTIC pre-filter threshold (skip exact check if
            loss above this).  The default was refit on a validation
            benchmark for the penalized loss (see results_penalized_saito/,
            incl. the n>=14 recall study), but the computed loss is an upper
            bound from a finite multistart: the threshold trades exact-check
            work against recall and certifies nothing either way.
        n_restarts: optimizer restarts in the loss pre-filter (default 10).
        verbose: print per-candidate diagnostics.
        target_exponents: optional (d1, d2) override for the loss filter.

    Returns:
        List of dicts, one per successful extension:
            {'arrangement': new free LineArrangement,
             'exponents': (1, d1, d2),
             'new_line': the added ProjectiveLine,
             'loss': pre-filter loss (≈0)}
    """
    candidates = _enumerate_extension_candidates(
        seed_arr, coord_range=coord_range, max_denominator=max_denominator)
    n_seed = len(seed_arr)
    if verbose:
        print(f"  {n_seed} seed lines, {len(candidates)} candidate extensions")

    successes = []
    n_passed_filter = 0
    n_passed_combinatorial = 0

    for idx, line in enumerate(candidates):
        new_arr = LineArrangement(list(seed_arr.lines) + [line])

        # Cheap combinatorial pre-filter: candidate exponents must exist
        cand_exps = new_arr.candidate_exponents()
        if cand_exps is None:
            continue
        n_passed_combinatorial += 1

        tgt = target_exponents if target_exponents is not None else cand_exps

        # Penalized loss pre-filter
        try:
            loss = saito_loss(new_arr, target_exponents=tgt,
                              profile='search', n_restarts=n_restarts)
        except Exception:
            continue
        if loss > loss_threshold:
            continue
        n_passed_filter += 1

        # Exact verification
        try:
            is_free, exps = new_arr.is_free()
        except Exception:
            continue
        if not is_free:
            continue

        successes.append({
            'arrangement': new_arr,
            'exponents': exps,
            'new_line': line,
            'loss': float(loss),
        })
        if verbose:
            print(f"  [{idx}/{len(candidates)}] FREE: line={line}, exps={exps}, loss={loss:.2e}")

    if verbose:
        print(f"  combinatorial: {n_passed_combinatorial}, "
              f"loss-filter passed: {n_passed_filter}, "
              f"exact-free: {len(successes)}")
    return successes


# ─────────────────────────────────────────────────────────────────────────────
# Δb2 prediction and targeted extension
# ─────────────────────────────────────────────────────────────────────────────

def _line_score_and_count(line, pts):
    """For a candidate line L and existing intersection points dict (point -> set of line indices),
    return (S, k) where:
        S = sum of multiplicities of existing points that L passes through
        k = number of distinct existing points that L passes through

    Used by extend_arrangement_targeted to predict Δb2 = n + k - S without
    actually building the new arrangement.
    """
    S = 0
    k = 0
    for pt, lines_through in pts.items():
        if line.passes_through(pt):
            S += len(lines_through)
            k += 1
    return S, k


def predicted_delta_b2(line, seed_arr):
    """Predict the change in b2 if `line` is added to `seed_arr`.

    Δb2 = n_seed + k - S, where S is the multiplicity sum and k is the
    number of existing intersection points L passes through.
    """
    n_seed = len(seed_arr)
    if n_seed < 2:
        return 0
    pts = seed_arr._structure()
    S, k = _line_score_and_count(line, pts)
    return n_seed + k - S


def extend_arrangement_targeted(
    seed_arr: LineArrangement,
    target_exponents,
    coord_range: int = 5,
    loss_threshold: float = 1e-6,
    n_restarts: int = 10,
    max_denominator: int = 1,
    verbose: bool = False,
):
    """Extend a free arrangement by one line, targeting a specific (d1', d2') for n+1.

    Differs from extend_arrangement in two ways:
      1. Pre-filters candidates by computing Δb2 = n + k - S and accepting only
         those whose Δb2 matches the target.
      2. Adapts the candidate pool to the required Δb2:
         - small Δb2 (target line through many points) -> singularity-driven
         - large Δb2 (target line avoids existing structure) -> small-integer pool
         - intermediate -> both

    Args:
        seed_arr: known free arrangement to extend.
        target_exponents: (d1', d2') for the n+1-line result.
        coord_range, loss_threshold, n_restarts, max_denominator, verbose: same as
            extend_arrangement.

    Returns:
        List of dicts (same format as extend_arrangement). Each dict has the
        new free arrangement, exponents, the added line, and the smooth loss
        value at filter time.
    """
    n_seed = len(seed_arr)
    b2_seed = seed_arr.b2()
    d1_t, d2_t = target_exponents
    b2_target = n_seed + d1_t * d2_t
    delta_required = b2_target - b2_seed

    if verbose:
        print(f"  seed: n={n_seed}, b2={b2_seed}; target exps=(1,{d1_t},{d2_t}), "
              f"b2_target={b2_target}, Δb2_required={delta_required}")

    # Sanity bounds: a single new line can change b2 by at most n+1 (k=1, S=0)
    # and at least 1 (k=0, S=0 only if the line meets nothing — impossible since
    # it always meets the n existing lines somewhere, but those somewheres can
    # land on existing points). Strict bound: 1 ≤ Δb2 ≤ n+1.
    if delta_required < 1 or delta_required > n_seed + 1:
        if verbose:
            print(f"  Δb2_required={delta_required} is out of range [1, {n_seed+1}] — impossible")
        return []

    # Decide which candidate strategies to use based on Δb2 regime
    # Small Δb2 -> need lines through many existing points (singularity-driven)
    # Large Δb2 -> need lines through few existing points (pool, large coord_range)
    use_singularity = delta_required <= n_seed - 2  # leaves room for k >= 1
    # For large Δb2, expand the integer pool to find more "generic" lines
    if delta_required >= n_seed - 1:
        # Want lines that avoid existing structure entirely
        candidates = _enumerate_extension_candidates(
            seed_arr, coord_range=max(coord_range, 10),
            include_singularity=False, max_denominator=1)
    else:
        candidates = _enumerate_extension_candidates(
            seed_arr, coord_range=coord_range,
            include_singularity=use_singularity,
            max_denominator=max_denominator)

    if verbose:
        print(f"  enumerated {len(candidates)} candidates "
              f"(singularity={use_singularity}, expanded_pool={delta_required >= n_seed - 1})")

    pts = seed_arr._structure() if n_seed >= 2 else {}

    # Δb2 pre-filter
    matched = []
    for line in candidates:
        S, k = _line_score_and_count(line, pts)
        delta = n_seed + k - S
        if delta == delta_required:
            matched.append(line)

    if verbose:
        print(f"  Δb2-filtered: {len(matched)}/{len(candidates)} match Δb2={delta_required}")

    successes = []
    for idx, line in enumerate(matched):
        new_arr = LineArrangement(list(seed_arr.lines) + [line])

        # Combinatorial check (must give the target exponents)
        cand_exps = new_arr.candidate_exponents()
        if cand_exps is None or cand_exps != target_exponents:
            continue

        # Penalized loss pre-filter
        try:
            loss = saito_loss(new_arr, target_exponents=target_exponents,
                              profile='search', n_restarts=n_restarts)
        except Exception:
            continue
        if loss > loss_threshold:
            continue

        # Exact verification
        try:
            is_free, exps = new_arr.is_free()
        except Exception:
            continue
        if not is_free:
            continue

        successes.append({
            'arrangement': new_arr,
            'exponents': exps,
            'new_line': line,
            'loss': float(loss),
        })
        if verbose:
            print(f"  [{idx}/{len(matched)}] FREE: line={line}, exps={exps}, loss={loss:.2e}")

    if verbose:
        print(f"  exact-free: {len(successes)}")
    return successes


# ─────────────────────────────────────────────────────────────────────────────
# Direct constructions for known free families
# ─────────────────────────────────────────────────────────────────────────────

def construct_near_pencil(n: int) -> LineArrangement:
    """Construct a near-pencil arrangement: n-1 lines through a single point + 1 transversal.

    The result is a free arrangement with exponents (1, 1, n-2). This is one of the
    classical free arrangement families and is trivially free by closed-form construction
    (no search or polish needed).

    The pencil apex is the point [0:0:1] (i.e., the point at z-infinity in chart z=1).
    The pencil consists of n-1 lines through that apex, parameterized as
    `i*x + j*y = 0` for distinct (i, j) pairs. The transversal is `z = 0`.

    Args:
        n: total number of lines (must be >= 3).

    Returns:
        LineArrangement with n lines, free with exponents (1, 1, n-2).
    """
    if n < 3:
        raise ValueError(f"near-pencil requires n >= 3, got {n}")
    # Apex at [0:0:1]: lines through it have c = 0, i.e. ax + by = 0.
    # Generate n-1 such lines with distinct (a:b) ratios using small integers.
    # We pick (1,0), (0,1), then (i, 1) for i = 1, 2, ..., -1, -2, ... and
    # (1, j) for j = 2, 3, ... as needed. This ensures projective distinctness.
    lines = []
    seen = set()

    def _add(a, b, c):
        try:
            line = ProjectiveLine(a, b, c)
        except (AssertionError, ValueError):
            return False
        if line.coords in seen:
            return False
        seen.add(line.coords)
        lines.append(line)
        return True

    # Always start with the two coordinate lines through [0:0:1]
    _add(1, 0, 0)
    _add(0, 1, 0)
    # Then add lines a*x + b*y = 0 with small a, b
    for s in range(1, 100):  # Spiral outward in size
        for a in range(-s, s + 1):
            for b in range(-s, s + 1):
                if abs(a) != s and abs(b) != s:
                    continue
                if a == 0 and b == 0:
                    continue
                if len(lines) >= n - 1:
                    break
                _add(a, b, 0)
            if len(lines) >= n - 1:
                break
        if len(lines) >= n - 1:
            break

    if len(lines) < n - 1:
        raise RuntimeError(f"Could not generate {n-1} pencil lines")

    # Add the transversal z = 0
    _add(0, 0, 1)
    return LineArrangement(lines)


def construct_supersolvable(n: int, d1: int) -> LineArrangement:
    """Construct a supersolvable free arrangement with exponents (1, d1, n-1-d1).

    Construction: two pencils sharing one common line.
      - Pencil 1 centered at [0:0:1] (lines of form a*x + b*y = 0): contains (d1 + 1) lines
      - Pencil 2 centered at [0:1:0] (lines of form a*x + c*z = 0): contains (n - d1) lines
      - The shared line is x = 0 (which lies in both pencils since (1,0,0) has both b=0 and c=0)
      - Total: (d1 + 1) + (n - d1) - 1 = n lines

    By Terao's supersolvability theorem, this arrangement is free with exponents
    (1, d1, n - 1 - d1). The smaller pencil contributes the d1 exponent.

    Args:
        n: total number of lines.
        d1: smaller exponent (1 ≤ d1 ≤ (n-1)//2). If d1=1 this gives (1,1,n-2)
            (same as construct_near_pencil but via the supersolvable construction).

    Returns:
        LineArrangement with n lines, free with exponents (1, d1, n - 1 - d1).
    """
    if d1 < 1 or d1 > n - 1 - d1:
        raise ValueError(f"need 1 <= d1 <= (n-1)//2, got d1={d1}, n={n}")

    n_pencil_1 = d1 + 1   # contributes d1 exponent
    n_pencil_2 = n - d1   # contributes (n - 1 - d1) exponent

    lines = []
    seen = set()

    def _add(a, b, c):
        try:
            line = ProjectiveLine(a, b, c)
        except (AssertionError, ValueError):
            return False
        if line.coords in seen:
            return False
        seen.add(line.coords)
        lines.append(line)
        return True

    # Shared line: x = 0  (a=1, b=0, c=0). It is in both pencils.
    _add(1, 0, 0)

    # Pencil 1 (through [0:0:1]): a*x + b*y = 0 with b ≠ 0 (else duplicates x=0)
    # Need (n_pencil_1 - 1) more lines (the shared one is already in).
    # Use small integer (a, b) ratios.
    target_p1 = n_pencil_1 - 1
    count = 0
    s = 1
    while count < target_p1 and s < 1000:
        for a in range(-s, s + 1):
            for b in range(-s, s + 1):
                if abs(a) != s and abs(b) != s:
                    continue
                if b == 0:  # exclude x=0 (already added)
                    continue
                if _add(a, b, 0):
                    count += 1
                    if count >= target_p1:
                        break
            if count >= target_p1:
                break
        s += 1

    # Pencil 2 (through [0:1:0]): a*x + c*z = 0 with c ≠ 0 (else duplicates x=0)
    # Need (n_pencil_2 - 1) more lines.
    target_p2 = n_pencil_2 - 1
    count = 0
    s = 1
    while count < target_p2 and s < 1000:
        for a in range(-s, s + 1):
            for c in range(-s, s + 1):
                if abs(a) != s and abs(c) != s:
                    continue
                if c == 0:  # exclude x=0 (already added)
                    continue
                if _add(a, 0, c):
                    count += 1
                    if count >= target_p2:
                        break
            if count >= target_p2:
                break
        s += 1

    if len(lines) != n:
        raise RuntimeError(f"Built {len(lines)} lines, expected {n}")
    return LineArrangement(lines)


