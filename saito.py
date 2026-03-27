"""
saito.py

Continuous Saito loss / reward shaping for the RL agent.

Three levels of signal:
  1. Combinatorial: Does b2(A) give integer candidate exponents?
     disc = (n-1)^2 - 4*(b2-(n-1)) must be >= 0 and a perfect square.

  2. Algebraic (smooth): For candidate exponents (d2, d3), search over the
     full null spaces ker(M_d2), ker(M_d3) to find derivations theta2, theta3
     minimizing ||det(Euler, theta2, theta3) - c*Q||^2 / ||Q||^2.
     Solved via Alternating Least Squares in coefficient space.

  3. Algebraic (hard): Does a Saito basis exist?
     (Only used for exact verification at episode end.)

The reward function R(A) returned to the RL agent is:
  R(A) = w_comb * score_comb(A)
       + w_alg  * score_alg(A)
       - w_pen  * is_pencil(A)
       + w_free * is_free_exact(A)  [terminal bonus]

where each score is in [-1, 1].
"""

import numpy as np
import sympy as sp
from sympy import Rational, Matrix
from arrangement import LineArrangement, ProjectiveLine


# ─────────────────────────────────────────────────────────────────────────────
# Combinatorial score
# ─────────────────────────────────────────────────────────────────────────────

def combinatorial_score(arr: LineArrangement) -> float:
    """
    Continuous score in [-1, 1] measuring how close b2(A) is to
    yielding integer candidate exponents.

    The discriminant for candidate exponents (1, d2, d3) is:
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

def _null_space_basis(M, tol=1e-10):
    """Compute orthonormal basis for ker(M) via SVD.

    Returns:
        V: (dim, k) matrix where columns are null-space basis vectors, or None
        k: dimension of null space
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
    # Number of singular values at or above threshold
    n_significant = int(np.sum(s > threshold))
    # Null space = columns of V corresponding to zero singular values
    # In Vt (shape cols x cols if full_matrices=True), rows n_significant..cols-1 are null space
    k = cols - n_significant
    if k == 0:
        return None, 0

    # Null space basis: last k rows of Vt, transposed to columns
    V = Vt[n_significant:].T  # shape (cols, k)
    return V, k


def _build_det_tensor(V2, V3, d2, d3, n):
    """Precompute bilinear tensor T mapping (alpha2, alpha3) -> det coefficients.

    det(Euler, theta2, theta3) = x*(g2*h3 - g3*h2) - y*(f2*h3 - f3*h2) + z*(f2*g3 - f3*g2)

    where theta2 = V2 @ alpha2 = (f2, g2, h2) and theta3 = V3 @ alpha3 = (f3, g3, h3).

    Each cross-term is a bilinear product of degree-d2 and degree-d3 polynomials (degree d2+d3=n-1),
    then multiplied by x, y, or z to give degree n.

    Returns T of shape (N_out, k2, k3) where N_out = C(n+2, 2).
    """
    _, monoms_d2 = _monomial_index_map(d2)
    _, monoms_d3 = _monomial_index_map(d3)
    idx_map_out, monoms_out = _monomial_index_map(n)

    N2 = len(monoms_d2)
    N3 = len(monoms_d3)
    N_out = len(monoms_out)
    k2 = V2.shape[1]
    k3 = V3.shape[1]

    # Extract component sub-matrices from V2 and V3
    # V2 has shape (3*N2, k2): rows [0:N2] = f2, [N2:2*N2] = g2, [2*N2:3*N2] = h2
    V2_f = V2[:N2]       # (N2, k2)
    V2_g = V2[N2:2*N2]   # (N2, k2)
    V2_h = V2[2*N2:]     # (N2, k2)
    V3_f = V3[:N3]       # (N3, k3)
    V3_g = V3[N3:2*N3]   # (N3, k3)
    V3_h = V3[2*N3:]     # (N3, k3)

    # Precompute the multiplication table for d2 * d3
    ia, ib, io = _poly_mult_table(d2, d3)
    _, monoms_nm1 = _monomial_index_map(d2 + d3)  # degree n-1
    N_nm1 = len(monoms_nm1)

    # Shift indices for multiplication by x, y, z (degree-1 monomials)
    # x = monomial (1,0,0), y = (0,1,0), z = (0,0,1)
    shift_x = np.array([idx_map_out[(a+1, b, c)] for a, b, c in monoms_nm1])
    shift_y = np.array([idx_map_out[(a, b+1, c)] for a, b, c in monoms_nm1])
    shift_z = np.array([idx_map_out[(a, b, c+1)] for a, b, c in monoms_nm1])

    def _cross_term_tensor(Va, Vb):
        """Compute tensor for one bilinear product term Va[ia]*Vb[ib] in coefficient space.

        Returns shape (N_nm1, ka, kb).
        """
        # Va: (N_d2, ka), Vb: (N_d3, kb)
        ka, kb = Va.shape[1], Vb.shape[1]
        T_term = np.zeros((N_nm1, ka, kb), dtype=np.float64)
        # For each multiplication table entry: out[io[l]] += Va[ia[l]] * Vb[ib[l]]
        for l in range(len(ia)):
            # outer product of Va[ia[l], :] and Vb[ib[l], :]
            T_term[io[l]] += np.outer(Va[ia[l]], Vb[ib[l]])
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

    # Shift to degree-n monomials
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
    """Minimize ||D(alpha2, alpha3) - c*q||^2 / ||q||^2 via ALS.

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


def smooth_saito_loss(arr):
    """Compute smooth Saito loss for a line arrangement.

    Searches over the full null spaces ker(M_d2), ker(M_d3) to find
    derivations theta2, theta3 minimizing:
        ||det(Euler, theta2, theta3) - c*Q||^2 / ||Q||^2

    Returns:
        loss: float in [0, 1], where 0 = free arrangement, 1 = far from free
    """
    n = len(arr)
    if n < 3:
        return 1.0

    exps = arr.candidate_exponents()
    if exps is None:
        return 1.0

    d2, d3 = exps

    # Build float derivation matrices
    M_d2 = _float_derivation_matrix(arr, d2)
    M_d3 = M_d2 if d2 == d3 else _float_derivation_matrix(arr, d3)

    # Extract full null space bases
    V2, k2 = _null_space_basis(M_d2)
    if V2 is None or k2 == 0:
        return 1.0

    if d2 == d3:
        V3, k3 = V2, k2
        if k3 < 2:
            return 1.0  # need 2 independent derivations from same space
    else:
        V3, k3 = _null_space_basis(M_d3)
        if V3 is None or k3 == 0:
            return 1.0

    # Compute Q coefficient vector
    q = _compute_Q_coefficients(arr)
    if np.dot(q, q) < 1e-30:
        return 1.0

    # Build bilinear tensor
    T = _build_det_tensor(V2, V3, d2, d3, n)

    # Optimize via ALS
    loss, _, _ = _als_minimize(T, q, n_iters=10, n_restarts=3)
    return float(np.clip(loss, 0.0, 1.0))


def algebraic_score(arr: LineArrangement) -> float:
    """
    Continuous score in [-1, 1] measuring progress toward freeness.

    Two-tier design:
      Tier 1 ([-1, 0]): discriminant proximity — how close is b2 to producing
        integer exponents?  Cheap arithmetic, always computable.
      Tier 2 ([0, 1]): smooth Saito loss — searches over full null spaces via
        ALS to find optimal derivations minimizing ||det - c*Q||/||Q||.

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
    product = b2 - (n - 1)       # = d2 * d3
    disc = (n - 1) ** 2 - 4 * product

    # ── Tier 1: discriminant proximity → [-1, 0] ────────────────────────────

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

    # ── Tier 2: smooth Saito loss → [0, 1] ──────────────────────────────────

    loss = smooth_saito_loss(arr)
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
    terminal_only_free_bonus: bool = True,
    skip_exact_above: int = 12,
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
        terminal_only_free_bonus: Only give w_free at terminal step (len==target_n).
        skip_exact_above: Skip costly sympy exact check for n > this value during training.
                          Instead give a partial bonus based on algebraic score.

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

    # Algebraic score
    n = len(arr)
    if n >= 3:
        reward += w_alg * algebraic_score(arr)

    # Interestingness bonus (rich singularity structure)
    if n >= 3:
        reward += w_interest * interestingness_score(arr)

    # ── Per-step shaping (dense signal) ──────────────────────────────────────
    if n >= 3:
        # Candidate exponent feasibility: reward staying on a viable path
        has_cand = arr.candidate_exponents() is not None
        if has_cand:
            reward += w_feasibility

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
            else:
                # Large n: give stronger partial bonus based on algebraic score
                alg = algebraic_score(arr)
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


