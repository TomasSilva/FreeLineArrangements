"""
Unit and mathematical regression tests for the penalized Saito functional.

Covers the 17 required categories of the migration spec (§6); the numbering
below matches that list.
"""

import numpy as np
import pytest
import sympy as sp
from sympy import Matrix, Rational, symbols

import saito
from arrangement import LineArrangement, ProjectiveLine
from certificates import (find_exact_saito_certificate, verify_certificate,
                          certificate_to_bw_vectors, certificate_to_json,
                          certificate_from_json)
from penalized_saito import (PenalizedSaitoEvaluator, penalized_saito_loss,
                             penalized_saito_loss_all_pairs,
                             kernel_diagnostics, admissible_degree_pairs)
from reference_impl import gamma_reference

x, y, z = symbols('x y z')


def lines_matrix(arr):
    return np.array([l.to_float() for l in arr.lines])


def loss(arr, d1, d2, **kw):
    kw.setdefault("profile", "search")
    return penalized_saito_loss(arr, d1, d2, **kw)


# ─── 1. ker(L_{A,d}) agrees with the exact logarithmic tangency equations ────

@pytest.mark.parametrize("fixture,d", [
    ("braid", 2), ("braid", 3), ("a2xa1", 1), ("a2xa1", 2),
    ("nonfree7", 3), ("pencil4", 0), ("pencil4", 3),
])
def test_kernel_dim_agrees_with_exact(request, fixture, d):
    arr = request.getfixturevalue(fixture)
    exact_dim = arr.derivation_space_dim(d)
    diag = kernel_diagnostics(arr, d)
    assert diag["numerical_kernel_dim"] == exact_dim


def test_exact_nullvectors_lie_in_ker_L(braid):
    ev = PenalizedSaitoEvaluator(braid, 2, 3)
    for d, L in ((2, ev.L1), (3, ev.L2)):
        null = braid.derivation_matrix(d).nullspace()
        assert null
        from penalized_saito import _bw_sqrt_weights, _monoms
        N = len(_monoms(d))
        sw = _bw_sqrt_weights(d)
        for v in null:
            c = np.array([float(t) for t in v], dtype=np.float64)
            w = np.concatenate([c[:N] / sw, c[N:2 * N] / sw, c[2 * N:] / sw])
            w /= np.linalg.norm(w)
            assert np.linalg.norm(L @ w) < 1e-12


# ─── 2. known free arrangements: loss ~ 0 for correct exponents ──────────────

@pytest.mark.parametrize("fixture,pair", [
    ("braid", (2, 3)), ("a2xa1", (1, 2)), ("pencil4", (0, 3)),
])
def test_free_loss_near_zero(request, fixture, pair):
    arr = request.getfixturevalue(fixture)
    assert loss(arr, *pair) < 1e-8


# ─── 3. the exact symbolic Saito pair gives Gamma = 1 ────────────────────────

@pytest.mark.parametrize("fixture", ["braid", "a2xa1", "pencil4"])
def test_exact_pair_gamma_one(request, fixture):
    arr = request.getfixturevalue(fixture)
    cert = find_exact_saito_certificate(arr)
    assert cert is not None
    assert verify_certificate(cert)
    u, v = certificate_to_bw_vectors(cert)
    ev = PenalizedSaitoEvaluator(arr, cert["d1"], cert["d2"])
    g, parts = ev.gamma(u, v, return_parts=True)
    assert g > 1 - 1e-10
    assert parts["residual_R"] < 1e-24


def test_certificate_json_roundtrip(braid):
    cert = find_exact_saito_certificate(braid)
    cert2 = certificate_from_json(certificate_to_json(cert))
    assert verify_certificate(cert2)


# ─── 4. known nonfree arrangements: strictly inside (0, 1) ───────────────────

@pytest.mark.parametrize("fixture,pair", [
    ("nonfree7", (3, 3)), ("nonfree7b", (3, 3)),
    ("generic4", (1, 2)), ("generic4", (0, 3)),
])
def test_nonfree_strictly_interior(request, fixture, pair):
    arr = request.getfixturevalue(fixture)
    val = loss(arr, *pair)
    assert 1e-6 < val < 1 - 1e-6


# ─── 5. free arrangement + wrong degree pair: not zero ───────────────────────

def test_wrong_pair_not_zero(braid):
    assert loss(braid, 1, 4) > 1e-2
    assert loss(braid, 0, 5) > 1e-2


# ─── 6. all-pairs envelope vanishes on free arrangements ─────────────────────

@pytest.mark.parametrize("fixture", ["braid", "a2xa1", "pencil4"])
def test_all_pairs_envelope_free(request, fixture):
    arr = request.getfixturevalue(fixture)
    val, det = penalized_saito_loss_all_pairs(arr, profile="search",
                                              return_details=True)
    assert val < 1e-8
    # envelope must include d1 = 0 (pencils / nonessential arrangements)
    assert (0, len(arr) - 1) in det["per_pair"]


# ─── 7. line rescaling and complex phase invariance ──────────────────────────

def test_line_rescaling_invariance(nonfree7):
    m = lines_matrix(nonfree7)
    scales = np.array([2.0, -3.0, 7.0, 0.5, -1.0, 11.0, 1 / 3])
    l0 = penalized_saito_loss(m, 3, 3, profile="search", seed=1)
    l1 = penalized_saito_loss(m * scales[:, None], 3, 3, profile="search",
                              seed=1)
    assert abs(l0 - l1) < 1e-9


def test_complex_phase_invariance(nonfree7):
    m = lines_matrix(nonfree7).astype(np.complex128)
    rng = np.random.default_rng(5)
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=m.shape[0]))
    l_real = penalized_saito_loss(lines_matrix(nonfree7), 3, 3,
                                  profile="search", seed=1)
    l_phase = penalized_saito_loss(m * phases[:, None], 3, 3,
                                   profile="search", seed=1,
                                   dtype=np.complex128)
    # complex spheres contain the real ones; with real data the optimum agrees
    assert abs(l_real - l_phase) < 1e-6


def test_complex_phase_invariance_free(braid):
    m = lines_matrix(braid).astype(np.complex128)
    rng = np.random.default_rng(6)
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=m.shape[0]))
    assert penalized_saito_loss(m * phases[:, None], 2, 3, profile="search",
                                dtype=np.complex128) < 1e-8


# ─── 8. line permutation invariance ──────────────────────────────────────────

def test_permutation_invariance(nonfree7, rng):
    m = lines_matrix(nonfree7)
    perm = rng.permutation(m.shape[0])
    l0 = penalized_saito_loss(m, 3, 3, profile="search", seed=2)
    l1 = penalized_saito_loss(m[perm], 3, 3, profile="search", seed=2)
    assert abs(l0 - l1) < 1e-9


# ─── 9. orthogonal/unitary coordinate changes preserve the score ─────────────

def _rotate(m, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3))
    Qmat, _ = np.linalg.qr(A)
    # line coefficients transform by Q^T under x -> Q x
    return m @ Qmat


def test_orthogonal_invariance_free(braid):
    m = _rotate(lines_matrix(braid), 11)
    assert penalized_saito_loss(m, 2, 3, profile="search") < 1e-8


def test_orthogonal_invariance_nonfree(nonfree7):
    m = lines_matrix(nonfree7)
    l0 = penalized_saito_loss(m, 3, 3, profile="benchmark", seed=3)
    l1 = penalized_saito_loss(_rotate(m, 12), 3, 3, profile="benchmark",
                              seed=3)
    assert abs(l0 - l1) < 1e-5


# ─── 10. projective (non-unitary) changes: zero set preserved, values not ────

SHEAR = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, -1.0], [0.5, 0.0, 3.0]])


def test_projective_change_preserves_zero_set(braid):
    m = lines_matrix(braid) @ SHEAR
    assert penalized_saito_loss(m, 2, 3, profile="search") < 1e-8


def test_projective_change_moves_positive_values(nonfree7):
    m = lines_matrix(nonfree7)
    l0 = penalized_saito_loss(m, 3, 3, profile="benchmark", seed=4)
    l1 = penalized_saito_loss(m @ SHEAR, 3, 3, profile="benchmark", seed=4)
    assert 0 < l0 < 1 and 0 < l1 < 1
    # positive values are NOT projectively invariant (this particular shear
    # moves the value by far more than optimizer noise)
    assert abs(l0 - l1) > 1e-4


# ─── 11. for fixed (u, v), Gamma is decreasing in lambda ─────────────────────

def test_lambda_monotone_pointwise(nonfree7, rng):
    ev = PenalizedSaitoEvaluator(nonfree7, 3, 3)
    u = rng.standard_normal(ev.dim_u)
    u /= np.linalg.norm(u)
    v = rng.standard_normal(ev.dim_v)
    v /= np.linalg.norm(v)
    lams = [1e-3, 1e-1, 1.0, 10.0, 1e3]
    gs = [ev.gamma(u, v, lam=l) for l in lams]
    for a, b in zip(gs, gs[1:]):
        assert a > b  # strict: R(u, v) > 0 for generic (u, v)


# ─── 12. optimized nonfree losses -> 1 as lambda grows ───────────────────────

def test_lambda_optimized_trend_to_one(nonfree7):
    lams = [1e-2, 1.0, 1e2, 1e4]
    ls = [loss(nonfree7, 3, 3, lam=l) for l in lams]
    for a, b in zip(ls, ls[1:]):
        assert b > a - 1e-6      # nondecreasing up to optimizer noise
    assert ls[-1] > 0.9


# ─── 13. free losses remain ~0 for all lambda ────────────────────────────────

def test_free_zero_all_lambda(braid):
    for lam in [1e-3, 1.0, 1e3, 1e6]:
        assert loss(braid, 2, 3, lam=lam) < 1e-7


# ─── 14. float64 agrees with high-precision / exact reference ────────────────

def test_float64_vs_reference_free_pair(a2xa1):
    cert = find_exact_saito_certificate(a2xa1)
    u_mono = [Rational(sp.nsimplify(t)) for t in cert["theta1"]]
    v_mono = [Rational(sp.nsimplify(t)) for t in cert["theta2"]]
    ref = gamma_reference(a2xa1, cert["d1"], cert["d2"], u_mono, v_mono,
                          lam=1.0, beta=0.5)
    u, v = certificate_to_bw_vectors(cert)
    ev = PenalizedSaitoEvaluator(a2xa1, cert["d1"], cert["d2"])
    g = ev.gamma(u, v, lam=1.0, beta=0.5)     # match the reference's beta
    assert abs(float(ref) - 1.0) < 1e-30      # exact pair: Gamma = 1 exactly
    assert abs(g - float(ref)) < 1e-10


def test_float64_vs_reference_nonfree_point(nonfree7):
    # deterministic rational test vectors (not near the kernel)
    d1 = d2 = 3
    from penalized_saito import _monoms, _bw_sqrt_weights
    N = len(_monoms(3))
    u_mono = [Rational((7 * k) % 11 - 5, 3) for k in range(3 * N)]
    v_mono = [Rational((5 * k) % 13 - 6, 4) for k in range(3 * N)]
    ref = float(gamma_reference(nonfree7, d1, d2, u_mono, v_mono,
                                lam=1.0, beta=0.5))
    # evaluator works in BW coordinates on normalized vectors
    sw = _bw_sqrt_weights(3)
    def to_bw(mono):
        c = np.array([float(t) for t in mono])
        w = np.concatenate([c[:N] / sw, c[N:2 * N] / sw, c[2 * N:] / sw])
        return w / np.linalg.norm(w)
    ev = PenalizedSaitoEvaluator(nonfree7, d1, d2)
    # the exact reference is computed at beta = 0.5; evaluate at the same
    # beta explicitly (the production default is 0.75)
    g = ev.gamma(to_bw(u_mono), to_bw(v_mono), lam=1.0, beta=0.5)
    assert abs(g - ref) < 1e-11 * max(1.0, abs(ref))


# ─── 15. analytic gradient matches finite differences ────────────────────────

def test_gradient_finite_difference(nonfree7, rng):
    ev = PenalizedSaitoEvaluator(nonfree7, 3, 3)
    for trial in range(3):
        u = rng.standard_normal(ev.dim_u)
        u /= np.linalg.norm(u)
        v = rng.standard_normal(ev.dim_v)
        v /= np.linalg.norm(v)
        g0, gu, gv = ev.gamma_and_grad(u, v)
        h = 1e-6
        for _ in range(4):
            du = rng.standard_normal(ev.dim_u)
            dv = rng.standard_normal(ev.dim_v)
            # central differences of gamma∘normalize; its directional
            # derivative is given by the Riemannian (projected) gradient
            fp = ev.gamma((u + h * du) / np.linalg.norm(u + h * du),
                          (v + h * dv) / np.linalg.norm(v + h * dv))
            fm = ev.gamma((u - h * du) / np.linalg.norm(u - h * du),
                          (v - h * dv) / np.linalg.norm(v - h * dv))
            gu_r = gu - (u @ gu) * u
            gv_r = gv - (v @ gv) * v
            analytic = gu_r @ du + gv_r @ dv
            numeric = (fp - fm) / (2 * h)
            assert abs(analytic - numeric) < 1e-5 * max(1.0, abs(analytic))


# ─── 16. the old exact-kernel construction is binary on exact examples ───────

def _exact_saito_det(arr, v1, v2, d1, d2):
    from certificates import _vec_to_components
    m1 = LineArrangement._monoms(d1)
    m2 = LineArrangement._monoms(d2)
    f1, g1, h1 = _vec_to_components(list(v1), m1)
    f2, g2, h2 = _vec_to_components(list(v2), m2)
    return sp.expand(Matrix([[x, f1, f2], [y, g1, g2], [z, h1, h2]]).det())


def test_legacy_exact_construction_is_binary_nonfree(nonfree7):
    """Nonfree: EVERY exact determinant of exact-kernel pairs is identically
    zero, so the exact angular score is 1 by convention — never intermediate."""
    d1 = d2 = 3
    null = nonfree7.derivation_matrix(3).nullspace()
    assert null      # kernel is nonempty, yet no Saito basis exists
    rng = np.random.default_rng(0)
    combos = list(null)
    for _ in range(5):    # random exact rational combinations
        coeffs = [Rational(int(rng.integers(-5, 6)), int(rng.integers(1, 4)))
                  for _ in null]
        combos.append(sum((c * v for c, v in zip(coeffs, null)),
                          sp.zeros(len(null[0]), 1)))
    for v1 in combos:
        for v2 in combos:
            det = _exact_saito_det(nonfree7, v1, v2, d1, d2)
            assert sp.simplify(det) == 0


def test_legacy_exact_construction_is_binary_free(braid):
    """Free: every exact determinant is c * Q (c possibly 0) — the angular
    score is 0 when c != 0, undefined/1 when c = 0.  Never intermediate."""
    Q = sp.expand(sp.prod(l.linear_form() for l in braid.lines))
    null2 = braid.derivation_matrix(2).nullspace()
    null3 = braid.derivation_matrix(3).nullspace()
    for v1 in null2:
        for v2 in null3:
            det = _exact_saito_det(braid, v1, v2, 2, 3)
            ratio = sp.cancel(det / Q)
            assert ratio.is_number     # det is ALWAYS a scalar multiple of Q


def test_legacy_score_still_available():
    arr = LineArrangement([ProjectiveLine(*c) for c in
                           [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0),
                            (1, 0, -1), (0, 1, -1)]])
    val = saito.legacy_invalid_angular_score(arr)
    assert 0.0 <= val <= 1.0


# ─── 17. new loss does not depend on the old SVD null-space machinery ────────

def test_new_loss_independent_of_svd_tolerance(nonfree7, monkeypatch):
    l0 = loss(nonfree7, 3, 3, seed=7)

    def _poisoned(*a, **k):
        raise AssertionError("penalized loss must not call _null_space_basis")

    monkeypatch.setattr(saito, "_null_space_basis", _poisoned)
    l1 = loss(nonfree7, 3, 3, seed=7)
    assert l0 == l1
    # production wrapper also avoids it
    assert saito.saito_loss(nonfree7, target_exponents=(3, 3)) == pytest.approx(
        saito.saito_loss(nonfree7, target_exponents=(3, 3)))


def test_legacy_min_extra_changes_legacy_not_new(nonfree7):
    new_a = loss(nonfree7, 3, 3, seed=8)
    leg_0 = saito.legacy_invalid_angular_score(nonfree7,
                                               target_exponents=(3, 3),
                                               min_extra=0)
    leg_8 = saito.legacy_invalid_angular_score(nonfree7,
                                               target_exponents=(3, 3),
                                               min_extra=8)
    new_b = loss(nonfree7, 3, 3, seed=8)
    assert new_a == new_b                      # unaffected by legacy knobs
    assert isinstance(leg_0, float) and isinstance(leg_8, float)


# ─── integration: production wrappers and reward path ────────────────────────

def test_saito_loss_wrapper(braid, nonfree7):
    assert saito.saito_loss(braid, target_exponents=(2, 3)) < 1e-8
    val = saito.saito_loss(nonfree7, target_exponents=(3, 3))
    assert 0.001 < val < 0.999


def test_smooth_saito_loss_deprecated_forwards_to_new(braid):
    with pytest.warns(DeprecationWarning):
        val = saito.smooth_saito_loss(braid, target_exponents=(2, 3))
    assert val < 1e-8


def test_bounds_hold_everywhere(request):
    for fixture in ["braid", "a2xa1", "pencil4", "nonfree7", "generic4"]:
        arr = request.getfixturevalue(fixture)
        n = len(arr)
        for pair in admissible_degree_pairs(n):
            val = penalized_saito_loss(arr, *pair, profile="rl")
            assert 0.0 <= val <= 1.0
