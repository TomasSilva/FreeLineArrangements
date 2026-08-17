"""
Audit tests for the penalized Saito functional (2026-08 revision).

Covers the audit-mandated cases not already in tests/test_penalized_saito.py:
construction without rank cutoffs, explicit R = 0 / base-locus branches,
Gamma in [0, 1], S_all = min (not average), beta = 0.5 nonsmooth support,
solver-floor convergence, seed reproducibility, and independence of exact
certification from the numerical reward.
"""

import numpy as np
import pytest

import penalized_saito
from penalized_saito import (PenalizedSaitoEvaluator, penalized_saito_loss,
                             penalized_saito_loss_all_pairs, _line_kernel_basis,
                             DEFAULT_BETA, admissible_degree_pairs)
from arrangement import LineArrangement, ProjectiveLine
from certificates import (find_certificate_fast, find_exact_saito_certificate,
                          certificate_to_bw_vectors, verify_certificate)


def arr_from(coords):
    return LineArrangement([ProjectiveLine(*c) for c in coords])


BRAID = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (1, 0, -1), (0, 1, -1)]
NONFREE7 = [(0, 1, -1), (1, 1, -1), (1, 1, 1), (1, 1, 2), (1, 1, 0),
            (1, 1, -2), (1, 0, -2)]


# ── audit config ─────────────────────────────────────────────────────────────

def test_production_default_beta():
    assert DEFAULT_BETA == 0.75


def test_diagnostics_report_conventions(braid_arr=None):
    ev = PenalizedSaitoEvaluator(arr_from(BRAID), 2, 3)
    res = ev.maximize(n_restarts=2, n_iters=10)
    assert res["optimization_field"] == "real"
    assert res["beta_smoothness"] == "differentiable"       # beta = 0.75
    assert res["mm_r_floor"] == penalized_saito._MM_R_FLOOR
    res05 = ev.maximize(beta=0.5, n_restarts=2, n_iters=10)
    assert res05["beta_smoothness"] == "nonsmooth_at_R0"


# ── 2. no numerical rank cutoff in the construction of L ─────────────────────

def test_construction_uses_no_rank_cutoff(monkeypatch, rng=None):
    """The residual operator is built from a closed-form projector identity;
    the a-priori-known rank of multiplication by a linear form means no
    tolerance/rank decision can occur.  Poison the rank-threshold helper and
    rebuild L for benign and badly-scaled lines: identical, no exception."""
    def _poisoned(*a, **k):
        raise AssertionError("rank-threshold helper must not be called "
                             "during construction of L")
    monkeypatch.setattr(penalized_saito, "kernel_diagnostics_from_operator",
                        _poisoned)
    lines = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (1, 0, -1),
             (0, 1, -1)]
    ev = PenalizedSaitoEvaluator(np.array([l for l in np.array(lines,
                                                               float)]), 2, 3)
    assert ev.L1.shape[1] == 3 * ev.N1
    # badly scaled lines: construction still exact-rank (normalization only)
    scaled = np.array(lines, float) * np.array([1e6, 1e-6, 1, 7, 1e5,
                                                1e-5])[:, None]
    ev2 = PenalizedSaitoEvaluator(scaled, 2, 3)
    assert np.allclose(ev.L1, ev2.L1, atol=1e-12)


def test_line_kernel_basis_known_rank():
    """The algebraic kernel of a nonzero linear form always has dimension
    exactly 2; the basis construction never makes a rank decision."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = rng.standard_normal(3) * 10.0 ** rng.integers(-8, 9)
        a = a / np.linalg.norm(a)
        u, w = _line_kernel_basis(a)
        assert abs(np.dot(a, u)) < 1e-12 and abs(np.dot(a, w)) < 1e-12
        assert abs(np.vdot(u, w)) < 1e-12
        assert abs(np.linalg.norm(u) - 1) < 1e-12


# ── 7. Gamma stays in [0, 1] pointwise ───────────────────────────────────────

def test_gamma_in_unit_interval():
    rng = np.random.default_rng(2)
    for coords, pair in [(BRAID, (2, 3)), (NONFREE7, (3, 3))]:
        ev = PenalizedSaitoEvaluator(arr_from(coords), *pair)
        for _ in range(50):
            u = rng.standard_normal(ev.dim_u)
            v = rng.standard_normal(ev.dim_v)
            g = ev.gamma(u / np.linalg.norm(u), v / np.linalg.norm(v))
            assert 0.0 <= g <= 1.0
        # at the certified pair the clipped value is exactly <= 1
        cert = find_exact_saito_certificate(arr_from(BRAID))
        ub, vb = certificate_to_bw_vectors(cert)
        evb = PenalizedSaitoEvaluator(arr_from(BRAID), 2, 3)
        assert 0.0 <= evb.gamma(ub, vb) <= 1.0


# ── 11. explicit R = 0 branches ──────────────────────────────────────────────

def test_r_zero_branches():
    """Exact-branch unit test via instance surgery: zeroed residual
    operators force R == 0.0 exactly."""
    ev = PenalizedSaitoEvaluator(arr_from(NONFREE7), 3, 3)
    ev.L1 = np.zeros_like(ev.L1)
    ev.L2 = ev.L1
    ev._L1tL1 = ev.L1.T @ ev.L1
    ev._L2tL2 = ev._L1tL1
    rng = np.random.default_rng(3)
    u = rng.standard_normal(ev.dim_u)
    u /= np.linalg.norm(u)
    v = rng.standard_normal(ev.dim_v)
    v /= np.linalg.norm(v)
    # R = 0, B != 0: denominator is ||B||^2 exactly; Gamma in [0, 1]
    g, parts = ev.gamma(u, v, return_parts=True)
    assert parts["residual_R"] == 0.0 and parts["B_norm"] > 0
    assert 0.0 <= g <= 1.0
    # R = 0, B ~ 0 (u = v with d1 = d2: det vanishes identically; float
    # leaves ~1e-15 dust): the value is a bounded dust ratio in [0, 1] —
    # deliberately NOT thresholded (that would be a hidden epsilon)
    g0, parts0 = ev.gamma(u, u, return_parts=True)
    assert parts0["B_norm"] < 1e-13
    assert 0.0 <= g0 <= 1.0
    # the EXACT base locus (den == 0.0) maps to exactly 0
    orig_Bv = ev.B_v_matrix
    ev.B_v_matrix = lambda vv: np.zeros((ev.N_out, ev.dim_u))
    g_exact, parts_exact = ev.gamma(u, v, return_parts=True)
    assert parts_exact["denominator"] == 0.0 and g_exact == 0.0
    ev.B_v_matrix = orig_Bv
    # gradient branch at R = 0: finite, penalty term contributes 0
    gg, gu, gv = ev.gamma_and_grad(u, v)
    assert np.all(np.isfinite(gu)) and np.all(np.isfinite(gv))


# ── 6. S_all is the minimum, never the average ───────────────────────────────

def test_all_pairs_is_min_not_average():
    arr = arr_from(BRAID)
    val, det = penalized_saito_loss_all_pairs(arr, profile="rl",
                                              return_details=True)
    per_pair = det["per_pair"]
    assert set(per_pair) == set(admissible_degree_pairs(len(arr)))
    assert val == min(per_pair.values())
    assert val < 1e-7                       # the free pair drives the min
    assert np.mean(list(per_pair.values())) > 0.01   # average would NOT vanish


# ── 12. beta = 0.5 remains supported (nonsmooth-labeled, MM-optimized) ───────

def test_beta_half_supported_and_sane():
    l_nonfree = penalized_saito_loss(arr_from(NONFREE7), 3, 3, beta=0.5,
                                     profile="search", seed=0)
    assert 1e-6 < l_nonfree < 1 - 1e-6
    l_free = penalized_saito_loss(arr_from(BRAID), 2, 3, beta=0.5,
                                  profile="search", seed=0)
    assert l_free < 1e-8


# ── 4 (numerical): solver floor is regularization-only; value converges ──────

def test_mm_floor_convergence(monkeypatch):
    """The MM linearization floor is solver-internal.  The reported value
    must be unchanged (to optimizer accuracy) as the floor tends to zero."""
    base = {}
    for floor in (1e-100, 1e-200, 1e-300):
        monkeypatch.setattr(penalized_saito, "_MM_R_FLOOR", floor)
        for name, coords, pair in [("free", BRAID, (2, 3)),
                                   ("nonfree", NONFREE7, (3, 3))]:
            val = penalized_saito_loss(arr_from(coords), *pair,
                                       profile="search", seed=4)
            base.setdefault(name, []).append(val)
    for name, vals in base.items():
        assert max(vals) - min(vals) < 1e-9, (name, vals)


# ── 13. reproducibility under fixed seeds ────────────────────────────────────

def test_seed_reproducibility():
    a = penalized_saito_loss(arr_from(NONFREE7), 3, 3, profile="search",
                             seed=11)
    b = penalized_saito_loss(arr_from(NONFREE7), 3, 3, profile="search",
                             seed=11)
    assert a == b
    c = penalized_saito_loss(arr_from(NONFREE7), 3, 3, profile="search",
                             seed=12)
    assert isinstance(c, float)             # different seed may differ; both valid


# ── 14. exact certification independent of the numerical reward ──────────────

def test_certification_independent_of_reward(monkeypatch):
    def _poisoned(*a, **k):
        raise AssertionError("certification must not touch the numerical "
                             "loss machinery")
    monkeypatch.setattr(penalized_saito, "penalized_saito_loss", _poisoned)
    monkeypatch.setattr(penalized_saito, "cached_penalized_loss", _poisoned)
    monkeypatch.setattr(penalized_saito.PenalizedSaitoEvaluator, "maximize",
                        _poisoned)
    cert, status = find_certificate_fast(arr_from(BRAID),
                                         target_exponents=(2, 3))
    assert status == "certified" and verify_certificate(cert)
    cert2, status2 = find_certificate_fast(arr_from(NONFREE7),
                                           target_exponents=(3, 3))
    assert cert2 is None and status2 in ("not_free_exact", "modp_reject")
