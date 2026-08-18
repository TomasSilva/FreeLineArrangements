"""Phase-3 numeric-layer tests: embeddings, dtype policy, cache identity,
exact Stage-C rebuild for quadratic-field arrangements."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from certificates import find_certificate_fast, certificate_to_bw_vectors
from known_arrangements import akn13, dual_hesse
from penalized_saito import (PenalizedSaitoEvaluator, penalized_saito_loss,
                             cached_penalized_loss, _LOSS_CACHE,
                             _field_dtype_tags, FUNCTIONAL_VERSION)


def test_akn13_loss_near_zero_real_embedding():
    loss = penalized_saito_loss(akn13(), 6, 6, profile="search", seed=0)
    assert 0.0 <= loss < 1e-8


def test_dual_hesse_loss_near_zero_complex():
    loss = penalized_saito_loss(dual_hesse(), 4, 4, profile="search", seed=0)
    assert 0.0 <= loss < 1e-8


def test_gamma_at_certificate_vectors():
    A = akn13()
    cert, _ = find_certificate_fast(A, target_exponents=(6, 6))
    u, v = certificate_to_bw_vectors(cert)
    assert 1 - PenalizedSaitoEvaluator(A, 6, 6).gamma(u, v) < 1e-10
    H = dual_hesse()
    cert, _ = find_certificate_fast(H, target_exponents=(4, 4))
    u, v = certificate_to_bw_vectors(cert)
    ev = PenalizedSaitoEvaluator(H, 4, 4, dtype=np.complex128)
    assert 1 - ev.gamma(u, v) < 1e-10


def test_stage_c_exact_rebuild_agrees():
    A = akn13()
    ev = PenalizedSaitoEvaluator(A, 6, 6)
    assert ev._exact_lines is not None and ev._field_d == 3
    cert, _ = find_certificate_fast(A, target_exponents=(6, 6))
    u, v = certificate_to_bw_vectors(cert)
    g_mp, diag = ev._gamma_mpmath(u, v, lam=1.0, beta=0.75, dps=80)
    g_fast = ev.gamma(u, v, lam=1.0, beta=0.75)
    assert abs(float(g_mp) - g_fast) < 1e-12


def test_complex_field_requires_complex_dtype():
    with pytest.raises(ValueError):
        penalized_saito_loss(dual_hesse(), 4, 4, dtype=np.float64)
    # auto-selection picks complex128
    tag, dts, dt = _field_dtype_tags(dual_hesse())
    assert (tag, dts, dt) == ("QQ(sqrt-3)", "complex128", np.complex128)


def test_cache_keys_separate_fields():
    """Same-shaped keys must differ across (field, dtype) combinations."""
    A = akn13()
    before = len(_LOSS_CACHE)
    v1 = cached_penalized_loss(A, 6, 6, profile="rl", seed=123)
    v2 = cached_penalized_loss(A, 6, 6, profile="rl", seed=123)   # hit
    assert v1 == v2
    assert len(_LOSS_CACHE) == before + 1
    keys = [k for k in _LOSS_CACHE if k[1] == 6 and k[2] == 6
            and k[8] == 123]
    assert any(k[9] == "QQ(sqrt3)" and k[12] == FUNCTIONAL_VERSION
               for k in keys)


def test_qq_cache_tag_unchanged_semantics():
    """QQ arrangements carry the QQ/float64 tags (goldens lock the values)."""
    from arrangement import LineArrangement, ProjectiveLine
    braid = LineArrangement([ProjectiveLine(*c) for c in
                             [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                              (1, -1, 0), (1, 0, -1), (0, 1, -1)]])
    tag, dts, dt = _field_dtype_tags(braid)
    assert (tag, dts, dt) == ("QQ", "float64", np.float64)
