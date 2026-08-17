"""
Production-safety pass tests: clipping accounting, deterministic repeated
evaluation, the tiny-loss-but-exactly-nonfree gate, and hardened
certificate verification.
"""

import copy
import json
import os

import numpy as np
import pytest
from sympy import Rational

import penalized_saito
from arrangement import LineArrangement, ProjectiveLine
from certificates import (find_certificate_fast, find_exact_saito_certificate,
                          verify_certificate, certificate_to_json,
                          certificate_from_json)
from penalized_saito import PenalizedSaitoEvaluator, runtime_provenance
from swap_search import double_pencil_seed, certify_state
from saito import construct_supersolvable, saito_loss


def arr_from(coords):
    return LineArrangement([ProjectiveLine(*c) for c in coords])


BRAID = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (1, 0, -1), (0, 1, -1)]


# ── item 3: clipping accounting ──────────────────────────────────────────────

def test_gamma_raw_recorded_and_clip_counted():
    ev = PenalizedSaitoEvaluator(arr_from(BRAID), 2, 3)
    res = ev.maximize(n_restarts=4, n_iters=40)
    p = res["parts"]
    tol = ev.gamma_tolerance
    for key in ("gamma_raw", "gamma_bounded", "numerical_status",
                "clip_applied", "clip_excess", "error_tolerance",
                "diagnostic_message"):
        assert key in p
    assert p["numerical_status"] in ("OK", "ROUNDING_CLIPPED", "RETRY_OK")
    assert 0.0 <= p["gamma_bounded"] <= 1.0
    # raw value sits within the scale-aware tolerance of the bounded value
    assert abs(p["gamma_raw"] - p["gamma_bounded"]) <= tol
    assert res["gamma_clip_max_excess"] <= tol
    assert res["numerical_error_count"] == 0
    assert res["functional_version"] == penalized_saito.FUNCTIONAL_VERSION
    # tolerance is dtype/dimension-aware, not the old fixed 1e-9
    assert tol == penalized_saito._gamma_error_tolerance(
        ev.dtype, ev.N_out, ev.dim_u, ev.dim_v)
    assert 1e-14 < tol < 1e-9


def test_two_ulp_excess_is_logged_and_safely_clipped():
    ev = PenalizedSaitoEvaluator(arr_from(BRAID), 2, 3)
    rng = np.random.default_rng(1)
    u = rng.standard_normal(ev.dim_u)
    u /= np.linalg.norm(u)
    v = rng.standard_normal(ev.dim_v)
    v /= np.linalg.norm(v)
    eps = np.finfo(np.float64).eps
    # simulate a two-ulp Cauchy-Schwarz excess by inflating q by (1 + eps)
    ev.q = ev.q * (1.0 + eps)
    before = ev._clip_count
    g, parts = ev.gamma(u, v, return_parts=True)
    # value is safely inside [0, 1]; any clip that occurred was logged
    assert 0.0 <= g <= 1.0
    assert parts["numerical_status"] in ("OK", "ROUNDING_CLIPPED")
    if parts["clip_applied"]:
        assert ev._clip_count == before + 1
        assert parts["clip_excess"] <= parts["error_tolerance"]


def test_substantial_gamma_violation_is_numerical_error(monkeypatch):
    from certificates import certificate_to_bw_vectors
    cert = find_exact_saito_certificate(arr_from(BRAID))
    u, v = certificate_to_bw_vectors(cert)   # fully aligned point (cos^2 = 1)
    ev = PenalizedSaitoEvaluator(arr_from(BRAID), 2, 3)
    # sabotage q to force a genuine (non-roundoff) violation; the mpmath
    # stage would REPAIR this (it rebuilds q from the raw lines), so also
    # simulate arbitrary-precision failure to reach the terminal error
    ev.q = ev.q * 2.0

    def _mp_fail(*a, **k):
        raise RuntimeError("mp unavailable")
    monkeypatch.setattr(ev, "_gamma_mpmath", _mp_fail)
    g, parts = ev.gamma(u, v, return_parts=True)
    assert g is None
    assert parts["numerical_status"] == "NUMERICAL_ERROR"
    assert parts["gamma_raw"] > 1.0 + parts["error_tolerance"]
    assert parts["gamma_bounded"] is None
    assert parts["retries"] >= 3          # compensated + two mp attempts
    with pytest.raises(penalized_saito.GammaNumericalError):
        ev.gamma(u, v)                    # plain call raises, never leaks


def test_arbitrary_precision_retry_repairs_corrupted_float_state():
    """Stage C: the mpmath rebuild reconstructs q_A, B, residuals and the
    denominator from the model inputs, so a corrupted float64 q is REPAIRED
    (RETRY_OK with a verified bounded value), not merely rejected."""
    from certificates import certificate_to_bw_vectors
    cert = find_exact_saito_certificate(arr_from(BRAID))
    u, v = certificate_to_bw_vectors(cert)
    ev = PenalizedSaitoEvaluator(arr_from(BRAID), 2, 3)
    ev.q = ev.q * 2.0                     # corrupt only the float64 q
    g, parts = ev.gamma(u, v, return_parts=True)
    assert parts["numerical_status"] == "RETRY_OK"
    assert "arbitrary-precision" in parts["diagnostic_message"]
    assert 0.0 <= g <= 1.0
    assert abs(g - 1.0) < 1e-9            # true value at the certified pair


def test_no_invalid_score_reaches_search_layer(monkeypatch):
    """Final-audit contract: a numerical failure is NEVER converted to a
    numeric loss (no pessimistic 1.0, no NaN).  saito_loss propagates the
    structured error; the swap env turns it into a recorded no-op with the
    separate numerical_failure_penalty component."""
    def _always_error(*a, **k):
        raise penalized_saito.GammaNumericalError("forced")
    monkeypatch.setattr("saito.penalized_saito_loss", _always_error)
    monkeypatch.setattr("saito.cached_penalized_loss", _always_error)
    with pytest.raises(penalized_saito.GammaNumericalError):
        saito_loss(construct_supersolvable(9, 3),
                   target_exponents=(3, 5), profile="rl", cached=True)


def test_compensated_retry_path_can_succeed(monkeypatch):
    """If the fast path is out of tolerance but the compensated path is
    valid, the evaluation succeeds with status RETRY_OK."""
    from certificates import certificate_to_bw_vectors
    cert = find_exact_saito_certificate(arr_from(BRAID))
    u, v = certificate_to_bw_vectors(cert)     # aligned: g_raw ~ 1
    ev = PenalizedSaitoEvaluator(arr_from(BRAID), 2, 3)
    # push g_raw just above 1 + tol (tiny q inflation), then emulate a
    # successful stable re-evaluation
    ev.q = ev.q * (1.0 + 1e-6)
    monkeypatch.setattr(ev, "_gamma_compensated",
                        lambda *a, **k: (0.5, 1.0, 1.0, 2.0))
    g, parts = ev.gamma(u, v, return_parts=True)
    assert parts["numerical_status"] == "RETRY_OK"
    assert g == 0.5
    assert parts["retries"] == 1
    assert ev._retry_count >= 1


def test_gamma_diagnostics_survive_serialization():
    import json as _json
    ev = PenalizedSaitoEvaluator(arr_from(BRAID), 2, 3)
    rng = np.random.default_rng(4)
    u = rng.standard_normal(ev.dim_u)
    u /= np.linalg.norm(u)
    v = rng.standard_normal(ev.dim_v)
    v /= np.linalg.norm(v)
    _, parts = ev.gamma(u, v, return_parts=True)
    round_tripped = _json.loads(_json.dumps(parts))
    assert round_tripped["numerical_status"] == parts["numerical_status"]
    assert round_tripped["gamma_raw"] == parts["gamma_raw"]
    assert round_tripped["error_tolerance"] == parts["error_tolerance"]


# ── item 7: deterministic repeated evaluation + provenance ───────────────────

def test_repeated_rl_evaluation_deterministic():
    arr = construct_supersolvable(10, 4)
    vals = [saito_loss(arr, target_exponents=(4, 5), profile="rl",
                       cached=True) for _ in range(3)]
    assert vals[0] == vals[1] == vals[2]
    # uncached path with the same seed is bitwise identical too
    a = saito_loss(arr, target_exponents=(4, 5), profile="rl", cached=False)
    b = saito_loss(arr, target_exponents=(4, 5), profile="rl", cached=False)
    assert a == b


def test_runtime_provenance_fields():
    prov = runtime_provenance(".")
    for key in ("functional_version", "code_commit", "dirty_tree",
                "source_content_hash", "dependency_versions", "python",
                "default_lambda", "default_beta",
                "optimization_field_default", "gamma_tolerance_model",
                "mm_r_floor", "profiles", "basis_convention"):
        assert key in prov
    assert len(prov["source_content_hash"]) == 64
    assert prov["dependency_versions"]["numpy"] != "absent"


# ── item 10: tiny-loss but exactly nonfree ⇒ no reward, no discovery ────────

@pytest.fixture(scope="module")
def tiny_loss_nonfree():
    """Perturb one line of a certified-free supersolvable by 1e-8: the loss
    is far below the 1e-6 heuristic gate, but the arrangement is EXACTLY
    nonfree with the target pair (proved by the exact negative certificate,
    not by search failure)."""
    base = construct_supersolvable(9, 3)
    a, b, c = base.lines[-1].coords
    t = Rational(1, 10**8)
    pert = LineArrangement(list(base.lines[:-1]) +
                           [ProjectiveLine(a + t, b + 2 * t, c - t)])
    cert, status = find_certificate_fast(pert, target_exponents=(3, 5))
    assert cert is None and status in ("not_target_free", "modp_reject")
    loss = saito_loss(pert, target_exponents=(3, 5), profile="search")
    assert loss < 1e-6          # passes the heuristic gate
    return pert


def test_tiny_loss_nonfree_gets_no_certificate_or_reward(tiny_loss_nonfree):
    # reward gate: certify_state is the ONLY path to the terminal bonus in
    # the swap env and the only path to certified.jsonl in campaigns
    assert certify_state(tiny_loss_nonfree, 3, 5) is None


def test_tiny_loss_nonfree_not_written_to_discoveries(tiny_loss_nonfree,
                                                      tmp_path):
    from experiments.run_swap_campaign import CampaignIO
    from novelty import lattice_wl_hash, coordinate_height
    io = CampaignIO(str(tmp_path), 9, 3, 5, "test", 0)
    rec = {
        "lines": [str(l) for l in tiny_loss_nonfree.lines],
        "n": 9, "d1": 3, "d2": 5, "loss": 1e-8,
        "b2": tiny_loss_nonfree.b2(),
        "m_max": tiny_loss_nonfree.max_multiplicity(),
        "height": coordinate_height(tiny_loss_nonfree),
        "lattice_hash": lattice_wl_hash(tiny_loss_nonfree),
        "engine": "test", "step": 0, "t": 0.0,
    }
    io.on_candidate(rec)
    # candidate is logged (numerically promising)…
    assert io.counters["candidates"] == 1
    # …but certification fails exactly, so nothing is certified/persisted
    assert io.counters["certified"] == 0
    assert io.counters["cert_failed"] == 1
    assert not os.path.exists(os.path.join(str(tmp_path), "certified.jsonl"))
    # and the repo-root discoveries.json is never touched by this pipeline
    assert not os.path.exists(os.path.join(str(tmp_path),
                                           "discoveries.json"))


# ── item 11: hardened certificate verification ───────────────────────────────

def test_certificate_requires_distinct_lines_and_degree_sum():
    cert = find_exact_saito_certificate(arr_from(BRAID))
    assert verify_certificate(cert)
    # tamper 1: duplicate a line (non-reduced arrangement)
    bad = copy.deepcopy(cert)
    bad["lines"] = list(bad["lines"])
    bad["lines"][0] = bad["lines"][1]
    assert not verify_certificate(bad)
    # tamper 2: wrong degree bookkeeping
    bad2 = copy.deepcopy(cert)
    bad2["d1"], bad2["d2"] = 1, 3            # 1 + 3 != n - 1 = 5
    assert not verify_certificate(bad2)
    # tamper 3: constant flipped to zero must fail (c != 0 required)
    bad3 = copy.deepcopy(cert)
    bad3["c"] = 0
    assert not verify_certificate(bad3)
    # JSON round-trip of the good certificate still verifies
    assert verify_certificate(certificate_from_json(
        json.loads(json.dumps(certificate_to_json(cert)))))
