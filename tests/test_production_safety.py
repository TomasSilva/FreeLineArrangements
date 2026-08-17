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
    assert "gamma_raw" in p
    # raw value sits within roundoff of the clipped value
    assert abs(p["gamma_raw"] - res["gamma"]) <= penalized_saito.GAMMA_CLIP_TOL
    assert res["gamma_clip_count"] >= 0
    assert res["gamma_clip_max_excess"] <= penalized_saito.GAMMA_CLIP_TOL
    assert res["functional_version"] == penalized_saito.FUNCTIONAL_VERSION


def test_large_gamma_violation_warns_not_silently_clipped():
    from certificates import certificate_to_bw_vectors
    cert = find_exact_saito_certificate(arr_from(BRAID))
    u, v = certificate_to_bw_vectors(cert)   # fully aligned point (cos^2 = 1)
    ev = PenalizedSaitoEvaluator(arr_from(BRAID), 2, 3)
    # sabotage q to force a genuine (non-roundoff) violation: with |q| = 2
    # the aligned point gives num ~ 4 ||B||^2 > den
    ev.q = ev.q * 2.0
    with pytest.warns(RuntimeWarning, match="Gamma exceeded 1"):
        g, parts = ev.gamma(u, v, return_parts=True)
    assert parts["gamma_raw"] > 1.0 + penalized_saito.GAMMA_CLIP_TOL
    assert g == parts["gamma_raw"]          # NOT silently clipped


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
                "default_lambda", "default_beta",
                "optimization_field_default", "gamma_clip_tol",
                "mm_r_floor", "profiles"):
        assert key in prov


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
    assert cert is None and status in ("not_free_exact", "modp_reject")
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
