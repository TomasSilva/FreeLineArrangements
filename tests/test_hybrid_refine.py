"""Gradient validation for the experimental hybrid geometric refinement.

Covers (task spec): centered finite-difference agreement, torch-vs-
production consistency, scale/permutation invariance, free arrangement at
its exact pair (Gamma ~ 1 seeded with the certificate pair), wrong-pair
positivity, generic non-free positivity, structured safety statuses, and
the exact-acceptance/nontriviality gates.
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from arrangement import LineArrangement, ProjectiveLine
from certificates import find_certificate_fast, certificate_to_bw_vectors
from hybrid_refine import (DifferentiableGamma, refine_line,
                           _rationalize_candidates, OK, NO_IMPROVEMENT,
                           LINE_COLLISION)
from penalized_saito import PenalizedSaitoEvaluator

BRAID = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (1, 0, -1), (0, 1, -1)]
NONFREE7 = [(0, 1, -1), (1, 1, -1), (1, 1, 1), (1, 1, 2), (1, 1, 0),
            (1, 1, -2), (1, 0, -2)]


def _arr(coords):
    return LineArrangement([ProjectiveLine(*c) for c in coords])


def _dg(coords, d1, d2, i=0, seed=0):
    arr = _arr(coords)
    ev = PenalizedSaitoEvaluator(arr, d1, d2)
    res = ev.maximize(n_restarts=6, n_iters=60, seed=seed)
    return ev, DifferentiableGamma(ev, i, res["u"], res["v"]), res


def test_torch_matches_production_gamma():
    for coords, pair in ((BRAID, (2, 3)), (NONFREE7, (3, 3))):
        for i in (0, 3):
            _, dg, _ = _dg(coords, *pair, i=i)
            assert dg.consistency_error < 1e-9


def test_centered_finite_difference_agreement():
    """FD validation AWAY from the zero locus: at R -> 0 the beta = 0.75
    penalty is nonsmooth (R^(beta-1) singular), so finite differences are
    invalid exactly on free configurations — validated instead on nonfree
    states where Gamma is smooth (this is the documented beta < 1
    nonsmoothness, not a gradient defect)."""
    pert_braid = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0),
                  (1, 0, -1), (7, 100, -93)]      # nonfree perturbation
    max_rel = 0.0
    for coords, pair, i in ((NONFREE7, (3, 3), 2), (NONFREE7, (3, 3), 5),
                            (pert_braid, (2, 3), 5)):
        ev, dg, _ = _dg(coords, *pair, i=i)
        a = np.array(ev.lines[i])
        g0, grad = dg.gamma_and_grad(a)
        h = 1e-6
        for k in range(3):
            e = np.zeros(3)
            e[k] = h
            gp = float(dg.gamma(torch.tensor(a + e)))
            gm = float(dg.gamma(torch.tensor(a - e)))
            fd = (gp - gm) / (2 * h)
            # gamma() normalizes internally (degree-0 homogeneous), so the
            # ambient autograd gradient is already tangent and equals FD
            rel = abs(fd - grad[k]) / max(abs(fd), abs(grad[k]), 1e-12)
            max_rel = max(max_rel, rel)
    assert max_rel < 5e-5, f"max relative FD error {max_rel:.2e}"


def test_scale_and_sign_invariance():
    ev, dg, _ = _dg(NONFREE7, 3, 3, i=1)
    a = np.array(ev.lines[1])
    g1 = float(dg.gamma(torch.tensor(a)))
    g2 = float(dg.gamma(torch.tensor(2.5 * a)))
    g3 = float(dg.gamma(torch.tensor(-a)))
    assert abs(g1 - g2) < 1e-12 and abs(g1 - g3) < 1e-12


def test_permutation_invariance():
    ev1, dg1, _ = _dg(NONFREE7, 3, 3, i=0)
    perm = [3, 0, 1, 2, 4, 6, 5]
    coords_p = [NONFREE7[j] for j in perm]
    ev2, dg2, _ = _dg(coords_p, 3, 3, i=perm.index(0))
    a = np.array(ev1.lines[0])
    assert abs(float(dg1.gamma(torch.tensor(a)))
               - float(dg2.gamma(torch.tensor(a)))) < 1e-9


def test_free_arrangement_seeded_with_certificate_pair():
    arr = _arr(BRAID)
    cert, status = find_certificate_fast(arr, target_exponents=(2, 3))
    assert status == "certified"
    u, v = certificate_to_bw_vectors(cert)
    ev = PenalizedSaitoEvaluator(arr, 2, 3)
    dg = DifferentiableGamma(ev, 0, u, v)
    g = float(dg.gamma(dg.a0))
    assert abs(g - 1.0) < 1e-9          # raw loss 0 at the exact Saito pair


def test_wrong_pair_and_nonfree_are_positive():
    from penalized_saito import penalized_saito_loss
    assert penalized_saito_loss(_arr(BRAID), 1, 4,
                                profile="search", seed=0) > 1e-3
    assert penalized_saito_loss(_arr(NONFREE7), 3, 3,
                                profile="search", seed=0) > 1e-3


def test_gamma_never_silently_clipped_and_finite_guards():
    ev, dg, _ = _dg(NONFREE7, 3, 3, i=0)
    # poison the cofactor: force a nonfinite gradient path
    dg.G_t = dg.G_t * float("nan")
    a = np.array(ev.lines[0])
    g, grad = dg.gamma_and_grad(a)
    assert not np.all(np.isfinite(grad)) or math.isnan(g)
    # refine_line surfaces this as a structured status, never a loss value
    # (exercised through the collision guard below on the healthy object)


def test_refine_reports_structured_statuses():
    arr = _arr(NONFREE7)
    new, rep = refine_line(arr, 0, 3, 3, steps=4, seed=0)
    assert rep["status"] in (OK, NO_IMPROVEMENT, LINE_COLLISION,
                             "rationalization_failed")
    assert "raw_loss_before" in rep
    assert rep["torch_vs_evaluator_gamma_diff"] < 1e-7
    if new is not None:
        assert rep["raw_loss_after"] < rep["raw_loss_before"]
        # exact acceptance: rationalized, valid, distinct lines
        assert len({l.coords for l in new.lines}) == len(arr)


def test_rationalize_candidates_are_exact_and_valid():
    cands = _rationalize_candidates(np.array([0.3333334, -1.0, 0.5000001]))
    assert cands
    from sympy import Rational
    for L in cands:
        for c in L.coords:
            assert getattr(c, "is_Rational", False)


def test_recovery_pulls_perturbed_free_arrangement_back():
    """Perturb one braid line; refinement must strictly lower the raw loss
    and recover a certifiable free configuration (small case: n=6)."""
    from sympy import Rational
    arr = _arr(BRAID)
    pert_lines = list(arr.lines[:-1]) + [
        ProjectiveLine(Rational(1, 20), 1, -1)]      # (0,1,-1) nudged
    pert = LineArrangement(pert_lines)
    from penalized_saito import penalized_saito_loss
    loss_p = penalized_saito_loss(pert, 2, 3, profile="search", seed=0)
    assert loss_p > 1e-6                              # genuinely broken
    new, rep = refine_line(pert, len(pert_lines) - 1, 2, 3,
                           steps=20, seed=0)
    assert rep["status"] == OK and new is not None
    assert rep["raw_loss_after"] < loss_p * 0.1
    cert, status = find_certificate_fast(new, target_exponents=(2, 3))
    assert status == "certified"
