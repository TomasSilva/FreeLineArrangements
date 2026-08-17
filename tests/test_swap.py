"""Tests for swap_search.py: seeds, proposals, Δb2 parity, validity,
recovery of perturbed free arrangements (Day-1 GO/NO-GO gate)."""

import numpy as np
import pytest

from arrangement import LineArrangement, ProjectiveLine
from saito import predicted_delta_b2
from swap_search import (double_pencil_seed, random_valid_seed, perturb_k_swaps,
                         propose_swaps, is_valid_state, ChainEvaluator,
                         greedy_search, certify_state)
from novelty import canonical_lineset_key


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(20260817)


# ── validity ─────────────────────────────────────────────────────────────────

def test_validity_guards():
    pencil = LineArrangement([ProjectiveLine(1, k, 0) for k in range(4)])
    assert not is_valid_state(pencil, 4)                 # non-essential pencil
    near = LineArrangement([ProjectiveLine(1, k, 0) for k in range(4)]
                           + [ProjectiveLine(0, 0, 1)])
    assert not is_valid_state(near, 5)                   # m_max = 4 = n - 1
    assert is_valid_state(double_pencil_seed(9, 3, 5), 9)
    assert not is_valid_state(double_pencil_seed(9, 3, 5), 10)  # wrong n


def test_random_valid_seed(rng):
    for n in (8, 11):
        arr = random_valid_seed(n, rng)
        assert is_valid_state(arr, n)


# ── seeds are certified free in-cell ─────────────────────────────────────────

@pytest.mark.parametrize("n,d1", [(7, 2), (9, 3), (10, 4)])
def test_double_pencil_seed_certified(n, d1):
    d2 = n - 1 - d1
    arr = double_pencil_seed(n, d1, d2)
    ev = ChainEvaluator(n, d1, d2)
    assert ev.screen_loss(arr) < 1e-8
    cert = certify_state(arr, d1, d2)
    assert cert is not None and cert["d1"] == d1 and cert["d2"] == d2


# ── Δb2 parity (batched predictor vs exact recomputation) ────────────────────

def test_delta_b2_parity(rng):
    arr = double_pencil_seed(11, 4, 6)
    n = len(arr)
    checked = 0
    for trial_idx in range(400):
        i = int(rng.integers(n))
        rest = LineArrangement([l for j, l in enumerate(arr.lines) if j != i])
        props = propose_swaps(arr, 4, 6, rng, n_remove=1, n_add_per_remove=4,
                              exact_frac=0.5, b2_slack=10 ** 6)
        for (_, line, trial) in props:
            rest2 = LineArrangement(trial.lines[:-1])
            assert trial.lines[-1] is line
            exact = trial.b2() - rest2.b2()
            assert exact == predicted_delta_b2(line, rest2)
            checked += 1
        if checked >= 400:
            break
    assert checked >= 400


# ── proposals ────────────────────────────────────────────────────────────────

def test_proposals_valid_and_tabu(rng):
    arr = double_pencil_seed(10, 3, 6)
    tabu = {canonical_lineset_key(arr)}
    props = propose_swaps(arr, 3, 6, rng, tabu=tabu)
    assert props
    n = len(arr)
    for (i, line, trial) in props:
        assert is_valid_state(trial, n)
        assert line.coords != arr.lines[i].coords          # L+ != L-
        assert canonical_lineset_key(trial) not in tabu    # tabu respected
        # exactly one line replaced
        old = {l.coords for l in arr.lines}
        new = {l.coords for l in trial.lines}
        assert len(old - new) == 1 and len(new - old) == 1


def test_exact_tier_restores_b2(rng):
    d1, d2 = 4, 6
    arr = double_pencil_seed(11, d1, d2)
    b2_star = (len(arr) - 1) + d1 * d2
    props = propose_swaps(arr, d1, d2, rng, exact_frac=1.0, b2_slack=0)
    assert props
    for (_, _, trial) in props:
        assert trial.b2() == b2_star


# ── perturbation ─────────────────────────────────────────────────────────────

def test_perturb_k_swaps(rng):
    arr = double_pencil_seed(10, 4, 5)
    for k in (1, 3):
        pert = perturb_k_swaps(arr, k, rng)
        assert is_valid_state(pert, len(arr))
        diff = {l.coords for l in arr.lines} - {l.coords for l in pert.lines}
        assert 1 <= len(diff) <= k


# ── Day-1 GO/NO-GO gate: greedy recovers a 1-swap-perturbed free seed ────────

def test_gate_greedy_recovers_perturbed_double_pencil(rng):
    n, d1, d2 = 12, 5, 6
    seed = double_pencil_seed(n, d1, d2)
    ev = ChainEvaluator(n, d1, d2)
    assert ev.screen_loss(seed) < 1e-8
    pert = perturb_k_swaps(seed, 1, rng)
    assert ev.screen_loss(pert) > 1e-4      # perturbation genuinely breaks it
    best, best_loss, _ = greedy_search(pert, d1, d2, ev, rng, steps=25,
                                       proposal_kwargs={"n_remove": 6,
                                                        "n_add_per_remove": 30})
    assert best_loss < 1e-8, f"greedy failed to recover: loss={best_loss:.3e}"
    cert = certify_state(best, d1, d2)
    assert cert is not None
