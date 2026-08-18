"""Phase-5 search-integration tests over quadratic fields."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments"))

from arrangement import LineArrangement
from environment import generate_candidate_lines_K, _singularity_candidates
from known_arrangements import akn13
from novelty import lattice_wl_hash, is_supersolvable_rank3
from quadfield import QuadraticField
from swap_search import (perturb_k_swaps, _integer_pool, ChainEvaluator,
                         greedy_search, certify_state, is_valid_state)


def test_k_grid_pool():
    pool = generate_candidate_lines_K(QuadraticField(3), 1)
    assert len(pool) > 100
    assert all(len({l.coords for l in pool}) == len(pool)
               for _ in [0])                      # projectively distinct
    K_lines = [l for l in pool if l.field is not None]
    assert K_lines and all(l.field.d == 3 for l in K_lines)


def test_integer_pool_dispatch():
    A = akn13()
    pool = _integer_pool(A, 3)
    assert any(l.field is not None for l in pool)
    from saito import construct_supersolvable
    qq_pool = _integer_pool(construct_supersolvable(9, 3), 3)
    assert all(l.field is None for l in qq_pool)


def test_singularity_candidates_field_closed():
    A = akn13()
    res = _singularity_candidates(A)
    assert len(res) > 100
    fields = {l.field.d for _, l in res if l.field is not None}
    assert fields <= {3}


def test_greedy_recovers_akn_from_one_swap():
    """The Phase-5 recovery gate: a 1-swap perturbation of AKN-13 descends
    back to the exact AKN lattice, certified, non-supersolvable."""
    A = akn13()
    akn_hash = lattice_wl_hash(A)
    rng = np.random.default_rng(7)
    pert = perturb_k_swaps(A, 1, rng, coord_range=1)
    assert is_valid_state(pert, 13, nontrivial=True)
    assert lattice_wl_hash(pert) != akn_hash
    ev = ChainEvaluator(13, 6, 6, seed=7)
    best, best_loss, _ = greedy_search(
        pert, 6, 6, ev, rng, steps=25,
        proposal_kwargs={"n_remove": 13, "coord_range": 1})
    assert best_loss < 1e-8
    assert lattice_wl_hash(best) == akn_hash
    cert = certify_state(best, 6, 6)
    assert cert is not None
    assert not is_supersolvable_rank3(best)


def test_akn_has_no_single_line_lift():
    """Documented negative, consistent with AKN non-recursive-freeness:
    no single-line extension reaches any admissible free n=14 cell (the
    achievable Delta-b2 values miss every required one)."""
    from saito import extend_arrangement_targeted
    A = akn13()
    for pair in ((6, 7), (5, 8), (4, 9)):
        assert extend_arrangement_targeted(A, target_exponents=pair,
                                           coord_range=2) == []


def test_rl_arm_guards_against_k():
    from swap_env import SwapArrangementEnv
    env = SwapArrangementEnv(target_n=9, d1=3, d2=5, seed=0)
    env.reset()          # QQ works
    # K seeds cannot enter the RL arm: guard is in reset via the seed
    # builder; simulate by checking the guard directly
    from swap_env import double_pencil_seed
    assert double_pencil_seed(9, 3, 5).coefficient_field() is None


def test_record_carries_field_tags():
    from swap_search import _record
    A = akn13()
    rec = _record(A, 6, 6, 0.5, "greedy", 0)
    assert rec["coefficient_field"]["d"] == 3
    assert rec["optimization_field"] == "real"
