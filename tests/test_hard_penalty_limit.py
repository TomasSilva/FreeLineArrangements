"""
Hard-penalty-limit tests (docs/hard_penalty_limit_proof.md).

Fast versions of the sweep's Part-5 checks: target-free controls stay at 0
for every lambda (seeded with their exact Saito pair, so optimizer failure
cannot masquerade as mathematical behavior); non-target-free cases —
including a FREE arrangement at a WRONG admissible pair — converge to 1
with the common-pool loss exactly nondecreasing in lambda.
"""

import numpy as np
import pytest

from arrangement import LineArrangement, ProjectiveLine
from penalized_saito import PenalizedSaitoEvaluator
from certificates import (find_exact_saito_certificate,
                          certificate_to_bw_vectors)

BRAID = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (1, 0, -1), (0, 1, -1)]
NONFREE7 = [(0, 1, -1), (1, 1, -1), (1, 1, 1), (1, 1, 2), (1, 1, 0),
            (1, 1, -2), (1, 0, -2)]
LAMS = [1e-2, 1.0, 1e2, 1e4, 1e6, 1e8]


def arr_from(coords):
    return LineArrangement([ProjectiveLine(*c) for c in coords])


def sweep_pool(ev, warm, lams, beta=0.75, n_restarts=4, n_iters=40):
    """Native runs per lambda + common-pool envelope over inits, finals,
    and warm starts.  Returns (native_S, pool_S) arrays ordered by lams."""
    pool = list(warm or [])
    pool.extend([(u, v) for (u, v, _) in ev._initial_points(
        np.random.default_rng(0), 6, warm, True, lam=1.0, beta=beta)])
    native = []
    for lam in lams:
        res = ev.maximize(lam=lam, beta=beta, n_restarts=n_restarts,
                          n_iters=n_iters, seed=0, warm_starts=warm)
        native.append(res["loss"])
        pool.append((res["u"], res["v"]))
    pool_S = []
    for lam in lams:
        g = max(ev.gamma(u, v, lam=lam, beta=beta) for (u, v) in pool)
        pool_S.append(1.0 - g)
    return np.array(native), np.array(pool_S)


@pytest.fixture(scope="module")
def braid_cert_pair():
    cert = find_exact_saito_certificate(arr_from(BRAID))
    return [certificate_to_bw_vectors(cert)]


def test_target_free_control_zero_for_all_lambda(braid_cert_pair):
    ev = PenalizedSaitoEvaluator(arr_from(BRAID), 2, 3)
    native, pool = sweep_pool(ev, braid_cert_pair, LAMS)
    assert np.max(pool) < 1e-8
    assert np.max(native) < 1e-8


def test_nonfree_converges_to_one_pool_monotone():
    ev = PenalizedSaitoEvaluator(arr_from(NONFREE7), 3, 3)
    native, pool = sweep_pool(ev, None, LAMS)
    # bounds
    assert np.all((pool >= -1e-15) & (pool <= 1 + 1e-15))
    # exact monotonicity of the common-pool loss (regression: each fixed
    # candidate's Gamma is nonincreasing in lambda)
    assert np.all(np.diff(pool) >= -1e-12)
    # convergence toward 1 with bounded lambda * gap
    assert 1.0 - pool[-1] < 1e-3
    gaps = LAMS * (1.0 - pool)
    assert gaps[-1] <= 3.0 * max(gaps[-2], 1e-300)


def test_free_wrong_pair_behaves_like_nonfree(braid_cert_pair):
    # braid is free with (2,3); at the admissible wrong pair (1,4) the loss
    # must converge to 1 (non-target-free), monotone over the common pool
    ev = PenalizedSaitoEvaluator(arr_from(BRAID), 1, 4)
    native, pool = sweep_pool(ev, None, LAMS)
    assert np.all(np.diff(pool) >= -1e-12)
    assert 1.0 - pool[-1] < 1e-3
    assert pool[0] < 1.0 - 1e-6          # strictly interior at small lambda


def test_pool_monotonicity_exact_on_random_pool():
    """Sharp regression: for ANY fixed candidate pool, the envelope loss is
    nondecreasing in lambda (up to <= 1e-15 float slack)."""
    rng = np.random.default_rng(3)
    ev = PenalizedSaitoEvaluator(arr_from(NONFREE7), 3, 3)
    pool = []
    for _ in range(12):
        u = rng.standard_normal(ev.dim_u)
        v = rng.standard_normal(ev.dim_v)
        pool.append((u / np.linalg.norm(u), v / np.linalg.norm(v)))
    lams = np.logspace(-3, 8, 23)
    S = np.array([1.0 - max(ev.gamma(u, v, lam=l, beta=0.75)
                            for (u, v) in pool) for l in lams])
    assert np.all(np.diff(S) >= -1e-15)
