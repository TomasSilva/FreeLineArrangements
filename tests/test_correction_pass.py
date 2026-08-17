"""
Final correction-pass tests: promotion pipeline (Part 3), freeness-status
taxonomy (Part 4), certificate tampering (Part 5), cache isolation (Part 7),
the exact inverse-linearity regression (Part 8), and the RL calibration
layer.
"""

import copy
import json
import multiprocessing
import os

import numpy as np
import pytest
from sympy import Rational

import penalized_saito
from arrangement import LineArrangement, ProjectiveLine
from certificates import (classify_freeness, find_exact_saito_certificate,
                          find_certificate_fast, verify_certificate,
                          certificate_to_json, FREE_TARGET, NOT_TARGET_FREE,
                          GLOBALLY_NONFREE, UNRESOLVED)
from promotion import (build_discovery_entry, promote, canonical_discovery_id,
                       certificate_hash)
from calibration import calibrated_loss, freeness_potential, compute_tau
from saito import construct_supersolvable


def arr_from(coords):
    return LineArrangement([ProjectiveLine(*c) for c in coords])


BRAID = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (1, 0, -1), (0, 1, -1)]
NONFREE7 = [(0, 1, -1), (1, 1, -1), (1, 1, 1), (1, 1, 2), (1, 1, 0),
            (1, 1, -2), (1, 0, -2)]
GENERIC5 = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, 3), (3, -1, 2)]


# ── Part 4: structured freeness statuses ─────────────────────────────────────

def test_free_target_status():
    r = classify_freeness(arr_from(BRAID), target_pair=(2, 3))
    assert r["status"] == FREE_TARGET
    assert verify_certificate(r["certificate"])


def test_wrong_pair_is_not_target_free_not_globally_nonfree():
    r = classify_freeness(arr_from(BRAID), target_pair=(1, 4))
    assert r["status"] == NOT_TARGET_FREE       # NOT GLOBALLY_NONFREE
    # the global classification of the same arrangement is free
    g = classify_freeness(arr_from(BRAID))
    assert g["status"] == FREE_TARGET and g["candidate_pair"] == (2, 3)


def test_globally_nonfree_by_factorization_obstruction():
    r = classify_freeness(arr_from(GENERIC5))
    assert r["status"] == GLOBALLY_NONFREE
    assert "terao" in r["evidence"]


def test_globally_nonfree_by_candidate_pair_C_zero():
    r = classify_freeness(arr_from(NONFREE7))
    assert r["status"] == GLOBALLY_NONFREE
    assert r["candidate_pair"] == (3, 3)
    assert "pair_matrix_zero" in r["evidence"]


def test_unresolved_never_a_nonfree_proof(monkeypatch):
    import certificates as C

    def _boom(*a, **k):
        raise TimeoutError("injected")
    monkeypatch.setattr(C, "find_certificate_fast", _boom)
    r = C.classify_freeness(arr_from(BRAID), target_pair=(2, 3))
    assert r["status"] == UNRESOLVED
    assert r["certificate"] is None


# ── Part 5: certificate tampering ────────────────────────────────────────────

@pytest.fixture(scope="module")
def braid_cert():
    return find_exact_saito_certificate(arr_from(BRAID))


def test_tampering_matrix(braid_cert):
    assert verify_certificate(braid_cert)
    # zero line
    bad = copy.deepcopy(braid_cert)
    bad["lines"] = list(bad["lines"])
    bad["lines"][0] = (0, 0, 0)
    assert not verify_certificate(bad)
    # proportional (duplicate after canonicalization) lines
    bad = copy.deepcopy(braid_cert)
    bad["lines"] = list(bad["lines"])
    bad["lines"][0] = tuple(2 * Rational(v) for v in bad["lines"][1])
    assert not verify_certificate(bad)
    # n mismatch via degree sum
    bad = copy.deepcopy(braid_cert)
    bad["d1"], bad["d2"] = 2, 4
    assert not verify_certificate(bad)
    # non-canonical order d1 > d2
    bad = copy.deepcopy(braid_cert)
    bad["d1"], bad["d2"] = 3, 2
    assert not verify_certificate(bad)
    # wrong stated degree / nonhomogeneous packing (truncated vector)
    bad = copy.deepcopy(braid_cert)
    bad["theta1"] = bad["theta1"][:-1]
    assert not verify_certificate(bad)
    # nonlogarithmic derivation (perturbed coefficient)
    bad = copy.deepcopy(braid_cert)
    bad["theta1"] = list(bad["theta1"])
    bad["theta1"][0] = bad["theta1"][0] + 1
    assert not verify_certificate(bad)
    # altered determinant constant
    bad = copy.deepcopy(braid_cert)
    bad["c"] = bad["c"] * 2
    assert not verify_certificate(bad)
    # c = 0
    bad = copy.deepcopy(braid_cert)
    bad["c"] = 0
    assert not verify_certificate(bad)
    # float input rejected (silent rationalization forbidden)
    bad = copy.deepcopy(braid_cert)
    bad["theta1"] = list(bad["theta1"])
    bad["theta1"][0] = 0.5
    assert not verify_certificate(bad)
    bad = copy.deepcopy(braid_cert)
    bad["lines"] = list(bad["lines"])
    bad["lines"][0] = (1.0, 0.0, 0.0)
    assert not verify_certificate(bad)
    # zero derivation
    bad = copy.deepcopy(braid_cert)
    bad["theta1"] = [0] * len(bad["theta1"])
    assert not verify_certificate(bad)


# ── Part 3: promotion pipeline ───────────────────────────────────────────────

def _entry_for(arr, pair, run_id="testrun"):
    cert = find_exact_saito_certificate(arr, target_exponents=pair)
    assert cert is not None
    return build_discovery_entry(cert, run_id=run_id, engine="test",
                                 search_params={"lambda": 1.0, "beta": 0.75,
                                                "field": "real"})


def test_promotion_end_to_end(tmp_path):
    store = str(tmp_path / "discoveries.json")
    # pre-existing legacy entry must survive
    legacy = {"arrangements": [{"n": 4, "exponents": [1, 1, 2],
                                "lines": ["(1x+0y+0z=0)"],
                                "source": "legacy"}],
              "index": {"legacykey": 0}}
    with open(store, "w") as f:
        json.dump(legacy, f)

    arr = construct_supersolvable(9, 3)
    e = _entry_for(arr, (3, 5))
    res = promote([e], store)
    assert res["promoted"] == 1 and not res["rejected"]
    data = json.load(open(store))
    assert len(data["arrangements"]) == 2          # legacy preserved
    promoted = data["arrangements"][1]
    assert promoted["verification_status"] == "verified_exact"
    assert promoted["discovery_id"] == canonical_discovery_id(arr)
    assert promoted["certificate_hash"] == \
        certificate_hash(promoted["certificate"])
    assert data["arrangements"][0].get("verification_status") == \
        "legacy_unverified_by_promoter"
    # duplicate promotion is idempotent
    res2 = promote([e], store)
    assert res2["promoted"] == 0 and res2["duplicates"] == 1
    assert len(json.load(open(store))["arrangements"]) == 2


def test_promotion_rejects_uncertified_and_baseline(tmp_path):
    store = str(tmp_path / "d.json")
    arr = construct_supersolvable(9, 3)
    e = _entry_for(arr, (3, 5))
    # tampered certificate fails re-verification at promotion time
    bad = copy.deepcopy(e)
    bad["certificate"]["c"] = "0"
    bad["certificate_hash"] = certificate_hash(bad["certificate"])
    res = promote([bad], store)
    assert res["promoted"] == 0
    assert res["rejected"][0][1] == "certificate_failed_reverification"
    # hash mismatch rejected
    bad2 = copy.deepcopy(e)
    bad2["certificate_hash"] = "0" * 64
    res = promote([bad2], store)
    assert res["rejected"][0][1] == "certificate_hash_mismatch"
    # baseline pair requires explicit allowance
    from saito import construct_near_pencil
    np_arr = construct_near_pencil(6)
    eb = _entry_for(np_arr, (1, 4))
    res = promote([eb], store)
    assert res["promoted"] == 0 and "baseline" in res["rejected"][0][1]
    assert promote([eb], store, allow_baseline=True)["promoted"] == 1


def test_promotion_never_promotes_exactly_nonfree(tmp_path):
    """Tiny-loss but exactly nonfree: no certificate exists, so nothing can
    even be staged — and a forged entry is caught by re-verification."""
    store = str(tmp_path / "d.json")
    base = construct_supersolvable(9, 3)
    a, b, c = base.lines[-1].coords
    t = Rational(1, 10**8)
    pert = LineArrangement(list(base.lines[:-1]) +
                           [ProjectiveLine(a + t, b + 2 * t, c - t)])
    cert, status = find_certificate_fast(pert, target_exponents=(3, 5))
    assert cert is None and status in ("not_target_free", "modp_reject")
    # forge an entry by pairing the nonfree lines with the free cert
    good = _entry_for(base, (3, 5))
    forged = copy.deepcopy(good)
    forged["lines"] = [str(l) for l in pert.lines]
    forged["discovery_id"] = canonical_discovery_id(pert)
    res = promote([forged], store)
    assert res["promoted"] == 0        # discovery_id vs certificate mismatch


def _promote_worker(args):
    store, k = args
    arr = construct_supersolvable(9 + (k % 2), 3 + (k % 2))
    pair = (3 + (k % 2), len(arr) - 1 - (3 + (k % 2)))
    cert = find_exact_saito_certificate(arr, target_exponents=pair)
    e = build_discovery_entry(cert, run_id=f"w{k}",
                              search_params={"lambda": 1.0, "beta": 0.75,
                                             "field": "real"})
    return promote([e], store)


def test_concurrent_promotion_no_corruption(tmp_path):
    store = str(tmp_path / "d.json")
    with multiprocessing.get_context("fork").Pool(4) as pool:
        results = pool.map(_promote_worker, [(store, k) for k in range(8)])
    data = json.load(open(store))                  # valid JSON, no corruption
    ids = [r["discovery_id"] for r in data["arrangements"]]
    assert len(ids) == len(set(ids)) == 2          # two distinct arrangements
    assert sum(r["promoted"] for r in results) == 2
    assert sum(r["duplicates"] for r in results) == 6


# ── Part 7: cache isolation ──────────────────────────────────────────────────

def test_cache_key_isolation():
    from penalized_saito import cached_penalized_loss, _LOSS_CACHE
    arr = arr_from(NONFREE7)
    base_kwargs = dict(d1=3, d2=3, lam=1.0, beta=0.75, profile="rl", seed=0)
    n0 = len(_LOSS_CACHE)
    cached_penalized_loss(arr, **base_kwargs)
    assert len(_LOSS_CACHE) == n0 + 1
    # one-parameter changes must each be a cache MISS (new key)
    for change in (dict(lam=2.0), dict(beta=0.5), dict(seed=1),
                   dict(profile="search")):
        kw = dict(base_kwargs)
        kw.update(change)
        cached_penalized_loss(arr, **kw)
    assert len(_LOSS_CACHE) == n0 + 5
    # repeat of the base call: HIT (no growth)
    cached_penalized_loss(arr, **base_kwargs)
    assert len(_LOSS_CACHE) == n0 + 5
    # line order / per-line scaling do NOT change the key (canonical quotient)
    perm = LineArrangement(list(reversed(arr.lines)))
    cached_penalized_loss(perm, **base_kwargs)
    assert len(_LOSS_CACHE) == n0 + 5


# ── Part 8: exact inverse-linearity in lambda at a fixed candidate ───────────

def test_inverse_gamma_linear_in_lambda():
    """For fixed (u, v) with N > 0 and R > 0:
        1/Gamma_lambda = ||B||^2/N + (R^beta/N) * lambda   (exactly),
        lambda * Gamma_lambda -> N / R^beta.
    Stronger than O(1/lambda), which a broken solver returning 0 would also
    'satisfy'."""
    ev = penalized_saito.PenalizedSaitoEvaluator(arr_from(NONFREE7), 3, 3)
    rng = np.random.default_rng(8)
    u = rng.standard_normal(ev.dim_u)
    u /= np.linalg.norm(u)
    v = rng.standard_normal(ev.dim_v)
    v /= np.linalg.norm(v)
    beta = 0.75
    g1, parts = ev.gamma(u, v, lam=1.0, beta=beta, return_parts=True)
    N = parts["raw_numerator"]
    Bsq = parts["B_norm"] ** 2
    Rb = parts["residual_R"] ** beta
    assert N > 0 and parts["residual_R"] > 0
    lams = np.logspace(-2, 10, 13)
    for lam in lams:
        g = ev.gamma(u, v, lam=lam, beta=beta)
        pred = 1.0 / (Bsq / N + (Rb / N) * lam)
        assert abs(g - pred) <= 1e-10 * max(pred, g)
    # asymptote of lambda * Gamma
    tail = [lam * ev.gamma(u, v, lam=lam, beta=beta) for lam in lams[-3:]]
    assert all(abs(t - N / Rb) <= 1e-6 * (N / Rb) for t in tail)


# ── calibration layer ────────────────────────────────────────────────────────

def test_calibration_endpoints_and_monotonicity():
    tau = 0.1
    assert calibrated_loss(0.0, tau) == 0.0
    assert abs(calibrated_loss(1.0, tau) - 1.0) < 1e-15
    assert abs(calibrated_loss(tau, tau) - (1 + tau) / 2) < 1e-12
    xs = np.linspace(0, 1, 101)
    ys = [calibrated_loss(s, tau) for s in xs]
    assert all(b > a for a, b in zip(ys, ys[1:]))      # strictly increasing
    with pytest.raises(ValueError):
        calibrated_loss(0.5, 0.0)


def test_calibration_preserves_action_ordering():
    rng = np.random.default_rng(9)
    raw = rng.uniform(0, 1, 50)
    for tau in (0.01, 0.2, 3.0):
        cal = np.array([calibrated_loss(s, tau) for s in raw])
        assert np.array_equal(np.argsort(raw), np.argsort(cal))
        pot = np.array([freeness_potential(s, tau) for s in raw])
        assert np.array_equal(np.argsort(raw), np.argsort(-pot))


def test_tau_deterministic_and_cached(tmp_path):
    cache = str(tmp_path / "tau.json")
    t1 = compute_tau(9, 3, 5, n_samples=6, cache_path=cache)
    t2 = compute_tau(9, 3, 5, n_samples=6, cache_path=cache)  # cache hit
    assert t1 == t2 > 0
    stored = json.load(open(cache))
    key = list(stored.keys())[0]
    assert "n9_d3_5" in key and "fv" in key
    # env records tau and logs raw + calibrated
    from swap_env import SwapArrangementEnv
    env = SwapArrangementEnv(target_n=9, d1=3, d2=5, seed=1,
                             episode_len=3, max_candidates=24, tau=t1)
    obs = env.reset()
    a = int(np.flatnonzero(env.action_mask() > 0)[0])
    _, r, _, info = env.step(a)
    assert info["tau"] == t1
    assert 0.0 <= info["raw_loss"] <= 1.0
    assert 0.0 <= info["calibrated_potential"] <= 1.0
