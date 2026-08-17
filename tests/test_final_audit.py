"""
Final production-audit regression tests: raw/calibrated separation,
terminal-potential convention, numerical-error containment at the
environment level, legacy migration, strict loading, and cache identity.
"""

import copy
import json
import os

import numpy as np
import pytest

import penalized_saito
from arrangement import LineArrangement, ProjectiveLine
from calibration import CalibrationError, compute_tau, freeness_potential
from certificates import find_exact_saito_certificate, verify_certificate
from promotion import (build_discovery_entry, promote, migrate_legacy_store,
                       load_verified_discoveries)
from saito import construct_supersolvable
from swap_env import SwapArrangementEnv
from swap_search import ChainEvaluator, double_pencil_seed


def arr_from(coords):
    return LineArrangement([ProjectiveLine(*c) for c in coords])


BRAID = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (1, 0, -1), (0, 1, -1)]


# ── raw/calibrated separation ────────────────────────────────────────────────

def test_default_is_raw_phi_no_implicit_calibration(tmp_path):
    env = SwapArrangementEnv(target_n=9, d1=3, d2=5, seed=0, episode_len=2,
                             max_candidates=16)
    assert env.tau is None                       # DEFAULT: raw potential
    obs = env.reset()
    a = int(np.flatnonzero(env.action_mask() > 0)[0])
    _, r, _, info = env.step(a)
    # raw phi: potential equals 1 - raw loss exactly
    assert info["tau"] is None
    assert abs(info["calibrated_potential"]
               - (1.0 - info["raw_loss"])) < 1e-15
    # a cached tau file on disk must NOT activate calibration implicitly
    cache = tmp_path / "tau_cache.json"
    cache.write_text(json.dumps({"any_key": {"tau": 0.1}}))
    env2 = SwapArrangementEnv(target_n=9, d1=3, d2=5, seed=0, episode_len=2,
                              max_candidates=16)
    assert env2.tau is None


def test_explicit_tau_activates_and_logs_both():
    env = SwapArrangementEnv(target_n=9, d1=3, d2=5, seed=0, episode_len=2,
                             max_candidates=16, tau=0.05)
    env.reset()
    a = int(np.flatnonzero(env.action_mask() > 0)[0])
    _, r, _, info = env.step(a)
    assert info["tau"] == 0.05
    assert info["raw_loss"] is not None
    assert abs(info["calibrated_potential"]
               - freeness_potential(info["raw_loss"], 0.05)) < 1e-15


def test_engines_and_gate_use_raw_loss():
    ev = ChainEvaluator(9, 3, 5)
    arr = construct_supersolvable(9, 3)
    loss = ev.screen_loss(arr)
    comp = ev.energy_components(arr, loss)
    for key in ("raw_saito_loss", "b2_shell_penalty", "b2_shell_weight",
                "total_energy"):
        assert key in comp
    assert comp["raw_saito_loss"] == loss        # RAW, never calibrated
    assert abs(comp["total_energy"] - (loss + comp["b2_shell_weight"]
                                       * comp["b2_shell_penalty"])) < 1e-15


def test_exact_verification_independent_of_calibration():
    cert = find_exact_saito_certificate(arr_from(BRAID))
    assert verify_certificate(cert)              # no calibration anywhere
    # certificates carry no tau/calibration fields
    assert "tau" not in cert and "calibrated" not in str(sorted(cert))


def test_resume_guard_refuses_mode_switch(tmp_path):
    from experiments.train_swap_policy import main as train_main
    out = tmp_path / "run"
    out.mkdir()
    (out / "manifest.json").write_text(json.dumps(
        {"args": {"tau_mode": "median"}, "tau": 0.07}))
    import sys
    argv = sys.argv
    sys.argv = ["prog", "--n", "9", "--d1", "3", "--d2", "5",
                "--updates", "1", "--tau-mode", "none", "--out", str(out)]
    try:
        with pytest.raises(SystemExit, match="resume conflict"):
            train_main()
    finally:
        sys.argv = argv


# ── terminal-potential convention ────────────────────────────────────────────

def test_terminal_phi_zero_convention():
    env = SwapArrangementEnv(target_n=9, d1=3, d2=5, seed=3, episode_len=1,
                             max_candidates=16, certify_below=-1.0)
    env.reset()                                   # certify gate disabled
    phi_prev = env._phi_prev
    a = int(np.flatnonzero(env.action_mask() > 0)[0])
    _, r, done, info = env.step(a)
    assert done
    # final transition uses Phi(terminal) = 0:
    # reward = eta * (gamma * 0 - Phi(s_prev))
    assert abs(r - env.eta * (0.0 - phi_prev)) < 1e-12
    # phi of the resulting state is still computed and logged
    assert info["calibrated_potential"] is not None


def test_terminal_bonus_independent_of_loss(monkeypatch):
    """The exact-certification bonus comes only from certify_state."""
    env = SwapArrangementEnv(target_n=9, d1=3, d2=5, seed=4, episode_len=1,
                             max_candidates=16, certify_below=2.0)
    env.reset()                                   # gate always open
    import swap_env as se
    monkeypatch.setattr(se, "certify_state", lambda *a: None)
    a = int(np.flatnonzero(env.action_mask() > 0)[0])
    _, r, done, info = env.step(a)
    assert "certificate" not in info              # no cert -> no bonus


# ── environment-level numerical-error containment ────────────────────────────

def test_env_numerical_error_is_structured_no_fabricated_phi(monkeypatch):
    env = SwapArrangementEnv(target_n=9, d1=3, d2=5, seed=5, episode_len=10,
                             max_candidates=16)
    env.reset()
    calls = {"n": 0}

    def _boom(arr):
        calls["n"] += 1
        raise penalized_saito.GammaNumericalError("forced")
    monkeypatch.setattr(env, "_raw_loss", _boom)
    rewards = []
    for _ in range(env.max_consecutive_errors):
        a = int(np.flatnonzero(env.action_mask() > 0)[0])
        _, r, done, info = env.step(a)
        rewards.append(r)
        assert info["numerical_error"] is True
        assert info["raw_loss"] is None           # NO fabricated loss
        assert info["calibrated_potential"] is None
        assert not np.isnan(r)                    # penalty component, not NaN
        assert r == -env.numerical_failure_penalty
        if done:
            break
    assert done and info["episode_status"] == "evaluator_error"
    assert env.numerical_error_count >= env.max_consecutive_errors


def test_errors_never_cached_or_in_tau(monkeypatch, tmp_path):
    from penalized_saito import _LOSS_CACHE

    def _always_error(*a, **k):
        raise penalized_saito.GammaNumericalError("forced")
    n0 = len(_LOSS_CACHE)
    monkeypatch.setattr(penalized_saito, "penalized_saito_loss",
                        _always_error)
    with pytest.raises(penalized_saito.GammaNumericalError):
        # unique seed -> guaranteed cache miss -> the poisoned evaluator runs
        penalized_saito.cached_penalized_loss(
            construct_supersolvable(9, 3), d1=3, d2=5, seed=777)
    assert len(_LOSS_CACHE) == n0                 # error NOT cached
    with pytest.raises(CalibrationError):
        compute_tau(9, 3, 5, n_samples=10,
                    cache_path=str(tmp_path / "t.json"))


# ── migration + strict loading ───────────────────────────────────────────────

def _mixed_store(tmp_path):
    arr = construct_supersolvable(9, 3)
    cert = find_exact_saito_certificate(arr, target_exponents=(3, 5))
    good = build_discovery_entry(cert, run_id="mig", search_params={
        "lambda": 1.0, "beta": 0.75, "field": "real"})
    store = str(tmp_path / "discoveries.json")
    promote([good], store)
    data = json.load(open(store))
    data["arrangements"].append({"n": 6, "exponents": [1, 2, 3],
                                 "lines": ["(1x+0y+0z=0)"],
                                 "source": "legacy_rl"})
    data["arrangements"].append({"broken": True})
    bad = copy.deepcopy(good)
    bad["certificate"]["c"] = "0"                 # fails reverification
    data["arrangements"].append(bad)
    with open(store, "w") as f:
        json.dump(data, f, default=str)
    return store, good


def test_migration_dry_run_and_mixed(tmp_path):
    store, good = _mixed_store(tmp_path)
    dry = migrate_legacy_store(store, dry_run=True)
    assert (dry["n_verified"], dry["n_legacy"], dry["n_malformed"]) \
        == (1, 1, 2)
    assert not os.path.exists(store + f".backup.{dry['source_checksum'][:16]}")
    res = migrate_legacy_store(store)
    assert os.path.exists(res["backup"])          # backup preserved
    ok, rejects = load_verified_discoveries(store)
    assert len(ok) == 1 and ok[0]["discovery_id"] == good["discovery_id"]
    legacy = json.load(open(str(tmp_path / "legacy_candidates.json")))
    assert len(legacy["arrangements"]) == 1
    assert legacy["arrangements"][0]["_migration"]["reason"] \
        == "legacy_unverified_by_promoter"
    quarantine = json.load(open(str(tmp_path
                                    / "legacy_quarantine_report.json")))
    assert len(quarantine["quarantined"]) == 2    # malformed + tampered
    # no data loss: total records preserved across the three files + backup
    assert 1 + 1 + 2 == res["n_total"]
    # restartable: second run is a no-op on the already-clean store
    res2 = migrate_legacy_store(store)
    assert res2["already_migrated"] and res2["n_verified"] == 1


def test_strict_loader_rejects_everything_unverified(tmp_path):
    store = str(tmp_path / "d.json")
    with open(store, "w") as f:
        json.dump({"arrangements": [
            {"n": 5, "lines": ["x"], "source": "legacy"},
            {"schema_version": "discovery-2.0",
             "verification_status": "legacy_unverified_by_promoter",
             "lines": ["x"]},
        ], "index": {}}, f)
    ok, rejects = load_verified_discoveries(store)
    assert ok == [] and len(rejects) == 2


# ── cache identity ───────────────────────────────────────────────────────────

def test_identity_hash_in_cache_key(monkeypatch):
    from penalized_saito import _LOSS_CACHE, cached_penalized_loss
    arr = arr_from(BRAID)
    cached_penalized_loss(arr, d1=2, d2=3, profile="rl", seed=0)
    n1 = len(_LOSS_CACHE)
    # changing the evaluator identity (source/dependency hash) is a MISS
    monkeypatch.setattr(penalized_saito, "_IDENTITY_HASH", "deadbeef")
    cached_penalized_loss(arr, d1=2, d2=3, profile="rl", seed=0)
    assert len(_LOSS_CACHE) == n1 + 1
