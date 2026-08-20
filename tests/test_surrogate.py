"""Surrogate-guided search: feature extraction, DKP bound helper, validity
ceiling, m-target energy, ranker discipline, training/checkpoint sanity."""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from arrangement import LineArrangement, ProjectiveLine
from known_arrangements import akn13
from saito import construct_supersolvable
from surrogate import (extract_features, FEATURE_NAMES, N_FEATURES,
                       train_surrogate, SurrogateRanker,
                       FEATURE_SCHEMA_VERSION)
from swap_search import (min_feasible_m, is_valid_state, ChainEvaluator,
                         propose_swaps)

BRAID = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (1, 0, -1), (0, 1, -1)]


def _arr(coords):
    return LineArrangement([ProjectiveLine(*c) for c in coords])


def test_feature_extraction_shape_and_determinism():
    braid = _arr(BRAID)
    x1 = extract_features(braid, 2, 3)
    x2 = extract_features(braid, 2, 3)
    assert x1.shape == (N_FEATURES,) == (len(FEATURE_NAMES),)
    assert np.array_equal(x1, x2) and np.all(np.isfinite(x1))
    # field one-hot: braid is QQ; akn13 is d=3
    qq_slot = FEATURE_NAMES.index("field_QQ")
    d3_slot = FEATURE_NAMES.index("field_3")
    assert x1[qq_slot] == 1.0 and x1[d3_slot] == 0.0
    xa = extract_features(akn13(), 6, 6)
    assert xa[qq_slot] == 0.0 and xa[d3_slot] == 1.0


def test_min_feasible_m_dkp_bound():
    # DKP Prop 3.1: m >= 2n/(d1+2)
    assert min_feasible_m(20, 9) == 4      # eps = 5 window at (20,9,10)
    assert min_feasible_m(18, 8) == 4
    assert min_feasible_m(21, 10) == 4
    assert min_feasible_m(13, 6) == 4      # matches DKP A13 (m = 4)
    assert min_feasible_m(14, 6) == 4      # matches DKP C14


def test_validity_ceiling():
    ss = construct_supersolvable(9, 3)     # has a high-multiplicity point
    assert is_valid_state(ss, 9, nontrivial=True)
    assert not is_valid_state(ss, 9, nontrivial=True, max_mult=3)


def test_m_target_energy_component():
    ev = ChainEvaluator(9, 3, 5, seed=0, m_target=3, w_m=0.1)
    ss = construct_supersolvable(9, 3)
    comps = ev.energy_components(ss, 0.5)
    assert comps["m_target_penalty"] == max(0, ss.max_multiplicity() - 3)
    assert comps["m_target_weight"] == 0.1
    assert abs(comps["total_energy"]
               - (0.5 + comps["b2_shell_weight"] * comps["b2_shell_penalty"]
                  + 0.1 * comps["m_target_penalty"])) < 1e-12
    # raw loss stays a separate, untouched component
    assert comps["raw_saito_loss"] == 0.5
    # without m_target the component is zero and energy is unchanged
    ev0 = ChainEvaluator(9, 3, 5, seed=0)
    c0 = ev0.energy_components(ss, 0.5)
    assert c0["m_target_penalty"] == 0.0 and c0["m_target_weight"] == 0.0


class _FakeRanker:
    def __init__(self):
        self.calls = 0

    def rank(self, arrangements, d1, d2):
        self.calls += 1
        # deterministic arbitrary scores (by lattice size proxy)
        return np.array([float(len(a.intersection_points()))
                         for a in arrangements])


def test_ranker_discipline_and_off_identity():
    arr = construct_supersolvable(9, 3)
    rng1 = np.random.default_rng(5)
    rng2 = np.random.default_rng(5)
    base = propose_swaps(arr, 3, 5, rng1, n_remove=4)
    base2 = propose_swaps(arr, 3, 5, rng2, n_remove=4)
    # ranker=None is deterministic and unchanged
    assert [(i, l.coords) for i, l, _ in base] == \
           [(i, l.coords) for i, l, _ in base2]
    fake = _FakeRanker()
    rng3 = np.random.default_rng(5)
    ranked = propose_swaps(arr, 3, 5, rng3, n_remove=4, ranker=fake,
                           ranked_k=8)
    assert fake.calls == 1
    assert 0 < len(ranked) <= 8
    for (i, line, trial) in ranked:
        assert is_valid_state(trial, 9, nontrivial=True)


def test_ranker_respects_max_mult():
    arr = construct_supersolvable(10, 4)
    rng = np.random.default_rng(1)
    props = propose_swaps(arr, 4, 5, rng, n_remove=6, max_mult=6)
    for (_, _, trial) in props:
        assert trial.max_multiplicity() <= 6


def _synthetic_dataset(path, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, N_FEATURES))
    signal = X[:, 4] * 2.0 + X[:, 5]            # planted structure
    y_log = signal + 0.1 * rng.standard_normal(n)
    y_cls = (y_log < -1.0).astype(float)
    holdout = np.zeros(n, dtype=bool)
    holdout[: n // 5] = True
    np.savez(path, X=X, y_log=y_log, y_cls=y_cls, holdout=holdout,
             cells=np.array(["9,3,5"] * n), manifest_hash="synthetic")
    return path


def test_training_and_checkpoint_roundtrip(tmp_path):
    ds = _synthetic_dataset(str(tmp_path / "ds.npz"))
    out = str(tmp_path / "model.pt")
    metrics = train_surrogate(ds, out, epochs=8, batch=512, verbose=False)
    assert metrics["r2_logloss"] > 0.5           # planted signal learned
    assert metrics["auc_certifiable"] > 0.7
    ranker = SurrogateRanker.load(out)
    assert ranker.provenance["holdout_metrics"]["r2_logloss"] > 0.5
    scores = ranker.rank([_arr(BRAID), akn13()], 2, 3)
    assert scores.shape == (2,) and np.all(np.isfinite(scores))
    # schema mismatch must refuse to load
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    ckpt["feature_schema"] = "wrong-schema"
    bad = str(tmp_path / "bad.pt")
    torch.save(ckpt, bad)
    with pytest.raises(ValueError):
        SurrogateRanker.load(bad)
