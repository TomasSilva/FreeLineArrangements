"""Non-divisional-directed search machinery (Terao candidates): lattice
helper, opt-in energy penalty, MAP-Elites descriptor bit, certification
priority, ordered seed dirs.  Everything must be inert at w_div = 0."""

import gzip
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from arrangement import LineArrangement, ProjectiveLine       # noqa: E402
from novelty import (n_divisional_lines, points_per_line,     # noqa: E402
                     expected_moduli_dim_rank3)
from swap_search import ChainEvaluator, descriptor, _record   # noqa: E402

DB = os.path.join(ROOT, "certified_free_arrangements.jsonl.gz")
NONDIV_15 = "9aed30e359fd"     # n=15, (7,7), rational, NOT divisionally free


def _db_arr(prefix):
    from certificates import _parse_exact_scalar
    from quadfield import QuadraticField
    with gzip.open(DB, "rt") as f:
        next(f)
        for line in f:
            if f'"lattice_hash": "{prefix}' not in line:
                continue
            e = json.loads(line)
            K = QuadraticField.from_json(e["field"])
            return LineArrangement(
                [ProjectiveLine(*[_parse_exact_scalar(v, K) for v in c])
                 for c in e["lines"]]), e
    raise AssertionError(prefix)


# ── lattice helper ───────────────────────────────────────────────────────────

def test_braid_is_divisional(braid):
    # braid A3, exponents (2, 3): every line carries 3 points -> 3-1 = 2
    assert points_per_line(braid) == [3] * 6
    assert n_divisional_lines(braid, 2, 3) == 6


@pytest.mark.skipif(not os.path.exists(DB), reason="database not present")
def test_known_nondivisional_lattice():
    arr, e = _db_arr(NONDIV_15)
    d1, d2 = e["exponents"][1], e["exponents"][2]
    assert (d1, d2) == (7, 7)
    assert n_divisional_lines(arr, d1, d2) == 0
    assert all(c - 1 not in (7,) for c in points_per_line(arr))


def test_expected_moduli_dim_braid(braid):
    assert expected_moduli_dim_rank3(braid) == 0


# ── energy: inert at w_div = 0, additive when on ─────────────────────────────

def test_energy_off_is_unchanged(braid):
    ev0 = ChainEvaluator(6, 2, 3)
    comps = ev0.energy_components(braid, 0.25)
    assert "divisional_lines" not in comps          # records byte-identical
    assert abs(ev0.energy(braid, 0.25) - comps["total_energy"]) < 1e-15


def test_energy_on_adds_penalty(braid):
    ev0 = ChainEvaluator(6, 2, 3)
    ev = ChainEvaluator(6, 2, 3, w_div=0.1)
    e0, e1 = ev0.energy(braid, 0.25), ev.energy(braid, 0.25)
    assert abs((e1 - e0) - 0.1 * 6 / 6) < 1e-15
    comps = ev.energy_components(braid, 0.25)
    assert comps["divisional_lines"] == 6
    assert comps["divisional_weight"] == 0.1
    assert abs(comps["total_energy"] - e1) < 1e-15
    assert comps["raw_saito_loss"] == 0.25          # raw loss untouched


# ── descriptor bit ───────────────────────────────────────────────────────────

def test_descriptor_default_unchanged(braid):
    d = descriptor(braid, 6)
    assert len(d) == 3
    assert descriptor(braid, 6, pair=(2, 3)) == d + (0,)   # divisional


@pytest.mark.skipif(not os.path.exists(DB), reason="database not present")
def test_descriptor_nondivisional_bit():
    arr, _ = _db_arr(NONDIV_15)
    assert descriptor(arr, 15, pair=(7, 7))[-1] == 1


# ── certification priority ───────────────────────────────────────────────────

def test_divisional_candidates_deferred_beyond_cap(braid, tmp_path):
    from run_swap_campaign import CampaignIO
    ev = ChainEvaluator(6, 2, 3, w_div=0.1)
    rec = _record(braid, 2, 3, 1e-12, "test", 0,
                  extra=ev.energy_components(braid, 1e-12))
    io = CampaignIO(str(tmp_path / "capped"), 6, 2, 3, "test", 0,
                    div_cert_cap=0)
    io.on_candidate(dict(rec))
    assert io.counters["certified"] == 0
    assert io.counters["div_cert_deferred"] == 1
    assert io.counters["candidates"] == 1           # recorded, never lost
    assert os.path.exists(tmp_path / "capped" / "candidates.jsonl")

    io2 = CampaignIO(str(tmp_path / "open"), 6, 2, 3, "test", 0,
                     div_cert_cap=5)
    io2.on_candidate(dict(rec))
    assert io2.counters["certified"] == 1
    entry = json.loads(open(tmp_path / "open" / "certified.jsonl").readline())
    assert entry["divisional"] is True and entry["divisional_lines"] == 6
    assert "expected_moduli_dim" in entry


def test_no_priority_screen_by_default(braid, tmp_path):
    from run_swap_campaign import CampaignIO
    ev = ChainEvaluator(6, 2, 3)
    rec = _record(braid, 2, 3, 1e-12, "test", 0,
                  extra=ev.energy_components(braid, 1e-12))
    io = CampaignIO(str(tmp_path / "default"), 6, 2, 3, "test", 0)
    io.on_candidate(dict(rec))
    assert io.counters["certified"] == 1
    entry = json.loads(
        open(tmp_path / "default" / "certified.jsonl").readline())
    assert "divisional" not in entry                # default records unchanged


# ── ordered seed directories ─────────────────────────────────────────────────

def test_seed_dirs_first_match_wins():
    import run_swap_campaign as rsc
    saved = list(rsc.SEEDS_DIRS)
    try:
        rsc.SEEDS_DIRS[:] = ["swap_nondiv_seeds", "swap_lift_seeds"]
        nd = rsc.load_lift_seeds(15, 7, 7, ROOT)
        assert len(nd) == 2
        assert all(n_divisional_lines(a, 7, 7) == 0 for a in nd)
        # no designated file for this cell -> falls back to general seeds
        fb = rsc.load_lift_seeds(29, 14, 14, ROOT)
        assert len(fb) >= 1
    finally:
        rsc.SEEDS_DIRS[:] = saved


def test_directed_seed_mode_uses_only_designated_seeds():
    import run_swap_campaign as rsc
    saved = list(rsc.SEEDS_DIRS)
    try:
        rsc.SEEDS_DIRS[:] = ["swap_nondiv_seeds", "swap_lift_seeds"]
        rng = np.random.default_rng(0)
        seeds = rsc.build_seeds(15, 7, 7, rng, 3, mode="directed",
                                repo_root=ROOT)
        assert len(seeds) >= 2
        # the two designated lattices come first, untouched
        assert all(n_divisional_lines(a, 7, 7) == 0 for a in seeds[:2])
    finally:
        rsc.SEEDS_DIRS[:] = saved
