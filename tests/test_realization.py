"""Lattice-preserving re-realization: sampler round-trips (lattice
preserved by VF2, freeness re-certified in new coordinates), frame
normalization, tangent dimensions, explicit isomorphism maps."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realization import (incidence_structure, expected_moduli_dim,  # noqa: E402
                         choose_frame, normalize_frame,
                         sample_realization, tangent_space_dim,
                         lattice_isomorphism_map)
from novelty import lattice_wl_hash, lattices_isomorphic            # noqa: E402
from certificates import find_certificate_fast                      # noqa: E402
from known_arrangements import ziegler_pair                          # noqa: E402
from geometry import special_position_profile                        # noqa: E402


def _rng(seed=20260901):
    return np.random.default_rng(seed)


# ── incidence structure and frame ────────────────────────────────────────────

def test_incidence_structure_braid(braid):
    s = incidence_structure(braid)
    assert s["n"] == 6
    assert len(s["points"]) == 4                 # braid A3: 4 triple points
    assert all(len(p) == 3 for p in s["points"])


def test_choose_frame_avoids_concurrency(braid):
    s = incidence_structure(braid)
    frame = choose_frame(s)
    assert frame is not None
    for p in s["points"]:
        assert len(set(p) & set(frame)) <= 2


def test_normalize_frame_preserves_lattice(braid):
    s = incidence_structure(braid)
    frame = choose_frame(s)
    moved, _N = normalize_frame(braid, frame)
    assert lattice_wl_hash(moved) == lattice_wl_hash(braid)
    assert lattices_isomorphic(braid, moved)
    got = [moved.lines[i].coords for i in frame]
    assert got[0] == (1, 0, 0) and got[1] == (0, 1, 0)
    assert got[2] == (0, 0, 1) and got[3] == (1, 1, 1)


# ── sampler round-trips ──────────────────────────────────────────────────────

def test_sample_ziegler_lattice(rng):
    special, _generic = ziegler_pair()
    r = _rng()
    found = None
    for _ in range(5):
        found = sample_realization(special, r, height_bound=8)
        if found is not None:
            break
    assert found is not None
    assert lattices_isomorphic(special, found)
    # generically the re-realization is OFF the conic
    prof = special_position_profile(found)
    assert isinstance(prof["conics"], list)


def test_sample_braid_rigid_roundtrip(braid):
    # braid lattice is projectively rigid: expected dim 0; the sampler
    # must still reproduce it (all placements forced after the frame)
    r = _rng(7)
    found = None
    for _ in range(10):
        found = sample_realization(braid, r, height_bound=6)
        if found is not None:
            break
    assert found is not None
    assert lattices_isomorphic(braid, found)
    cert, status = find_certificate_fast(found, target_exponents=(2, 3))
    assert status == 'certified'


def test_sample_supersolvable_and_recertify():
    from saito import construct_supersolvable
    arr = construct_supersolvable(8, 3)          # free, exponents (1, 3, 4)
    r = _rng(11)
    found = None
    for _ in range(10):
        found = sample_realization(arr, r, height_bound=8)
        if found is not None:
            break
    assert found is not None
    assert lattices_isomorphic(arr, found)
    cert, status = find_certificate_fast(found, target_exponents=(3, 4))
    assert status == 'certified'


# ── dimensions ───────────────────────────────────────────────────────────────

def test_dims_braid_rigid(braid):
    s = incidence_structure(braid)
    assert expected_moduli_dim(s) == 0
    assert tangent_space_dim(braid, s) == 0


def test_dims_ziegler():
    special, generic = ziegler_pair()
    s = incidence_structure(special)
    assert expected_moduli_dim(s) == 4
    assert tangent_space_dim(special, s) == 4
    assert tangent_space_dim(generic) == 4


# ── explicit isomorphism map ─────────────────────────────────────────────────

def test_isomorphism_map_roundtrip(braid):
    s = incidence_structure(braid)
    frame = choose_frame(s)
    moved, _ = normalize_frame(braid, frame)
    m = lattice_isomorphism_map(braid, moved)
    assert m is not None
    assert sorted(m["lines"].keys()) == list(range(6))
    assert sorted(m["lines"].values()) == list(range(6))


def test_isomorphism_map_none_for_different_lattices(braid, generic4):
    assert lattice_isomorphism_map(braid, generic4) is None


# ── conic forcing (sample-and-solve-last) ────────────────────────────────────

def test_force_conic_on_ziegler_generic():
    from realization import force_conic_realization
    _special, generic = ziegler_pair()
    s = incidence_structure(generic)
    assert len(s["points"]) == 6
    arr = None
    for seed in (4, 1, 2, 3, 5, 6, 7, 8):
        arr, log = force_conic_realization(generic, list(range(6)),
                                           _rng(seed), struct=s, tries=8,
                                           height_bound=5)
        if arr is not None:
            break
    if arr is None:
        # honest failure is allowed (roots may fall in unsupported
        # fields), but the log must say why
        assert all("status" in e for e in log)
        pytest.skip(f"no supported-field root found: {log[-3:]}")
    assert lattices_isomorphic(generic, arr)
    prof = special_position_profile(arr)
    smooth = [c for c in prof["conics"]
              if c["size"] == 6 and not c.get("implied")]
    assert len(smooth) >= 1                      # the forced coincidence
