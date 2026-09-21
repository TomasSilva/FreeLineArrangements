"""Ziegler-pair fixture gate: same lattice, special (6 triple points on a
smooth conic) vs generic realization; both provably non-free (W1); the
conic detector is the only separator.  Negative control for the whole
Terao-stress pipeline."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from known_arrangements import ziegler_pair, validate_ziegler_pair  # noqa: E402
from novelty import lattices_isomorphic, lattice_wl_hash            # noqa: E402
from geometry import special_position_profile                       # noqa: E402
from certificates import (nonfreeness_certificate,                  # noqa: E402
                          verify_nonfreeness_certificate)


def test_gate():
    special, generic = ziegler_pair()
    assert validate_ziegler_pair(special, generic)


def test_lattices_isomorphic():
    special, generic = ziegler_pair()
    assert lattice_wl_hash(special) == lattice_wl_hash(generic)
    assert lattices_isomorphic(special, generic)


def test_conic_detector_separates():
    special, generic = ziegler_pair()
    ps = special_position_profile(special)
    pg = special_position_profile(generic)
    assert ps['n_multiple_points'] == 6 == pg['n_multiple_points']
    smooth = [c for c in ps['conics']
              if c['size'] == 6 and not c['degenerate'] and not c['implied']]
    assert len(smooth) == 1
    assert pg['conics'] == []


def test_both_nonfree_with_w1_witness():
    for arr in ziegler_pair():
        w, status = nonfreeness_certificate(arr)
        assert status == 'nonfree_certified'
        assert w['method'] == 'terao_factorization_obstruction'
        assert verify_nonfreeness_certificate(w)


def test_loss_positive_on_both():
    from penalized_saito import penalized_saito_loss_all_pairs
    for arr in ziegler_pair():
        loss = penalized_saito_loss_all_pairs(arr, n_restarts=4, n_iters=40,
                                              seed=0)
        assert float(loss) > 1e-3
