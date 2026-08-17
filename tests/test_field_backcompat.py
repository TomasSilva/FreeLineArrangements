"""
Phase-0 backward-compatibility goldens for the quadratic-field extension.

These values were computed at commit 5349a97 (FUNCTIONAL_VERSION 2.4.0,
pure-QQ pipeline) BEFORE any field-extension change.  The QQ code path must
remain byte-identical: exact float equality on losses, exact string
equality on canonical keys/hashes/ids.  If a field-extension change breaks
one of these, the QQ fast path drifted — fix the change, never the golden.
"""

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arrangement import LineArrangement, ProjectiveLine
from penalized_saito import cached_penalized_loss
from certificates import (find_exact_saito_certificate, certificate_to_json,
                          certificate_from_json, verify_certificate)
from novelty import canonical_lineset_key, lattice_wl_hash
from promotion import canonical_discovery_id
from saito import construct_supersolvable

BRAID = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (1, 0, -1), (0, 1, -1)]
NONFREE7 = [(0, 1, -1), (1, 1, -1), (1, 1, 1), (1, 1, 2), (1, 1, 0),
            (1, 1, -2), (1, 0, -2)]

GOLDEN = {
    "loss_braid_23": "1.1102230246251565e-16",
    "loss_nf7_33": "0.10040349410524196",
    "loss_ss9_35": "0.0",
    "key_braid": "('(0, 0, 1)', '(0, 1, -1)', '(0, 1, 0)', '(1, -1, 0)', "
                 "'(1, 0, -1)', '(1, 0, 0)')",
    "wl_braid": "b6dd740d384757b5d4098af8bf2f61a3",
    "wl_ss9": "0c117966ca680b556bb22b708e4ee77f",
    "did_braid":
        "0e232012bd1ba4c94c2737e0d78fb47aa1da863c23b4e7a39999f2ccd16ab2bb",
    "cert_braid_c": "-1",
}


def _arr(coords):
    return LineArrangement([ProjectiveLine(*c) for c in coords])


def test_qq_loss_goldens_exact():
    braid, nf7 = _arr(BRAID), _arr(NONFREE7)
    ss9 = construct_supersolvable(9, 3)
    assert repr(cached_penalized_loss(braid, 2, 3, profile="rl",
                                      seed=0)) == GOLDEN["loss_braid_23"]
    assert repr(cached_penalized_loss(nf7, 3, 3, profile="rl",
                                      seed=0)) == GOLDEN["loss_nf7_33"]
    assert repr(cached_penalized_loss(ss9, 3, 5, profile="rl",
                                      seed=0)) == GOLDEN["loss_ss9_35"]


def test_qq_canonical_keys_golden():
    braid = _arr(BRAID)
    assert canonical_lineset_key(braid) == GOLDEN["key_braid"]
    assert lattice_wl_hash(braid) == GOLDEN["wl_braid"]
    assert lattice_wl_hash(construct_supersolvable(9, 3)) == GOLDEN["wl_ss9"]
    assert canonical_discovery_id(braid) == GOLDEN["did_braid"]


def test_qq_certificate_roundtrip_golden():
    cert = find_exact_saito_certificate(_arr(BRAID))
    cj = certificate_to_json(cert)
    assert str(cj["c"]) == GOLDEN["cert_braid_c"]
    assert (cj.get("d1"), cj.get("d2")) == (2, 3)
    assert cj["lines"][0] == ["1", "0", "0"]
    assert verify_certificate(certificate_from_json(cj))


def test_qq_path_never_constructs_quadelem():
    """After Phase 1, rational arrangements must stay pure-Rational."""
    try:
        from quadfield import QuadElem
    except ImportError:
        pytest.skip("quadfield not yet present (pre-Phase-1)")
    braid = _arr(BRAID)
    for line in braid.lines:
        for c in line.coords:
            assert not isinstance(c, QuadElem)
    for p in braid.intersection_points():
        for c in p:
            assert not isinstance(c, QuadElem)


def test_qq_hot_path_timing_guard():
    """Loose guard: ProjectiveLine + lattice structure must stay fast."""
    t0 = time.time()
    for _ in range(200):
        arr = _arr(NONFREE7)
        arr._structure()
    assert time.time() - t0 < 5.0
