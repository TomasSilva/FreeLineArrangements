"""Non-freeness witnesses: soundness, completeness on fixtures, JSON
round-trip, tamper detection.  A witness must be IMPOSSIBLE on free input
and re-verifiable from its lines alone."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from certificates import (nonfreeness_certificate,          # noqa: E402
                          verify_nonfreeness_certificate,
                          nonfreeness_to_json, nonfreeness_from_json,
                          free_module_dims, _dim_S)


def _roundtrip(w):
    return nonfreeness_from_json(json.loads(json.dumps(nonfreeness_to_json(w))))


# ── nonfree fixtures must get valid witnesses ────────────────────────────────

def test_nonfree7_witness(nonfree7):
    w, status = nonfreeness_certificate(nonfree7)
    assert status == 'nonfree_certified'
    assert w['forced_pair'] == [3, 3]
    assert w['method'] in ('forced_pair_degree_dim',
                           'forced_pair_matrix_zero')
    assert verify_nonfreeness_certificate(_roundtrip(w))


def test_nonfree7b_witness(nonfree7b):
    w, status = nonfreeness_certificate(nonfree7b)
    assert status == 'nonfree_certified'
    assert verify_nonfreeness_certificate(_roundtrip(w))


def test_generic4_gets_terao_obstruction_witness(generic4):
    w, status = nonfreeness_certificate(generic4)
    assert status == 'nonfree_certified'
    assert w['method'] == 'terao_factorization_obstruction'
    assert w['forced_pair'] is None
    assert verify_nonfreeness_certificate(_roundtrip(w))


# ── free fixtures must NEVER get a witness ───────────────────────────────────

def test_braid_no_witness(braid):
    w, status = nonfreeness_certificate(braid)
    assert w is None
    assert status == 'free_or_unresolved'


def test_a2xa1_no_witness(a2xa1):
    w, status = nonfreeness_certificate(a2xa1)
    assert w is None
    assert status == 'free_or_unresolved'


# ── tamper detection ─────────────────────────────────────────────────────────

def test_tampered_dim_fails_verify(nonfree7):
    w, status = nonfreeness_certificate(nonfree7)
    assert status == 'nonfree_certified'
    t = dict(w)
    if t['method'] == 'forced_pair_degree_dim':
        t['dim_computed'] = int(t['dim_computed']) + 1
    else:
        t['b2'] = int(t['b2']) + 1
    assert not verify_nonfreeness_certificate(t)


def test_tampered_lines_fail_verify(nonfree7, braid):
    w, status = nonfreeness_certificate(nonfree7)
    assert status == 'nonfree_certified'
    t = dict(w)
    t['lines'] = [str(l) for l in braid.lines]      # free arrangement's lines
    assert not verify_nonfreeness_certificate(t)


def test_tampered_method_fails_verify(generic4):
    w, status = nonfreeness_certificate(generic4)
    t = dict(w)
    t['method'] = 'nonsense_method'
    assert not verify_nonfreeness_certificate(t)


# ── dimension arithmetic sanity on certified free fixtures ───────────────────

def test_free_dimension_matches_prediction(braid):
    # braid: free, exponents (1, 2, 3)
    assert braid.candidate_exponents() == (2, 3)
    assert braid.derivation_space_dim(2) == free_module_dims(2, 2, 3)
    assert braid.derivation_space_dim(3) == free_module_dims(3, 2, 3)


def test_trivial_dim_formula():
    assert _dim_S(0) == 1
    assert _dim_S(1) == 3
    assert _dim_S(-1) == 0
    assert free_module_dims(3, 3, 3) == _dim_S(2) + 2 * _dim_S(0)
