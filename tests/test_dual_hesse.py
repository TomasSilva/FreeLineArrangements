"""
End-to-end ground truth: the dual Hesse arrangement over Q(sqrt(-3)) —
9 lines, 12 triple points, free with exponents (1, 4, 4); the canonical
free-but-not-inductively-free arrangement, not realizable over R.
Exercises the COMPLEX quadratic path.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from certificates import (find_certificate_fast, verify_certificate,
                          certificate_to_json, certificate_from_json,
                          certificate_to_bw_vectors)
from known_arrangements import dual_hesse, validate_dual_hesse_lattice
from novelty import is_supersolvable_rank3, is_essential, coordinate_height
from quadfield import QuadraticField


def test_lattice_validates():
    H = dual_hesse()
    assert H.coefficient_field() == QuadraticField(-3)
    assert validate_dual_hesse_lattice(H)
    assert is_essential(H)
    assert not is_supersolvable_rank3(H)
    assert coordinate_height(H) >= 1


def test_exact_certificate_and_roundtrip():
    H = dual_hesse()
    cert, status = find_certificate_fast(H, target_exponents=(4, 4))
    assert status == "certified"
    assert verify_certificate(cert)
    cj = json.loads(json.dumps(certificate_to_json(cert)))
    assert cj["field"]["d"] == -3
    assert verify_certificate(certificate_from_json(cj))


def test_bw_vectors_are_complex():
    H = dual_hesse()
    cert, _ = find_certificate_fast(H, target_exponents=(4, 4))
    u, v = certificate_to_bw_vectors(cert)
    assert u.dtype == np.complex128 and v.dtype == np.complex128
    assert abs(np.linalg.norm(u) - 1) < 1e-12
    assert abs(np.linalg.norm(v) - 1) < 1e-12


def test_float_embedding_raises_for_complex_field():
    H = dual_hesse()
    with pytest.raises(TypeError):
        H.lines[1].to_float()
    emb = H.lines[1].embed()
    assert emb.dtype == np.complex128
