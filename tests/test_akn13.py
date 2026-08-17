"""
End-to-end ground truth: the Abe-Kawanoue-Nozawa A(13) arrangement
(arXiv:1406.5820) over Q(sqrt(3)) — free with exponents (1, 6, 6), NOT
inductively free (the paper proves the stronger: not recursively free).
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments"))

from sympy import Rational

from arrangement import LineArrangement, ProjectiveLine
from certificates import (find_certificate_fast, verify_certificate,
                          certificate_to_json, certificate_from_json,
                          classify_freeness, FREE_TARGET)
from known_arrangements import akn13, validate_akn13_lattice
from novelty import (is_supersolvable_rank3, is_near_pencil, is_essential,
                     lattice_wl_hash, parse_line_str)
from quadfield import QuadraticField, QuadElem


def test_lattice_validates_and_gate_has_teeth():
    A = akn13()
    assert A.coefficient_field() == QuadraticField(3)
    assert validate_akn13_lattice(A)
    assert validate_akn13_lattice(akn13(Rational(1, 5)))
    # degenerate parameters must be rejected
    assert not validate_akn13_lattice(akn13(0))
    assert not validate_akn13_lattice(akn13(Rational(1, 2)))


def test_family_screens():
    A = akn13()
    assert is_essential(A)
    assert not is_near_pencil(A)
    assert not is_supersolvable_rank3(A)


def test_exact_certificate_and_roundtrip():
    A = akn13()
    cert, status = find_certificate_fast(A, target_exponents=(6, 6))
    assert status == "certified"
    assert verify_certificate(cert)
    # c is genuinely irrational: a QuadElem with nonzero sqrt(3) part
    assert isinstance(cert["c"], QuadElem) and cert["c"].b != 0
    cj = json.loads(json.dumps(certificate_to_json(cert)))
    assert cj["field"]["d"] == 3 and cj["field"]["embedding"] == "principal"
    assert verify_certificate(certificate_from_json(cj))
    # classify agrees
    res = classify_freeness(A, target_pair=(6, 6))
    assert res["status"] == FREE_TARGET


def test_not_inductively_free():
    from triage_swap import inductive_freeness_status
    A = akn13()
    verdict, _ = inductive_freeness_status(A, 6, 6,
                                           deadline=time.time() + 300)
    assert verdict == "not_inductively_free"


def test_perturbed_negative_control():
    A = akn13()
    K = QuadraticField(3)
    bad = ProjectiveLine(K.element(1, 1), Rational(2), Rational(1))
    pert = LineArrangement(list(A.lines[:-1]) + [bad])
    cert, status = find_certificate_fast(pert, target_exponents=(6, 6))
    assert cert is None
    assert status in ("modp_reject", "not_target_free")


def test_prescreen_soundness_on_free_fixture():
    """The mod-p block prescreen must never fire on the free fixture."""
    from certificates import modp_nullity_reject
    assert not modp_nullity_reject(akn13(), 6, 6)


def test_line_string_roundtrip_and_wl_stability():
    A = akn13()
    K = QuadraticField(3)
    for line in A.lines:
        assert parse_line_str(repr(line), field=K) == line
    B = LineArrangement([parse_line_str(repr(l), field=K) for l in A.lines])
    assert lattice_wl_hash(B) == lattice_wl_hash(A)
