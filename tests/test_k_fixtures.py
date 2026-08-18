"""Reflection/CEVA family fixtures (overnight K-campaign seeds).

Fast combinatorial gates for all five; exact certification for the two
cheap ones (the full set certified during prep: hesse12/g443/g413/h3_15/
ceva6 all 'certified' + verified, see reference_hashes_k_fixtures.json).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from certificates import find_certificate_fast, verify_certificate
from known_arrangements import (ceva6, hesse12, g443, g413, h3_15,
                                validate_fixture, FIXTURE_GATES,
                                FIELD_SEED_REGISTRY)
from novelty import is_supersolvable_rank3
from quadfield import QuadraticField


def test_all_fixture_gates():
    for name, f in (("ceva6", ceva6), ("hesse12", hesse12), ("g443", g443),
                    ("g413", g413), ("h3_15", h3_15)):
        assert validate_fixture(name, f()), name


def test_fixture_fields():
    assert ceva6().coefficient_field() == QuadraticField(-3)
    assert hesse12().coefficient_field() == QuadraticField(-3)
    assert g443().coefficient_field() == QuadraticField(-1)
    assert g413().coefficient_field() == QuadraticField(-1)
    assert h3_15().coefficient_field() == QuadraticField(5)


def test_nonss_flags():
    # the free-but-not-inductively-free relatives are non-supersolvable
    assert not is_supersolvable_rank3(ceva6())
    assert not is_supersolvable_rank3(g443())
    assert not is_supersolvable_rank3(h3_15())
    # the full monomial groups G(r,1,3) are supersolvable
    assert is_supersolvable_rank3(hesse12())
    assert is_supersolvable_rank3(g413())


def test_cheap_fixture_certificates():
    for name, f in (("hesse12", hesse12), ("g443", g443)):
        exps = FIXTURE_GATES[name][3]
        cert, status = find_certificate_fast(f(), target_exponents=exps)
        assert status == "certified", name
        assert verify_certificate(cert), name


def test_registry_covers_target_range():
    ns = sorted(n for (_, n) in FIELD_SEED_REGISTRY)
    assert 18 in ns and 15 in ns and 13 in ns and 12 in ns
    for (d, n), factories in FIELD_SEED_REGISTRY.items():
        for f in factories:
            arr = f()
            assert len(arr) == n
            K = arr.coefficient_field()
            assert K is not None and K.d == d
