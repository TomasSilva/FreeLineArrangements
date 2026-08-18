"""Phase-4 store-integrity tests: schema 2.1, field validation, K
promotion round-trip, discovery-id field separation."""

import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import Rational

from arrangement import LineArrangement, ProjectiveLine
from certificates import find_certificate_fast
from known_arrangements import akn13, dual_hesse
from novelty import arrangement_from_record, lattice_wl_hash
from promotion import (build_discovery_entry, promote,
                       load_verified_discoveries, canonical_discovery_id)
from quadfield import QuadraticField


def _akn_entry():
    A = akn13()
    cert, status = find_certificate_fast(A, target_exponents=(6, 6))
    assert status == "certified"
    return A, build_discovery_entry(
        cert, run_id="test", engine="fixture",
        search_params={"lambda": 1.0, "beta": 0.75, "field": "real"},
        lattice_hash=lattice_wl_hash(A))


def test_k_promotion_roundtrip(tmp_path):
    A, entry = _akn_entry()
    assert entry["schema_version"] == "discovery-2.1"
    assert entry["field"]["d"] == 3
    assert entry["coefficient_field"] == entry["field"]
    assert entry["optimization_field"] == "real"
    store = str(tmp_path / "store.json")
    r1 = promote([entry], store)
    r2 = promote([entry], store)
    assert r1["promoted"] == 1 and r2["duplicates"] == 1
    ok, rejects = load_verified_discoveries(store)
    assert len(ok) == 1 and not rejects
    back = arrangement_from_record(ok[0])
    assert back.coefficient_field() == QuadraticField(3)
    assert lattice_wl_hash(back) == entry["lattice_hash"]


def test_promotion_rejections(tmp_path):
    _, entry = _akn_entry()
    store = str(tmp_path / "store.json")
    bad_d = copy.deepcopy(entry)
    bad_d["field"] = {"type": "quadratic", "d": 7, "name": "QQ(sqrt7)",
                      "embedding": "principal"}
    bad_missing = copy.deepcopy(entry)
    bad_missing["field"] = "QQ"          # 2.1 without an explicit field
    r = promote([bad_d, bad_missing], store)
    reasons = [reason for _, reason in r["rejected"]]
    assert any("unsupported_field" in s for s in reasons)
    assert any("schema_2.1_requires_explicit_field" in s for s in reasons)
    assert r["promoted"] == 0


def test_loader_rejects_bad_field(tmp_path):
    _, entry = _akn_entry()
    store = str(tmp_path / "store.json")
    promote([entry], store)
    data = json.load(open(store))
    data["arrangements"][0]["field"] = {"type": "quadratic", "d": 7,
                                        "name": "QQ(sqrt7)",
                                        "embedding": "principal"}
    json.dump(data, open(store, "w"))
    ok, rejects = load_verified_discoveries(store, reverify=False)
    assert not ok and rejects and "unsupported_field" in rejects[0][1]


def test_discovery_id_separates_fields():
    """Identical (a, b) strings under different d must not collide."""
    a3 = ProjectiveLine(QuadraticField(3).element(0, 1), Rational(2),
                        Rational(1))
    a5 = ProjectiveLine(QuadraticField(5).element(0, 1), Rational(2),
                        Rational(1))
    id3 = canonical_discovery_id(LineArrangement([a3]))
    id5 = canonical_discovery_id(LineArrangement([a5]))
    assert id3 != id5


def test_qq_entries_keep_schema_20(tmp_path):
    from saito import construct_supersolvable
    arr = construct_supersolvable(9, 3)
    cert, status = find_certificate_fast(arr, target_exponents=(3, 5))
    assert status == "certified"
    entry = build_discovery_entry(cert, run_id="test", engine="fixture")
    assert entry["schema_version"] == "discovery-2.0"
    assert entry["field"] == "QQ"
    store = str(tmp_path / "store.json")
    assert promote([entry], store)["promoted"] == 1
    ok, rejects = load_verified_discoveries(store)
    assert len(ok) == 1 and not rejects
