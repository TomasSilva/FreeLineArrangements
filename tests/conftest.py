import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arrangement import LineArrangement, ProjectiveLine  # noqa: E402


def arr_from(coords):
    return LineArrangement([ProjectiveLine(*c) for c in coords])


# ── known free arrangements (exactly certified in test_certificates) ─────────

BRAID_A3 = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (1, 0, -1), (0, 1, -1)]
# n = 6, free, exponents (1, 2, 3)

A2xA1 = [(1, 0, 0), (0, 1, 0), (1, -1, 0), (0, 0, 1)]
# n = 4, free, exponents (1, 1, 2)

PENCIL4 = [(1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 2, 0)]
# n = 4, pencil through (0:0:1), free (nonessential), exponents (1, 0, 3)

# ── known nonfree arrangements WITH integer candidate exponents ──────────────

NONFREE7_A = [(0, 1, -1), (1, 1, -1), (1, 1, 1), (1, 1, 2), (1, 1, 0),
              (1, 1, -2), (1, 0, -2)]
# n = 7, candidate exponents (3, 3), exactly verified nonfree

NONFREE7_B = [(1, 0, -1), (1, 0, -2), (2, 0, 1), (0, 0, 1), (1, -2, 0),
              (0, 1, 2), (1, 0, 0)]
# n = 7, candidate exponents (3, 3), exactly verified nonfree

# ── nonfree without candidate exponents (generic) ────────────────────────────

GENERIC4 = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, 3)]
# n = 4, b2 = 6, no candidate exponents, nonfree; D(A) needs > 3 minimal
# generators (generic arrangement)


@pytest.fixture(scope="session")
def braid():
    return arr_from(BRAID_A3)


@pytest.fixture(scope="session")
def a2xa1():
    return arr_from(A2xA1)


@pytest.fixture(scope="session")
def pencil4():
    return arr_from(PENCIL4)


@pytest.fixture(scope="session")
def nonfree7():
    return arr_from(NONFREE7_A)


@pytest.fixture(scope="session")
def nonfree7b():
    return arr_from(NONFREE7_B)


@pytest.fixture(scope="session")
def generic4():
    return arr_from(GENERIC4)


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(20260816)
