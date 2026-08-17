"""
quadfield.py

Exact arithmetic over real and complex quadratic fields K = Q(sqrt(d)),
d in {2, 3, 5, -1, -3}, and K-linear algebra by Weil restriction to Q.

Design:
  * `QuadElem(field, a, b)` represents a + b*sqrt(d) with a, b sympy
    Rational.  Elements are canonical by construction (a pair is its own
    normal form), so `__eq__`/`__hash__` are exact — this is the property
    the incidence layer of `arrangement.py` needs and which raw sympy
    expressions do not have (`(1+sqrt(5))/2 == 2/(sqrt(5)-1)` is False
    structurally).
  * Collapse invariant: any operation whose result has b == 0 returns a
    plain sympy Rational.  A `QuadElem` in the wild always has b != 0,
    so rational arrangements never contain QuadElems and mixed
    Rational/QuadElem tuples have consistent equality and hashing.
  * All elimination-shaped linear algebra (nullspace, rank, mod-p) is
    delegated to the existing, audited rational sympy path via the block
    (Weil restriction) construction:
        M = M0 + sqrt(d)*M1  over K   ~->   B = [[M0, d*M1], [M1, M0]] over Q
    with ker_Q(B) ~= ker_K(M) as a Q-space of twice the K-dimension.
  * Embeddings are principal and fixed: sqrt(d) -> +sqrt(d) for d > 0,
    sqrt(d) -> +i*sqrt(|d|) for d < 0.  Certificates record this.

Kept dependency-light (sympy + numpy only) so HPC nodes can import it.
"""

import math
import re
from fractions import Fraction

import sympy as sp
from sympy import Rational, Matrix

SUPPORTED_DISCRIMINANTS = (2, 3, 5, -1, -3)


class QuadraticField:
    """The field Q(sqrt(d)) with the principal embedding."""

    _registry = {}

    def __new__(cls, d):
        d = int(d)
        if d not in SUPPORTED_DISCRIMINANTS:
            raise ValueError(f"unsupported quadratic field d={d}; "
                             f"supported: {SUPPORTED_DISCRIMINANTS}")
        if d not in cls._registry:
            obj = super().__new__(cls)
            obj.d = d
            obj.is_real = d > 0
            obj.name = f"QQ(sqrt{d})"
            cls._registry[d] = obj
        return cls._registry[d]

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return isinstance(other, QuadraticField) and other.d == self.d

    def __hash__(self):
        return hash(("QuadraticField", self.d))

    def element(self, a, b=0):
        """a + b*sqrt(d), collapsing to Rational when b == 0."""
        return _make(self, Rational(a), Rational(b))

    @property
    def sqrt(self):
        """The element sqrt(d)."""
        return QuadElem(self, Rational(0), Rational(1))

    def embed_value(self):
        """Numeric value of sqrt(d) under the principal embedding."""
        if self.d > 0:
            return math.sqrt(self.d)
        return complex(0.0, math.sqrt(-self.d))

    def to_json(self):
        return {"type": "quadratic", "d": self.d, "name": self.name,
                "embedding": "principal"}

    @classmethod
    def from_json(cls, tag):
        if tag in (None, "QQ"):
            return None
        if isinstance(tag, dict) and tag.get("type") == "quadratic":
            return cls(tag["d"])
        raise ValueError(f"unrecognized coefficient_field tag: {tag!r}")


def _make(field, a, b):
    if b == 0:
        return a
    return QuadElem(field, a, b)


def _coerce_rational(v):
    """Rational from int/Fraction/Rational/decimal string, else None."""
    if isinstance(v, Rational):
        return v
    if isinstance(v, (int, Fraction)):
        return Rational(v)
    if isinstance(v, str):
        try:
            return Rational(v)
        except (TypeError, ValueError):
            return None
    if getattr(v, "is_Rational", False):
        return Rational(v)
    return None


class QuadElem:
    """a + b*sqrt(d), a,b Rational, b != 0 (else collapsed to Rational)."""

    __slots__ = ("field", "a", "b")

    # Higher than sympy Expr (10.0): sympy's decorated binary ops defer to
    # our reflected methods instead of sympifying (which would silently
    # degrade to Float via __float__ — the exact corruption we must forbid).
    _op_priority = 100.0

    def __init__(self, field, a, b):
        self.field = field
        self.a = a
        self.b = b

    # ── coercion helpers ─────────────────────────────────────────────────

    def _lift(self, other):
        """Return (a, b) pair for `other` in this field, or None."""
        if isinstance(other, QuadElem):
            if other.field.d != self.field.d:
                raise ValueError(
                    f"mixing quadratic fields d={self.field.d} and "
                    f"d={other.field.d} is not supported")
            return other.a, other.b
        r = _coerce_rational(other)
        if r is None:
            return None
        return r, Rational(0)

    # ── arithmetic ───────────────────────────────────────────────────────

    def __add__(self, other):
        o = self._lift(other)
        if o is None:
            return NotImplemented
        return _make(self.field, self.a + o[0], self.b + o[1])

    __radd__ = __add__

    def __neg__(self):
        return QuadElem(self.field, -self.a, -self.b)

    def __sub__(self, other):
        o = self._lift(other)
        if o is None:
            return NotImplemented
        return _make(self.field, self.a - o[0], self.b - o[1])

    def __rsub__(self, other):
        o = self._lift(other)
        if o is None:
            return NotImplemented
        return _make(self.field, o[0] - self.a, o[1] - self.b)

    def __mul__(self, other):
        o = self._lift(other)
        if o is None:
            return NotImplemented
        a1, b1, (a2, b2), d = self.a, self.b, o, self.field.d
        return _make(self.field, a1 * a2 + d * b1 * b2, a1 * b2 + b1 * a2)

    __rmul__ = __mul__

    def inverse(self):
        n = self.a * self.a - self.field.d * self.b * self.b
        if n == 0:
            raise ZeroDivisionError("QuadElem division by zero")
        return _make(self.field, self.a / n, -self.b / n)

    def __truediv__(self, other):
        o = self._lift(other)
        if o is None:
            return NotImplemented
        a2, b2 = o
        if b2 == 0:
            if a2 == 0:
                raise ZeroDivisionError("QuadElem division by zero")
            return _make(self.field, self.a / a2, self.b / a2)
        return self * QuadElem(self.field, a2, b2).inverse()

    def __rtruediv__(self, other):
        o = self._lift(other)
        if o is None:
            return NotImplemented
        num = _make(self.field, o[0], o[1])
        return num * self.inverse()

    def __pow__(self, n):
        if not isinstance(n, int) or n < 0:
            return NotImplemented
        result = Rational(1)
        base = self
        while n:
            if n & 1:
                result = base * result
            base = base * base
            n >>= 1
        return result

    def conjugate(self):
        """Galois conjugate a - b*sqrt(d)."""
        return QuadElem(self.field, self.a, -self.b)

    def norm(self):
        """Field norm a^2 - d*b^2 (a Rational)."""
        return self.a * self.a - self.field.d * self.b * self.b

    # ── identity ─────────────────────────────────────────────────────────

    def __eq__(self, other):
        if isinstance(other, QuadElem):
            return (other.field.d == self.field.d and other.a == self.a
                    and other.b == self.b)
        r = _coerce_rational(other)
        if r is None:
            return NotImplemented
        return False        # invariant: b != 0, so never equal to a rational

    def __ne__(self, other):
        res = self.__eq__(other)
        if res is NotImplemented:
            return res
        return not res

    def __hash__(self):
        return hash(("QuadElem", self.field.d, self.a, self.b))

    def __bool__(self):
        return True         # b != 0 by invariant

    # ── conversion / printing ────────────────────────────────────────────

    def to_sympy(self):
        return self.a + self.b * sp.sqrt(self.field.d)

    def embed(self):
        """Numeric value under the principal embedding (float or complex)."""
        s = self.field.embed_value()
        if self.field.is_real:
            return float(self.a) + float(self.b) * s
        return complex(float(self.a), 0.0) + complex(self.b) * s

    def __float__(self):
        if not self.field.is_real:
            raise TypeError(f"cannot convert element of {self.field.name} "
                            "(complex embedding) to float")
        return self.embed()

    def __complex__(self):
        return complex(self.embed())

    def sort_key(self):
        """Deterministic (non-numeric) total order key within one field."""
        return (sp.default_sort_key(self.a), sp.default_sort_key(self.b))

    def sign(self):
        """Exact sign of the real number a + b*sqrt(d) (real fields only)."""
        if not self.field.is_real:
            raise TypeError("sign() only defined for real quadratic fields")
        a, b, d = self.a, self.b, self.field.d
        if a == 0:
            return 1 if b > 0 else -1
        if a > 0 and b > 0:
            return 1
        if a < 0 and b < 0:
            return -1
        # opposite signs: compare a^2 with d*b^2; sign follows the winner
        if a * a > d * b * b:
            return 1 if a > 0 else -1
        return 1 if b > 0 else -1   # a^2 == d*b^2 impossible (d nonsquare)

    def __repr__(self):
        """Canonical bracket token, e.g. [1/2+3/2s], [-1s], [2-1/3s]."""
        if self.a == 0:
            return f"[{self.b}s]"
        bs = f"{self.b}s" if self.b < 0 else f"+{self.b}s"
        return f"[{self.a}{bs}]"

    __str__ = __repr__


_BRACKET_RE = re.compile(
    r'^([+-]?\d+(?:/\d+)?(?=[+-]))?([+-]?\d+(?:/\d+)?)s$')


def parse_quad_token(token, field):
    """Parse a bracket token body 'a+bs' / '-bs' / 'a-bs' into an element.

    `token` excludes the surrounding brackets.  Requires an explicit field.
    """
    if field is None:
        raise ValueError(f"quadratic token {token!r} requires a declared "
                         "coefficient field")
    if not isinstance(field, QuadraticField):
        field = QuadraticField(field)
    m = _BRACKET_RE.match(token.strip())
    if not m:
        raise ValueError(f"cannot parse quadratic token {token!r}")
    a = Rational(m.group(1)) if m.group(1) else Rational(0)
    b = Rational(m.group(2))
    return _make(field, a, b)


# Safety net: if sympify(QuadElem) is ever invoked (e.g. building a sympy
# Matrix directly from QuadElems), convert EXACTLY, never through float.
from sympy.core.sympify import converter as _sympy_converter  # noqa: E402
_sympy_converter[QuadElem] = lambda q: q.to_sympy()


def scalar_field(values):
    """The QuadraticField shared by any QuadElem in `values`, else None.

    Raises on mixed discriminants.
    """
    field = None
    for v in values:
        if isinstance(v, QuadElem):
            if field is None:
                field = v.field
            elif field.d != v.field.d:
                raise ValueError("mixed quadratic fields in one object")
    return field


def split_parts(v):
    """(a, b) with v = a + b*sqrt(d); rationals give (v, 0)."""
    if isinstance(v, QuadElem):
        return v.a, v.b
    return v, Rational(0)


# ── Linear algebra by Weil restriction ──────────────────────────────────


def block_matrix(rows, field):
    """B = [[M0, d*M1], [M1, M0]] over Q for M = M0 + sqrt(d)*M1."""
    d = field.d
    m0_rows, m1_rows = [], []
    for row in rows:
        r0, r1 = [], []
        for v in row:
            a, b = split_parts(v)
            r0.append(a)
            r1.append(b)
        m0_rows.append(r0)
        m1_rows.append(r1)
    top = [r0 + [d * v for v in r1] for r0, r1 in zip(m0_rows, m1_rows)]
    bot = [r1 + r0 for r0, r1 in zip(m0_rows, m1_rows)]
    return Matrix(top + bot)


def k_rank(rows, field):
    """Rank over K of a matrix given as rows of Rational/QuadElem."""
    if not rows:
        return 0
    B = block_matrix(rows, field)
    r = B.rank()
    assert r % 2 == 0, "Weil-restriction rank must be even"
    return r // 2


def _k_vector_from_block(w, ncols, field):
    """K-vector u0 + sqrt(d)*u1 from a length-2n rational kernel vector."""
    return [_make(field, Rational(w[i]), Rational(w[ncols + i]))
            for i in range(ncols)]


def k_nullspace(rows, field):
    """K-basis of the kernel of M (rows of Rational/QuadElem entries).

    Returns a list of K-vectors (lists of Rational/QuadElem).  The block
    kernel over Q has twice the K-dimension; a K-basis is extracted
    greedily with exact K-rank tests (also via Weil restriction).
    """
    ncols = len(rows[0]) if rows else 0
    B = block_matrix(rows, field)
    null_q = B.nullspace()
    assert len(null_q) % 2 == 0, "Weil-restriction nullity must be even"
    target = len(null_q) // 2
    basis = []
    for w in null_q:
        if len(basis) == target:
            break
        cand = _k_vector_from_block(list(w), ncols, field)
        if k_rank(basis + [cand], field) > len(basis):
            basis.append(cand)
    assert len(basis) == target, "failed to extract a K-basis of the kernel"
    return basis
