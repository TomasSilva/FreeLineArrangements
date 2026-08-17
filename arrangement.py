"""
arrangement.py

Core mathematics for line arrangements in CP^2 (central in C^3).
Works over Q (sympy Rationals) for exact arithmetic.
"""

import numpy as np
import sympy as sp
from sympy import Rational, Matrix, symbols, binomial
from itertools import combinations
from collections import defaultdict

from quadfield import QuadElem, scalar_field


x, y, z = symbols('x y z')


def _coerce_coord(v):
    """Rational for rational inputs (unchanged QQ path); QuadElem passes."""
    if isinstance(v, QuadElem):
        return v
    return Rational(v)


# ─── Projective line ──────────────────────────────────────────────────────────

class ProjectiveLine:
    """Line ax+by+cz=0 in CP^2, represented by projective point [a:b:c].

    Coordinates are sympy Rationals (field QQ) or quadfield.QuadElem
    elements of one quadratic field Q(sqrt(d)); `self.field` is that
    QuadraticField, or None for QQ.
    """

    def __init__(self, a, b, c):
        coords = (_coerce_coord(a), _coerce_coord(b), _coerce_coord(c))
        assert not all(c == 0 for c in coords), "Zero line"
        self.coords = self._canonical(coords)
        self.field = scalar_field(self.coords)

    @staticmethod
    def _canonical(coords):
        for c in coords:
            if c != 0:
                return tuple(v / c for v in coords)
        return coords

    def __eq__(self, other):
        return self.coords == other.coords

    def __hash__(self):
        return hash(self.coords)

    def __repr__(self):
        a, b, c = self.coords
        return f"({a}x+{b}y+{c}z=0)"

    def intersect(self, other):
        """Intersection point in CP^2 (cross product), or None if same line."""
        a1, b1, c1 = self.coords
        a2, b2, c2 = other.coords
        px = b1 * c2 - b2 * c1
        py = c1 * a2 - c2 * a1
        pz = a1 * b2 - a2 * b1
        if px == 0 and py == 0 and pz == 0:
            return None
        # Normalize to canonical projective point
        for coord in (px, py, pz):
            if coord != 0:
                return (px / coord, py / coord, pz / coord)

    def to_float(self):
        return np.array([float(v) for v in self.coords])

    def embed(self):
        """Numeric coordinates under the field's principal embedding.

        float64 for QQ / real quadratic fields, complex128 for complex ones.
        """
        if self.field is not None and not self.field.is_real:
            return np.array([complex(v) for v in self.coords],
                            dtype=np.complex128)
        return np.array([float(v) for v in self.coords])

    def linear_form(self):
        a, b, c = self.coords
        if self.field is not None:
            a, b, c = (v.to_sympy() if isinstance(v, QuadElem) else v
                       for v in (a, b, c))
        return a * x + b * y + c * z

    @classmethod
    def from_two_points(cls, p1, p2):
        """Line through two projective points (cross product in CP^2).

        Points are tuples (x, y, z) of Rationals.  Returns None if the
        points coincide (cross product is zero).
        """
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        a = y1 * z2 - y2 * z1
        b = z1 * x2 - z2 * x1
        c = x1 * y2 - x2 * y1
        if a == 0 and b == 0 and c == 0:
            return None
        return cls(a, b, c)

    def passes_through(self, point):
        """Does this line contain the projective point?"""
        return sum(a * p for a, p in zip(self.coords, point)) == 0


# ─── Line arrangement ─────────────────────────────────────────────────────────

class LineArrangement:
    """Central hyperplane arrangement in C^3 / line arrangement in CP^2."""

    def __init__(self, lines=None):
        self.lines = list(lines) if lines else []
        self._cache = None

    def add_line(self, line: ProjectiveLine):
        self.lines.append(line)
        self._cache = None

    def remove_last(self):
        if self.lines:
            self.lines.pop()
            self._cache = None

    def remove_line(self, index):
        """Remove the line at the given index (clears intersection cache)."""
        del self.lines[index]
        self._cache = None

    def copy(self):
        arr = LineArrangement(list(self.lines))
        return arr

    def __len__(self):
        return len(self.lines)

    def coefficient_field(self):
        """QuadraticField shared by the lines, or None for QQ.

        Derived from the coordinates (never stored on the arrangement), so
        it cannot desynchronize under add/remove.  Mixing fields raises.
        """
        field = None
        for line in self.lines:
            f = getattr(line, "field", None)
            if f is not None:
                if field is None:
                    field = f
                elif field.d != f.d:
                    raise ValueError("mixed quadratic fields in arrangement")
        return field

    # ── Intersection structure ────────────────────────────────────────────────

    def _structure(self):
        if self._cache is not None:
            return self._cache
        pts = {}
        for i, j in combinations(range(len(self.lines)), 2):
            p = self.lines[i].intersect(self.lines[j])
            if p is None:
                continue
            if p not in pts:
                pts[p] = set()
            pts[p].add(i)
            pts[p].add(j)
        self._cache = pts
        return pts

    def intersection_points(self):
        """Return dict: point_key -> frozenset of line indices through it."""
        return {k: frozenset(v) for k, v in self._structure().items()}

    def multiplicities(self):
        """List of multiplicities of all intersection points."""
        return [len(v) for v in self._structure().values()]

    def max_multiplicity(self):
        """Multiplicity of the most singular point (0 if no intersections)."""
        mults = self.multiplicities()
        return max(mults) if mults else 0

    def n_intersection_points(self):
        """Total number of distinct intersection points."""
        return len(self._structure())

    def b2(self):
        """b2(A) = sum_p (m_p - 1).

        This is the coefficient of t in the characteristic polynomial:
          chi(A, t) = t^3 - n*t^2 + b2*t - b3

        For a free arrangement with exponents (1, d1, d2):
          b2 = (n-1) + d1*d2,  so  d1*d2 = b2 - (n-1).
        """
        return sum(m - 1 for m in self.multiplicities())

    def t2(self):
        """Alias for backward compatibility. Returns b2."""
        return self.b2()

    def is_pencil(self):
        n = len(self.lines)
        return n >= 3 and any(len(v) == n for v in self._structure().values())

    def has_duplicate(self, line: ProjectiveLine):
        return line in self.lines

    # ── Exponent candidates ───────────────────────────────────────────────────

    @staticmethod
    def all_exponent_types(n):
        """All valid (d1, d2) with 1 <= d1 <= d2 and d1+d2 = n-1."""
        return [(d1, n - 1 - d1) for d1 in range(1, (n - 1) // 2 + 1)]

    def candidate_exponents(self):
        """
        Necessary condition for freeness with exponents (1, d1, d2):
          d1 + d2 = n-1,  d1*d2 = b2 - (n-1)
        Returns (d1, d2) as ints, or None.

        Derivation: chi(A, t) = (t-1)(t-d1)(t-d2)
          => b2 = (n-1) + d1*d2
          => d1, d2 are roots of  t^2 - (n-1)*t + (b2-(n-1)) = 0
        """
        n = len(self.lines)
        if n < 2:
            return None
        b2 = self.b2()
        product = b2 - (n - 1)   # = d1 * d2
        if product < 0:
            return None
        # d1 + d2 = n-1, d1*d2 = product  => disc = (n-1)^2 - 4*product
        disc = (n - 1) ** 2 - 4 * product
        if disc < 0:
            return None
        sq = int(disc ** 0.5 + 0.5)
        if sq * sq != disc:
            return None
        d1 = ((n - 1) - sq) // 2
        d2 = ((n - 1) + sq) // 2
        if d1 + d2 == n - 1 and d1 * d2 == product and d1 >= 0:
            return int(d1), int(d2)
        return None

    # ── Polynomial derivation space ───────────────────────────────────────────

    @staticmethod
    def _monoms(d):
        """All exponent triples (a,b,c) with a+b+c=d."""
        result = []
        for a in range(d + 1):
            for b in range(d - a + 1):
                result.append((a, b, d - a - b))
        return result

    @staticmethod
    def _ker_basis(a, b, c):
        """Two K-rational basis vectors for ker([a,b,c])."""
        a, b, c = _coerce_coord(a), _coerce_coord(b), _coerce_coord(c)
        if a != 0:
            u = [-b, a, Rational(0)]
            v = [-c, Rational(0), a]
        elif b != 0:
            u = [Rational(1), Rational(0), Rational(0)]
            v = [Rational(0), -c, b]
        else:
            u = [Rational(1), Rational(0), Rational(0)]
            v = [Rational(0), Rational(1), Rational(0)]
        return u, v

    @staticmethod
    def _mono_param(u, v, ma, mb, mc, p, q):
        """
        Coefficient of s^p*t^q in (s*u0+t*v0)^ma*(s*u1+t*v1)^mb*(s*u2+t*v2)^mc.
        """
        res = Rational(0)
        for i in range(ma + 1):
            for j in range(mb + 1):
                for k in range(mc + 1):
                    if i + j + k == p:
                        c = (binomial(ma, i) * binomial(mb, j) * binomial(mc, k)
                             * u[0]**i * v[0]**(ma-i)
                             * u[1]**j * v[1]**(mb-j)
                             * u[2]**k * v[2]**(mc-k))
                        res += c
        return res

    def _derivation_rows(self, d):
        """Rows of the derivation-condition matrix (entries in K).

        Same construction as `derivation_matrix`, but returned as plain
        Python lists so quadratic-field entries (QuadElem) survive; the QQ
        path wraps them in a sympy Matrix exactly as before.
        """
        monoms = self._monoms(d)
        N = len(monoms)
        rows = []
        for line in self.lines:
            a, b, c = line.coords
            u, v = self._ker_basis(a, b, c)
            for p in range(d + 1):
                row = [Rational(0)] * (3 * N)
                for idx, (ma, mb, mc) in enumerate(monoms):
                    coeff = self._mono_param(u, v, ma, mb, mc, p, d - p)
                    row[idx]       += a * coeff
                    row[N + idx]   += b * coeff
                    row[2*N + idx] += c * coeff
                rows.append(row)
        return rows

    def derivation_matrix(self, d):
        """
        Build matrix M (over Q) such that ker(M) = D(A)_d.
        Variables: [f_coeffs | g_coeffs | h_coeffs], each of length C(d+2,2).
        For each line alpha_i, condition alpha_i | (a_i*f + b_i*g + c_i*h).

        QQ only — quadratic-field arrangements use `_derivation_rows` with
        `quadfield.k_rank`/`k_nullspace` (Weil restriction) instead.
        """
        return Matrix(self._derivation_rows(d))

    def derivation_space_dim(self, d):
        """Dimension over K of D(A)_d."""
        K = self.coefficient_field()
        if K is not None:
            from quadfield import k_rank
            rows = self._derivation_rows(d)
            return 3 * len(self._monoms(d)) - k_rank(rows, K)
        M = self.derivation_matrix(d)
        # rank + nullity = 3 * C(d+2, 2)
        return 3 * len(self._monoms(d)) - M.rank()

    # ── Freeness check ────────────────────────────────────────────────────────

    def is_free(self):
        """
        Check freeness using Saito's criterion.
        Returns (is_free, exponents) where exponents = (1, d1, d2) or None.
        """
        exps = self.candidate_exponents()
        if exps is None:
            return False, None
        d1, d2 = exps
        n = len(self.lines)

        # For free arrangement: D(A)_d1 has dimension C(d1+1,2)+1 (if d1<=d2)
        # We check by finding the null space of the derivation matrix
        K = self.coefficient_field()
        if K is not None:
            from quadfield import k_nullspace
            null_d1 = k_nullspace(self._derivation_rows(d1), K)
        else:
            M_d1 = self.derivation_matrix(d1)
            null_d1 = M_d1.nullspace()

        if not null_d1:
            return False, None

        # Similarly for d2
        if d1 == d2:
            null_d2 = null_d1
        elif K is not None:
            from quadfield import k_nullspace
            null_d2 = k_nullspace(self._derivation_rows(d2), K)
            if not null_d2:
                return False, None
        else:
            M_d2 = self.derivation_matrix(d2)
            null_d2 = M_d2.nullspace()
            if not null_d2:
                return False, None

        # Build Euler derivation (always in D(A)_1)
        monoms1 = self._monoms(1)
        N_euler = len(monoms1)
        euler = [Rational(0)] * (3 * N_euler)
        for idx, (ma, mb, mc) in enumerate(monoms1):
            if (ma, mb, mc) == (1, 0, 0): euler[idx] = Rational(1)
            elif (ma, mb, mc) == (0, 1, 0): euler[N_euler + idx] = Rational(1)
            elif (ma, mb, mc) == (0, 0, 1): euler[2*N_euler + idx] = Rational(1)

        # Defining polynomial
        Q = sp.expand(sp.prod(line.linear_form() for line in self.lines))

        # Try all pairs from null_d1 x null_d2
        monoms_d1 = self._monoms(d1)
        monoms_d2 = self._monoms(d2)
        N1, N2 = len(monoms_d1), len(monoms_d2)

        def vec_to_poly(vec, monoms):
            p = sp.Integer(0)
            N = len(monoms)
            for i, (ma, mb, mc) in enumerate(monoms):
                mono = x**ma * y**mb * z**mc
                p += vec[i] * mono
            return p, vec_to_poly_component(vec, monoms, N, 1), vec_to_poly_component(vec, monoms, N, 2)

        def vec_to_poly_component(vec, monoms, N, comp):
            p = sp.Integer(0)
            for i, (ma, mb, mc) in enumerate(monoms):
                p += vec[comp*N + i] * x**ma * y**mb * z**mc
            return p

        def build_theta(vec, monoms):
            N = len(monoms)
            f = sum(vec[i] * x**ma * y**mb * z**mc for i, (ma, mb, mc) in enumerate(monoms))
            g = sum(vec[N+i] * x**ma * y**mb * z**mc for i, (ma, mb, mc) in enumerate(monoms))
            h = sum(vec[2*N+i] * x**ma * y**mb * z**mc for i, (ma, mb, mc) in enumerate(monoms))
            return f, g, h

        # Euler: f=x, g=y, h=z
        ef, eg, eh = x, y, z

        for v2 in null_d1:
            f2, g2, h2 = build_theta(list(v2), monoms_d1)
            for v3 in null_d2:
                f3, g3, h3 = build_theta(list(v3), monoms_d2)
                # Saito matrix det
                S = Matrix([[ef, f2, f3], [eg, g2, g3], [eh, h2, h3]])
                det_S = sp.expand(S.det())
                ratio = sp.cancel(det_S / Q)
                if ratio.is_number and ratio != 0:
                    return True, (1, d1, d2)

        return False, None

    # ── Defining polynomial ───────────────────────────────────────────────────

    def defining_polynomial(self):
        if not self.lines:
            return sp.Integer(1)
        return sp.expand(sp.prod(line.linear_form() for line in self.lines))

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self):
        n = len(self.lines)
        b2 = self.b2()
        exps = self.candidate_exponents()
        pencil = self.is_pencil()
        mults = sorted(self.multiplicities(), reverse=True)
        return {
            'n': n,
            'b2': b2,
            'candidate_exponents': exps,
            'is_pencil': pencil,
            'multiplicity_profile': mults,
        }


# Module-level convenience alias
def all_exponent_types(n):
    """All valid (d1, d2) with 1 <= d1 <= d2 and d1+d2 = n-1."""
    return LineArrangement.all_exponent_types(n)
