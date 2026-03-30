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


x, y, z = symbols('x y z')


# ─── Projective line ──────────────────────────────────────────────────────────

class ProjectiveLine:
    """Line ax+by+cz=0 in CP^2, represented by projective point [a:b:c]."""

    def __init__(self, a, b, c):
        coords = (Rational(a), Rational(b), Rational(c))
        assert not all(c == 0 for c in coords), "Zero line"
        self.coords = self._canonical(coords)

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

    def linear_form(self):
        a, b, c = self.coords
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

    def copy(self):
        arr = LineArrangement(list(self.lines))
        return arr

    def __len__(self):
        return len(self.lines)

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

        For a free arrangement with exponents (1, d2, d3):
          b2 = (n-1) + d2*d3,  so  d2*d3 = b2 - (n-1).
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
        """All valid (d2, d3) with 1 <= d2 <= d3 and d2+d3 = n-1."""
        return [(d2, n - 1 - d2) for d2 in range(1, (n - 1) // 2 + 1)]

    def candidate_exponents(self):
        """
        Necessary condition for freeness with exponents (1, d2, d3):
          d2 + d3 = n-1,  d2*d3 = b2 - (n-1)
        Returns (d2, d3) as ints, or None.

        Derivation: chi(A, t) = (t-1)(t-d2)(t-d3)
          => b2 = (n-1) + d2*d3
          => d2, d3 are roots of  t^2 - (n-1)*t + (b2-(n-1)) = 0
        """
        n = len(self.lines)
        if n < 2:
            return None
        b2 = self.b2()
        product = b2 - (n - 1)   # = d2 * d3
        if product < 0:
            return None
        # d2 + d3 = n-1, d2*d3 = product  => disc = (n-1)^2 - 4*product
        disc = (n - 1) ** 2 - 4 * product
        if disc < 0:
            return None
        sq = int(disc ** 0.5 + 0.5)
        if sq * sq != disc:
            return None
        d2 = ((n - 1) - sq) // 2
        d3 = ((n - 1) + sq) // 2
        if d2 + d3 == n - 1 and d2 * d3 == product and d2 >= 0:
            return int(d2), int(d3)
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
        """Two rational basis vectors for ker([a,b,c])."""
        a, b, c = Rational(a), Rational(b), Rational(c)
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

    def derivation_matrix(self, d):
        """
        Build matrix M (over Q) such that ker(M) = D(A)_d.
        Variables: [f_coeffs | g_coeffs | h_coeffs], each of length C(d+2,2).
        For each line alpha_i, condition alpha_i | (a_i*f + b_i*g + c_i*h).
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
        return Matrix(rows)

    def derivation_space_dim(self, d):
        """Dimension of D(A)_d."""
        M = self.derivation_matrix(d)
        # rank + nullity = 3 * C(d+2, 2)
        return 3 * len(self._monoms(d)) - M.rank()

    # ── Freeness check ────────────────────────────────────────────────────────

    def is_free(self):
        """
        Check freeness using Saito's criterion.
        Returns (is_free, exponents) where exponents = (1, d2, d3) or None.
        """
        exps = self.candidate_exponents()
        if exps is None:
            return False, None
        d2, d3 = exps
        n = len(self.lines)

        # For free arrangement: D(A)_d2 has dimension C(d2+1,2)+1 (if d2<=d3)
        # We check by finding the null space of the derivation matrix
        M_d2 = self.derivation_matrix(d2)
        null_d2 = M_d2.nullspace()

        if not null_d2:
            return False, None

        # Similarly for d3
        if d2 == d3:
            null_d3 = null_d2
        else:
            M_d3 = self.derivation_matrix(d3)
            null_d3 = M_d3.nullspace()
            if not null_d3:
                return False, None

        # Build Euler derivation (always in D(A)_1)
        monoms1 = self._monoms(1)
        N1 = len(monoms1)
        euler = [Rational(0)] * (3 * N1)
        for idx, (ma, mb, mc) in enumerate(monoms1):
            if (ma, mb, mc) == (1, 0, 0): euler[idx] = Rational(1)
            elif (ma, mb, mc) == (0, 1, 0): euler[N1 + idx] = Rational(1)
            elif (ma, mb, mc) == (0, 0, 1): euler[2*N1 + idx] = Rational(1)

        # Defining polynomial
        Q = sp.expand(sp.prod(line.linear_form() for line in self.lines))

        # Try all pairs from null_d2 x null_d3
        monoms_d2 = self._monoms(d2)
        monoms_d3 = self._monoms(d3)
        N2, N3 = len(monoms_d2), len(monoms_d3)

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

        for v2 in null_d2:
            f2, g2, h2 = build_theta(list(v2), monoms_d2)
            for v3 in null_d3:
                f3, g3, h3 = build_theta(list(v3), monoms_d3)
                # Saito matrix det
                S = Matrix([[ef, f2, f3], [eg, g2, g3], [eh, h2, h3]])
                det_S = sp.expand(S.det())
                ratio = sp.cancel(det_S / Q)
                if ratio.is_number and ratio != 0:
                    return True, (1, d2, d3)

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
    """All valid (d2, d3) with 1 <= d2 <= d3 and d2+d3 = n-1."""
    return LineArrangement.all_exponent_types(n)
