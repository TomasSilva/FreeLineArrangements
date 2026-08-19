"""
hybrid_refine.py — EXPERIMENTAL continuous geometric refinement of a single
line's coefficients under the penalized Saito score.  Not wired into any
production engine; used only by experiments/hybrid_smoke.py behind its
--refine flag.  Real slice (float64) only.

Math: with the exponent pair (d1, d2) fixed and the inner variables (u, v)
temporarily frozen at the production optimizer's output, the compact score

    Gamma(a) = |<B(u,v), q_A(a)>_BW|^2 / (||B||^2 + lam * R_A(a)^beta)

is a smooth function of the refined line's coefficient vector a on the unit
sphere (away from line collisions): B is independent of the arrangement;
q_A(a) enters through one polynomial multiplication of the fixed cofactor
G = prod_{j != i} alpha_j by alpha(a) = a/||a||; and only line i's rows of
the residual operator depend on a, through a locally smooth orthonormal
frame of ker(a) (the BW norm of the restriction is frame-invariant, so the
frozen-pivot frame choice does not affect the value or the gradient).

The ascent is projected/Riemannian on the unit sphere (tangent projection
(I - a a^T) g; the real slice has no residual phase gauge beyond the sign,
which normalization absorbs).  The mathematical denominator carries NO
epsilon; numerical guards return structured failure statuses and every
acceptance decision is made by RE-SCORING THE EXACT RAW OBJECTIVE with the
production evaluator on an exactly rationalized arrangement.

This is heuristic local refinement of the penalized Saito score — no claim
of a canonical gradient flow to the free locus.
"""

import math
import time

import numpy as np
import sympy as sp
from sympy import Rational

from arrangement import LineArrangement, ProjectiveLine
from penalized_saito import (PenalizedSaitoEvaluator, _monoms,
                             _bw_sqrt_weights, _mult_table,
                             _restriction_expansion, _comb,
                             DEFAULT_LAMBDA, DEFAULT_BETA)

# statuses (structured; never encoded as loss values)
OK = "ok"
GRAD_NONFINITE = "grad_nonfinite"
GAMMA_EXCEEDS_ONE = "gamma_exceeds_one"      # reported, never clipped here
LINE_COLLISION = "line_collision"
NO_IMPROVEMENT = "no_improvement"
RATIONALIZATION_FAILED = "rationalization_failed"
CONSISTENCY_FAILED = "torch_vs_evaluator_inconsistent"

_COLLISION_COS = 1.0 - 1e-9      # |cos| above this = nearly coincident


def _torch():
    import torch
    torch.set_default_dtype(torch.float64)
    return torch


def _kernel_frame_torch(torch, a_unit, pivot):
    """Smooth orthonormal frame {u, w} of ker(a) with a FROZEN pivot axis
    (chosen once per refinement session at the start point, so the frame is
    smooth along the whole ascent path)."""
    e = torch.zeros(3, dtype=a_unit.dtype)
    e[pivot] = 1.0
    u = torch.linalg.cross(e, a_unit)
    u = u / torch.linalg.norm(u)
    w = torch.linalg.cross(a_unit, u)
    return u, w


class DifferentiableGamma:
    """Torch re-implementation of Gamma as a function of ONE line's
    coefficients, mirroring PenalizedSaitoEvaluator's conventions exactly
    (unit lines, BW-unit q, sw-scaled restriction rows, row scaling,
    1/sqrt(n)).  Validated against the production evaluator at the start
    point on every construction (CONSISTENCY_FAILED otherwise)."""

    def __init__(self, ev: PenalizedSaitoEvaluator, i: int,
                 u_bw: np.ndarray, v_bw: np.ndarray,
                 lam=DEFAULT_LAMBDA, beta=DEFAULT_BETA):
        torch = _torch()
        assert not ev.iscomplex, "hybrid refinement: real slice only"
        self.torch = torch
        self.ev = ev
        self.i = int(i)
        self.lam = float(lam)
        self.beta = float(beta)
        n, d1, d2 = ev.n, ev.d1, ev.d2

        # ── constants independent of a ──────────────────────────────────
        B = ev.B_bw(u_bw, v_bw)                       # line-independent
        self.B_t = torch.tensor(B)
        self.B_sq = float(B @ B)
        self.q_sw_out_inv = torch.tensor(ev._sw_out_inv)

        # cofactor G = monomial coeffs of prod_{j != i} alpha_j (unit lines)
        lines = ev.lines                              # already unit rows
        c = np.array([1.0])
        deg = 0
        for j in range(n):
            if j == self.i:
                continue
            ia, ib, io = _mult_table(deg, 1)
            out = np.zeros(len(_monoms(deg + 1)))
            lin = np.zeros(3)
            for idx, m in enumerate(_monoms(1)):
                lin[idx] = lines[j][{(1, 0, 0): 0, (0, 1, 0): 1,
                                     (0, 0, 1): 2}[m]]
            np.add.at(out, io, c[ia] * lin[ib])
            c = out
            deg += 1
        self.G_t = torch.tensor(c)                    # degree n-1
        ia, ib, io = _mult_table(n - 1, 1)
        self.mul_ia = torch.tensor(ia, dtype=torch.long)
        self.mul_ib = torch.tensor(ib, dtype=torch.long)
        self.mul_io = torch.tensor(io, dtype=torch.long)
        self.lin_order = [{(1, 0, 0): 0, (0, 1, 0): 1, (0, 0, 1): 2}[m]
                          for m in _monoms(1)]
        self.N_out = ev.N_out

        # residual constants: contributions of all OTHER lines
        # (recomputed exactly: R_other = R_total(a0) - r_i(a0))
        _, parts = ev.gamma(u_bw, v_bw, lam=lam, beta=beta, return_parts=True)
        self.R_total0 = float(parts["residual_R"])
        self.gamma_prod0 = float(parts["gamma_raw"])

        # per-degree data for line-i rows
        self.sides = []
        for (d, vec) in ((d1, u_bw), (d2, v_bw)):
            monoms = _monoms(d)
            N = len(monoms)
            sw = _bw_sqrt_weights(d)
            vec = np.asarray(vec, dtype=np.float64)
            blocks = np.stack([vec[:N] * sw, vec[N:2 * N] * sw,
                               vec[2 * N:] * sw])          # (3, N) monomial
            row_scale = np.array([1.0 / math.sqrt(_comb(d, p))
                                  for p in range(d + 1)])
            expans = []
            for p in range(d + 1):
                terms = []
                for m_idx, (ma, mb, mc) in enumerate(monoms):
                    for (ii, jj, kk, bc) in _restriction_expansion(
                            ma, mb, mc, p):
                        terms.append((m_idx, ii, ma - ii, jj, mb - jj,
                                      kk, mc - kk, float(bc)))
                idx = torch.tensor([t[0] for t in terms], dtype=torch.long)
                pw = torch.tensor([[t[1], t[2], t[3], t[4], t[5], t[6]]
                                   for t in terms])
                bc = torch.tensor([t[7] for t in terms])
                expans.append((idx, pw, bc))
            self.sides.append({
                "d": d, "N": N,
                "blocks": torch.tensor(blocks),
                "row_scale": torch.tensor(row_scale),
                "expans": expans,
            })

        a0 = np.asarray(lines[self.i], dtype=np.float64)
        self.pivot = int(np.argmin(np.abs(a0)))       # frozen frame pivot
        self.a0 = torch.tensor(a0)
        self.n = n

        # R_other = residual of all OTHER lines, a constant along the path:
        # R_total(a0) from the production evaluator minus torch r_i(a0)
        with torch.no_grad():
            r_i0 = float(self._r_line(self.a0 / torch.linalg.norm(self.a0)))
        self.R_other = max(self.R_total0 - r_i0, 0.0)

        # consistency check at the start point (torch vs production)
        g0 = float(self.gamma(self.a0))
        self.consistency_error = abs(g0 - self.gamma_prod0)

    # ── the differentiable score ────────────────────────────────────────

    def _r_line(self, a_unit):
        """Residual contribution of line i at coefficients a (unit)."""
        torch = self.torch
        ku, kw = _kernel_frame_torch(torch, a_unit, self.pivot)
        total = torch.zeros((), dtype=a_unit.dtype)
        for side in self.sides:
            d = side["d"]
            for p in range(d + 1):
                idx, pw, bc = side["expans"][p]
                # coeff_m = sum bc * ku_x^i kw_x^(ma-i) ku_y^j kw_y^(mb-j)...
                t = (bc
                     * ku[0] ** pw[:, 0] * kw[0] ** pw[:, 1]
                     * ku[1] ** pw[:, 2] * kw[1] ** pw[:, 3]
                     * ku[2] ** pw[:, 4] * kw[2] ** pw[:, 5])
                coeffs = torch.zeros(side["N"], dtype=a_unit.dtype)
                coeffs = coeffs.index_add(0, idx, t)
                # sw-scaling is already inside side["blocks"]
                dot = (a_unit[0] * (coeffs @ side["blocks"][0])
                       + a_unit[1] * (coeffs @ side["blocks"][1])
                       + a_unit[2] * (coeffs @ side["blocks"][2]))
                total = total + (side["row_scale"][p] * dot) ** 2
        return total / self.n

    def gamma(self, a):
        """Gamma as a torch scalar; `a` any nonzero 3-vector (normalized
        internally, so the value is scale/sign invariant)."""
        torch = self.torch
        a_unit = a / torch.linalg.norm(a)
        # q(a): multiply cofactor G by the linear form a_unit
        lin = a_unit[self.lin_order]
        c_out = torch.zeros(self.N_out, dtype=a.dtype)
        c_out = c_out.index_add(0, self.mul_io,
                                self.G_t[self.mul_ia] * lin[self.mul_ib])
        q_bw = c_out * self.q_sw_out_inv
        q_bw = q_bw / torch.linalg.norm(q_bw)
        inner = q_bw @ self.B_t
        num = inner ** 2
        # R(a) = R_other (constant) + r_i(a); no epsilon anywhere
        R = self.R_other + self._r_line(a_unit)
        penalty = torch.where(R > 0, self.lam * R ** self.beta,
                              torch.zeros_like(R))
        den = self.B_sq + penalty
        return num / den

    def gamma_and_grad(self, a_np):
        """(gamma, riemannian_grad) at a numpy 3-vector (unit)."""
        torch = self.torch
        a = torch.tensor(np.asarray(a_np, dtype=np.float64),
                         requires_grad=True)
        g = self.gamma(a)
        g.backward()
        grad = a.grad.detach().numpy()
        a_unit = np.asarray(a_np) / np.linalg.norm(a_np)
        tangent = grad - (grad @ a_unit) * a_unit      # sphere projection
        return float(g.detach()), tangent


def _rationalize_candidates(a_np, denominators=(4, 8, 12, 16, 24, 32, 48)):
    """Small-height exact rational representatives approximating a float
    line (scale-normalized first; per-coordinate limit_denominator)."""
    a = np.asarray(a_np, dtype=np.float64)
    k = int(np.argmax(np.abs(a)))
    a = a / a[k]
    from fractions import Fraction
    out, seen = [], set()
    for D in denominators:
        coords = tuple(Fraction(float(x)).limit_denominator(D)
                       for x in a)
        if all(c == 0 for c in coords) or coords in seen:
            continue
        seen.add(coords)
        try:
            out.append(ProjectiveLine(Rational(coords[0].numerator,
                                               coords[0].denominator),
                                      Rational(coords[1].numerator,
                                               coords[1].denominator),
                                      Rational(coords[2].numerator,
                                               coords[2].denominator)))
        except AssertionError:
            continue
    return out


def refine_line(arr: LineArrangement, i: int, d1: int, d2: int,
                lam=DEFAULT_LAMBDA, beta=DEFAULT_BETA,
                steps=12, lr0=0.5, seed=0,
                n_restarts=8, n_iters=80,
                consistency_tol=1e-7):
    """One-line hybrid refinement.  Returns (new_arr_or_None, report).

    new_arr is an EXACT (rationalized) arrangement accepted ONLY when its
    production raw loss strictly improves on the input's and all validity
    guards pass; otherwise None with the structured status in the report.
    """
    report = {"status": OK, "i": int(i), "gamma_trace": [],
              "grad_norms": [], "gamma_exceeds_one": 0,
              "steps_accepted": 0, "eval_calls": 0, "warnings": []}
    t0 = time.time()

    ev = PenalizedSaitoEvaluator(arr, d1, d2)
    res = ev.maximize(lam=lam, beta=beta, n_restarts=n_restarts,
                      n_iters=n_iters, seed=seed)
    report["eval_calls"] += 1
    if res.get("status") != "ok":
        report["status"] = "numerical_error_baseline"
        return None, report
    u_bw, v_bw = res["u"], res["v"]
    raw0 = float(res["loss"])
    report["raw_loss_before"] = raw0

    dg = DifferentiableGamma(ev, i, u_bw, v_bw, lam=lam, beta=beta)
    report["torch_vs_evaluator_gamma_diff"] = dg.consistency_error
    if dg.consistency_error > consistency_tol:
        report["status"] = CONSISTENCY_FAILED
        return None, report

    others = np.array([ev.lines[j] for j in range(len(arr)) if j != i])
    a = np.array(ev.lines[i], dtype=np.float64)
    g_cur, grad = dg.gamma_and_grad(a)
    report["gamma_trace"].append(g_cur)
    report["alignment0"] = (abs(float(np.vdot(ev.q, ev.B_bw(u_bw, v_bw))))
                            / max(math.sqrt(dg.B_sq), 1e-300))

    lr = lr0
    for _ in range(steps):
        if not np.all(np.isfinite(grad)):
            report["status"] = GRAD_NONFINITE
            break
        gn = float(np.linalg.norm(grad))
        report["grad_norms"].append(gn)
        if gn < 1e-14:
            break
        stepped = False
        while lr > 1e-8:
            cand = a + lr * grad
            cand = cand / np.linalg.norm(cand)
            if np.max(np.abs(others @ cand)) > _COLLISION_COS:
                lr *= 0.5                          # projected away
                continue
            g_new, grad_new = dg.gamma_and_grad(cand)
            if not math.isfinite(g_new):
                lr *= 0.5
                continue
            if g_new > g_cur:
                a, g_cur, grad = cand, g_new, grad_new
                report["gamma_trace"].append(g_cur)
                if g_cur > 1.0 + 1e-9:
                    report["gamma_exceeds_one"] += 1   # reported, not clipped
                report["steps_accepted"] += 1
                lr = min(lr * 1.6, 4.0)
                stepped = True
                break
            lr *= 0.5
        if not stepped:
            break
    if report["status"] == GRAD_NONFINITE:
        return None, report
    if np.max(np.abs(others @ a)) > _COLLISION_COS:
        report["status"] = LINE_COLLISION
        return None, report
    if report["steps_accepted"] == 0:
        report["status"] = NO_IMPROVEMENT
        return None, report

    # exact rationalization + RAW re-scoring (acceptance authority)
    best = (None, raw0)
    from swap_search import is_valid_state
    for cand_line in _rationalize_candidates(a):
        trial_lines = [l for j, l in enumerate(arr.lines) if j != i]
        if any(cand_line.coords == l.coords for l in trial_lines):
            continue
        trial = LineArrangement(trial_lines + [cand_line])
        if not is_valid_state(trial, len(arr), nontrivial=True):
            continue
        ev_t = PenalizedSaitoEvaluator(trial, d1, d2)
        res_t = ev_t.maximize(lam=lam, beta=beta, n_restarts=4,
                              n_iters=n_iters, seed=seed,
                              warm_starts=[(u_bw, v_bw)])
        report["eval_calls"] += 1
        if res_t.get("status") != "ok":
            continue
        if float(res_t["loss"]) < best[1]:
            best = (trial, float(res_t["loss"]))
    report["raw_loss_after"] = best[1]
    report["gamma_float_final"] = g_cur
    report["seconds"] = round(time.time() - t0, 2)
    if best[0] is None:
        if report["status"] == OK:
            report["status"] = (RATIONALIZATION_FAILED
                                if best[1] >= raw0 else NO_IMPROVEMENT)
        return None, report
    report["status"] = OK
    return best[0], report
