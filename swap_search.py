"""
swap_search.py

Fixed-cardinality replacement-move search for free line arrangements.

State: an arrangement of exactly n distinct lines with a target exponent pair
(d1, d2), d1 + d2 = n - 1.  Move:  A' = (A \\ {L-}) ∪ {L+}.

The corrected penalized Saito loss (penalized_saito.py) is defined at every
such state (the degree condition holds identically at fixed n), so it serves
as a dense search energy; the numerical loss NEVER certifies — every claimed
discovery carries an exact symbolic certificate (certificates.py) and is
re-verified before being persisted.

This module is torch-free (numpy/sympy/networkx only) so HPC workers stay
light.  Engines here are the "volume" searchers (greedy / random walk /
simulated annealing); the archive engine (MAP-Elites) and the learned policy
build on these primitives.

Conventions (paper revision):
  * essential states only; m_max <= n - 2 in nontrivial cells (pencils and
    near-pencils excluded); d1 = 1 cells are near-pencil-only controls;
  * candidates (numerically promising) and discoveries (exactly certified)
    are separate output streams; nothing here ever writes the repo-root
    discoveries.json.
"""

import json
import os
import time
from math import exp

import numpy as np
from sympy import Rational

from arrangement import LineArrangement, ProjectiveLine
from environment import (_singularity_candidates, generate_candidate_lines,
                         generate_candidate_lines_K)

_K_POOL_CACHE = {}


def _integer_pool(arr, coord_range):
    """QQ: historical integer grid.  Quadratic field: small O_K grid
    (cached per (d, range)); the field-closed singularity pool remains the
    primary K proposal source."""
    K = arr.coefficient_field()
    if K is None:
        return generate_candidate_lines(coord_range)
    key = (K.d, min(int(coord_range), 1))
    if key not in _K_POOL_CACHE:
        _K_POOL_CACHE[key] = generate_candidate_lines_K(K, key[1])
    return _K_POOL_CACHE[key]
from penalized_saito import PenalizedSaitoEvaluator, cached_penalized_loss
from saito import construct_supersolvable, predicted_delta_b2
from novelty import (lattice_wl_hash, is_essential, coordinate_height,
                     canonical_lineset_key, arrangement_from_record,
                     iter_corpus_records)

__all__ = [
    "SwapState", "ChainEvaluator",
    "double_pencil_seed", "random_valid_seed", "corpus_seeds",
    "perturb_k_swaps", "propose_swaps", "is_valid_state",
    "greedy_search", "random_walk", "simulated_annealing",
    "map_elites", "descriptor",
    "certify_state",
]

LOSS_CANDIDATE_THRESHOLD = 1e-6     # matches the refit extension pre-filter


# ─────────────────────────────────────────────────────────────────────────────
# Validity
# ─────────────────────────────────────────────────────────────────────────────

def is_valid_state(arr: LineArrangement, n: int, nontrivial: bool = True):
    """Exactly n distinct lines, essential; in nontrivial cells additionally
    m_max <= n - 2 (no pencil / near-pencil)."""
    if len(arr) != n:
        return False
    if len({l.coords for l in arr.lines}) != n:
        return False
    if not is_essential(arr):
        return False
    if nontrivial and arr.max_multiplicity() > n - 2:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Seeding
# ─────────────────────────────────────────────────────────────────────────────

def double_pencil_seed(n: int, d1: int, d2: int) -> LineArrangement:
    """Supersolvable double-pencil seed with exponents (1, d1, d2).

    d1 lines through one point, d2 through another, plus the connector
    (this is construct_supersolvable's construction).  Exists for every
    admissible cell; m_max = max(d1, d2) + 1 <= n - 2 whenever d1 >= 2.
    """
    assert d1 + d2 == n - 1 and 1 <= d1 <= d2
    return construct_supersolvable(n, d1)


def random_valid_seed(n: int, rng, coord_range: int = 3, nontrivial=True,
                      max_tries: int = 2000) -> LineArrangement:
    """Rejection-sample a valid random n-subset of the integer pool."""
    pool = generate_candidate_lines(coord_range)
    for _ in range(max_tries):
        idx = rng.choice(len(pool), size=n, replace=False)
        arr = LineArrangement([pool[i] for i in idx])
        if is_valid_state(arr, n, nontrivial):
            return arr
    raise RuntimeError(f"no valid random seed found for n={n}")


def corpus_seeds(n: int, d1: int, d2: int, limit: int = 20,
                 max_height: int = 50, repo_root=".", paths=None):
    """Low-height corpus arrangements in the (n, d1, d2) cell (seeding only —
    corpus freeness is NOT trusted for any claim)."""
    out, seen = [], set()
    for _, rec in iter_corpus_records(paths, repo_root):
        if rec.get("n") != n:
            continue
        exps = rec.get("exponents")
        if not exps or (int(exps[1]), int(exps[2])) != (d1, d2):
            continue
        key = str(tuple(sorted(rec["lines"])))
        if key in seen:
            continue
        seen.add(key)
        try:
            arr = arrangement_from_record(rec)
        except Exception:
            continue
        if coordinate_height(arr) > max_height:
            continue
        if is_valid_state(arr, n, nontrivial=(d1 >= 2)):
            out.append(arr)
            if len(out) >= limit:
                break
    return out


def perturb_k_swaps(arr: LineArrangement, k: int, rng, coord_range: int = 3,
                    nontrivial=True, max_tries: int = 200):
    """Apply k random valid swaps (uniform L-, random valid L+)."""
    n = len(arr)
    cur = arr.copy()
    pool = _integer_pool(arr, coord_range)
    for _ in range(k):
        for _ in range(max_tries):
            i = int(rng.integers(n))
            newline = pool[int(rng.integers(len(pool)))]
            if newline in cur.lines:
                continue
            trial = LineArrangement([l for j, l in enumerate(cur.lines)
                                     if j != i] + [newline])
            if is_valid_state(trial, n, nontrivial):
                cur = trial
                break
        else:
            break
    return cur


# ─────────────────────────────────────────────────────────────────────────────
# Proposals (two-tier Δb2 masking)
# ─────────────────────────────────────────────────────────────────────────────

def propose_swaps(arr: LineArrangement, d1: int, d2: int, rng,
                  coord_range: int = 3, n_remove: int = 4,
                  n_add_per_remove: int = 24, exact_frac: float = 0.7,
                  b2_slack: int = 2, tabu=None, nontrivial=True):
    """Sample candidate swaps (i_minus, L_plus).

    For each of `n_remove` random removal choices, build the Δb2-tiers over
    L+ candidates (singularity lines of A \\ {L-} + integer pool): the exact
    tier restores b2 to b2* = (n-1) + d1*d2; the slack tier lands within
    `b2_slack`.  Roughly `exact_frac` of returned proposals come from the
    exact tier when it is nonempty.  `tabu` is a set of canonical line-set
    keys to skip (revisits are pointless — the loss cache would be hit, but
    the move wastes a slot).
    """
    n = len(arr)
    b2_star = (n - 1) + d1 * d2
    proposals = []
    removals = rng.permutation(n)[:n_remove]
    for i in removals:
        rest = LineArrangement([l for j, l in enumerate(arr.lines) if j != i])
        b2_rest = rest.b2()
        required = b2_star - b2_rest
        # candidate L+ pool: singularity lines (score-sorted) + random pool
        cands = [line for _, line in _singularity_candidates(rest)[:120]]
        pool = _integer_pool(rest, coord_range)
        cands.extend(pool[j] for j in rng.choice(len(pool),
                                                 size=min(60, len(pool)),
                                                 replace=False))
        existing = {l.coords for l in rest.lines}
        removed_coords = arr.lines[i].coords
        exact_tier, slack_tier = [], []
        seen = set()
        for line in cands:
            c = line.coords
            if c in existing or c == removed_coords or c in seen:
                continue
            seen.add(c)
            delta = predicted_delta_b2(line, rest)
            gap = abs(delta - required)
            if gap == 0:
                exact_tier.append(line)
            elif gap <= b2_slack:
                slack_tier.append(line)
        rng.shuffle(exact_tier)
        rng.shuffle(slack_tier)
        take = n_add_per_remove
        k_exact = min(len(exact_tier), max(1, int(round(take * exact_frac)))) \
            if exact_tier else 0
        chosen = exact_tier[:k_exact] + slack_tier[:take - k_exact]
        for line in chosen:
            trial = LineArrangement(list(rest.lines) + [line])
            if not is_valid_state(trial, n, nontrivial):
                continue
            if tabu is not None and canonical_lineset_key(trial) in tabu:
                continue
            proposals.append((int(i), line, trial))
    rng.shuffle(proposals)
    return proposals


# ─────────────────────────────────────────────────────────────────────────────
# Energy
# ─────────────────────────────────────────────────────────────────────────────

class ChainEvaluator:
    """Loss/energy evaluation for one chain at fixed (n, d1, d2).

    Screening uses the shared cached 'rl'-profile loss; refinement runs the
    evaluator directly with warm starts carried across accepted states
    (dimension-compatible at fixed degrees).  Energy adds a small pull toward
    the target b2 shell so off-shell excursions are allowed but bounded.
    """

    def __init__(self, n, d1, d2, w_b2=0.05, seed=0,
                 refine_restarts=8, refine_iters=80):
        self.n, self.d1, self.d2 = n, d1, d2
        self.b2_star = (n - 1) + d1 * d2
        self.w_b2 = w_b2
        self.seed = seed
        self.refine_restarts = refine_restarts
        self.refine_iters = refine_iters
        self.warm = None          # (u, v) from the last refined evaluation
        self.n_screen = 0
        self.n_refine = 0
        self.n_numerical_errors = 0

    def screen_loss(self, arr) -> float:
        """Raw penalized loss (rl profile, cached).  Raises
        GammaNumericalError on evaluation failure — engines catch it per
        proposal, skip the move, and count the event; a failure is never a
        loss value."""
        self.n_screen += 1
        return cached_penalized_loss(arr, d1=self.d1, d2=self.d2,
                                     profile='rl', seed=self.seed)

    def screen_loss_or_none(self, arr):
        from penalized_saito import GammaNumericalError
        try:
            return self.screen_loss(arr)
        except GammaNumericalError:
            self.n_numerical_errors += 1
            return None

    def refined_loss(self, arr) -> float:
        """Search-profile evaluation with warm start; updates the warm pair."""
        self.n_refine += 1
        ev = PenalizedSaitoEvaluator(arr, self.d1, self.d2)
        warm = [self.warm] if self.warm is not None else None
        res = ev.maximize(n_restarts=self.refine_restarts,
                          n_iters=self.refine_iters, seed=self.seed,
                          warm_starts=warm)
        self.warm = (res["u"], res["v"])
        return res["loss"]

    def energy(self, arr, loss=None) -> float:
        if loss is None:
            loss = self.screen_loss(arr)
        b2_gap = abs(arr.b2() - self.b2_star) / max(1, self.b2_star)
        return loss + self.w_b2 * b2_gap

    def energy_components(self, arr, loss) -> dict:
        """Separately logged energy components (audit requirement): the
        RAW Saito loss, the b2-shell penalty and weight, and the total.
        Calibration NEVER appears here — engines rank by raw loss."""
        b2_pen = abs(arr.b2() - self.b2_star) / max(1, self.b2_star)
        return {"raw_saito_loss": float(loss),
                "b2_shell_penalty": float(b2_pen),
                "b2_shell_weight": float(self.w_b2),
                "total_energy": float(loss + self.w_b2 * b2_pen)}

    def stats(self):
        return {"screen_evals": self.n_screen, "refine_evals": self.n_refine,
                "numerical_errors": self.n_numerical_errors}


# ─────────────────────────────────────────────────────────────────────────────
# Certification (exact; sympy path — fast mod-p path lands in certificates.py)
# ─────────────────────────────────────────────────────────────────────────────

def certify_state(arr, d1, d2):
    """Exact certificate or None (fast path: point-evaluation pair scan with
    a symbolically verified positive and an exact sound negative).  Every
    positive is additionally re-verified from scratch before being returned."""
    from certificates import find_certificate_fast, verify_certificate
    cert, status = find_certificate_fast(arr, target_exponents=(d1, d2))
    if cert is None:
        return None
    if not verify_certificate(cert):
        return None
    return cert


# ─────────────────────────────────────────────────────────────────────────────
# Engines (chain-level; the campaign runner parallelizes chains)
# ─────────────────────────────────────────────────────────────────────────────

def _record(arr, d1, d2, loss, engine, step, extra=None):
    from penalized_saito import DEFAULT_LAMBDA, DEFAULT_BETA
    rec = {
        "lines": [str(l) for l in arr.lines],
        "n": len(arr), "d1": d1, "d2": d2,
        "loss": float(loss), "b2": arr.b2(),
        "m_max": arr.max_multiplicity(),
        "height": coordinate_height(arr),
        "lattice_hash": lattice_wl_hash(arr),
        "engine": engine, "step": int(step), "t": time.time(),
        "lambda": DEFAULT_LAMBDA, "beta": DEFAULT_BETA,
        "optimization_field": ("complex" if (arr.coefficient_field()
                               and not arr.coefficient_field().is_real)
                               else "real"),
        "coefficient_field": ("QQ" if arr.coefficient_field() is None
                              else arr.coefficient_field().to_json()),
    }
    if extra:
        rec.update(extra)
    return rec


def greedy_search(state, d1, d2, evaluator, rng, steps=200,
                  proposal_kwargs=None, loss_tol=1e-9, on_candidate=None):
    """Greedy best-of-k swap descent.  Returns (best_arr, best_loss, log)."""
    pk = dict(proposal_kwargs or {})
    cur = state
    seed_loss = evaluator.screen_loss_or_none(cur)
    if seed_loss is None:
        raise RuntimeError("greedy_search: seed state failed numerical "
                           "evaluation")
    cur_energy = evaluator.energy(cur, seed_loss)
    best = (cur, seed_loss)
    log = []
    tabu = {canonical_lineset_key(cur)}
    for step in range(steps):
        proposals = propose_swaps(cur, d1, d2, rng, tabu=tabu, **pk)
        if not proposals:
            log.append({"step": step, "event": "no_proposals"})
            break
        scored = []
        for (i, line, trial) in proposals:
            loss = evaluator.screen_loss_or_none(trial)
            if loss is None:
                continue          # numerical failure: skip move, counted
            scored.append((evaluator.energy(trial, loss), loss, trial))
        if not scored:
            log.append({"step": step, "event": "all_proposals_failed"})
            break
        scored.sort(key=lambda t: t[0])
        e_new, loss_new, arr_new = scored[0]
        if e_new >= cur_energy:
            log.append({"step": step, "event": "local_min",
                        "energy": cur_energy})
            break
        cur, cur_energy = arr_new, e_new
        tabu.add(canonical_lineset_key(cur))
        if loss_new < best[1]:
            best = (cur, loss_new)
        if on_candidate is not None and loss_new < LOSS_CANDIDATE_THRESHOLD:
            on_candidate(_record(cur, d1, d2, loss_new, "greedy", step,
                                 extra=evaluator.energy_components(cur, loss_new)))
        if loss_new < loss_tol:
            break
    return best[0], best[1], log


def random_walk(state, d1, d2, evaluator, rng, steps=500,
                proposal_kwargs=None, on_candidate=None):
    """Uniform random valid swaps (baseline)."""
    pk = dict(proposal_kwargs or {})
    cur = state
    seed_loss = evaluator.screen_loss_or_none(cur)
    if seed_loss is None:
        raise RuntimeError("random_walk: seed state failed numerical "
                           "evaluation")
    best = (cur, seed_loss)
    for step in range(steps):
        proposals = propose_swaps(cur, d1, d2, rng, n_remove=1,
                                  n_add_per_remove=1, **pk)
        if not proposals:
            continue
        _, _, nxt = proposals[0]
        loss = evaluator.screen_loss_or_none(nxt)
        if loss is None:
            continue              # numerical failure: skip, counted
        cur = nxt
        if loss < best[1]:
            best = (cur, loss)
        if on_candidate is not None and loss < LOSS_CANDIDATE_THRESHOLD:
            on_candidate(_record(cur, d1, d2, loss, "walk", step,
                                 extra=evaluator.energy_components(cur, loss)))
    return best[0], best[1], []


def descriptor(arr, n):
    """MAP-Elites behavior descriptor: (m_max clamped to [3, n-2],
    #points of multiplicity >= 3 binned by 2, sign of b2 drift is handled by
    the energy, so the third slot is the count of >= 4-fold points binned
    by 2).  Coarse on purpose: descriptor cells hold lattice reservoirs."""
    mults = arr.multiplicities()
    m_max = max(3, min(arr.max_multiplicity(), n - 2))
    t3p = sum(1 for m in mults if m >= 3)
    t4p = sum(1 for m in mults if m >= 4)
    return (m_max, t3p // 2, t4p // 2)


def map_elites(seeds, d1, d2, evaluator, rng, generations=3000,
               reservoir_size=8, on_candidate=None, loss_tol=1e-9,
               proposal_kwargs=None, archive=None, on_snapshot=None,
               snapshot_every=500):
    """Quality-diversity archive over swap moves.

    archive: {descriptor: [elite dicts]} where each descriptor cell keeps up
    to `reservoir_size` elites with DISTINCT lattice hashes, ordered by
    (not certified, loss, height, canonical key) — so coordinate-level
    improvements can never evict lattice diversity.  Elite dicts store the
    line coordinate strings; arrangements are rebuilt on selection.

    Deterministic for a fixed rng seed and seed list (single process, total
    order tie-breaks).  Returns (archive, best_arr, best_loss).
    """
    from novelty import parse_line_str, is_supersolvable_rank3
    from math import log10
    pk = dict(proposal_kwargs or {})
    n = len(seeds[0])
    archive = {} if archive is None else archive
    best = (None, 1.0)

    def elite_sort_key(e):
        # certified first; then loss ORDER OF MAGNITUDE; within a magnitude
        # bucket prefer NON-supersolvable elites (novelty pressure); then
        # exact loss, height, canonical key (total order, deterministic)
        bucket = int(log10(max(e["loss"], 1e-16)))
        return (not e["certified"], bucket, e.get("ss", True), e["loss"],
                e["height"], e["key"])

    def add_to_archive(arr, loss, certified=False):
        nonlocal best
        d = str(descriptor(arr, n))
        K_arr = arr.coefficient_field()
        rec = {
            "lines": [str(l) for l in arr.lines],
            "coefficient_field": ("QQ" if K_arr is None else K_arr.to_json()),
            "loss": float(loss), "b2": arr.b2(),
            "m_max": arr.max_multiplicity(),
            "height": coordinate_height(arr),
            "lattice_hash": lattice_wl_hash(arr),
            "key": canonical_lineset_key(arr),
            "certified": bool(certified),
            "ss": bool(is_supersolvable_rank3(arr)),
            "descriptor": d,
        }
        cell = archive.setdefault(d, [])
        same_lat = [e for e in cell if e["lattice_hash"] == rec["lattice_hash"]]
        if same_lat:
            incumbent = same_lat[0]
            if elite_sort_key(rec) < elite_sort_key(incumbent):
                cell.remove(incumbent)
                cell.append(rec)
        else:
            cell.append(rec)
        cell.sort(key=elite_sort_key)
        del cell[reservoir_size:]
        if loss < best[1]:
            best = (arr, loss)
        return rec

    for s in seeds:
        sl = evaluator.screen_loss_or_none(s)
        if sl is not None:
            add_to_archive(s, sl)

    for gen in range(generations):
        # deterministic parent selection: uniform over cells, then over elites
        cells = sorted(archive.keys())
        cell = archive[cells[int(rng.integers(len(cells)))]]
        parent_rec = cell[int(rng.integers(len(cell)))]
        parent_field = None
        cf = parent_rec.get("coefficient_field")
        if cf and cf != "QQ":
            from quadfield import QuadraticField
            parent_field = QuadraticField.from_json(cf)
        parent = LineArrangement([parse_line_str(s, field=parent_field)
                                  for s in parent_rec["lines"]])
        r = rng.random()
        if r < 0.80:
            k_swaps = 1
        elif r < 0.95:
            k_swaps = 2
        else:
            k_swaps = 3
        child = parent
        for _ in range(k_swaps):
            props = propose_swaps(child, d1, d2, rng, n_remove=2,
                                  n_add_per_remove=4, **pk)
            if not props:
                break
            child = props[0][2]
        if child is parent:
            continue
        loss = evaluator.screen_loss_or_none(child)
        if loss is None:
            continue              # numerical failure: never archived
        rec = add_to_archive(child, loss)
        if on_candidate is not None and loss < LOSS_CANDIDATE_THRESHOLD:
            on_candidate(_record(child, d1, d2, loss, "map_elites", gen,
                                 extra=evaluator.energy_components(child, loss)))
        if best[1] < loss_tol:
            break
        if on_snapshot is not None and (gen + 1) % snapshot_every == 0:
            on_snapshot(archive, gen)
    return archive, best[0], best[1]


def simulated_annealing(state, d1, d2, evaluator, rng, steps=2000,
                        t0=0.05, t_min=1e-4, cooling=0.999,
                        reheat_after=300, proposal_kwargs=None,
                        on_candidate=None, loss_tol=1e-9):
    """Metropolis on the energy with geometric cooling, tabu on visited
    line-sets, and reheats on stagnation.  Returns (best_arr, best_loss, log)."""
    pk = dict(proposal_kwargs or {})
    cur = state
    seed_loss = evaluator.screen_loss_or_none(cur)
    if seed_loss is None:
        raise RuntimeError("simulated_annealing: seed state failed "
                           "numerical evaluation")
    cur_energy = evaluator.energy(cur, seed_loss)
    best = (cur, seed_loss)
    T = t0
    tabu = {canonical_lineset_key(cur)}
    since_improve = 0
    log = []
    for step in range(steps):
        proposals = propose_swaps(cur, d1, d2, rng, n_remove=2,
                                  n_add_per_remove=6, tabu=tabu, **pk)
        if not proposals:
            since_improve += 1
            T = min(t0, T / cooling)
            continue
        i, line, trial = proposals[0]
        loss = evaluator.screen_loss_or_none(trial)
        if loss is None:
            since_improve += 1    # numerical failure: skip move, counted
            continue
        e_new = evaluator.energy(trial, loss)
        accept = e_new <= cur_energy or \
            rng.random() < exp(-(e_new - cur_energy) / max(T, 1e-12))
        if accept:
            cur, cur_energy = trial, e_new
            tabu.add(canonical_lineset_key(cur))
            if len(tabu) > 20000:
                tabu.clear()
            if loss < best[1]:
                best = (cur, loss)
                since_improve = 0
            else:
                since_improve += 1
            if on_candidate is not None and loss < LOSS_CANDIDATE_THRESHOLD:
                on_candidate(_record(cur, d1, d2, loss, "anneal", step,
                                     extra=evaluator.energy_components(cur, loss)))
            if loss < loss_tol:
                break
        else:
            since_improve += 1
        T = max(t_min, T * cooling)
        if since_improve >= reheat_after:
            T = t0
            since_improve = 0
            log.append({"step": step, "event": "reheat"})
    return best[0], best[1], log
