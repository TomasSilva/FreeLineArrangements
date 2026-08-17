"""
environment.py

Gym-like RL environment for building free line arrangements in CP^2.

Two modes:
  1. Pool mode (original):  agent picks from a fixed pool of small-integer-
     coordinate lines.
  2. Singularity-aware mode (new):  candidates are generated dynamically at
     each step from the intersection structure of the current arrangement.
     Lines through pairs of existing high-multiplicity points are proposed
     first, mixed with a random sample from the standard pool for diversity.

State (dict):
  - selected_coords: float32 (max_n, 3) — selected line coordinates, zero-padded
  - candidate_coords: float32 (max_candidates, 3) — current candidate lines
  - scalars: float32 (scalar_dim,) — global arrangement features
  - n_selected: int — number of lines selected so far
  - mask: float32 (max_candidates,) — 1 = valid candidate

Action:
  - Index into the current candidate list (masked for invalid positions).

Reward:
  - Shaped via Saito criterion (see saito.py).
  - Large bonus at episode end if arrangement is free and not a pencil.
  - Large penalty at any step for creating a pencil.
  - Interestingness bonus for rich singularity structure.
"""

import numpy as np
from collections import Counter
from arrangement import LineArrangement, ProjectiveLine
from saito import saito_reward, combinatorial_score, algebraic_score


# ─────────────────────────────────────────────────────────────────────────────
# Candidate line pool (static, for bootstrap / diversity / legacy mode)
# ─────────────────────────────────────────────────────────────────────────────

def generate_candidate_lines(coord_range=3):
    """
    Generate all projectively distinct lines [a:b:c] with integer coordinates
    in [-coord_range, coord_range], not all zero.

    Returns list of ProjectiveLine.
    """
    candidates = []
    seen = set()
    r = coord_range
    for a in range(-r, r + 1):
        for b in range(-r, r + 1):
            for c in range(-r, r + 1):
                if a == 0 and b == 0 and c == 0:
                    continue
                try:
                    line = ProjectiveLine(a, b, c)
                except AssertionError:
                    continue
                if line.coords not in seen:
                    seen.add(line.coords)
                    candidates.append(line)
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Singularity-driven candidate generation
# ─────────────────────────────────────────────────────────────────────────────

def _singularity_candidates(arr):
    """
    Generate candidate lines from the intersection structure.

    For each pair of intersection points, compute the line through them
    (cross product in CP^2).  Deduplicate and exclude lines already in arr.

    Returns list of (score, ProjectiveLine) sorted by descending score.
    Score = sum of multiplicities of intersection points that the line
    passes through — higher means the line reinforces more singularities.
    """
    if len(arr) < 2:
        return []

    pts = arr._structure()  # point -> set of line indices
    pt_list = list(pts.keys())
    mults_list = [len(pts[pt]) for pt in pt_list]
    existing = set(l.coords for l in arr.lines)

    # Build float arrays for batch scoring
    P = len(pt_list)
    if P < 2:
        return []
    pts_float = np.array([[float(c) for c in pt] for pt in pt_list], dtype=np.float64)
    mults_arr = np.array(mults_list, dtype=np.float64)

    # Generate candidate lines from pairs (exact arithmetic for dedup)
    candidates = {}  # coords -> (line, line_float)
    for i in range(P):
        for j in range(i + 1, P):
            line = ProjectiveLine.from_two_points(pt_list[i], pt_list[j])
            if line is None or line.coords in existing or line.coords in candidates:
                continue
            candidates[line.coords] = (line, line.to_float())

    if not candidates:
        return []

    # Batch score all candidates at once: dot product with all points
    cand_list = list(candidates.values())
    cand_float = np.array([cf for _, cf in cand_list], dtype=np.float64)  # (C, 3)
    dots = cand_float @ pts_float.T  # (C, P)
    on_line = np.abs(dots) < 1e-10  # boolean (C, P)
    scores = on_line @ mults_arr  # (C,) — sum of mults for points on each line

    # Build scored result
    result = [(scores[k], cand_list[k][0]) for k in range(len(cand_list))]
    result.sort(key=lambda x: -x[0])
    return result


def _score_line_against_structure(line, arr):
    """Score a line by how many existing intersection points it passes through,
    weighted by their multiplicity."""
    if len(arr) < 2:
        return 0.0
    pts = arr._structure()
    score = 0.0
    for pt, lines_through in pts.items():
        if line.passes_through(pt):
            score += len(lines_through)
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction (scalar features)
# ─────────────────────────────────────────────────────────────────────────────

SCALAR_DIM = 17   # 14 base + 3 exponent-targeting features

def extract_scalars(arr: LineArrangement, target_n: int,
                    target_exponents=None, score_mode: str = 'penalized'
                    ) -> np.ndarray:
    """
    Build scalar feature vector for the policy network.

    Original 11 features + 3 singularity-aware + 3 exponent-targeting:
      [0]  n / target_n
      [1]  b2 / (n*(n-1)/2 + 1)
      [2]  disc_norm (tanh-scaled discriminant)
      [3]  m2 / max  (double points, normalized)
      [4]  m3 / max  (triple points, normalized)
      [5]  m4p / max (quadruple+ points, normalized)
      [6]  is_pencil
      [7]  combinatorial_score
      [8]  algebraic_score
      [9]  max_mult / target_n
      [10] n_pts / (n*(n-1)/2)
      [11] n_triple_plus / n  (fraction of points with mult >= 3)
      [12] mult_entropy  (entropy of multiplicity distribution, normalized)
      [13] singularity_density = sum(C(m,2)) / C(n,2)
      --- exponent targeting ---
      [14] d1_norm = target_d1 / (target_n - 1)
      [15] d2_norm = target_d2 / (target_n - 1)
      [16] b2_progress = 1 - |b2 - target_b2| / max_b2
    """
    n = len(arr)
    if n < 2:
        b2 = 0
        disc_norm = 0.0
        m2, m3, m4p = 0, 0, 0
        mults = []
    else:
        b2 = arr.b2()
        product = b2 - (n - 1)
        disc = (n - 1) ** 2 - 4 * product
        disc_norm = np.tanh(disc / max(1, (n - 1) ** 2))
        mults = arr.multiplicities()
        m2 = sum(1 for m in mults if m == 2)
        m3 = sum(1 for m in mults if m == 3)
        m4p = sum(1 for m in mults if m >= 4)

    max_mult = arr.max_multiplicity() if n >= 2 else 0
    n_pts = arr.n_intersection_points() if n >= 2 else 0

    # --- new singularity-aware features ---
    if mults:
        n_triple_plus = sum(1 for m in mults if m >= 3)
        triple_ratio = n_triple_plus / max(1, len(mults))

        # Entropy of multiplicity distribution
        mult_counts = Counter(mults)
        total = sum(mult_counts.values())
        entropy = -sum(
            (c / total) * np.log(c / total + 1e-10)
            for c in mult_counts.values()
        )
        max_entropy = np.log(max(2, len(mult_counts)))
        norm_entropy = entropy / (max_entropy + 1e-10) if max_entropy > 0 else 0.0

        # Singularity density: fraction of line-pairs meeting at a high-mult point
        # sum C(m,2) over all points / C(n,2)
        sing_density = sum(m * (m - 1) / 2 for m in mults) / max(1, n * (n - 1) / 2)
    else:
        triple_ratio = 0.0
        norm_entropy = 0.0
        sing_density = 0.0

    # Exponent-targeting features
    if target_exponents is not None:
        d1_t, d2_t = target_exponents
        d1_norm = d1_t / max(1, target_n - 1)
        d2_norm = d2_t / max(1, target_n - 1)
        target_b2 = (target_n - 1) + d1_t * d2_t
        scale = max(1, target_b2)
        b2_progress = max(0.0, 1.0 - abs(b2 - target_b2) / scale) if n >= 2 else 0.0
    else:
        d1_norm = 0.0
        d2_norm = 0.0
        b2_progress = 0.0

    return np.array([
        n / target_n,
        b2 / max(1, n * (n - 1) / 2),
        disc_norm,
        m2 / max(1, n * (n - 1) / 2),
        m3 / max(1, n),
        m4p / max(1, n),
        float(arr.is_pencil()),
        combinatorial_score(arr),
        # feature [8]: algebraic score under the active reward mode.
        # 'penalized' (default) = corrected penalized-loss score;
        # 'legacy' = explicitly-invalid old angular score (regression arm);
        # anything else = 0.0 so score-free arms leak no loss signal via obs.
        (algebraic_score(arr, target_exponents=target_exponents,
                         use_legacy=(score_mode == 'legacy'))
         if n >= 3 and score_mode in ('penalized', 'legacy') else 0.0),
        max_mult / max(1, target_n),
        n_pts / max(1, n * (n - 1) / 2),
        # singularity-aware
        triple_ratio,
        norm_entropy,
        sing_density,
        # exponent-targeting
        d1_norm,
        d2_norm,
        b2_progress,
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class FreeArrangementEnv:
    """
    Episode: build an arrangement of exactly `target_n` lines, one per step.

    In **pool mode** (singularity_aware=False, the original behaviour), the
    agent picks from a fixed pool of small-integer-coordinate lines.

    In **singularity-aware mode** (singularity_aware=True), candidates are
    regenerated every step from the intersection structure:
      - Lines through pairs of high-multiplicity points
      - A random sample from the static pool (for diversity / bootstrapping)
    The first ``bootstrap_steps`` lines are always drawn from the static pool
    so the arrangement has some intersection points to work with.
    """

    def __init__(
        self,
        target_n: int = 6,
        coord_range: int = 3,
        max_n: int = None,
        max_candidates: int = 200,
        singularity_aware: bool = False,
        bootstrap_steps: int = 3,
        pool_sample_frac: float = 0.3,
        w_comb: float = 0.3,
        w_alg: float = 0.5,
        w_pencil: float = 5.0,
        w_free: float = 10.0,
        w_mult: float = 2.0,
        w_interest: float = 1.0,
        skip_exact_above: int = 12,
        seed: int = 0,
        reward_mode: str = 'penalized',
        gamma_shaping: float = 0.99,
    ):
        # reward_mode arms (for controlled comparisons, see §8 of the
        # penalized-loss migration report):
        #   'penalized'     composite reward with the corrected loss (default)
        #   'potential'     potential-based shaping gamma*Phi' - Phi with
        #                   Phi = Gamma_hat, plus exact terminal bonus
        #   'combinatorial' composite with w_alg = 0 (no algebraic signal)
        #   'terminal'      binary exact-freeness bonus at terminal only
        #   'random'        zero reward (unguided search baseline)
        #   'legacy'        composite with legacy_invalid_angular_score
        #                   (explicitly invalid regression baseline)
        if reward_mode not in ('penalized', 'potential', 'combinatorial',
                               'terminal', 'random', 'legacy'):
            raise ValueError(f"unknown reward_mode {reward_mode!r}")
        self.reward_mode = reward_mode
        self.gamma_shaping = gamma_shaping
        self._phi_prev = 0.0
        self.target_n = target_n
        self.max_n = max_n if max_n is not None else target_n
        self.max_candidates = max_candidates
        self.singularity_aware = singularity_aware
        self.bootstrap_steps = bootstrap_steps
        self.pool_sample_frac = pool_sample_frac
        self.w_comb = w_comb
        self.w_alg = w_alg
        self.w_pencil = w_pencil
        self.w_free = w_free
        self.w_mult = w_mult
        self.w_interest = w_interest
        self.skip_exact_above = skip_exact_above
        self.target_exponents = None  # (d1, d2) or None

        # Static pool (always available for bootstrap / diversity)
        self.pool = generate_candidate_lines(coord_range)
        self.pool_size = len(self.pool)
        self._pool_coords_set = {l.coords for l in self.pool}
        self._pool_index = {l.coords: i for i, l in enumerate(self.pool)}

        # Fixed float coords for the static pool (used in pool-only mode)
        self._pool_float = np.array(
            [l.to_float() for l in self.pool], dtype=np.float32
        )

        self.rng = np.random.default_rng(seed)
        self.reset()

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self, target_n: int = None, random_start: bool = False,
              target_exponents=None):
        if target_n is not None:
            self.target_n = target_n
        self.target_exponents = target_exponents
        self.arr = LineArrangement()
        self.selected_pool = np.zeros(self.pool_size, dtype=bool)
        self.step_count = 0
        self.done = False
        self.episode_reward = 0.0
        self._phi_prev = 0.0

        # Current dynamic candidates (regenerated each step in sing-aware mode)
        self.current_candidates = []
        self.current_candidate_coords = np.zeros(
            (self.max_candidates, 3), dtype=np.float32
        )
        self.current_mask = np.zeros(self.max_candidates, dtype=np.float32)

        if random_start and self.pool_size >= 2:
            n_start = self.rng.integers(1, min(3, self.target_n))
            for _ in range(n_start):
                self._add_random_line()

        self._update_candidates()
        return self._obs()

    # ── Candidate generation ─────────────────────────────────────────────────

    def _update_candidates(self):
        """Rebuild the candidate list for the current step."""
        if not self.singularity_aware:
            # Legacy pool mode: candidates = full pool, mask = not-yet-selected
            self.current_candidates = list(self.pool)
            self.current_candidate_coords = self._pool_float.copy()
            self.current_mask = (~self.selected_pool).astype(np.float32)
            # Pad / truncate to max_candidates
            if self.pool_size < self.max_candidates:
                pad = self.max_candidates - self.pool_size
                self.current_candidate_coords = np.pad(
                    self.current_candidate_coords, ((0, pad), (0, 0))
                )
                self.current_mask = np.pad(self.current_mask, (0, pad))
            elif self.pool_size > self.max_candidates:
                self.current_candidate_coords = self.current_candidate_coords[:self.max_candidates]
                self.current_mask = self.current_mask[:self.max_candidates]
                self.current_candidates = self.current_candidates[:self.max_candidates]
            return

        # ── Singularity-aware mode ───────────────────────────────────────────
        existing_coords = set(l.coords for l in self.arr.lines)
        seen = set(existing_coords)
        scored_candidates = []

        in_bootstrap = self.step_count < self.bootstrap_steps

        if not in_bootstrap:
            # Singularity candidates: lines through pairs of intersection points
            sing_cands = _singularity_candidates(self.arr)
            for score, line in sing_cands:
                if line.coords not in seen:
                    seen.add(line.coords)
                    scored_candidates.append((score, line))

        # Pool candidates (for diversity / bootstrap)
        # During bootstrap or when few singularity candidates, use more pool lines
        if in_bootstrap:
            n_pool = self.max_candidates
        else:
            n_pool = max(
                10,
                int(self.max_candidates * self.pool_sample_frac)
            )

        available_pool = [
            i for i in range(self.pool_size)
            if not self.selected_pool[i] and self.pool[i].coords not in seen
        ]
        if len(available_pool) > n_pool:
            pool_indices = self.rng.choice(
                available_pool, size=n_pool, replace=False
            )
        else:
            pool_indices = available_pool

        for idx in pool_indices:
            line = self.pool[idx]
            if line.coords not in seen:
                seen.add(line.coords)
                score = _score_line_against_structure(line, self.arr)
                scored_candidates.append((score, line))

        # Sort by descending score, take top max_candidates
        scored_candidates.sort(key=lambda x: -x[0])
        selected = scored_candidates[:self.max_candidates]

        # Build arrays
        self.current_candidates = []
        self.current_candidate_coords = np.zeros(
            (self.max_candidates, 3), dtype=np.float32
        )
        self.current_mask = np.zeros(self.max_candidates, dtype=np.float32)
        for i, (score, line) in enumerate(selected):
            self.current_candidates.append(line)
            self.current_candidate_coords[i] = line.to_float()
            self.current_mask[i] = 1.0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _add_random_line(self):
        available = np.where(~self.selected_pool)[0]
        if len(available) == 0:
            return
        idx = self.rng.choice(available)
        self.arr.add_line(self.pool[idx])
        self.selected_pool[idx] = True
        self.step_count += 1

    def _obs(self):
        sel = np.zeros((self.max_n, 3), dtype=np.float32)
        n = len(self.arr)
        for i, line in enumerate(self.arr.lines):
            sel[i] = line.to_float()

        if self.reward_mode in ('penalized', 'potential'):
            score_mode = 'penalized'
        elif self.reward_mode == 'legacy':
            score_mode = 'legacy'
        else:
            score_mode = 'none'   # blind the score-free arms
        scalars = extract_scalars(self.arr, self.target_n,
                                  target_exponents=self.target_exponents,
                                  score_mode=score_mode)

        return {
            'selected_coords': sel,
            'candidate_coords': self.current_candidate_coords,
            'scalars': scalars,
            'n_selected': n,
            'mask': self.current_mask,
        }

    def action_mask(self):
        return self.current_mask.copy()

    # ── Reward modes ──────────────────────────────────────────────────────────

    def _exact_free_terminal(self):
        """Exact symbolic terminal check (gated by skip_exact_above)."""
        if self.arr.candidate_exponents() is None:
            return False
        if len(self.arr) > self.skip_exact_above:
            return False
        is_free, _ = self.arr.is_free()
        return bool(is_free)

    def _potential(self):
        """Phi(A) = Gamma_hat(A) for potential-based shaping.

        When the episode has target exponents and the arrangement is at full
        size (d1 + d2 = n - 1), score against that pair — the loss is defined
        there regardless of candidate-exponent arithmetic.  During growth (or
        without a target), fall back to the arrangement's own candidate
        exponents when they exist, else 0.  Any state potential yields
        policy-invariant shaping; this choice keeps the hot path cheap.
        """
        from saito import saito_loss
        arr = self.arr
        n = len(arr)
        if n < 3:
            return 0.0
        te = self.target_exponents
        if te is not None and te[0] + te[1] == n - 1:
            return 1.0 - saito_loss(arr, target_exponents=tuple(te),
                                    profile='rl', cached=True)
        if arr.candidate_exponents() is None:
            return 0.0
        return 1.0 - saito_loss(arr, profile='rl', cached=True)

    def _compute_reward(self, prev_arr, is_pencil, is_terminal):
        mode = self.reward_mode
        if mode == 'random':
            return 0.0
        if mode == 'terminal':
            if is_terminal and not is_pencil and self._exact_free_terminal():
                return self.w_free
            return 0.0
        if mode == 'potential':
            if is_pencil:
                self._phi_prev = 0.0
                return -self.w_pencil
            phi_new = self._potential()
            reward = self.gamma_shaping * phi_new - self._phi_prev
            self._phi_prev = phi_new
            if is_terminal and self._exact_free_terminal():
                reward += self.w_free
            return reward
        # composite modes: 'penalized' (default), 'combinatorial', 'legacy'
        return saito_reward(
            self.arr,
            target_n=self.target_n,
            prev_arr=prev_arr,
            w_comb=self.w_comb,
            w_alg=0.0 if mode == 'combinatorial' else self.w_alg,
            w_pencil=self.w_pencil,
            w_free=self.w_free,
            w_mult=self.w_mult,
            w_interest=self.w_interest,
            terminal_only_free_bonus=True,
            skip_exact_above=self.skip_exact_above,
            target_exponents=self.target_exponents,
            use_legacy=(mode == 'legacy'),
            # score-free arm: the exact terminal bonus stays, but the
            # algebraic-score-based partial bonus for n > skip_exact_above
            # must not leak loss signal into this arm
            terminal_alg_bonus=(mode != 'combinatorial'),
        )

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(self, action: int):
        """
        Add the candidate at index ``action`` to the arrangement.
        Returns: obs, reward, done, info
        """
        assert not self.done, "Episode already done"
        assert action < len(self.current_candidates), (
            f"Action {action} out of range ({len(self.current_candidates)} candidates)"
        )
        assert self.current_mask[action] > 0, f"Action {action} is masked"

        line = self.current_candidates[action]

        # Snapshot for per-step shaping (before adding line)
        prev_arr = self.arr.copy() if len(self.arr) >= 2 else None

        self.arr.add_line(line)
        self.step_count += 1

        # Mark in static pool if applicable
        if line.coords in self._pool_index:
            self.selected_pool[self._pool_index[line.coords]] = True

        is_pencil = self.arr.is_pencil()
        is_terminal = (self.step_count == self.target_n)
        self.done = is_terminal or is_pencil

        reward = self._compute_reward(prev_arr, is_pencil, is_terminal)

        info = {
            'n': len(self.arr),
            't2': self.arr.b2(),
            'is_pencil': is_pencil,
            'is_terminal': is_terminal,
            'candidate_exponents': self.arr.candidate_exponents(),
            'target_exponents': self.target_exponents,
        }

        if is_terminal and not is_pencil:
            exps = self.arr.candidate_exponents()
            if exps is not None:
                is_free, exponents = self.arr.is_free()
                info['is_free'] = is_free
                info['exponents'] = exponents
            else:
                info['is_free'] = False
                info['exponents'] = None

        self.episode_reward += reward

        # Regenerate candidates for next step (unless done)
        if not self.done:
            self._update_candidates()

        return self._obs(), reward, self.done, info

    # ── Render ────────────────────────────────────────────────────────────────

    def render(self):
        s = self.arr.summary()
        n_sing = len(self.current_candidates)
        print(f"n={s['n']} | b2={s['b2']} | exps={s['candidate_exponents']} | "
              f"pencil={s['is_pencil']} | mults={s['multiplicity_profile'][:5]} | "
              f"candidates={n_sing}")
        for i, line in enumerate(self.arr.lines):
            print(f"  L{i}: {line}")
