"""
swap_env.py

Gym-like fixed-cardinality swap environment for the learned-policy arm.

State: exactly n lines + target exponents (d1, d2).  Action: a single flat
index a = i_minus * max_candidates + j_plus decoding to the replacement
A' = (A \\ {L_{i_minus}}) ∪ {C_{j_plus}} over the current candidate list
(joint-flatten factorization — keeps the PPO plumbing single-Categorical).

Reward: potential shaping  gamma*Phi(A') - Phi(A)  with
Phi = 1 - saito_loss(A; d1, d2, profile='rl', cached), plus an exact-
certificate bonus R_EXACT that dominates any cumulative shaping (the bonus
is granted only by the exact symbolic certificate — never by the numerical
loss).  Episodes run for `episode_len` swaps (pure truncation); at the
final step gamma*Phi(terminal) is added so the missing bootstrap value is
approximated by the potential itself (documented approximation, adequate
for the smoke-scale comparison).

Observation dict matches FreeArrangementEnv's schema (same 17 scalars, so
checkpoints stay loadable) with `mask` of width max_n * max_candidates and
an extra `n_candidates` int.  Fresh arrays every step (no aliasing).
"""

import numpy as np

from arrangement import LineArrangement
from environment import extract_scalars, generate_candidate_lines, \
    _singularity_candidates
from swap_search import (double_pencil_seed, perturb_k_swaps, is_valid_state,
                         certify_state)
from novelty import canonical_lineset_key
from saito import saito_loss

R_EXACT = 50.0


class SwapArrangementEnv:
    def __init__(self, target_n=12, d1=None, d2=None, coord_range=3,
                 max_candidates=64, episode_len=None, k_perturb=2,
                 gamma_shaping=0.99, seed=0, seed_mode="perturbed",
                 certify_below=1e-3, max_n=None, tau=None, eta=1.0):
        # tau: optional FROZEN calibration constant (calibration.py).  When
        # set, the shaping potential is 1 - calibrated_loss(raw; tau); the
        # RAW loss is always logged alongside (info['raw_loss']).  tau is
        # chosen before training and never changed mid-run.
        self.tau = tau
        self.eta = eta
        # numerical-failure policy: a failed Gamma evaluation carries NO
        # Saito loss.  The action becomes a recorded no-op with the SEPARATE
        # named reward component `numerical_failure_penalty` (default 0);
        # after `max_consecutive_errors` consecutive failures the episode
        # aborts with episode_status='evaluator_error'.
        self.numerical_failure_penalty = 0.0
        self.max_consecutive_errors = 5
        self.numerical_error_count = 0
        self._consecutive_errors = 0
        self.target_n = target_n
        self.max_n = max_n or target_n
        self.d1 = d1 if d1 is not None else (target_n - 1) // 2
        self.d2 = d2 if d2 is not None else target_n - 1 - self.d1
        assert self.d1 + self.d2 == target_n - 1
        self.coord_range = coord_range
        self.max_candidates = max_candidates
        self.episode_len = episode_len or 2 * target_n
        self.k_perturb = k_perturb
        self.gamma_shaping = gamma_shaping
        self.seed_mode = seed_mode
        self.certify_below = certify_below
        self.pool = generate_candidate_lines(coord_range)
        self.rng = np.random.default_rng(seed)
        self.certified_keys = set()
        self.reset()

    # ── helpers ─────────────────────────────────────────────────────────────

    def _raw_loss(self, arr):
        """Raw penalized loss.  Raises GammaNumericalError on evaluation
        failure — the caller handles it structurally (never a loss)."""
        return saito_loss(arr, target_exponents=(self.d1, self.d2),
                          profile='rl', cached=True)

    def _phi(self, arr):
        s = self._raw_loss(arr)
        self._last_raw_loss = s
        if self.tau is not None:
            from calibration import freeness_potential
            return freeness_potential(s, self.tau)
        return 1.0 - s

    def _build_candidates(self):
        """Candidate L+ list for the CURRENT arrangement: top singularity
        lines + random pool sample, excluding lines already present."""
        existing = {l.coords for l in self.arr.lines}
        cands = []
        for _, line in _singularity_candidates(self.arr)[:self.max_candidates]:
            if line.coords not in existing:
                cands.append(line)
        n_fill = self.max_candidates - len(cands)
        if n_fill > 0:
            idx = self.rng.choice(len(self.pool),
                                  size=min(2 * n_fill, len(self.pool)),
                                  replace=False)
            for j in idx:
                line = self.pool[j]
                if line.coords not in existing and \
                        all(line.coords != c.coords for c in cands):
                    cands.append(line)
                    if len(cands) >= self.max_candidates:
                        break
        self.candidates = cands[:self.max_candidates]
        # cheap joint mask: real removal slots x real candidates (duplicates
        # were excluded above); full validity (essentiality, m_max cap) is
        # enforced at step time — an invalid swap is a penalized no-op.
        n = len(self.arr)
        mask = np.zeros((self.max_n, self.max_candidates), dtype=np.float32)
        mask[:n, :len(self.candidates)] = 1.0
        self.joint_mask = mask.reshape(-1)

    def _obs(self):
        sel = np.zeros((self.max_n, 3), dtype=np.float32)
        for i, line in enumerate(self.arr.lines):
            sel[i] = line.to_float()
        cand = np.zeros((self.max_candidates, 3), dtype=np.float32)
        for j, line in enumerate(self.candidates):
            cand[j] = line.to_float()
        scalars = extract_scalars(self.arr, self.target_n,
                                  target_exponents=(self.d1, self.d2),
                                  score_mode='penalized')
        return {
            'selected_coords': sel,
            'candidate_coords': cand,
            'scalars': scalars.astype(np.float32),
            'n_selected': len(self.arr),
            'mask': self.joint_mask.copy(),
            'n_candidates': len(self.candidates),
        }

    # ── API ─────────────────────────────────────────────────────────────────

    def reset(self, target_n=None, random_start=None, target_exponents=None):
        if target_exponents is not None:
            self.d1, self.d2 = target_exponents
        base = double_pencil_seed(self.target_n, self.d1, self.d2)
        if base.coefficient_field() is not None:
            raise NotImplementedError(
                "the RL arm is QQ-only: float32 observations cannot carry "
                "quadratic-field coordinates; use the greedy/anneal/"
                "MAP-Elites engines (--field-d) for K campaigns")
        if self.seed_mode == "perturbed":
            self.arr = perturb_k_swaps(base, self.k_perturb, self.rng,
                                       coord_range=self.coord_range)
        elif self.seed_mode == "supersolvable":
            self.arr = base
        else:
            from swap_search import random_valid_seed
            self.arr = random_valid_seed(self.target_n, self.rng,
                                         coord_range=self.coord_range)
        self.t = 0
        self.done = False
        from penalized_saito import GammaNumericalError
        for _ in range(4):
            try:
                self._phi_prev = self._phi(self.arr)
                break
            except GammaNumericalError:
                self.numerical_error_count += 1
                self.arr = perturb_k_swaps(self.arr, 1, self.rng,
                                           coord_range=self.coord_range)
        else:
            raise RuntimeError("swap env reset: seed states repeatedly "
                               "failed numerical evaluation")
        self.episode_reward = 0.0
        self._build_candidates()
        return self._obs()

    def action_mask(self):
        return self.joint_mask.copy()

    def step(self, action):
        assert not self.done
        i, j = divmod(int(action), self.max_candidates)
        assert self.joint_mask[action] > 0, "masked action"
        line = self.candidates[j]
        new_lines = [l for k, l in enumerate(self.arr.lines) if k != i] \
            + [line]
        trial = LineArrangement(new_lines)
        self.t += 1
        self.done = self.t >= self.episode_len
        if not is_valid_state(trial, len(self.arr),
                              nontrivial=(self.d1 >= 2)):
            # invalid swap: penalized no-op (state unchanged)
            obs = self._obs()
            info = {'n': len(self.arr), 't2': self.arr.b2(),
                    'is_pencil': False, 'is_terminal': self.done,
                    'candidate_exponents': self.arr.candidate_exponents(),
                    'target_exponents': (self.d1, self.d2),
                    'invalid_swap': True,
                    'raw_loss': getattr(self, '_last_raw_loss', 1.0),
                    'tau': self.tau,
                    'best_loss': getattr(self, '_last_raw_loss', 1.0)}
            reward = -0.5
            if self.done:
                # terminal convention Phi(terminal) = 0: close the
                # telescope for the unchanged state
                reward += self.eta * (0.0 - self._phi_prev)
            self.episode_reward += reward
            return obs, reward, self.done, info
        from penalized_saito import GammaNumericalError
        try:
            phi_new_candidate = self._phi(trial)
        except GammaNumericalError:
            # numerical failure: NOT a Saito loss.  Recorded no-op with the
            # separate named penalty component; abort after too many.
            self.numerical_error_count += 1
            self._consecutive_errors += 1
            if self._consecutive_errors >= self.max_consecutive_errors:
                self.done = True
            obs = self._obs()
            info = {'n': len(self.arr), 't2': self.arr.b2(),
                    'is_pencil': False, 'is_terminal': self.done,
                    'candidate_exponents': self.arr.candidate_exponents(),
                    'target_exponents': (self.d1, self.d2),
                    'numerical_error': True, 'raw_loss': None,
                    'calibrated_potential': None, 'tau': self.tau,
                    'episode_status': ('evaluator_error' if self.done
                                       else 'numerical_error_deferred'),
                    'numerical_failure_penalty':
                        -self.numerical_failure_penalty}
            reward = -self.numerical_failure_penalty
            self.episode_reward += reward
            return obs, reward, self.done, info
        self._consecutive_errors = 0
        self.arr = trial
        phi_new = phi_new_candidate

        # Potential-based shaping with the episodic absorbing convention
        # Phi(terminal) := 0 (Ng et al.): the final transition uses 0 for
        # the next-state potential, so shaping telescopes over the episode
        # and policy invariance holds for the episodic task.  phi_new is
        # still computed and logged; it just does not enter the terminal
        # shaping term.
        phi_next_for_shaping = 0.0 if self.done else phi_new
        reward = self.eta * (self.gamma_shaping * phi_next_for_shaping
                             - self._phi_prev)
        self._phi_prev = phi_new

        info = {'n': len(self.arr), 't2': self.arr.b2(),
                'is_pencil': False, 'is_terminal': self.done,
                'candidate_exponents': self.arr.candidate_exponents(),
                'target_exponents': (self.d1, self.d2),
                'raw_loss': self._last_raw_loss,
                'calibrated_potential': phi_new, 'tau': self.tau,
                'best_loss': self._last_raw_loss}
        # exact certification gate (never granted by the numerical loss)
        if self._last_raw_loss < self.certify_below:
            key = canonical_lineset_key(self.arr)
            if key not in self.certified_keys:
                self.certified_keys.add(key)
                cert = certify_state(self.arr, self.d1, self.d2)
                if cert is not None:
                    reward += R_EXACT
                    info['is_free'] = True
                    info['exponents'] = (1, self.d1, self.d2)
                    info['certificate'] = True
        self.episode_reward += reward
        if not self.done:
            self._build_candidates()
        return self._obs(), reward, self.done, info
