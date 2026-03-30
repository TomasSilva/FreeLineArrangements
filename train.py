"""
train.py

PPO training loop for discovering free line arrangements in CP^2.

Usage:
    # Fixed n
    python train.py --n 6 --steps 200000

    # Curriculum: train across a range of n values
    python train.py --n-min 6 --n-max 15 --total-steps 1000000

PPO hyperparameters follow standard recommendations.
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.optim as optim
from collections import deque
from arrangement import all_exponent_types
from environment import FreeArrangementEnv
from model import TransformerActorCritic
from discoveries import log_discovery
from vec_env import SubprocVecEnv


# ─────────────────────────────────────────────────────────────────────────────
# Rollout buffer
# ─────────────────────────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self):
        self.obs = []       # list of dicts
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.masks = []

    def store(self, obs, action, log_prob, reward, value, done, mask):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.masks.append(mask)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


def _collate_obs(obs_list):
    """Stack a list of obs dicts into a batched dict of tensors."""
    return {
        'selected_coords': torch.FloatTensor(
            np.stack([o['selected_coords'] for o in obs_list])
        ),
        'candidate_coords': torch.FloatTensor(
            np.stack([o['candidate_coords'] for o in obs_list])
        ),
        'scalars': torch.FloatTensor(
            np.stack([o['scalars'] for o in obs_list])
        ),
        'n_selected': torch.tensor(
            [o['n_selected'] for o in obs_list], dtype=torch.long
        ),
    }


def compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95):
    """Generalized Advantage Estimation."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        next_val = last_value if t == n - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns


# ─────────────────────────────────────────────────────────────────────────────
# PPO update
# ─────────────────────────────────────────────────────────────────────────────

def ppo_update(
    model: TransformerActorCritic,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    last_value: float,
    n_epochs: int = 4,
    batch_size: int = 64,
    clip_eps: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
):
    n = len(buffer)
    rewards = np.array(buffer.rewards, dtype=np.float32)
    values_np = np.array([v.item() for v in buffer.values], dtype=np.float32)
    dones_np = np.array(buffer.dones, dtype=np.float32)

    advantages, returns = compute_gae(rewards, values_np, dones_np, last_value, gamma, gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Batch all observations into tensors
    obs_t = _collate_obs(buffer.obs)
    acts_t = torch.LongTensor(buffer.actions)
    old_lp_t = torch.stack(buffer.log_probs).detach()
    ret_t = torch.FloatTensor(returns)
    adv_t = torch.FloatTensor(advantages)
    masks_t = torch.FloatTensor(np.array(buffer.masks))

    total_loss = 0.0
    for epoch in range(n_epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            mb = idx[start: start + batch_size]
            mb_obs = {k: v[mb] for k, v in obs_t.items()}
            mb_acts = acts_t[mb]
            mb_old_lp = old_lp_t[mb]
            mb_ret = ret_t[mb]
            mb_adv = adv_t[mb]
            mb_masks = masks_t[mb]

            new_lp, new_val, entropy = model.evaluate(mb_obs, mb_acts, mb_masks)

            ratio = torch.exp(new_lp - mb_old_lp)
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * ratio.clamp(1 - clip_eps, 1 + clip_eps)
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()

            value_loss = 0.5 * (new_val - mb_ret).pow(2).mean()
            entropy_loss = -entropy.mean()

            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

    return total_loss


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive curriculum sampling
# ─────────────────────────────────────────────────────────────────────────────

def _sample_curriculum_n(n_min, n_max, stats, rng, threshold=0.15):
    """
    Sample target_n weighted toward the learning frontier.

    Prioritizes the smallest n where:
      - The previous n (n-1) has success rate above threshold, AND
      - This n still has low success rate (room to improve)

    Falls back to uniform sampling if no frontier is found.
    """
    success = stats['curriculum_success']
    episodes = stats['curriculum_episodes']

    # Need some data first — bootstrap with uniform
    total_eps = sum(episodes.values())
    if total_eps < (n_max - n_min + 1) * 5:
        return int(rng.integers(n_min, n_max + 1))

    weights = []
    for n in range(n_min, n_max + 1):
        s = success.get(n, 0.0)
        # Priority: high for low-success n whose predecessor is mastered
        if n == n_min:
            prev_ok = True
        else:
            prev_ok = success.get(n - 1, 0.0) >= threshold
        if prev_ok:
            w = max(0.05, 1.0 - s)  # focus on what's not yet solved
        else:
            w = 0.05  # small chance even if predecessor not mastered
        weights.append(w)

    weights = np.array(weights)
    weights /= weights.sum()
    return int(rng.choice(range(n_min, n_max + 1), p=weights))


def _all_curriculum_triples(n_min, n_max):
    """Enumerate all (n, d2, d3) triples for the curriculum."""
    triples = []
    for n in range(n_min, n_max + 1):
        for d2, d3 in all_exponent_types(n):
            triples.append((n, d2, d3))
    return triples


def _sample_curriculum_triple(triples, stats, rng, threshold=0.15):
    """
    Sample (target_n, d2, d3) weighted toward under-explored triples.

    Prioritizes triples with low success rate. Triples with zero discoveries
    get an additional boost.
    """
    triple_success = stats['triple_success']
    triple_episodes = stats['triple_episodes']

    # Bootstrap: uniform sampling until we have some data
    total_eps = sum(triple_episodes.values())
    if total_eps < len(triples) * 3:
        idx = int(rng.integers(0, len(triples)))
        return triples[idx]

    weights = []
    for triple in triples:
        s = triple_success.get(triple, 0.0)
        eps = triple_episodes.get(triple, 0)
        w = max(0.05, 1.0 - s)
        # Extra boost for zero-discovery triples
        if eps > 5 and s < 0.01:
            w *= 2.0
        # Also boost triples that have had very few episodes
        if eps < 10:
            w *= 1.5
        weights.append(w)

    weights = np.array(weights)
    weights /= weights.sum()
    idx = int(rng.choice(len(triples), p=weights))
    return triples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Collect rollout
# ─────────────────────────────────────────────────────────────────────────────

def collect_rollout(
    env: FreeArrangementEnv,
    model: TransformerActorCritic,
    buffer: RolloutBuffer,
    n_steps: int,
    stats: dict,
    n_min: int = None,
    n_max: int = None,
    rng: np.random.Generator = None,
):
    """
    Collect n_steps transitions across episodes.

    If n_min/n_max are provided, samples a new (target_n, d2, d3) triple
    at each episode reset for exponent-targeted curriculum learning.
    """
    n_min = n_min or env.target_n
    n_max = n_max or env.target_n
    rng = rng or np.random.default_rng(0)

    use_curriculum = (n_min != n_max)
    triples = stats.get('_triples')  # precomputed list of (n, d2, d3)

    def _sample_next():
        if use_curriculum and triples:
            n, d2, d3 = _sample_curriculum_triple(triples, stats, rng)
            return n, (d2, d3)
        elif use_curriculum:
            return _sample_curriculum_n(n_min, n_max, stats, rng), None
        else:
            return n_min, None

    target_n, target_exponents = _sample_next()
    obs = env.reset(target_n=target_n, random_start=True,
                    target_exponents=target_exponents)
    ep_reward = 0.0

    for _ in range(n_steps):
        mask = obs['mask']
        action, log_prob, value = model.act(obs)

        next_obs, reward, done, info = env.step(action)
        buffer.store(obs, action, log_prob, reward, value, done, mask)
        obs = next_obs
        ep_reward += reward

        if done:
            stats['ep_rewards'].append(ep_reward)
            stats['n_episodes'] += 1
            stats['n_episodes_since_log'] += 1

            # Update curriculum stats (per-n for backward compat)
            has_cand_exps = info.get('candidate_exponents') is not None
            ema_alpha = 0.05
            prev = stats['curriculum_success'].get(target_n, 0.0)
            stats['curriculum_success'][target_n] = (
                prev * (1 - ema_alpha) + float(has_cand_exps) * ema_alpha
            )
            stats['curriculum_episodes'][target_n] = (
                stats['curriculum_episodes'].get(target_n, 0) + 1
            )

            # Update per-triple curriculum stats
            if target_exponents is not None:
                triple_key = (target_n,) + tuple(target_exponents)
                prev_t = stats['triple_success'].get(triple_key, 0.0)
                stats['triple_success'][triple_key] = (
                    prev_t * (1 - ema_alpha) + float(has_cand_exps) * ema_alpha
                )
                stats['triple_episodes'][triple_key] = (
                    stats['triple_episodes'].get(triple_key, 0) + 1
                )

            # Track free arrangements found (exact check for small n)
            if info.get('is_free') and not info.get('is_pencil'):
                stats['free_found'] += 1
                stats['free_found_since_log'] += 1
                n_val = info.get('n', env.target_n)
                stats['free_by_n'][n_val] = stats['free_by_n'].get(n_val, 0) + 1
                stats['last_free_arr'] = [str(l) for l in env.arr.lines]
                stats['last_free_exps'] = info.get('exponents')
                arr_summary = env.arr.summary()
                log_discovery(
                    lines=env.arr.lines,
                    exponents=info.get('exponents'),
                    b2=arr_summary['b2'],
                    n=n_val,
                    max_mult=env.arr.max_multiplicity(),
                    mult_profile=sorted(arr_summary['multiplicity_profile'], reverse=True),
                    n_pts=env.arr.n_intersection_points(),
                    source="train",
                    target_exponents=target_exponents,
                )
            ep_reward = 0.0
            target_n, target_exponents = _sample_next()
            obs = env.reset(target_n=target_n, random_start=True,
                            target_exponents=target_exponents)

    # Bootstrap last value for GAE
    with torch.no_grad():
        _, last_val = model.forward(
            torch.FloatTensor(obs['selected_coords']).unsqueeze(0),
            torch.FloatTensor(obs['candidate_coords']).unsqueeze(0),
            torch.FloatTensor(obs['scalars']).unsqueeze(0),
            torch.FloatTensor(obs['mask']).unsqueeze(0),
            torch.tensor([obs['n_selected']], dtype=torch.long),
        )
    last_value = last_val.item()

    recent = list(stats['ep_rewards'])[-50:]
    if recent:
        stats['mean_reward'] = float(np.mean(recent))

    return last_value


# ─────────────────────────────────────────────────────────────────────────────
# Vectorized rollout collection
# ─────────────────────────────────────────────────────────────────────────────

class VecRolloutBuffer:
    """Rollout buffer for multiple environments, stored as (n_steps, n_envs)."""

    def __init__(self, n_envs):
        self.n_envs = n_envs
        self.obs = []        # list of lists: obs[step][env]
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.masks = []

    def store_step(self, obs_list, actions, log_probs, rewards, values, dones, masks):
        self.obs.append(list(obs_list))
        self.actions.append(list(actions))
        self.log_probs.append(log_probs)     # tensor (n_envs,)
        self.rewards.append(list(rewards))
        self.values.append(values)            # tensor (n_envs,)
        self.dones.append(list(dones))
        self.masks.append(list(masks))

    def compute_gae_and_flatten(self, last_values, gamma=0.99, gae_lambda=0.95):
        """Compute per-env GAE then flatten all transitions for PPO."""
        T = len(self.obs)
        N = self.n_envs

        # Convert to arrays for GAE
        rewards_arr = np.array(self.rewards, dtype=np.float32)   # (T, N)
        values_arr = np.stack([v.numpy() for v in self.values])  # (T, N)
        dones_arr = np.array(self.dones, dtype=np.float32)       # (T, N)
        last_vals = np.array(last_values, dtype=np.float32)      # (N,)

        advantages = np.zeros((T, N), dtype=np.float32)
        for i in range(N):
            adv, _ = compute_gae(
                rewards_arr[:, i].tolist(),
                values_arr[:, i].tolist(),
                dones_arr[:, i].tolist(),
                last_vals[i],
                gamma, gae_lambda,
            )
            advantages[:, i] = adv

        returns = advantages + values_arr

        # Flatten (T, N) -> (T*N,)
        flat_obs = [self.obs[t][i] for t in range(T) for i in range(N)]
        flat_actions = [self.actions[t][i] for t in range(T) for i in range(N)]
        flat_log_probs = torch.cat([self.log_probs[t] for t in range(T)])
        flat_values = torch.cat([self.values[t] for t in range(T)])
        flat_masks = [self.masks[t][i] for t in range(T) for i in range(N)]
        flat_advantages = advantages.ravel()
        flat_returns = returns.ravel()

        return {
            'obs': flat_obs,
            'actions': flat_actions,
            'log_probs': flat_log_probs,
            'values': flat_values,
            'masks': flat_masks,
            'advantages': flat_advantages,
            'returns': flat_returns,
        }

    def clear(self):
        self.__init__(self.n_envs)

    def __len__(self):
        return len(self.obs) * self.n_envs


def _sample_next_triple(triples, stats, rng, n_min, n_max):
    """Sample next (target_n, target_exponents) for an env reset."""
    if triples:
        n, d2, d3 = _sample_curriculum_triple(triples, stats, rng)
        return n, (d2, d3)
    elif n_min != n_max:
        return _sample_curriculum_n(n_min, n_max, stats, rng), None
    else:
        return n_min, None


def _update_episode_stats(stats, info, ep_reward, target_n, target_exponents):
    """Update stats when an episode ends. Used by both single and vec rollouts."""
    stats['ep_rewards'].append(ep_reward)
    stats['n_episodes'] += 1
    stats['n_episodes_since_log'] += 1

    has_cand_exps = info.get('candidate_exponents') is not None
    ema_alpha = 0.05
    prev = stats['curriculum_success'].get(target_n, 0.0)
    stats['curriculum_success'][target_n] = (
        prev * (1 - ema_alpha) + float(has_cand_exps) * ema_alpha
    )
    stats['curriculum_episodes'][target_n] = (
        stats['curriculum_episodes'].get(target_n, 0) + 1
    )

    if target_exponents is not None:
        triple_key = (target_n,) + tuple(target_exponents)
        prev_t = stats['triple_success'].get(triple_key, 0.0)
        stats['triple_success'][triple_key] = (
            prev_t * (1 - ema_alpha) + float(has_cand_exps) * ema_alpha
        )
        stats['triple_episodes'][triple_key] = (
            stats['triple_episodes'].get(triple_key, 0) + 1
        )

    if info.get('is_free') and not info.get('is_pencil'):
        stats['free_found'] += 1
        stats['free_found_since_log'] += 1
        n_val = info.get('n', target_n)
        stats['free_by_n'][n_val] = stats['free_by_n'].get(n_val, 0) + 1
        stats['last_free_arr'] = info.get('arr_lines', [])
        stats['last_free_exps'] = info.get('exponents')
        # Log discovery — use data from info (worker enriched it)
        log_discovery(
            lines=info.get('arr_lines', []),
            exponents=info.get('exponents'),
            b2=info.get('arr_summary', {}).get('b2'),
            n=n_val,
            max_mult=info.get('arr_max_mult'),
            mult_profile=sorted(
                info.get('arr_summary', {}).get('multiplicity_profile', []),
                reverse=True,
            ),
            n_pts=info.get('arr_n_pts'),
            source="train",
            target_exponents=target_exponents,
        )


def collect_rollout_vec(
    vec_env: SubprocVecEnv,
    model: TransformerActorCritic,
    buffer: VecRolloutBuffer,
    n_steps: int,
    stats: dict,
    n_min: int,
    n_max: int,
    rng: np.random.Generator,
):
    """Collect n_steps transitions from all envs in parallel."""
    n_envs = vec_env.n_envs
    triples = stats.get('_triples')

    # Current (target_n, target_exponents) per env
    env_triples = [
        _sample_next_triple(triples, stats, rng, n_min, n_max)
        for _ in range(n_envs)
    ]

    # Initial reset
    reset_kwargs = [
        {'target_n': tn, 'random_start': True, 'target_exponents': te}
        for tn, te in env_triples
    ]
    obs_list = vec_env.reset_all(reset_kwargs)
    ep_rewards = [0.0] * n_envs

    for step in range(n_steps):
        masks = [obs['mask'] for obs in obs_list]
        actions, log_probs, values = model.act_batch(obs_list)

        # Pre-sample next triples for auto-reset
        next_triples = [
            _sample_next_triple(triples, stats, rng, n_min, n_max)
            for _ in range(n_envs)
        ]
        next_reset_args = [(tn, te) for tn, te in next_triples]

        next_obs_list, rewards, dones, infos = vec_env.step(actions,
                                                            next_reset_args)

        buffer.store_step(obs_list, actions, log_probs, rewards, values,
                          dones, masks)

        for i in range(n_envs):
            ep_rewards[i] += rewards[i]
            if dones[i]:
                tn, te = env_triples[i]
                _update_episode_stats(stats, infos[i], ep_rewards[i], tn, te)
                ep_rewards[i] = 0.0
                env_triples[i] = next_triples[i]

        obs_list = next_obs_list

    # Bootstrap values for all envs
    _, _, last_values = model.act_batch(obs_list)

    recent = list(stats['ep_rewards'])[-50:]
    if recent:
        stats['mean_reward'] = float(np.mean(recent))

    return last_values.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# PPO update (from pre-computed flat data)
# ─────────────────────────────────────────────────────────────────────────────

def ppo_update_flat(
    model, optimizer, flat_data,
    n_epochs=4, batch_size=64, clip_eps=0.2,
    vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
):
    """PPO update from pre-flattened rollout data (used with VecRolloutBuffer)."""
    obs_list = flat_data['obs']
    actions_list = flat_data['actions']
    old_log_probs = flat_data['log_probs']
    advantages = torch.FloatTensor(flat_data['advantages'])
    returns = torch.FloatTensor(flat_data['returns'])
    all_masks = flat_data['masks']

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n_samples = len(obs_list)
    total_loss = 0.0

    for epoch in range(n_epochs):
        indices = np.random.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            mb_idx = indices[start:end]

            mb_obs = [obs_list[i] for i in mb_idx]
            mb_actions = torch.tensor([actions_list[i] for i in mb_idx], dtype=torch.long)
            mb_old_lp = old_log_probs[mb_idx]
            mb_adv = advantages[mb_idx]
            mb_ret = returns[mb_idx]
            mb_masks = torch.FloatTensor(
                np.stack([all_masks[i] for i in mb_idx])
            )

            batch_obs = _collate_obs(mb_obs)
            new_lp, new_val, entropy = model.evaluate(batch_obs, mb_actions, mb_masks)

            ratio = (new_lp - mb_old_lp).exp()
            pg_loss1 = -mb_adv * ratio
            pg_loss2 = -mb_adv * ratio.clamp(1 - clip_eps, 1 + clip_eps)
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()

            value_loss = 0.5 * (new_val - mb_ret).pow(2).mean()
            entropy_loss = -entropy.mean()

            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_loss += loss.item()

    return total_loss


# ─────────────────────────────────────────────────────────────────────────────
# Main training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    n_min = getattr(args, 'n_min', args.n)
    n_max = getattr(args, 'n_max', args.n)
    n_envs = getattr(args, 'n_envs', 1)

    singularity_aware = getattr(args, 'singularity_aware', False)
    max_candidates = getattr(args, 'max_candidates', 200)

    # Auto-scale coord_range for large n if user left default
    if args.coord_range == 3 and n_max >= 12:
        args.coord_range = max(3, n_max // 3 + 1)
        print(f"Auto-scaled coord_range to {args.coord_range} for n_max={n_max}")

    env_kwargs = dict(
        target_n=n_min,
        coord_range=args.coord_range,
        max_n=n_max,
        max_candidates=max_candidates,
        singularity_aware=singularity_aware,
        bootstrap_steps=getattr(args, 'bootstrap_steps', 3),
        pool_sample_frac=getattr(args, 'pool_sample_frac', 0.3),
        w_comb=args.w_comb,
        w_alg=args.w_alg,
        w_pencil=args.w_pencil,
        w_free=args.w_free,
        w_mult=args.w_mult,
        w_interest=getattr(args, 'w_interest', 1.0),
        skip_exact_above=getattr(args, 'skip_exact_above', 12),
    )

    use_vec = n_envs > 1
    vec_env = None
    env = None

    if use_vec:
        print(f"Building {n_envs} parallel environments: n_min={n_min}, n_max={n_max}, "
              f"coord_range={args.coord_range}, singularity_aware={singularity_aware}")
        env_kwargs_list = [{**env_kwargs, 'seed': 42 + i} for i in range(n_envs)]
        vec_env = SubprocVecEnv(env_kwargs_list)
        print(f"Vectorized: {n_envs} workers, steps_per_update={args.n_steps * n_envs}")
    else:
        print(f"Building environment: n_min={n_min}, n_max={n_max}, "
              f"coord_range={args.coord_range}, singularity_aware={singularity_aware}")
        env = FreeArrangementEnv(**env_kwargs)
        print(f"Pool size: {env.pool_size} candidate lines (coord_range={args.coord_range})")
        print(f"max_n (padding): {env.max_n}")

    model = TransformerActorCritic(
        max_n=n_max,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        scalar_dim=17,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f"Resumed from {args.resume}")

    buffer = VecRolloutBuffer(n_envs) if use_vec else RolloutBuffer()
    rng = np.random.default_rng(42)

    # Precompute all (n, d2, d3) curriculum triples
    triples = _all_curriculum_triples(n_min, n_max)
    print(f"Curriculum triples: {len(triples)} (n, d2, d3) targets")

    stats = {
        'free_found': 0,
        'n_episodes': 0,
        'mean_reward': 0.0,
        'ep_rewards': deque(maxlen=200),
        'last_free_arr': None,
        'last_free_exps': None,
        'free_by_n': {},
        'free_found_since_log': 0,
        'n_episodes_since_log': 0,
        'curriculum_success': {},
        'curriculum_episodes': {},
        'triple_success': {},
        'triple_episodes': {},
        '_triples': triples,
    }

    total_steps = 0
    update = 0
    t0 = time.time()
    steps_per_update = args.n_steps * n_envs if use_vec else args.n_steps

    print(f"\nStarting PPO training ({'vec x' + str(n_envs) if use_vec else 'single env'})...")
    print(f"{'Update':>8} | {'Steps':>10} | {'MeanRew':>10} | {'Free':>6} | {'Rate':>8} | {'Time':>8}")
    print("-" * 67)

    try:
        while total_steps < args.total_steps:
            model.train()

            if use_vec:
                last_values = collect_rollout_vec(
                    vec_env, model, buffer, args.n_steps, stats,
                    n_min=n_min, n_max=n_max, rng=rng,
                )
                total_steps += steps_per_update
                flat_data = buffer.compute_gae_and_flatten(
                    last_values, args.gamma, args.gae_lambda,
                )
                loss = ppo_update_flat(
                    model, optimizer, flat_data,
                    n_epochs=args.n_epochs,
                    batch_size=args.batch_size,
                    clip_eps=args.clip_eps,
                    vf_coef=args.vf_coef,
                    ent_coef=args.ent_coef,
                    max_grad_norm=args.max_grad_norm,
                )
            else:
                last_val = collect_rollout(
                    env, model, buffer, args.n_steps, stats,
                    n_min=n_min, n_max=n_max, rng=rng,
                )
                total_steps += steps_per_update
                loss = ppo_update(
                    model, optimizer, buffer, last_val,
                    n_epochs=args.n_epochs,
                    batch_size=args.batch_size,
                    clip_eps=args.clip_eps,
                    vf_coef=args.vf_coef,
                    ent_coef=args.ent_coef,
                    max_grad_norm=args.max_grad_norm,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                )

            buffer.clear()
            update += 1

            if update % args.log_every == 0:
                elapsed = time.time() - t0
                n_eps = stats['n_episodes_since_log']
                rate_str = (
                    f"{stats['free_found_since_log'] / n_eps:.4f}"
                    if n_eps > 0 else "   n/a"
                )
                print(f"{update:>8} | {total_steps:>10} | {stats['mean_reward']:>10.3f} | "
                      f"{stats['free_found']:>6} | {rate_str:>8} | {elapsed:>7.1f}s")
                print(f"  Free by n: {dict(sorted(stats['free_by_n'].items()))}")
                if stats['triple_episodes']:
                    active = sum(1 for v in stats['triple_episodes'].values() if v > 0)
                    explored = sum(1 for v in stats['triple_success'].values() if v > 0.01)
                    print(f"  Triples: {active}/{len(triples)} active, {explored} with signal")
                if stats['last_free_arr']:
                    print(f"  Last free: exps={stats['last_free_exps']}")
                    for lstr in stats['last_free_arr']:
                        print(f"    {lstr}")
                stats['free_found_since_log'] = 0
                stats['n_episodes_since_log'] = 0

            if args.save_every and update % args.save_every == 0:
                ckpt_path = f"checkpoint_step{total_steps}.pt"
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

    finally:
        if vec_env is not None:
            vec_env.close()

    print("\nTraining complete.")
    print(f"Total free arrangements found: {stats['free_found']}")
    if stats['free_by_n']:
        print(f"  By n: {dict(sorted(stats['free_by_n'].items()))}")

    if args.save:
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, args.save)
        print(f"Model saved to {args.save}")

    return model, env, stats


# ─────────────────────────────────────────────────────────────────────────────
# Greedy evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_greedy(
    model: TransformerActorCritic,
    env: FreeArrangementEnv,
    n_episodes: int = 100,
    target_n: int = None,
):
    """Run greedy evaluation and collect found free arrangements."""
    found = []
    model.eval()
    for _ in range(n_episodes):
        obs = env.reset(target_n=target_n)
        done = False
        while not done:
            action, _, _ = model.act(obs, deterministic=True)
            obs, _, done, info = env.step(action)
        if info.get('is_free') and not info.get('is_pencil'):
            found.append({
                'lines': [str(l) for l in env.arr.lines],
                'exponents': info.get('exponents'),
                't2': info.get('t2'),
                'n': len(env.arr),
            })
    return found


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def get_parser():
    p = argparse.ArgumentParser(description="RL for free line arrangements in CP^2")
    p.add_argument("--n", type=int, default=6, help="Target number of lines (fixed n; overridden by --n-min/--n-max)")
    p.add_argument("--n-min", type=int, default=None, help="Min n for curriculum (defaults to --n)")
    p.add_argument("--n-max", type=int, default=None, help="Max n for curriculum (defaults to --n)")
    p.add_argument("--coord-range", type=int, default=3, help="Coordinate range for candidate lines")
    p.add_argument("--total-steps", type=int, default=500_000, help="Total env steps")
    p.add_argument("--n-steps", type=int, default=2048, help="Steps per rollout")
    p.add_argument("--n-epochs", type=int, default=4, help="PPO epochs per update")
    p.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.02)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--d-model", type=int, default=128, help="Transformer model dimension")
    p.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    p.add_argument("--n-layers", type=int, default=3, help="Number of Transformer encoder layers")
    p.add_argument("--w-comb", type=float, default=0.3)
    p.add_argument("--w-alg", type=float, default=0.5)
    p.add_argument("--w-pencil", type=float, default=5.0)
    p.add_argument("--w-free", type=float, default=10.0)
    p.add_argument("--w-mult", type=float, default=2.0)
    p.add_argument("--w-interest", type=float, default=1.0)
    p.add_argument("--singularity-aware", action='store_true',
                   help="Use singularity-driven dynamic candidate generation")
    p.add_argument("--max-candidates", type=int, default=200,
                   help="Max candidates per step (singularity-aware mode)")
    p.add_argument("--bootstrap-steps", type=int, default=3,
                   help="Initial steps drawn from static pool (singularity-aware mode)")
    p.add_argument("--pool-sample-frac", type=float, default=0.3,
                   help="Fraction of candidates from static pool (singularity-aware mode)")
    p.add_argument("--skip-exact-above", type=int, default=12,
                   help="Skip exact Saito check during training for n > this value")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=1,
                   help="Number of parallel environments (1 = single-threaded)")
    p.add_argument("--save", type=str, default="model_final.pt")
    p.add_argument("--resume", type=str, default="")
    return p


if __name__ == "__main__":
    args = get_parser().parse_args()
    # Resolve n_min / n_max from --n if not set
    if args.n_min is None:
        args.n_min = args.n
    if args.n_max is None:
        args.n_max = args.n

    model, env, stats = train(args)

    print("\nRunning greedy evaluation (100 episodes)...")
    found = evaluate_greedy(model, env, n_episodes=100)
    print(f"Found {len(found)} free arrangements in greedy evaluation")
    for i, f in enumerate(found[:5]):
        print(f"\nArrangement {i+1}: exps={f['exponents']}, t2={f['t2']}, n={f['n']}")
        for lstr in f['lines']:
            print(f"  {lstr}")
