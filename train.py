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
from environment import FreeArrangementEnv
from model import TransformerActorCritic
from discoveries import log_discovery


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

    If n_min/n_max are provided, samples a new target_n at each episode reset
    for curriculum learning.
    """
    n_min = n_min or env.target_n
    n_max = n_max or env.target_n
    rng = rng or np.random.default_rng(0)

    use_curriculum = (n_min != n_max)
    target_n = (
        _sample_curriculum_n(n_min, n_max, stats, rng) if use_curriculum
        else n_min
    )
    obs = env.reset(target_n=target_n, random_start=True)
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

            # Update curriculum stats
            has_cand_exps = info.get('candidate_exponents') is not None
            ema_alpha = 0.05
            prev = stats['curriculum_success'].get(target_n, 0.0)
            stats['curriculum_success'][target_n] = (
                prev * (1 - ema_alpha) + float(has_cand_exps) * ema_alpha
            )
            stats['curriculum_episodes'][target_n] = (
                stats['curriculum_episodes'].get(target_n, 0) + 1
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
                )
            ep_reward = 0.0
            target_n = (
                _sample_curriculum_n(n_min, n_max, stats, rng) if use_curriculum
                else n_min
            )
            obs = env.reset(target_n=target_n, random_start=True)

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
# Main training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    n_min = getattr(args, 'n_min', args.n)
    n_max = getattr(args, 'n_max', args.n)

    singularity_aware = getattr(args, 'singularity_aware', False)
    max_candidates = getattr(args, 'max_candidates', 200)

    # Auto-scale coord_range for large n if user left default
    if args.coord_range == 3 and n_max >= 12:
        args.coord_range = max(3, n_max // 3 + 1)
        print(f"Auto-scaled coord_range to {args.coord_range} for n_max={n_max}")

    print(f"Building environment: n_min={n_min}, n_max={n_max}, coord_range={args.coord_range}"
          f", singularity_aware={singularity_aware}")
    env = FreeArrangementEnv(
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
    print(f"Pool size: {env.pool_size} candidate lines (coord_range={args.coord_range})")
    print(f"max_n (padding): {env.max_n}")

    model = TransformerActorCritic(
        max_n=n_max,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        scalar_dim=14,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(f"Resumed from {args.resume}")

    buffer = RolloutBuffer()
    rng = np.random.default_rng(42)
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
        # Adaptive curriculum: EMA of "has candidate exponents" per n
        'curriculum_success': {},  # n -> EMA of success rate
        'curriculum_episodes': {},  # n -> total episodes attempted
    }

    total_steps = 0
    update = 0
    t0 = time.time()

    print("\nStarting PPO training...")
    print(f"{'Update':>8} | {'Steps':>10} | {'MeanRew':>10} | {'Free':>6} | {'Rate':>8} | {'Time':>8}")
    print("-" * 67)

    while total_steps < args.total_steps:
        model.train()
        last_val = collect_rollout(
            env, model, buffer, args.n_steps, stats,
            n_min=n_min, n_max=n_max, rng=rng,
        )
        total_steps += args.n_steps

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
