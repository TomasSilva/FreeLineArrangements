"""
Self-contained PPO trainer for the fixed-cardinality swap policy
(joint-flatten action space; SwapArrangementEnv + TransformerActorCritic
mode='swap_joint').

Deliberately standalone: the grow-one-line machinery in train.py is left
untouched; this trainer is the learned arm of the multi-engine comparison.

Usage:
  python experiments/train_swap_policy.py --n 12 --d1 5 --d2 6 \
      --updates 40 --episodes-per-update 8 --out <dir> [--seed 0]
Writes: policy.pt, training_log.jsonl, discoveries certified during
training to certified.jsonl (exact certificates only).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import TransformerActorCritic
from swap_env import SwapArrangementEnv


def collate(obs_list):
    return {
        'selected_coords': torch.FloatTensor(
            np.stack([o['selected_coords'] for o in obs_list])),
        'candidate_coords': torch.FloatTensor(
            np.stack([o['candidate_coords'] for o in obs_list])),
        'scalars': torch.FloatTensor(
            np.stack([o['scalars'] for o in obs_list])),
        'n_selected': torch.tensor([o['n_selected'] for o in obs_list],
                                   dtype=torch.long),
        'mask': torch.FloatTensor(np.stack([o['mask'] for o in obs_list])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--d1", type=int, required=True)
    ap.add_argument("--d2", type=int, required=True)
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--episodes-per-update", type=int, default=8)
    ap.add_argument("--episode-len", type=int, default=16)
    ap.add_argument("--k-perturb", type=int, default=2)
    ap.add_argument("--max-candidates", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-eps", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = SwapArrangementEnv(target_n=args.n, d1=args.d1, d2=args.d2,
                             seed=args.seed, episode_len=args.episode_len,
                             k_perturb=args.k_perturb,
                             max_candidates=args.max_candidates)
    model = TransformerActorCritic(max_n=args.n, mode="swap_joint")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    log_path = os.path.join(args.out, "training_log.jsonl")
    cert_path = os.path.join(args.out, "certified.jsonl")
    n_certified = 0
    total_steps = 0
    t0 = time.time()

    for update in range(args.updates):
        obs_buf, act_buf, logp_buf, val_buf, rew_buf, done_buf = \
            [], [], [], [], [], []
        ep_rewards = []
        for _ in range(args.episodes_per_update):
            obs = env.reset()
            done = False
            ep_r = 0.0
            while not done:
                a, lp, v = model.act(obs)
                nobs, r, done, info = env.step(a)
                obs_buf.append(obs)
                act_buf.append(a)
                logp_buf.append(float(lp))
                val_buf.append(float(v))
                rew_buf.append(r)
                done_buf.append(done)
                obs = nobs
                ep_r += r
                total_steps += 1
                if info.get('certificate'):
                    n_certified += 1
                    with open(cert_path, "a") as f:
                        f.write(json.dumps({
                            "lines": [str(l) for l in env.arr.lines],
                            "n": args.n, "d1": args.d1, "d2": args.d2,
                            "update": update, "step": total_steps,
                        }) + "\n")
            ep_rewards.append(ep_r)

        # GAE (episodes end by truncation; the env folds the tail potential
        # into the final reward, so bootstrap value 0 at done is consistent)
        adv = np.zeros(len(rew_buf), dtype=np.float64)
        last = 0.0
        for t in reversed(range(len(rew_buf))):
            next_val = 0.0 if done_buf[t] else val_buf[t + 1]
            delta = rew_buf[t] + args.gamma * next_val - val_buf[t]
            last = delta + args.gamma * args.gae_lambda * \
                (0.0 if done_buf[t] else last)
            adv[t] = last
        ret = adv + np.array(val_buf)
        adv_t = torch.FloatTensor((adv - adv.mean()) / (adv.std() + 1e-8))
        ret_t = torch.FloatTensor(ret)
        acts_t = torch.LongTensor(act_buf)
        old_lp = torch.FloatTensor(logp_buf)
        batch = collate(obs_buf)

        for _ in range(args.epochs):
            lp, vals, ent = model.evaluate(batch, acts_t, batch['mask'])
            ratio = torch.exp(lp - old_lp)
            s1 = ratio * adv_t
            s2 = torch.clamp(ratio, 1 - args.clip_eps,
                             1 + args.clip_eps) * adv_t
            loss = (-torch.min(s1, s2).mean()
                    + 0.5 * ((vals - ret_t) ** 2).mean()
                    - args.ent_coef * ent.mean())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

        rec = {"update": update, "steps": total_steps,
               "mean_ep_reward": float(np.mean(ep_rewards)),
               "certified": n_certified,
               "wall_s": time.time() - t0}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"u{update:03d} steps={total_steps} "
              f"meanR={rec['mean_ep_reward']:+.3f} cert={n_certified} "
              f"({rec['wall_s']:.0f}s)", flush=True)

    torch.save({"model": model.state_dict(), "args": vars(args)},
               os.path.join(args.out, "policy.pt"))
    print(f"done: {n_certified} certified during training; "
          f"{total_steps} env steps")


if __name__ == "__main__":
    main()
