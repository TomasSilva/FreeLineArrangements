"""
model.py

Transformer-based Actor-Critic for discovering free line arrangements in CP^2.

Architecture:
  - LineEncoder: projects raw [a,b,c] line coords -> d_model embedding.
  - A learned scalar-summary token fuses the 11 global scalar features.
  - TransformerEncoder processes (scalar token + selected line tokens), with a
    key-padding mask to ignore zero-padded positions.
  - Cross-attention: each candidate line queries the context to get its score.
  - Actor head: per-candidate score -> masked softmax over pool.
  - Critic head: scalar token (global summary) -> value.

The model operates directly on line coordinates, so it generalises across
different pool sizes and coord_range values without any architecture changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LineEncoder(nn.Module):
    """Projects a raw [a, b, c] line coordinate to a d_model embedding."""

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        # x: (..., 3)
        return self.net(x)


class TransformerActorCritic(nn.Module):
    """
    Combined Actor-Critic using a Transformer over selected lines.

    Inputs (all tensors, batch-first):
      selected_coords : (B, max_n, 3)  — selected lines, zero-padded
      candidate_coords: (B, pool_size, 3) — full candidate pool
      scalars         : (B, scalar_dim)  — global arrangement features
      mask            : (B, pool_size)   — 1 = valid action
      n_selected      : (B,) long        — true number of selected lines

    Outputs:
      logits: (B, pool_size) — action logits (invalid actions masked to -inf)
      value : (B,)           — state value estimate
    """

    def __init__(
        self,
        max_n: int = 20,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        scalar_dim: int = 14,
    ):
        super().__init__()
        self.max_n = max_n
        self.d_model = d_model
        self.scalar_dim = scalar_dim

        # Line coordinate encoder (shared for selected and candidate lines)
        self.line_encoder = LineEncoder(d_model)

        # Learned type embeddings to distinguish selected vs. candidate lines
        self.selected_type_emb = nn.Parameter(torch.zeros(d_model))
        self.candidate_type_emb = nn.Parameter(torch.zeros(d_model))

        # Project scalar features to a summary token prepended to the context
        self.scalar_proj = nn.Linear(scalar_dim, d_model)

        # Transformer encoder over (scalar token + selected lines)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,  # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Cross-attention: candidate lines query the context
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.0
        )

        # Actor: per-candidate score
        self.actor_head = nn.Linear(d_model, 1)

        # Critic: global summary -> value
        self.critic_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.constant_(self.actor_head.bias, 0.0)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)

    def forward(self, selected_coords, candidate_coords, scalars, mask, n_selected):
        """
        Args:
            selected_coords : (B, max_n, 3)
            candidate_coords: (B, pool_size, 3)
            scalars         : (B, scalar_dim)
            mask            : (B, pool_size)  float, 1=valid
            n_selected      : (B,) long
        Returns:
            logits: (B, pool_size)
            value : (B,)
        """
        B = selected_coords.shape[0]
        device = selected_coords.device

        # ── Encode selected lines ─────────────────────────────────────────────
        sel_emb = self.line_encoder(selected_coords) + self.selected_type_emb
        # (B, max_n, d_model)

        # ── Scalar summary token (always the first token, never padded) ───────
        scalar_token = self.scalar_proj(scalars).unsqueeze(1)  # (B, 1, d_model)

        # Concatenate: [scalar_token | selected_lines]
        context_in = torch.cat([scalar_token, sel_emb], dim=1)  # (B, 1+max_n, d_model)

        # ── Padding mask for Transformer (True = ignore position) ─────────────
        # Positions 1+ns .. 1+max_n-1 are padding
        positions = torch.arange(self.max_n, device=device).unsqueeze(0)  # (1, max_n)
        line_pad_mask = positions >= n_selected.unsqueeze(1)              # (B, max_n)
        scalar_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)
        ctx_pad_mask = torch.cat([scalar_pad, line_pad_mask], dim=1)      # (B, 1+max_n)

        # ── Transformer over context ──────────────────────────────────────────
        context = self.transformer(context_in, src_key_padding_mask=ctx_pad_mask)
        # (B, 1+max_n, d_model)

        # ── Encode candidate lines ────────────────────────────────────────────
        cand_emb = self.line_encoder(candidate_coords) + self.candidate_type_emb
        # (B, pool_size, d_model)

        # ── Cross-attention: candidates query the context ─────────────────────
        cand_ctx, _ = self.cross_attn(
            cand_emb, context, context,
            key_padding_mask=ctx_pad_mask,
        )  # (B, pool_size, d_model)

        # ── Actor logits ──────────────────────────────────────────────────────
        logits = self.actor_head(cand_ctx).squeeze(-1)  # (B, pool_size)
        logits = logits + (1.0 - mask) * (-1e9)

        # ── Critic value ──────────────────────────────────────────────────────
        # Use the scalar summary token as the global state representation
        global_ctx = context[:, 0, :]               # (B, d_model)
        value = self.critic_head(global_ctx).squeeze(-1)  # (B,)

        return logits, value

    @torch.no_grad()
    def act(self, obs: dict, deterministic: bool = False):
        """
        Sample (or greedily pick) an action from the policy.

        Args:
            obs: dict from FreeArrangementEnv._obs()
            deterministic: if True, take argmax instead of sampling

        Returns:
            action   (int)
            log_prob (tensor scalar, with grad for PPO)
            value    (tensor scalar, with grad for PPO)
        """
        sel   = torch.FloatTensor(obs['selected_coords']).unsqueeze(0)
        cand  = torch.FloatTensor(obs['candidate_coords']).unsqueeze(0)
        sc    = torch.FloatTensor(obs['scalars']).unsqueeze(0)
        mask  = torch.FloatTensor(obs['mask']).unsqueeze(0)
        n_sel = torch.tensor([obs['n_selected']], dtype=torch.long)

        logits, value = self.forward(sel, cand, sc, mask, n_sel)
        dist = torch.distributions.Categorical(logits=logits)

        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.squeeze(0), value.squeeze(0)

    def evaluate(self, batch_obs: dict, actions: torch.Tensor, masks: torch.Tensor):
        """
        Evaluate a batch of (obs, action) pairs for the PPO update.

        Args:
            batch_obs: dict with batched tensors (B, ...)
            actions  : (B,) long
            masks    : (B, pool_size) float

        Returns:
            log_probs: (B,)
            values   : (B,)
            entropy  : (B,)
        """
        logits, values = self.forward(
            batch_obs['selected_coords'],
            batch_obs['candidate_coords'],
            batch_obs['scalars'],
            masks,
            batch_obs['n_selected'],
        )
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy
