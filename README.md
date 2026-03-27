# FreeLineArrangements

Reinforcement learning system for discovering **free line arrangements** in the complex projective plane CP^2, using a Transformer-based actor-critic trained with PPO.

## The Mathematical Problem

A **line arrangement** in CP^2 is a finite collection of lines `L_i: a_i*x + b_i*y + c_i*z = 0`. The arrangement is **free** (in the sense of Terao) if its module of logarithmic derivations D(A) splits as a direct sum of graded components. Equivalently, by **Saito's criterion**, the arrangement with n lines is free with exponents (1, d2, d3) if there exist polynomial derivations theta2, theta3 of degrees d2, d3 such that:

```
det(theta_E, theta2, theta3) = c * Q(x, y, z)
```

where theta_E = (x, y, z) is the Euler derivation, Q = product of all linear forms is the defining polynomial, and c is a nonzero constant.

**Why this is hard:** For n >= 10, the search space of possible line arrangements grows combinatorially, and verifying freeness requires solving polynomial systems. This project uses RL to learn structural patterns that guide the search.

## Architecture

```
main.py              CLI entry point (train, search, explore, verify)
arrangement.py       Core math: ProjectiveLine, LineArrangement, exact Saito check
saito.py             Smooth Saito loss (ALS in coefficient space) + reward shaping
environment.py       Gym-like RL environment for building arrangements step-by-step
model.py             Transformer Actor-Critic (cross-attention over candidate lines)
train.py             PPO training loop with adaptive curriculum learning
discoveries.py       Persistent JSON log of found free arrangements
```

### Reward Pipeline

The reward signal combines multiple levels:

1. **Combinatorial score** -- Does b2(A) yield integer candidate exponents? Pure arithmetic, always computable.

2. **Smooth Saito loss** -- For candidate exponents (d2, d3), parameterize derivations theta2, theta3 via the null spaces of the derivation matrices and minimize `||det(Euler, theta2, theta3) - c*Q||^2 / ||Q||^2` using Alternating Least Squares in polynomial coefficient space. This gives a smooth, exact-in-the-limit measure of distance to freeness.

3. **Terminal exact check** -- For small n (n <= 12 by default), the exact Saito criterion is verified via sympy at episode end.

4. **Shaping signals** -- Interestingness bonus (rich singularity structure), multiplicity penalties (near-pencil avoidance), feasibility bonuses, and per-step multiplicity growth rewards.

### Model

The **TransformerActorCritic** processes the current arrangement:

- **LineEncoder** projects raw [a, b, c] coordinates to d_model embeddings
- A **scalar summary token** fuses 14 global features (b2, discriminant, multiplicity profile, algebraic score, etc.)
- **TransformerEncoder** processes [scalar_token | selected_lines] with padding masks
- **Cross-attention**: each candidate line queries the context to produce action logits
- **Critic head**: scalar token to value estimate

### Environment Modes

- **Pool mode** (default): Agent picks from a fixed pool of small-integer-coordinate lines
- **Singularity-aware mode** (`--singularity-aware`): Candidates are dynamically generated from the intersection structure -- lines through pairs of high-multiplicity points are proposed first, mixed with random pool lines for diversity

## Setup

```bash
conda create -n free_arr python=3.11
conda activate free_arr
pip install torch numpy sympy
```

## Usage

### Quick Verification

Verify the freeness checker on known examples (Braid B3, A2 x A1, Boolean A3):

```bash
python main.py verify
```

### Random Search (Baseline)

Brute-force search for free arrangements of n lines:

```bash
python main.py search --n 6 --coord-range 3 --max-check 10000
python main.py search --n 6 --coord-range 3 --exhaustive  # all combinations
```

### Training

Fixed n:

```bash
python main.py train --n 6 --total-steps 500000
```

Curriculum learning (recommended for large n):

```bash
python main.py train \
  --n-min 6 --n-max 15 \
  --coord-range 5 \
  --total-steps 2000000 \
  --singularity-aware \
  --skip-exact-above 12 \
  --save model_n15.pt
```

Key training flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--n-min`, `--n-max` | `--n` | Curriculum range (adaptive sampling toward frontier) |
| `--coord-range` | 3 | Integer coordinate range for candidate lines (auto-scales for large n) |
| `--singularity-aware` | off | Dynamic candidate generation from intersection structure |
| `--skip-exact-above` | 12 | Use smooth loss instead of exact sympy check for n > this |
| `--total-steps` | 500000 | Total environment steps |
| `--n-steps` | 2048 | Steps per PPO rollout |
| `--lr` | 3e-4 | Learning rate |
| `--ent-coef` | 0.02 | Entropy coefficient (exploration) |
| `--w-free` | 10.0 | Terminal bonus weight for verified free arrangements |
| `--w-alg` | 0.5 | Weight for algebraic (smooth Saito) score |
| `--save` | model_final.pt | Output model path |
| `--resume` | | Resume from checkpoint |

### Exploration

Run a trained model greedily to collect free arrangements:

```bash
python main.py explore --n 10 --model model_final.pt --episodes 1000
```

### Post-hoc Verification (Large n)

For n > 12, training uses the smooth loss proxy. Verify candidates exactly:

```bash
python main.py verify-found \
  --n 15 \
  --model model_n15.pt \
  --coord-range 5 \
  --episodes 5000 \
  --singularity-aware
```

### View Discoveries

```bash
python main.py discoveries
```

## How the Smooth Saito Loss Works

The key innovation enabling large-n search. For a given arrangement with candidate exponents (d2, d3):

1. **Build derivation matrices** M_d2, M_d3 (float64) encoding the divisibility constraints
2. **Extract full null space bases** V2, V3 via SVD (not just one vector -- the null spaces can be 50+ dimensional for large n)
3. **Precompute bilinear tensor** T such that `det_coeffs = T @ (alpha2, alpha3)` maps null-space parameters to the Saito determinant's coefficient vector
4. **Alternating Least Squares**: minimize `||det - c*Q||^2 / ||Q||^2` by alternating between solving for alpha2 (fixing alpha3) and vice versa, each step a small SVD
5. **Loss = 0** iff the arrangement is free; smoothly positive otherwise

Performance: ~0.2ms (n=6), ~1.4ms (n=15), ~2.5ms (n=20).

## Project Structure

```
FreeLineArrangements/
  main.py               Entry point and CLI
  arrangement.py         ProjectiveLine, LineArrangement, intersection structure,
                         candidate exponents, exact Saito check (sympy)
  saito.py               Smooth Saito loss (ALS), combinatorial score,
                         interestingness score, reward composition
  environment.py         FreeArrangementEnv (pool + singularity-aware modes),
                         scalar feature extraction (14 features)
  model.py               TransformerActorCritic (LineEncoder, cross-attention)
  train.py               PPO training, GAE, curriculum sampling, greedy eval
  discoveries.py         JSON persistence, deduplication, summary
  discoveries.json       Accumulated discovery log
  model_final.pt         Trained model checkpoint
```

## Key Invariants

For an arrangement of n lines:

- **b2(A)** = sum of (multiplicity - 1) over all intersection points
- **Candidate exponents** (d2, d3): d2 + d3 = n - 1, d2 * d3 = b2 - (n - 1). These must be non-negative integers (necessary condition for freeness).
- **Pencil**: all lines concurrent (trivially free but uninteresting, penalized by the reward)
- **Multiplicity profile**: sorted list of intersection point multiplicities -- captures the combinatorial type

## References

- Orlik, P. and Terao, H. *Arrangements of Hyperplanes*. Springer, 1992.
- Saito, K. "Theory of logarithmic differential forms and logarithmic vector fields." *J. Fac. Sci. Univ. Tokyo*, 1980.
- Schulman, J. et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347, 2017.
