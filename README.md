# FreeLineArrangements

A toolkit for discovering **free line arrangements** in the complex projective plane CP². Combines a Transformer-based PPO agent with classical algebraic geometry constructions and a hybrid bootstrap-extension search. Designed to run on HPC clusters with parallel environments.

## At a Glance

The repo evolved through three discovery strategies, each addressing a regime where the previous one failed:

| Strategy | Best for | Tool | Verified results |
|---|---|---|---|
| **Pure RL (PPO + Transformer)** | n ≤ 12 | `train`, `explore`, `verify-found` | 9,869 free arrangements at n=6..13 in 81h on HPC |
| **Hybrid bootstrap extension** | n ≥ 14 | `extend` | 1,602 arrangements at n=13..18 + 1,774 at n=19 in <24h locally |
| **Direct supersolvable construction** | All (n, d1, d2) cells | `construct` | One example per cell, instant, closed form |
| **Δb2-targeted extension** | Filling unbalanced exponent cells | `extend --target-new-exponents` / `--all-targets` | Single n=12 supersolvable seed → 1,162 free n=13 arrangements covering all 6 exponent types |

For **n ≥ 14** the recommended path is `construct --family all-supersolvable` (instant per-cell coverage) followed by `extend --all-targets` (rich non-supersolvable examples in every cell). See [Comprehensive Coverage](#comprehensive-coverage-of-all-exponent-types) below.

## Quickstart

If you just want to discover free line arrangements for n up to 20 with full (d1, d2) coverage, skip RL entirely:

```bash
# 1. Install
conda create -n free_arr python=3.11 && conda activate free_arr
pip install torch numpy sympy scipy

# 2. Seed every (n, d1, d2) cell with a closed-form supersolvable example (instant)
for N in 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  python main.py construct --family all-supersolvable --n $N
done

# 3. Cascade with targeted extension to find non-supersolvable examples (hours, not days)
for N in 12 13 14 15 16 17 18 19; do
  python main.py extend --n-from $N --all-targets
done

# 4. Inspect what you found
python main.py discoveries
```

Or on HPC: `qsub pbs/step5_coverage.pbs` — single self-contained job that does steps 2 and 3 with `xargs -P 16` parallelism per level.

## Empirical Results from the Cascade

Local cascade (no HPC, no GPU) starting from 105 n=12 seeds produced by an earlier RL run:

| n | Free arrangements found | Exponent types found | Notes |
|---|---|---|---|
| 13 | 1,162 (targeted) / 177 (unfiltered) | All 6 of (1,1,11)..(1,6,6) | Targeted run filled every cell |
| 14 | 73 (unfiltered) | (1, 6, 7) only | Unfiltered drifts to balanced types |
| 15 | 175 | (1, 7, 7) | |
| 16 | 192 | (1, 7, 8) | |
| 17 | 386 | (1, 7, 9) | |
| 18 | 599 | (1, 8, 9) | |
| 19 | 1,774 | (1, 9, 9), (1, 8, 10), (1, 7, 11) | First cascade level to find off-balanced types naturally |

For comparison, the original 81-hour RL training on HPC found **9,869** free arrangements but **all** were at n ≤ 13. The hybrid extension closes the n ≥ 14 gap entirely; the targeted variant additionally fills every (d1, d2) cell.

## The Mathematical Problem

A **line arrangement** A in CP² is a finite collection of lines L_i: a_i x + b_i y + c_i z = 0. The arrangement is **free** (in the sense of Terao) if its module of logarithmic derivations D(A) splits as a direct sum of line bundles:

D(A) = O(-1) + O(-d1) + O(-d2)

where (1, d1, d2) are the **exponents** of the arrangement with d1 + d2 = n - 1.

By **Saito's criterion**, this splitting holds if and only if there exist polynomial derivations theta2, theta3 of degrees d1, d2 such that:

```
det(theta_E, theta2, theta3) = c * Q(x, y, z)
```

where theta_E = (x, y, z) is the Euler derivation, Q is the product of all defining linear forms, and c is a nonzero constant.

**Why RL?** For n >= 10, the space of possible arrangements grows combinatorially. Verifying freeness requires solving polynomial systems over the full null spaces of derivation matrices. This project trains an RL agent to learn structural patterns of free arrangements and guide the search.

### Candidate Exponents

For an arrangement of n lines with second Betti number b2 = sum(m_p - 1) over intersection points p, the candidate exponents (d1, d2) must satisfy:

- d1 + d2 = n - 1
- d1 * d2 = b2 - (n - 1)

These are necessary (but not sufficient) conditions for freeness. The discriminant (n-1)^2 - 4*(b2 - (n-1)) must be a non-negative perfect square.

## Architecture Overview

```
main.py              CLI entry point (train, search, explore, verify, verify-found, extend, construct)
arrangement.py       Core math: ProjectiveLine, LineArrangement, intersection lattice, exact Saito check
saito.py             Smooth Saito loss (ALS), reward shaping, polish_arrangement, extend_arrangement,
                     extend_arrangement_targeted, construct_near_pencil, construct_supersolvable
environment.py       Gym-like RL environment with pool and singularity-aware candidate modes
model.py             Transformer Actor-Critic with cross-attention over candidate lines
train.py             PPO training with adaptive triple curriculum and vectorized environments
vec_env.py           Subprocess-based parallel environment (one worker per CPU core)
discoveries.py       Persistent JSON log with deduplication
pbs/                 HPC job scripts (step1_train, step2_explore, step3_verify, step4_extend, step5_coverage)
```

## The Loss Function

The core challenge is that freeness is a discrete algebraic property: either the Saito determinant condition holds or it doesn't. The system converts this into a smooth, differentiable reward signal through a multi-level pipeline.

### Level 1: Combinatorial Score

A fast arithmetic check on b2. Computes the discriminant for candidate exponents:

```
disc = (n-1)^2 - 4*(b2 - (n-1))
```

Returns a score in [-1, 1]:
- **1.0** if disc >= 0, is a perfect square, and yields valid integer exponents (d1, d2)
- Smooth interpolation toward -1.0 based on distance from a valid discriminant
- Computed for every step, no linear algebra needed

### Level 2: Smooth Saito Loss (the key innovation)

For an arrangement with candidate exponents (d1, d2), this measures how far the arrangement is from satisfying Saito's criterion -- continuously, in coefficient space.

**Step 1 -- Derivation matrices.** For degree d, build the float64 matrix M_d encoding the divisibility constraints: for each line alpha_i and each parameterization degree (p, q = d - p), the matrix rows encode alpha_i | theta(alpha_i) evaluated on ker(alpha_i). This produces M_d1 of shape (n * (d1+1), 3 * C(d1+2,2)) and similarly M_d2.

**Step 2 -- Null space extraction.** Compute full orthonormal null space bases V2, V3 via SVD of M_d1, M_d2. These null spaces can be high-dimensional (50+ for large n), capturing all derivations that satisfy the divisibility constraints. Any theta2 = V2 @ alpha2, theta3 = V3 @ alpha3 (for parameter vectors alpha2, alpha3) is a valid logarithmic derivation of degree d1, d2 respectively.

**Step 3 -- Bilinear determinant tensor.** Precompute a tensor T of shape (N_out, k2, k3) where N_out = C(n+2, 2) is the number of degree-n monomials, and k2, k3 are the null space dimensions. The tensor encodes:

```
det(theta_E, theta2, theta3) = T[alpha2, alpha3]
```

expanding the 3x3 determinant via cofactor expansion along the Euler row:

```
det = x*(g2*h3 - g3*h2) - y*(f2*h3 - f3*h2) + z*(f2*g3 - f3*g2)
```

where theta2 = (f2, g2, h2), theta3 = (f3, g3, h3). Each cross-term is a bilinear product of degree-d1 and degree-d2 polynomials (yielding degree n-1), then multiplied by x, y, or z to reach degree n. The multiplication uses precomputed sparse tables mapping monomial pairs to output indices.

**Step 4 -- Alternating Least Squares (ALS).** Minimize:

```
L = ||T[alpha2, alpha3] - c * Q||^2 / ||Q||^2
```

where Q is the defining polynomial (product of all linear forms). Since T is bilinear in alpha2, alpha3, fixing one and solving for the other (plus scalar c) reduces to a linear least squares problem solvable by SVD of an augmented system [A | -q]. The algorithm alternates:

1. Fix alpha3, solve for alpha2 via SVD of the augmented matrix
2. Fix alpha2, solve for alpha3 via SVD of the augmented matrix
3. Evaluate loss = 1 - cos^2(angle between T[alpha2, alpha3] and Q)

This runs for 10 iterations with 3 random restarts. The loss is exactly 0 if and only if the arrangement is free. For non-free arrangements, it smoothly measures the angular distance between the best achievable determinant and the target polynomial Q.

**Performance:** ~0.2ms (n=6), ~1.4ms (n=15), ~2.5ms (n=20).

### Level 3: Exact Saito Check

For small n (n <= 12 by default), the exact Saito criterion is verified via sympy over the rationals at episode end. For larger n, this is replaced by a graded bonus based on the algebraic score from Level 2, avoiding the expensive symbolic computation during training.

### Algebraic Score (Tier 1 + Tier 2 combined)

The `algebraic_score()` function combines Levels 1 and 2 into a single score in [-1, 1]:

| Score range | Meaning |
|---|---|
| -1.0 | b2 too far from any valid exponents |
| [-1.0, -0.5] | Discriminant negative (b2 too large) |
| [-0.5, 0.0) | Discriminant non-negative but not a perfect square |
| [0.0, 1.0] | Valid exponents exist; 1.0 = arrangement is free |

When target exponents (d1, d2) are specified, Tier 1 instead measures the normalized distance from b2 to the specific target b2 = (n-1) + d1 * d2, scaled by target_b2 itself (not the maximum possible b2) so that high-b2 targets get sharper gradient signal.

### Full Reward Composition

The reward returned to the RL agent at each step is:

```
R = w_comb   * combinatorial_score(A)         # 0.3 -- b2 yields integer exponents?
  + w_alg    * algebraic_score(A)              # 0.5 -- smooth Saito proximity
  + w_mult   * multiplicity_penalty(A)         # 2.0 -- penalize near-pencil
  + w_interest * interestingness_score(A)      # 1.0 -- rich singularity structure
  + w_feasibility * has_candidate_exponents    # 0.5 -- on a viable path
  + w_b2_traj * b2_trajectory_bonus(A)         # 1.5 -- b2 moving toward target
  + w_mult_growth * new_triple_points          # 0.3 -- creating triple+ intersections
  - w_pencil * is_pencil(A)                    # 5.0 -- all lines concurrent
  + w_free   * is_free(A)                      # 10.0 -- terminal bonus (verified free)
```

For n > `skip_exact_above` (default 12), the terminal bonus is replaced by a graded algebraic score bonus: 80% of w_free when alg_score > 0.95, 40% when > 0.80.

## Model

The **TransformerActorCritic** (689K parameters by default) processes the arrangement being built:

1. **LineEncoder** -- MLP projecting raw [a, b, c] coordinates to d_model=128 embeddings, with LayerNorm and GELU
2. **Scalar summary token** -- 17 global features (b2, discriminant, multiplicity stats, algebraic score, exponent targets, b2 progress) projected to d_model and prepended to the sequence
3. **TransformerEncoder** (3 layers, 4 heads, pre-norm) -- processes [scalar_token | selected_lines] with padding masks
4. **Cross-attention** -- each candidate line queries the context to produce action logits
5. **Critic head** -- scalar token to value estimate via 2-layer MLP

### Scalar Features (17 dimensions)

| Feature | Description |
|---|---|
| n / target_n | Build progress |
| b2 / max_b2 | Normalized second Betti number |
| disc_norm | Normalized discriminant |
| m2, m3, m4+ | Double, triple, quadruple+ point counts |
| is_pencil | Binary pencil indicator |
| comb_score | Combinatorial score |
| alg_score | Algebraic score (Tier 1 + 2) |
| max_mult / target_n | Normalized max multiplicity |
| n_pts / max_pts | Intersection point density |
| triple_ratio, entropy, sing_density | Singularity-aware features |
| d1_norm, d2_norm | Target exponent encoding |
| b2_progress | Distance to target b2 |

## Training

### PPO with Adaptive Triple Curriculum

Training uses Proximal Policy Optimization with Generalized Advantage Estimation. The curriculum system samples (n, d1, d2) triples -- not just n values -- weighted by:

1. **Inverse success rate** -- under-explored triples get more samples
2. **b2 difficulty boost** -- up to 3x weight for hard exponent types (d1 * d2 near maximum), preventing the agent from only finding easy exponents like (1, 1, n-2)
3. **Promotion** -- successful episodes at easier n values trigger sampling at harder ones

### Vectorized Environments

For HPC, training uses `SubprocVecEnv` with one worker process per CPU core (e.g., 16 on par16 queue). Each worker runs its own `FreeArrangementEnv` with auto-reset, while the main process batches observations for a single model forward pass.

### Environment Modes

- **Pool mode** (default) -- agent picks from a fixed pool of projectively distinct lines with integer coordinates in [-coord_range, coord_range]
- **Singularity-aware mode** (`--singularity-aware`) -- candidates are dynamically regenerated each step from the intersection structure: lines through pairs of high-multiplicity points, mixed with random pool lines for diversity

## Setup

```bash
conda create -n free_arr python=3.11
conda activate free_arr
pip install torch numpy sympy scipy
```

`scipy` is used by `polish_arrangement` (L-BFGS-B / Nelder-Mead in coefficient space) and is required for the hybrid pipeline. `torch` is only needed for the RL pipeline; the extension and construction commands work without it.

## Usage

### Verify Known Arrangements

```bash
python main.py verify
```

Checks the Braid B3 (n=6, exponents 1,2,3), A2 x A1 (n=4), Boolean A3 (n=3), and a non-free example.

### Train

Fixed n:

```bash
python main.py train --n 6 --total-steps 500000
```

Curriculum learning across n values (recommended):

```bash
python main.py train \
  --n-min 6 --n-max 20 \
  --coord-range 5 \
  --total-steps 5000000 \
  --n-envs 16 \
  --singularity-aware \
  --skip-exact-above 12 \
  --save model_n20.pt
```

| Flag | Default | Description |
|---|---|---|
| `--n-min`, `--n-max` | `--n` | Curriculum range |
| `--n-envs` | 1 | Parallel environments (set to number of CPU cores) |
| `--coord-range` | 3 | Integer coordinate range for candidate lines |
| `--singularity-aware` | off | Dynamic candidate generation from intersection structure |
| `--skip-exact-above` | 12 | Use smooth loss instead of exact sympy for n > this |
| `--total-steps` | 500000 | Total environment steps |
| `--n-steps` | 2048 | Steps per PPO rollout |
| `--batch-size` | 128 | Minibatch size for PPO updates |
| `--lr` | 3e-4 | Learning rate |
| `--ent-coef` | 0.02 | Entropy coefficient |
| `--w-free` | 10.0 | Terminal freeness bonus weight |
| `--w-alg` | 0.5 | Algebraic score weight |
| `--w-comb` | 0.3 | Combinatorial score weight |
| `--save` | model_final.pt | Output model path |
| `--save-every` | 0 | Checkpoint interval (updates) |
| `--resume` | | Resume from checkpoint |

### Explore with Trained Model

Run the trained policy to collect free arrangements:

```bash
python main.py explore --n 10 --model model_n20.pt --episodes 5000

# Target specific exponent types
python main.py explore --n 20 --model model_n20.pt --episodes 5000 --target-exponents 9 10
```

### Post-hoc Exact Verification

For n > 12, training uses the smooth loss proxy. Verify candidates exactly with sympy:

```bash
python main.py verify-found --n 15 --model model_n20.pt --episodes 5000 --singularity-aware
```

## Why Pure RL Fails Above n=13

An 81-hour HPC training run with 16 parallel environments (5M PPO steps, full curriculum n=6..20, all default reward shaping) found **9,869 free arrangements at n ∈ {6, ..., 13} but ZERO at n ≥ 14**. Subsequent exploration runs (60 jobs × 5,000 episodes targeting specific (d1, d2) for n=14..20) also found nothing. Three independent root causes converge at the n=13→14 cliff:

1. **Reward signal collapses for n > 12.** With `--skip-exact-above 12`, the strong `+10` exact-free terminal bonus is replaced by a graded bonus only when `algebraic_score > 0.95`. The smooth Saito loss is empirically chaotic with the default 3 ALS restarts: nearby arrangements get loss values that jump between 0 and 1 due to ALS local minima and hard SVD null-space dimension thresholds. Effectively the agent gets random reward for n>12 attempts.

2. **Search space is too large for discrete RL.** For n=20 with ~200 candidates per step, the trajectory space is ≈ 200²⁰. Free arrangements form a measure-zero manifold inside this space; without a strong reward gradient, PPO is reduced to random search.

3. **`smooth_saito_loss` is the wrong tool for the last mile.** It's a *signal*, not an *optimizer*. The discrete RL agent picks lines from a finite pool, but free arrangements live in a continuous parameter space — even when the agent is structurally close, every available candidate line is wrong by a small amount and there is no continuous knob to turn.

The fix is to abandon "build from scratch with RL" for n ≥ 14 and instead use the next two strategies.

### Bootstrap Extension (recommended for n >= 14)

Take a known free arrangement, enumerate candidate lines that could extend it, pre-filter cheaply, and exact-verify the survivors. Adding **one good line** to a known free arrangement is exponentially easier than discovering a free arrangement of n+1 lines from scratch — and discoveries cascade: today's n=13 result becomes tomorrow's n=14 seed.

The `extend` command implements this: it takes a seed arrangement at n_from, enumerates candidate lines from three sources (lines through pairs of existing intersection points, the small-integer pool, and optionally rational lines through multiple points), pre-filters via `smooth_saito_loss`, and exact-verifies the survivors with sympy. **Empirical result**: starting from existing n=12 seeds (105 arrangements), a local cascade in under 24 hours produced 1,602 free arrangements at n=13..18 and 1,774 more at n=19, covering ground that the 81-hour RL training never reached.

```bash
# Extend known n=12 arrangements to find n=13 free arrangements
python main.py extend --n-from 12 --seeds-file discoveries.json

# Cascade: each step uses the previous step's discoveries as seeds
for N in 12 13 14 15 16 17 18 19; do
  python main.py extend --n-from $N --seeds-file discoveries.json
done
```

| Flag | Default | Description |
|---|---|---|
| `--n-from` | required | Source n: load free arrangements with this n as seeds |
| `--seeds-file` | discoveries.json | Input JSON with seed arrangements |
| `--output` | same as seeds | Where to save new discoveries |
| `--coord-range` | 5 | Integer pool range for new candidate lines |
| `--max-denominator` | 1 | If >1, also generate rational lines through existing multiple points |
| `--loss-threshold` | 0.05 | Pre-filter: skip exact check if smooth loss above this |
| `--n-restarts` | 10 | ALS restarts in the smooth loss pre-filter |
| `--max-seeds` | None | Limit seeds (for testing) |
| `--target-exponents` | None | Filter seeds by their exponents |
| `--target-new-exponents` | None | Target a specific (d1, d2) for the n+1 result. Uses Δb2 pre-filter. |
| `--all-targets` | off | Iterate over ALL valid (d1, d2) types for n+1. Comprehensive coverage. |

This enumerate-and-verify approach is what enables discovery of arrangements with n >= 14 in this codebase. The cascading structure means a few hundred n=12 seeds can grow into thousands of arrangements at higher n values.

### Comprehensive Coverage of All Exponent Types

The unfiltered `extend` cascade tends to drift toward "balanced" exponent types: starting from n=12 seeds, the local cascade finds (1, 6, 6) at n=13, (1, 6, 7) at n=14, (1, 7, 7) at n=15, etc., while the unbalanced types (1, 1, n-2), (1, 2, n-3), ... never appear. The reason is geometric: "structural" candidate lines (those passing through several existing intersection points) decrease Δb2, and the algorithm's `_singularity_candidates` enumerator never produces lines that pass through few or zero existing points. To get **comprehensive coverage** — at least one example per (n, d1, d2) cell, including the near-pencil tail — combine two complementary tools.

**1. Direct construction** (`construct` command). Closed-form constructions of known free arrangement families. No search, no optimization — just emit the arrangement and save it. Provides one supersolvable example per cell instantly.

```bash
# Near-pencil: free with exponents (1, 1, n-2). Single function call.
python main.py construct --family near-pencil --n 20

# Supersolvable: free with exponents (1, d1, n-1-d1) for any d1 in [1, (n-1)//2].
# Construction: two pencils sharing one common line. The smaller pencil contributes
# the d1 exponent; by Terao's supersolvability theorem the result is free.
python main.py construct --family supersolvable --n 20 --d1 5

# All-supersolvable: emit one supersolvable for EVERY valid (d1, d2) at level n.
# Single command to fill the entire row of the (n, d1, d2) coverage table.
python main.py construct --family all-supersolvable --n 20
```

The supersolvable construction is implemented in `construct_supersolvable(n, d1)` ([saito.py](saito.py)). Two pencils centered at [0:0:1] and [0:1:0] share the common line `x = 0`; the first pencil contributes (d1+1) lines and the second contributes (n-d1) lines. Verified to produce exactly the right exponents `(1, d1, n-1-d1)` for every valid (n, d1) pair tested up to n=20.

**2. Targeted extension** (`extend --target-new-exponents D1 D2`). Uses the **Δb2 pre-filter** to efficiently search for non-supersolvable examples in a specific cell. Much faster than the unfiltered cascade because most candidates are rejected by an integer comparison before any algebraic work.

The Δb2 formula: for a candidate line L added to seed arrangement A of n lines, let `S(L, A)` be the sum of multiplicities of existing intersection points L passes through and `k(L, A)` be the number of distinct such points. Then exactly:
```
Δb2 = n + k - S
```
(derivation: L meets each of n existing lines somewhere; S of those meetings reuse existing points (each bumping a multiplicity by 1, contributing +1 to b2 per distinct point), the other n−S create brand-new simple points). For a target exponent (d1', d2') at level n+1, the required `Δb2 = (n + d1'·d2') − b2_seed`. This must lie in `[1, n+1]` to be achievable; outside that range the targeted extension exits in 0 ms with no candidates considered.

```bash
# Find all n=14 arrangements of type (1, 3, 10) starting from n=13 seeds
python main.py extend --n-from 13 --target-new-exponents 3 10

# Iterate over all 6 target types for n=14 — comprehensive coverage in one command
python main.py extend --n-from 13 --all-targets
```

**Empirical result.** Starting from just 5 supersolvable seeds at n=12 (one per exponent type), the targeted extension found **1,162 free n=13 arrangements covering all 6 exponent types**:

| Type | Count | Notes |
|---|---|---|
| (1, 1, 11) | 28 | near-pencil |
| (1, 2, 10) | 797 | most numerous |
| (1, 3, 9) | 131 | |
| (1, 4, 8) | 91 | |
| (1, 5, 7) | 75 | |
| (1, 6, 6) | 36 | balanced |

Compare this to the original unfiltered cascade, which only ever found (1, 5, 7) and (1, 6, 6) at n=13 — the four unbalanced types were completely missing. All 8 spot-checked unbalanced discoveries verified exactly free with sympy.

**Recommended workflow** for filling the entire (n, d1, d2) coverage table for n up to 20:

```bash
# PHASE A: instant per-cell supersolvable seeds
for N in 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  python main.py construct --family all-supersolvable --n $N
done

# PHASE B: cascade with targeted extension to find non-supersolvable examples
for N in 12 13 14 15 16 17 18 19; do
  python main.py extend --n-from $N --all-targets
done
```

The supersolvable arrangements from phase A are very symmetric (they have a modular line of high multiplicity, so their multiplicity profiles are dominated by one large point); phase B finds richer combinatorial types in adjacent cells. For HPC, use `pbs/step5_coverage.pbs` which parallelizes phase B across the (n_from, d1, d2) cells using xargs -P 16 with race-free intermediate files.

### View Discoveries

```bash
python main.py discoveries
```

### Visualize an Arrangement

[visualize_new.ipynb](visualize_new.ipynb) plots a line arrangement in the affine chart `z = 1`. Each projective line `ax + by + cz = 0` becomes `ax + by + c = 0` in this chart. The single exception is the line at infinity `z = 0` (i.e., a line with `(a, b) = (0, 0)` after projective normalization), which is rendered as a dotted boundary circle. Intersection points are marked with marker size scaled by multiplicity. Pairs of parallel affine lines correctly do NOT show an affine intersection — their common point lies at infinity.

```python
from visualize_new import draw_arrangement   # or open the notebook directly
draw_arrangement([
    "(1x+0y+0z=0)",   # x = 0
    "(0x+1y+0z=0)",   # y = 0
    "(1x+1y+-1z=0)",  # x + y = 1
    "(0x+0y+1z=0)",   # line at infinity z = 0
], xlim=(-2, 2), ylim=(-2, 2))
```

## HPC Deployment

PBS job scripts are in `pbs/`. They fall into two pipelines:

**Pipeline A — RL training and exploration** (worked for n ≤ 13 only):

| Script | Purpose |
|---|---|
| `step1_train.pbs` | PPO curriculum training n=6 to 20, 16 parallel environments, ~80h runtime |
| `step2_explore.pbs` | Parallel exponent-targeted greedy rollouts of the trained model (all (d1, d2) cells) |
| `step3_verify.pbs` | Parallel post-hoc exact sympy verification of model-found candidates |

**Pipeline B — Hybrid extension and direct construction** (works for ALL n):

| Script | Purpose |
|---|---|
| `step4_extend.pbs` | Sequential cascade extension n=12 → 20 (unfiltered; finds balanced types) |
| `step5_coverage.pbs` | Comprehensive coverage: phase A constructs one supersolvable per (n, d1, d2) cell sequentially; phase B runs targeted Δb2 extension in parallel for every cell, level by level, with race-free intermediate files |

**Recommended HPC workflow** for new runs (skip the slow training entirely):

```bash
qsub pbs/step5_coverage.pbs    # ~24-72h: complete coverage table for n=6..20
```

If you want to also collect the unfiltered cascade discoveries (which produce many balanced examples per cell):

```bash
qsub pbs/step5_coverage.pbs    # first
qsub pbs/step4_extend.pbs      # then, using step5's discoveries.json as input
```

Race-condition safety in `step5_coverage.pbs`: parallel jobs at each level only **read** from a frozen snapshot `coverage_intermediate/seeds_n${N_FROM}.json` and **write** to disjoint per-target output files `coverage_intermediate/results_n${N_TO}_d${D1}_${D2}.json`. After each level completes, a sequential merge step calls `log_discoveries` (which deduplicates by canonical key) to fold the results into the central `discoveries.json` before moving to the next level.

All scripts use `PYTHONUNBUFFERED=1` for real-time log monitoring:

```bash
tail -f logs/train.log              # monitor training
tail -f logs/explore_*.log          # monitor RL exploration
tail -f logs/verify_*.log           # monitor RL verification
tail -f logs/coverage_n*.log        # monitor targeted-extension coverage jobs
tail -f logs/extend_n*.log          # monitor unfiltered cascade
```

## Key Invariants

- **b2(A)** = sum of (m_p - 1) over all intersection points p
- **Candidate exponents** (d1, d2): d1 + d2 = n - 1, d1 * d2 = b2 - (n - 1)
- **Pencil**: all lines concurrent (trivially free, penalized)
- **Multiplicity profile**: sorted list of intersection point multiplicities


## Citation
```
@misc{silva2026semicontinuousrelaxationsaitoscriterion,
      title={A semicontinuous relaxation of Saito's criterion and freeness as angular minimization}, 
      author={Tomás S. R. Silva},
      year={2026},
      eprint={2604.02995},
      archivePrefix={arXiv},
      primaryClass={math.AG},
      url={https://arxiv.org/abs/2604.02995}, 
}
```
