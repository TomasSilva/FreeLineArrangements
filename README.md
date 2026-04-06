# FreeLineArrangements

Reinforcement learning system for discovering **free line arrangements** in the complex projective plane CP², using a Transformer-based actor-critic trained with PPO. Designed to run on HPC clusters with parallel environments.

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
main.py              CLI entry point (train, search, explore, verify, verify-found)
arrangement.py       Core math: ProjectiveLine, LineArrangement, intersection lattice, exact Saito check
saito.py             Smooth Saito loss (ALS), reward shaping, combinatorial/algebraic scores
environment.py       Gym-like RL environment with pool and singularity-aware candidate modes
model.py             Transformer Actor-Critic with cross-attention over candidate lines
train.py             PPO training with adaptive triple curriculum and vectorized environments
vec_env.py           Subprocess-based parallel environment (one worker per CPU core)
discoveries.py       Persistent JSON log with deduplication
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
pip install torch numpy sympy
```

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

### View Discoveries

```bash
python main.py discoveries
```

## HPC Deployment

PBS job scripts are in `pbs/`:

| Script | Purpose |
|---|---|
| `step1_train.pbs` | Curriculum training n=6 to 20 with 16 parallel environments |
| `step2_explore.pbs` | Parallel exponent-targeted exploration (all (d1, d2) pairs per n) |
| `step3_verify.pbs` | Parallel exact Saito verification per exponent type |

Steps 2 and 3 run up to 16 jobs in parallel via `xargs -P 16`, each targeting a specific exponent type with `--target-exponents D1 D2`. All scripts use `PYTHONUNBUFFERED=1` for real-time log monitoring:

```bash
tail -f logs/train.log           # monitor training
tail -f logs/explore_*.log       # monitor exploration
tail -f logs/verify_*.log        # monitor verification
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
