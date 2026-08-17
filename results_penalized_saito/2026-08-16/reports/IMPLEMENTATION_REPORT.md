# Penalized Saito functional — implementation report

**Date:** 2026-08-16 · **Baseline:** git `39a26ea` · **Env:** conda `free_arr`
(Python 3.10, numpy 2.2.6, sympy 1.14.0, torch 2.11.0, scipy 1.15.3)

## 1. What was replaced and why

The production search signal was the "smooth Saito loss": the ALS-optimized
angle between Saito determinants and Q(A), evaluated over SVD null-space
bases of float derivation matrices. **In exact arithmetic that quantity is
binary.** For exact logarithmic derivations u, v of degrees d1 + d2 = n − 1,
every defining form α_i divides B(u, v) = det M(θ_E, u, v) while
deg B = n = deg Q, hence B = c·Q identically, with c ≠ 0 possible only in
the free case (Saito's criterion). So the exact angular score is 0 on free
and 1 on nonfree arrangements; the intermediate float values the old code
returned measured numerical violation of logarithmicity, SVD tolerances,
and conditioning. This is demonstrated *symbolically* in the test suite
(`test_legacy_exact_construction_is_binary_*`: on a nonfree n = 7
arrangement every exact kernel pair — basis vectors and random rational
combinations — has determinant identically 0; on the free braid every pair
gives a scalar multiple of Q). The old score was removed from every
production path and survives verbatim as `saito.legacy_invalid_angular_score`
for regression comparisons only.

## 2. The functional the code now computes

All norms are Bombieri–Weyl (BW): ‖f‖² = Σ_γ |c_γ|²/multinom(d; γ) on S_d,
component sums on E_d = S_d³; every line form is normalized to ‖α‖ = 1.
Mathematically the construction is over C; the code computes on the real
slice for the repo's rational arrangements and is dtype-generic
(complex128 supported and tested for phase invariance).

**Canonical logarithmic residual.**
ρ_{α,d}(u) = (I − Π_{α,d})(a₀f₀ + a₁f₁ + a₂f₂), with Π_{α,d} the
BW-orthogonal projector of S_d onto α·S_{d−1} (Π_{α,0} = 0), and

L_{A,d} = n^{−1/2} ⊕_i ρ_{α_i,d},  with **ker L_{A,d} = D(A)_d exactly**.

Implementation: the closed-form restriction identity — for a Hermitian-
orthonormal basis {u_i, w_i} of the algebraic kernel of α, write
θ(α)(s·u_i + t·w_i) = Σ_p λ_p s^p t^{d−p}; then
‖ρ_{α,d}(u)‖² = Σ_p |λ_p|²/binom(d, p). (Proof: [conj(a) | u_i | w_i] is
unitary; BW norms are unitary-substitution invariant; after rotating α to
x₀ the complement of x₀·S_{d−1} is spanned by the x₀-free monomials.) **No
SVD null-space basis and no rank tolerance enter the definition.** The
identity was independently re-derived and cross-checked against an explicit
projector construction (exact rational Gram/normal equations) to ≤ 5e−16
by adversarial review, and against an exact-rational reference
implementation in `tests/reference_impl.py` (agreement ~1e−18 at nontrivial
points).

**Penalized objective.** With q = Q/‖Q‖ and unit u ∈ E_{d1}, v ∈ E_{d2},
d1 + d2 = n − 1 (any such pair; candidate-exponent arithmetic is only a
cheap pre-filter, and an all-pairs envelope min over d1 ≤ d2, including
d1 = 0, is provided):

    R(u, v) = ‖L_{A,d1} u‖² + ‖L_{A,d2} v‖²
    Γ_{λ,β}(u, v) = |⟨B(u,v), q⟩|² / (‖B(u,v)‖² + λ·R(u,v)^β),  Γ := 0 on 0/0
    S_{λ,β}(A; d1, d2) = 1 − sup_{‖u‖=‖v‖=1} Γ_{λ,β}(u, v)

Baseline β = 1/2, λ = 1; both configurable; β ∈ {0.5, 0.75, 0.9} swept in
the benchmark. There is **no epsilon in the denominator**; the only internal
regularization is a linearization floor (1e−300) on R₀ inside the MM
surrogate *weights* — step proposals only, never the objective, and the
safeguard line search evaluates the exact Γ (tested; see §4).

Verified properties (test numbers refer to `tests/test_penalized_saito.py`):
bounds 0 ≤ S ≤ 1 (everywhere); S ≈ 0 exactly on free arrangements at their
exponents (≤ 3e−15 across 12 certified free benchmark items, ≤ 2e−12 at
n = 13); nonfree values strictly interior (observed band 0.04–0.24 at λ=1);
wrong exponent pair on a free arrangement bounded away from 0 (0.40 for
braid at (1,4)); λ-monotonicity pointwise and after optimization, → 1 as
λ → ∞, free stays 0 for all λ up to 1e6; permutation and rescaling
invariance ≤ 1e−9; complex-phase invariance; orthogonal-change invariance
≤ 1e−5; projective changes preserve the zero set but move positive values.
S is **upper semicontinuous** on the space of arrangements — the integrand
is jointly lower semicontinuous and the sup runs over fixed compact
spheres, so no constant-rank stratification is needed (this replaces the
old "SVD bases vary polynomially" argument, which was false).

The numerical value is `1 − Γ̂ ≥ S`: an **upper bound** produced by a
nonconvex maximizer — a search signal, never a certificate of freeness or
nonfreeness. Exact certificates (only accepted proof):
`certificates.py` finds and re-verifies det M(θ_E, θ1, θ2) = c·Q, c ≠ 0
with exact rational logarithmic θ's.

## 3. Optimizer

Multistart MM ascent on the product of unit spheres: for fixed v, the
concave tangent overestimate of R^β at the current iterate turns the
denominator into u^H D u with D = B_v^H B_v + λβR₀^{β−1} L₁^H L₁ + λσI
(σ > 0), and the surrogate |q^H B_v u|²/(u^H D u) is maximized exactly by
u* ∝ D^{−1} B_v^H q; symmetric v-step; safeguarded backtracking on the
exact Γ enforces monotonicity (adversarial instrumentation: 64 runs,
worst accepted change −8.9e−16, i.e. within the declared 1e−15 slack; the
closed form matched a generalized eigensolve to 6e−16). β = 1/2
nonsmoothness at R = 0 is handled by the MM linearization floor + safeguard
(no denominator modification). Contractions B_v, B_u are assembled
matrix-free from sparse multiplication tables per iteration (no explicit
(N_out × 3N₁ × 3N₂) tensor). Gradients (`gamma_and_grad`) are exact off
R = 0 and match central finite differences to ≤ 4.8e−12 (adversarial) /
1e−5 (suite tolerance).

Initialization: (i) top singular pairs of the q-contracted bilinear map
A_q (A_q ≠ 0 always: ⟨z·f·g, Q⟩ ≡ 0 for all f, g would force ∂Q/∂z = 0);
(ii) **kernel-pair inits** — smallest right singular vectors of L₁, L₂
(a heuristic; for a free arrangement a generic kernel pair already attains
Γ = 1). This was added after the first extension rerun exposed optimizer
gaps on free arrangements with balanced exponents (losses up to 0.33 at
n = 10–13; ≤ 2e−12 after; the pre-fix benchmark is preserved in
`benchmark_v1_pre_kernel_init/`); (iii) Sobol sphere points (scipy QMC,
random-normal fallback); (iv) warm starts. Profiles: `rl` 4×40 (reward hot
path), `search` 8×80 (pre-filter), `benchmark` 20×150.

## 4. Adversarial verification

A 5-agent adversarial review (independent probes, no repo edits) returned:
residual identity **sound** (independent projector construction, exact
sympy kernels on 7 arrangements × d = 0..4, 33/33 dimension matches,
duplicated-line and (0,0,1)-line corners, complex path); determinant map
**sound** (independent sympy determinants to 8e−16, Aq consistency,
d1 = d2 antisymmetry, d1 = 0); optimizer **sound** (surrogate inequality
0/200 failures, monotonicity, generalized-eigen agreement, R₀ ≈ 0 and
b = 0 edge cases, escape from kernel warm starts on nonfree inputs);
properties stress **pre-fix optimizer gap confirmed fixed** (the flagged
free (4,4) example now scores 1.9e−13 on all seeds) with one expected
observation: near lattice degenerations (two lines approaching
coincidence) the loss of exactly-nonfree arrangements decays ∝ ε, so **no
fixed threshold separates free from nonfree** — consistent with upper
semicontinuity and precisely why thresholds gate exact checks rather than
certify; integration review found the `combinatorial`-arm terminal leak
(fixed via `terminal_alg_bonus=False`), the zero-weight wasted evaluation
(fixed), and the `gamma_shaping`/`--gamma` mismatch (fixed).

## 5. Tests

`tests/test_penalized_saito.py`: **45 tests, all passing** (~8–30 s),
covering the 17 required categories: exact-kernel agreement; free ≈ 0;
certificate pair Γ = 1 (and residual ≤ 1e−24); nonfree strictly interior;
wrong-pair nonzero; all-pairs envelope (with d1 = 0); rescaling/complex
phase; permutation; orthogonal invariance; projective zero-set vs positive
values; λ-monotonicity (pointwise and optimized); free-for-all-λ; float64
vs exact reference (6.5e−19 at the certified pair, 1.7e−18 at a nontrivial
nonfree point — the reference uses the *projector definition*, so this
also validates the restriction identity); gradient checks; symbolic
binarity of the legacy construction; independence from the legacy SVD
machinery (poisoned-monkeypatch and `min_extra` probes). Baseline recorded
before any edit: no test suite existed; `python main.py verify` passed
(log in `baseline/baseline_checks.log`).

## 6. Validation benchmark (§7) — key numbers

Suite: 19 items — 12 free (braid A3, A2×A1, pencil, near-pencils n = 6–10,
supersolvable n = 8–12 over five exponent types; **each carries an exact
certificate generated at build time**) and 7 nonfree (4 with integer
candidate exponents, exactly verified nonfree; 3 generic without candidate
exponents, scored at prescribed pairs — the loss needs no candidate
arithmetic). Outputs in `benchmark/` (JSON/CSV + full per-restart
diagnostics: ‖B‖, |⟨B,q⟩|, ‖L₁u‖, ‖L₂v‖, denominator, projected-gradient
norm, stop reason, σ_min/cond of L, λ, β, timings), plots in `plots/`.

- **Separation at λ = 1, β = 1/2:** free ≤ 3e−15; nonfree ∈ [0.084, 0.241].
- **λ sweep:** nonfree rise monotonically 1e−4 → ~1 across λ ∈ [1e−3, 1e4];
  free stay ≤ 1e−11 everywhere (`plots/lambda_sweep.png`).
- **Degeneration path** (supersolvable n = 9, one line perturbed by t):
  the computed loss became small along the tested perturbation path
  (monotone through six decades t ∈ [1e−6, 1e−1]; no metric-distance or
  proportionality theorem is claimed);
  the legacy score on the same path is non-monotone and jumps 1.0 → 5.8e−6
  between t = 1e−2 and 1e−3 (`plots/perturbation.png`) — the promised
  "genuinely graded" behavior vs the old score's chaos.
- **Legacy failure showcase:** the free pencil scores **1.0** under the
  legacy score (worst possible) and 0.0 under the new loss; one nonfree
  n = 9 item scores 2.2e−3 legacy (would pass the old 0.05 filter) vs
  0.084 new.
- **Reference check:** float64 vs exact-rational Γ at fixed points:
  ≤ 1.7e−18.
- **Budgets:** `search`-profile losses match `benchmark`-profile to the
  reported digits on all items; restart study shows free items solved from
  1 restart (kernel init), nonfree stable spread ≤ 1e−3 by 8 restarts.
- **Cost** (single core): 0.02–0.7 s per `benchmark` evaluation for
  n = 4–12; legacy score 0.002–0.05 s (different, invalid computation);
  exact `is_free` 0.03–0.8 s in this range.

## 7. Extension pre-filter rerun + threshold refit (§8)

Labeled data: every combinatorially-admissible one-line extension of seven
seeds (braid, two exactly-nonfree n = 7 seeds, supersolvable n = 9–12) plus
80 random integer arrangements with candidate exponents (n = 7–10), each
labeled by exact `is_free`. Splits by disjoint seeds: validation 147 rows
(133 free), test 86 rows (67 free). Files in `experiments/extension/`.

- A striking structural fact: **every** admissible one-line extension of
  these free seeds is itself free (addition-theorem territory) — the
  negative class comes from the random arrangements.
- **Separation: free ≤ 7.4e−13 vs nonfree ≥ 3.8e−2 — eleven orders of
  magnitude.**
- **Refit threshold (validation rule: 2 × max free loss): τ = 1.47e−12.**
  Test metrics at τ: precision 1.0, recall 1.0, precision-at-k 1.0,
  22% of exact checks avoided on this mix. The legacy score at its old
  0.05 default false-passes 4/33 nonfree candidates (precision 0.971).
- **Shipped default** `--loss-threshold 1e-6` (saito.py, main.py, pbs
  step4/step5): recall-first — 6 orders above observed free noise
  (robust to optimizer degradation at larger n), 4 orders below every
  observed nonfree value. The strict refit value and the full
  threshold-metric table are in `extension_report.json`.
- **Exact certification rate 1.0**: all 200 free extensions/arrangements
  certified (`extension_certificates.json`).
- Cost per candidate: new filter 0.37 s vs exact 0.65 s vs legacy 0.033 s.
  The filter's value grows with n as the exact check gets more expensive.

## 8. RL reward-arm comparison (§8) — smoke scale

Design: 6 arms (`penalized`, `potential`, `combinatorial`, `terminal`,
`random`, `legacy`) × 5 seeds × 30,000 env steps at n = 9, coord range 3,
identical budgets, each run in an isolated working directory (no writes to
the repo's live `discoveries.json`); every logged discovery re-verified
with an exact symbolic certificate. Results in `experiments/rl/`
(`rl_comparison_summary.json` per arm + plots). **This is a smoke-scale
comparison** — the full-scale retraining (n up to 20, 5M steps, 16 cores,
est. 90–110 h) is queued with exact commands in REPRODUCE.md.

Results (mean exactly-certified discoveries per seed; all discoveries carry
exact symbolic certificates re-verified from scratch):

| arm | certified mean | per-seed | hit rate | distinct mult-profiles | s/env-step |
|---|---|---|---|---|---|
| legacy | **102.4** | 65, 89, 267, 10, 81 | 5/5 | 3–4 | 0.0267 |
| penalized | **28.2** | 10, 11, 91, 5, 24 | 5/5 | 2–4 | 0.0275 |
| combinatorial | 2.8 | 1, 0, 9, 1, 3 | 4/5 | ≤3 | 0.0259 |
| terminal | 0.4 | 0, 2, 0, 0, 0 | 1/5 | ≤1 | 0.0241 |
| random | 0.2 | 0, 1, 0, 0, 0 | 1/5 | ≤1 | 0.0241 |
| potential | 0.0 | 0, 0, 0, 0, 0 | 0/5 | 0 | 0.0264 |

Honest reading of the smoke-scale numbers:

1. **Score-guided composite arms dominate score-free arms** by 1–2 orders of
   magnitude in certified discoveries — the Tier-2 algebraic signal is what
   drives discovery at this scale; binary terminal reward and unguided search
   almost never hit.
2. **The legacy composite beats the corrected composite on RAW counts at
   n = 9** (102.4 vs 28.2). This is reported without varnish, with two
   caveats that matter: (a) at n ≤ 12 both arms share the exact terminal
   bonus, so the invalid score's role here is per-step shaping, where its
   systematic false-optimism (near-zero "loss" on many nonfree states)
   behaves like an optimistic exploration bonus on b2-feasible states;
   (b) **distinct multiplicity-profile diversity is nearly identical (3–4 vs
   2–4)** — the raw-count gap is coordinate-level volume, not combinatorial
   diversity. The regime where the legacy score is provably uninformative
   (n > 12: binary in exact arithmetic, no exact bonus available) is exactly
   the regime this smoke test cannot reach; the queued full-scale runs
   (REPRODUCE.md) are the decisive comparison.
3. **The potential arm found nothing.** Two identified causes: the
   `_potential()` implementation ignored the episode's target exponents
   (fixed in the follow-up work: environment.py `_potential` now passes
   `target_exponents`), and pure potential shaping (|r| ≤ ~1) is dominated
   by the −5 pencil penalty while lacking the composite's combinatorial
   scaffolding. A re-run with the fix is part of the swap-search follow-up.
4. Per-step cost differences across arms are within ~14% (0.024–0.028 s);
   the corrected loss with caching adds ~4% over the legacy arm at this n.

Full per-seed data: `experiments/rl/<arm>/rl_comparison.json`; discovery
certificates: `experiments/rl/<arm>/certs_<arm>_seed<k>.json`; plots:
`plots/rl_certified_by_arm.png`, `plots/rl_discovery_curves.png`.

## 9. Honest limitations

1. The numerical loss is an upper bound from a nonconvex maximizer; rare
   optimizer gaps on free arrangements were observed pre-kernel-init and
   cannot be excluded at larger n — which is exactly why the pipeline never
   treats low loss as proof and never drops candidates on loss alone at
   recall-critical thresholds.
2. Near lattice degenerations, exactly-nonfree arrangements can have
   arbitrarily small loss (∝ distance to the degenerate locus). Expected
   from upper semicontinuity; documented; the reason thresholds gate exact
   checks only.
3. The β = 1/2 baseline is nonsmooth at R = 0; the MM/safeguard handles it,
   but β sweeps show β ∈ {0.75, 0.9} give smoother (slightly less sharply
   separated) landscapes — recorded in `beta_sweep.json`.
4. The RL comparison in-session is at smoke scale (n = 9); n ≥ 14 claims
   about RL viability with the corrected reward remain open until the
   queued full-scale run.
5. Historical discovery *counts* stand on their exact certificates, but
   any historical claim about the old score's *values* (timings,
   thresholds, reward quality) is withdrawn (see CLAIMS_AUDIT.md).
