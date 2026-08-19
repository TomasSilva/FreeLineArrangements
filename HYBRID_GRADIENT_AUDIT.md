# Hybrid gradient strategy — audit and experimental prototype

Date: 2026-08-18 · scope: read-only audit (Phase A) + isolated flagged
prototype (Phase B) · production campaigns untouched (separate output
directory `results_local/hybrid_gradient_experiment/`, per-run cache
clearing, no writes to any production store).

## Question 1 — is the hybrid strategy already implemented?

**Overall: PARTIAL.** Per component (evidence from source, not names):

| component | status | evidence |
|---|---|---|
| Discrete replacements A' = (A \ {L-}) ∪ {L+} | **YES** | `swap_search.py:175` `propose_swaps` (Δb2-tiered), consumed by `greedy_search`/`simulated_annealing`/`map_elites`; fixed cardinality enforced by construction |
| Fixed prescribed pair (1, d1, d2) per campaign | **YES** | `experiments/run_swap_campaign.py` requires `--d1/--d2`; `ChainEvaluator(n, d1, d2)` (`swap_search.py:240`) pins the pair for every evaluation. The all-pairs envelope exists only as a library function (`penalized_saito.py: penalized_saito_loss_all_pairs`) and is not used by any campaign |
| Raw compact penalized Saito functional as search energy | **YES** (with one declared addition) | `ChainEvaluator.screen_loss` (`swap_search.py:262`) is the raw cached loss; `energy` (`:290`) = raw loss + `w_b2 * |b2-b2*|/b2*` (combinatorial shell pull, components logged separately at `:296`; certification gate uses raw loss only). Calibration exists (`calibration.py`) but defaults off (`tau=None`) and never enters engines |
| Continuous gradient refinement of LINE COEFFICIENTS | **NO** | The only analytic gradient is w.r.t. (u, v): `penalized_saito.py:870` `gamma_and_grad` ("Real dtype only", raises `NotImplementedError` for complex at `:878`); grep confirms no code differentiates Γ w.r.t. any line coefficient. The float→exact "polish" (`saito.py:1067ff` `limit_denominator` sweeps) is rounding, not gradient descent. PPO backprop (grow-RL arm) updates policy weights, not coefficients |
| Alternating refinement of (u, v) and the arrangement | **NO (mechanism exists, unused)** | `ChainEvaluator.refined_loss` (`swap_search.py:279`) carries a warm (u, v) across evaluations, but has **zero call sites** — engines use `screen_loss` only, so (u, v) are re-solved fresh per state and no coefficient-side step exists |
| Exact Saito certification of every accepted discovery | **YES** | `swap_search.py:315` `certify_state` = `find_certificate_fast` + independent `verify_certificate`; `CampaignIO` certifies once per lattice; `promotion.py` re-verifies before the store |
| Exact rejection/classification of trivial classes | **YES** | `swap_search.py:80` `is_valid_state`: duplicate lines via canonical coords set, essential rank 3 (`novelty.py:142`, exact over K), `m_max <= n-2` excludes pencils and near-pencils; campaign policy `d1 >= 2` (`run_swap_campaign.py`, `--allow-baseline` gate); supersolvability exact via the modular-point criterion (`novelty.py:169`) |

Other audit items:

- **Engines using the penalized loss**: greedy, annealing, MAP-Elites, and
  the RL swap environment (`swap_env.py`, raw loss with optional labeled
  calibration) all rank by it; no continuous optimizer over coefficients
  existed before this prototype.
- **Line parameterization**: exact sympy `Rational` / `quadfield.QuadElem`
  triples, canonical projective normalization by first nonzero coordinate
  (`arrangement.py:29-33`); the numerical evaluator renormalizes rows to
  unit Hermitian norm (`penalized_saito.py` constructor).
- **Complex support**: the evaluator is Hermitian/complex128-capable
  end-to-end (Stage B/C included), but `gamma_and_grad` and
  `projected_grad_norm` are real-only (`:878`, `:913`). **The prototype
  therefore runs on the real slice; this limitation is stated, not
  worked around.**
- **Freeness classification**: `certificates.classify_freeness` with exact
  statuses (FREE_TARGET / NOT_TARGET_FREE / GLOBALLY_NONFREE / UNRESOLVED);
  supersolvability exact; inductive freeness via the bounded memoized
  deletion recursion `experiments/triage_swap.py:54` — its
  `not_inductively_free` verdicts are produced only when the exhaustion
  over the addition-deletion condition completes, `timeout` otherwise
  (maps onto NOT_INDUCTIVELY_FREE_CERTIFIED / UNKNOWN; positive verdicts
  currently do not persist the deletion-chain witness — noted as a gap).
  **Recursive freeness is not implemented anywhere** — all recursive
  statuses are UNKNOWN.
- **`discoveries.json`**: verified-exact-only. `discoveries.DEFAULT_PATH`
  redirects legacy writers to a staging file; the store is written only by
  `promotion.promote`, which re-verifies every certificate; the strict
  loader re-verifies again on load.

**Data flow (production):** CLI campaign (pair policy gate) → engines
(`swap_search`) rank swaps by raw penalized loss + b2-shell → candidates
below 1e-6 → `certify_state` (exact; sound negatives incl. mod-p block
prescreen) → per-run `certified.jsonl` → `promotion` (re-verify, atomic,
idempotent) → `discoveries.json`.

## Question 2 — can a small isolated prototype improve performance?

**Prototype built** (`hybrid_refine.py`, used only by
`experiments/hybrid_smoke.py`; no production default changed, no engine
imports it):

1. after each accepted discrete swap, (u, v) are optimized by the
   existing MM optimizer;
2. holding (u, v) fixed, projected/Riemannian gradient ascent on Γ with
   respect to the swapped-in line's coefficients — a torch float64
   re-implementation of Γ(a) that mirrors the evaluator's conventions
   exactly (unit lines, BW-unit q, sw-scaled restriction rows, 1/sqrt(n));
   the construction is validated against the production evaluator at the
   start point on every call (tolerance 1e-7; measured ~1e-10);
3. sphere projection removes the radial direction (real slice: the only
   gauge is the sign, absorbed by normalization — complex phase removal
   is N/A and stated as such);
4. collision guard: |cos angle| > 1 - 1e-9 against any other line rejects
   the step (structured status `line_collision`);
5. (u, v) are re-optimized with the previous pair as a warm start;
6. acceptance authority is EXACT: the float endpoint is rationalized
   (small-denominator sweep), each exact candidate is re-scored by the
   production evaluator, and the update is accepted only if the raw loss
   STRICTLY improves and `is_valid_state` (nontrivial, essential,
   duplicate-free) passes.  The mathematical denominator carries no
   epsilon; failures are structured statuses, never loss values; Γ > 1
   excursions along the float path are counted and reported, never
   clipped.

Gradient validation (tests/test_hybrid_refine.py, 10 tests):

- centered finite differences agree with autograd to max rel. error
  < 5e-5 (measured ~1e-8) on nonfree states;
- FD is documented invalid ON the free locus: at R → 0 the β = 0.75
  penalty has R^(β-1) → ∞ (the paper's β < 1 nonsmoothness) — observed
  and encoded in the test, not hidden;
- torch Γ matches the production evaluator to < 1e-9;
- scale/sign and line-permutation invariance to 1e-9;
- braid seeded with its exact certificate pair: Γ = 1 (raw loss 0) to
  1e-9; wrong pair and generic nonfree: strictly positive;
- structured statuses exercised (nonfinite gradients, collisions,
  no-improvement, rationalization failure);
- recovery: a perturbed braid line is pulled back to a CERTIFIED free
  configuration by refinement alone.

## Smoke experiment

`results_local/hybrid_gradient_experiment/`: n = 14, pairs
(2,11), (4,9), (6,7), seeds {0,1,2}, two arms (baseline / refine),
equal budget of 180 production Saito evaluations per run (refinement's
exact re-scoring evaluations are charged against the same budget; wall
clock reported separately), identical seeds and initial states, per-run
cache clearing.  Plus the n=14 recovery test on a verified lift-seed
discovery with one line perturbed.  Metrics, candidates (with the
four-level novelty labels; every certificate independently re-verified),
failures, manifest with commit + source-content hash, and plots live in
that directory.  RESULTS: see `metrics.json` / `RESULTS.md` there (filled
by the run; summarized in the final response of the session).

## Limitations (mathematical and numerical)

- Real slice only (complex Wirtinger refinement not implemented — the
  production analytic (u,v) gradient is also real-only).
- Γ is nonsmooth exactly on the zero locus for β < 1; the ascent works in
  the smooth region and hands off to exact rationalization + raw
  re-scoring, so the nonsmoothness affects step efficiency near
  convergence, never correctness.
- The frozen-pivot kernel frame is smooth locally; a very long ascent path
  crossing pivot degeneracy would need re-framing (paths here are short).
- Rationalization is a small-denominator snap; a refined float optimum
  with no nearby low-height rational representative is reported as
  `rationalization_failed` (counted).
- No claim of a canonical gradient flow to the free locus — this is
  heuristic local refinement of the penalized Saito score.

## Recommendation

**Experimental opt-in.**  Keep production engines unchanged.  The
refinement preserves exact-certification integrity by construction
(acceptance = exact re-scoring + the existing validity gates + the
existing certificate pipeline), its gradients are validated, and its
failure modes are structured and counted.  Whether it earns a production
flag should follow the equal-budget smoke metrics (see acceptance
criteria in the task): enable only if it wins on certificates / unique
lattices / time-to-first-certificate at equal evaluation budgets.
