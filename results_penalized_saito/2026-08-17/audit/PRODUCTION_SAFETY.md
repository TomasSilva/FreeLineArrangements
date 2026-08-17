# Production-safety pass — final report (2026-08-17)

Follow-up to AUDIT_REPORT.md; no functional rewrite. Functional version
2.2.0. Tests: 88 passing (`python -m pytest tests/ -q`). Studies:
`python benchmarks/production_safety_studies.py --out results_penalized_saito/2026-08-17/audit/safety`.

## 1–2. Explicit exponent-pair policy; d1 ≥ 2 nontrivial

The library keeps full d1 = 0 capability (`admissible_degree_pairs`,
evaluator, envelope). Campaigns now enforce an explicit policy:
`run_swap_campaign.py` classifies every cell (`nontrivial` d1 ≥ 2 /
`baseline_near_pencil` d1 = 1 / `baseline_pencil` d1 = 0), refuses baseline
classes without `--allow-baseline`, and records `pair_class` +
`allowed_pairs_policy` in the manifest. The PBS job generator emits only
d1 ≥ 2 cells; baseline classes never count toward nontrivial discovery
totals (report generator labels `pair_class` per cell).

## 3. Raw Gamma + calibrated clipping

SUPERSEDED by the final correction pass (functional version 2.3.0):
`gamma()` now returns a structured result (gamma_raw, gamma_bounded,
numerical_status, clip_applied, clip_excess, error_tolerance,
diagnostic_message).  The tolerance is dtype/dimension/scale-aware
(64·max(64, N_out+dim_u+dim_v)·eps — ~7e-12 at typical sizes, never a fixed
1e-9 "roundoff" constant); roundoff-sized violations are clipped into
[0, 1] and counted (ROUNDING_CLIPPED); larger violations are retried on a
compensated exactly-summed path (RETRY_OK) and otherwise rejected as
NUMERICAL_ERROR — the public loss never returns a value outside [0, 1]
and the search layer receives the pessimistic 1.0 with a warning instead
of any invalid score.  Tests: test_gamma_raw_recorded_and_clip_counted,
test_two_ulp_excess_is_logged_and_safely_clipped,
test_substantial_gamma_violation_is_numerical_error,
test_no_invalid_score_reaches_search_layer,
test_compensated_retry_path_can_succeed,
test_gamma_diagnostics_survive_serialization.

## 4. Clipping is not manufacturing the exact-zero free losses

Study A (`safety/clip_verification.json`), 12 exact-certified free items,
per-item raw Γ, ‖B‖, alignment |⟨B,q⟩|, and both logarithmic residuals:

- max raw excess over 1: **4.44e-16 (one ulp)**; max raw deficit: **0.0**;
- 10/12 items clipped exactly at that one-ulp scale; max clip excess seen
  4.44e-16;
- residuals at the optima are ~1e-16 (R ~ 1e-32) and ‖B‖ = |⟨B,q⟩| to
  machine precision.

Reading: unclipped free losses would be within ±4.4e-16 of zero; the clip
symmetrizes one-ulp roundoff and cannot create zeros that are not already
there to machine precision. Raw values ship in every diagnostics dict.

## 5. The 1e-6 prefilter is a heuristic — measured at n ≥ 14

Wording updated in `saito.py` (and README): the gate trades exact-check
work against recall and certifies nothing; **17/17 observed recall on this
benchmark** — the prefilter remains a heuristic with possible false
negatives from optimizer failure on unseen instances. Study B
(`safety/prefilter_recall.json`): 17 exactly-free supersolvable items
covering **every** target pair d1 = 2..⌊(n−1)/2⌋ at n = 14, 15, 16, each
under: base `search` budget, a unitary variant, two ill-conditioned integer
GL variants (cond ≈ 50 and ≈ 500), and deliberately low budgets (4×40 and
2×20). Exactly-nonfree perturbations (t = 1e-4, 1e-8 of one line) are
labeled by the **exact negative certificate** (34/34 labeled; zero
unresolved).

| condition | recall@1e-6 (free) | max free loss |
|---|---|---|
| base search (8×80) | 1.00 | 0.0 |
| unitary variant | 1.00 | 1.1e-16 |
| GL cond ≈ 50 | 1.00 | 7.8e-16 |
| GL cond ≈ 500 | 1.00 | **2.4e-7** |
| low budget 4×40 | 1.00 | 0.0 |
| very low 2×20 | 1.00 | 0.0 |

| perturbation | nonfree pass rate @1e-6 | min nonfree loss |
|---|---|---|
| t = 1e-4 | 0.18 | 3.5e-7 |
| t = 1e-8 | 1.00 | 3.9e-13 |

Two honest observations: (i) at cond ≈ 500 the free losses inflate to
within ~4× of the gate — heavily ill-conditioned coordinates erode the
margin, so keep coordinates low-height (the campaigns do) or raise τ for
such inputs; (ii) near-degenerate exactly-nonfree arrangements **pass the
gate** (the computed loss became small along the tested perturbation
paths; no metric-distance or proportionality theorem is claimed) and are eliminated by
mandatory exact certification — a cost, never a soundness issue.

## 6. β = 0.5 vs β = 0.75 action-ranking agreement

Study C (`safety/beta_ranking_agreement.json`): complete one-line
replacement neighborhoods (all removals × the full coord-range-2 candidate
pool, validity-filtered) at three states — perturbed double-pencils in
(13,6,6) and (14,6,7) and a random valid n = 13 state — losses under both
betas at fixed budget (rl 4×40) and seed, real field:

| state | actions | Spearman ρ | Kendall τ | top-20 overlap | best action same |
|---|---|---|---|---|---|
| perturbed double-pencil (13,6,6) | 481 | 0.986 | 0.901 | 0.70 | no |
| perturbed double-pencil (14,6,7) | 518 | 0.941 | 0.802 | 0.85 | yes |
| random valid (13,6,6) | 598 | 0.984 | 0.896 | 0.70 | no |

Reading: the two beta values produced strongly correlated rankings in the
three tested neighborhoods (ρ ≥ 0.94, τ ≥ 0.80, top-20 overlap ≥ 0.70),
while the best action differed in two of the three cases. No multi-step RL
behavior is inferred from these single-state rankings — that would require
campaign-level ablations. Threshold labels alone would have hidden both the
agreement and the residual disagreement, which is why both are reported.

## 7. Deterministic repeated evaluation + provenance

Repeated RL/potential evaluations are deterministic: exact canonical-
line-set cache plus fixed evaluator seeds (bitwise-equal repeats tested,
cached and uncached: `test_repeated_rl_evaluation_deterministic`).
`penalized_saito.runtime_provenance()` records functional version, code
commit, dirty-tree state, λ, β, field, clip tolerance, MM floor, and
profile budgets; campaign manifests embed it plus seeds, target pair, and
budgets. `maximize()` output now carries `functional_version`.

## 8. Historical β labeling

`make_swap_report.py` reads β from each unit manifest; units whose
manifests predate the audit patch are labeled `unknown_pre_audit` and their
**numerical loss statistics are never aggregated** with revised units
(`beta_groups` per cell in the report/CSV). Certified counts and distinct-
lattice counts are β-independent (exact certificates) and aggregate freely.
Session provenance indicates the pre-audit units ran at β = 0.5 (module
default bound at process start), but absent an in-manifest record they stay
labeled unknown. Exact discovery SETS may be deduplicated across historical
β values (certificates are β-independent), but discovery yield, rate, and
search productivity are never compared or aggregated across mixed-β
policies.

## 9. How "exactly verified nonfree" is certified

Documented in `certificates.py` (module docstring): the negative is a
**proof**, not a search failure. For exact kernel bases {v_i}, {w_j}, every
determinant of degree-(d1, d2) logarithmic derivations equals c(a, b)·Q
with c(a, b) = aᵀC b bilinear, C_ij = c(v_i, w_j). If every basis pair has
c(v_i, w_j) = 0 then C = 0 and no Saito basis with these degrees exists;
conversely a free arrangement always has a basis pair with c ≠ 0. Each
c(v_i, w_j) is decided exactly (point evaluation at a rational point where
Q ≠ 0, or full symbolic expansion in `is_free`). Candidate-exponent
failure separately excludes freeness via Terao factorization. Benchmark
"nonfree" labels come from these exact criteria; Study B's perturbed items
were 34/34 exactly labeled.

## 10. Tiny-loss nonfree ⇒ no reward, no discovery (integrated test)

`test_tiny_loss_nonfree_*`: a 1e-8 perturbation of a certified-free
supersolvable has loss < 1e-6 (passes the heuristic gate) and an **exact**
nonfree certificate for the target pair; `certify_state` returns None (no
terminal reward path), the campaign IO logs it only as a candidate,
writes nothing to `certified.jsonl`, and the repo-root `discoveries.json`
is untouched (the swap pipeline never writes it).

## 11. discoveries entry requirements

`verify_certificate` now additionally enforces pairwise-distinct projective
lines and d1 + d2 = n − 1 (with d1 ≥ 0) on top of the existing exact
logarithmicity of both derivations (coefficients in the S_{d1}/S_{d2}
bases, bounding the stated degrees) and det M(θ_E, θ1, θ2) = c·Q with
exact c ≠ 0. Tampering tests: duplicated line, wrong degree sum, and c = 0
all fail verification; JSON round-trips verify.

## 12. Regularity documentation

Module docstring and README now state explicitly: upper semicontinuity is
a property of the **ideal** loss S; the finite-multistart evaluator returns
an upper approximation Ŝ ≥ S whose value depends on initialization and
budget, and Ŝ need not inherit any semicontinuity. No global-supremum claim
is made for the multistart; no freeness or nonfreeness claim is ever made
from a floating-point loss.
