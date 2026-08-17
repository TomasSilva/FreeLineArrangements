# Final production audit — penalized Saito RL system (2026-08-17)

**1. Commits.** Start: `b889a91`. End: see the closing commit of this pass
(functional version 2.4.0). **2. Tree state:** committed clean at the end;
`source_content_hash` embedded in every provenance block and cache key
(commit + dirty flag alone is no longer the identity).

**3. Modified files.** `penalized_saito.py` (staged evaluation, identity
hash, cache lock, d=0 in the definition, tolerance wording),
`saito.py` (error propagation, Tier-2 structured deferral + counter),
`swap_search.py` (per-proposal error skipping + counters, energy-component
logging), `swap_env.py` (structured error containment, terminal Φ=0,
`numerical_failure_penalty`), `calibration.py` (validation, cohort
fingerprint, `CalibrationError`), `promotion.py` (dir-fsync durability,
strict loader, legacy migration), `discoveries.py` (staging redirect),
`novelty.py` (corpus paths), `experiments/train_swap_policy.py` (resume
guard), `experiments/run_swap_campaign.py` (engine-failure resilience),
`docs/hard_penalty_limit_proof.md` (8 corrections), tests (3 files
updated, `tests/test_final_audit.py` new), `benchmarks/production_smoke.py`
new, `.gitignore`.

## 4. Production call graph (audited)

`run_swap_campaign.py` CLI (pair policy gate: nontrivial d1≥2, baseline
needs `--allow-baseline`; manifest embeds λ/β/field + `runtime_provenance`)
→ engines in `swap_search.py` (greedy/anneal/MAP-Elites; **raw** loss +
b2-shell energy with separately logged components; per-proposal
`screen_loss_or_none` skips numerical failures and counts them) →
`cached_penalized_loss` (full-identity key; errors propagate, never cached)
→ `PenalizedSaitoEvaluator.maximize` → `gamma()` staged A/B/C → raw loss →
[RL arm only: `SwapArrangementEnv` — Φ = 1−raw by default; calibrated Φ_τ
only with an explicit frozen τ; terminal Φ:=0; certification gate on RAW
loss] → candidates.jsonl (numerically promising) → `certify_state` (exact
only) → certified.jsonl staging → `promotion.promote` (re-verify + atomic +
locked + dir-fsync) → `discoveries.json` (verified-exact-only store).

## 5. Repository searches and findings (all resolved)

| search | finding | resolution |
|---|---|---|
| NUMERICAL_ERROR → numeric loss | `saito_loss` mapped errors to 1.0 with a warning | removed; `GammaNumericalError` propagates; consumers handle structurally |
| warning-only invalid Γ | same site | replaced by staged retry + structured status |
| float-only APIs discarding status | `saito_loss`/`screen_loss`/`_raw_loss` | still float-typed but raise on error (status never silently dropped); env/engine catch |
| caching failed evaluations | not occurring (raise precedes insert) | regression-tested (`test_errors_never_cached_or_in_tau`) |
| calibration cohorts with failures | `except: continue` silently | failures excluded AND counted; cohort validated; `CalibrationError` on insufficiency |
| legacy score in production | docstrings/legacy body only | confirmed clean |
| direct `discoveries.json` writes | grow-RL `log_discovery` default path | `discoveries.DEFAULT_PATH` → `discoveries_staging.json`; verified store written only by the promoter |
| consumers accepting `legacy_unverified…` | records lived inside discoveries.json | migrated out (below); strict loader rejects |
| cache keys missing identity | no source/dependency hash | `_identity_hash()` (source sha256 + dependency versions) in loss and τ keys |
| terminal potential | truncation added γΦ(terminal) | replaced by the absorbing convention Φ(terminal)=0 |
| hard-coded clip tolerances | none remaining | scale-aware engineering-policy tolerance, documented as such |
| "exact summation" for fsum | 2 docstring sites | renamed "compensated/high-accuracy floating summation" |

## Staged Γ evaluation (Part 3)

Stage A: stable float64 (normalized contractions, `np.linalg.norm` scaled
sum-of-squares, no normal equations; raw components retained). Stage B:
compensated floating summation (`math.fsum`) — explicitly not exact, not
extended precision. Stage C: **arbitrary-precision mpmath rebuild of the
complete objective** (lines renormalized in mp; L₁u/L₂v via the restriction
identity in mp; q_A from the mp product of lines; B(u,v) via the exact
multiplication tables in mp; ⟨B,q⟩, ‖B‖², R, R^β, denominator, Γ) at 80
then 160 digits — never a re-division of corrupted float64 intermediates;
real and complex conventions supported; precision and before/after values
recorded in the diagnostics. Outcomes: in-tolerance → RETRY_OK with the
verified bounded value; else NUMERICAL_ERROR. Demonstrated: a corrupted
float64 q is *repaired* by Stage C
(`test_arbitrary_precision_retry_repairs_corrupted_float_state`), and with
mp disabled the same corruption terminates as NUMERICAL_ERROR carrying no
loss.

## 6. Tests

```
python -m pytest tests/ -q     → 124 passed (~26 s)
```
(111 pre-audit + 13 new/updated: staged retry semantics, env containment,
raw/calibrated separation, terminal convention, resume guard, migration,
strict loader, identity-hash cache miss, error-never-cached/never-in-τ.)

## 7. Smoke tests (all pass; artifacts under `final_audit/smokes/`)

| smoke | artifact | result |
|---|---|---|
| A evaluator statuses | `smoke_A.json` | OK / ROUNDING_CLIPPED(1-ulp) / RETRY_OK (mp repair, message + retries recorded) / NUMERICAL_ERROR (γ=None); no negative loss; cache unchanged by errors |
| B verified pipeline | `smoke_B.json` | n=14 (6,7) exact certificate → promoted once → reloaded via strict loader with re-verification → exactly one canonical entry; tiny-loss exactly-not-target-free has NO certificate to promote (`not_target_free`) |
| C legacy migration | below | dry-run then real |
| D deterministic calibrated reward | `smoke_D.json` | two identical repeats (status, raw, calibrated, τ, reward); Φ matches the formula; cohort fingerprint has 0 numerical errors |
| E n=14 (6,7) swap rollout | `smoke_E.json` | nontrivial pair policy; constant cardinality; raw/calibrated logged separately (τ=None default); 0 numerical errors; no discoveries written; ~1 s |
| F hard-penalty evaluator | `smoke_F.json` | inverse-linearity 1/Γ_λ = ‖B‖²/N + (R^β/N)λ to max rel. dev < 1e-9 over 12 decades; common-pool monotonicity violation ≤ 1e-15 |
| G ablation labeling | `audit/calibration_ablation/ablation_summary.json` | relaunched after a launcher `pgrep` portability bug (it had never started); labeled SMOKE-SCALE IMPLEMENTATION TEST, no RL-performance conclusion; the n=14 calibrated-wiring check is covered by smokes D/E |

## 8. Numerical-error statistics

Across the full suite + smokes: forced-error paths behave as designed
(errors counted per evaluator: `numerical_error_count`, `retry_count`;
Tier-2 deferrals counted in `saito._TIER2_NUMERICAL_ERRORS`; engines report
`numerical_errors` in `ChainEvaluator.stats()`). In natural (unforced)
evaluations across all smokes: **0 numerical errors** — the staged path
exists as a guard, not as a hot path.

## 9. Legacy migration (Smoke C, real store)

Dry run validated first (identical partition). Real run:
source checksum `8a8f349d…a765ed2b`, 120,027 records → **0 verified /
120,027 legacy / 0 malformed**; backup preserved at
`discoveries.json.backup.8a8f349d20b4afc6` (145.4 MB); legacy candidates at
`legacy_candidates.json` (115.4 MB, each with original index, migration
timestamp, reason `legacy_unverified_by_promoter`, source checksum);
post-migration `discoveries.json` = verified-exact-only store (currently 0
entries — honest; legacy records are candidates, not discoveries, and may
re-enter only through the certificate-checked promoter); strict loader: 0
rejects; corpus readers (seeding/screening only) unaffected (134,015
records reachable); restart is a no-op. **No data loss: 120,027 =
0 + 120,027 + 0, plus the checksummed backup.**

## 10. Sample verified entry / 11. sample quarantined record

Verified (from smoke B's temp store; schema-2.0): discovery_id
`sha256(sorted canonical rational lines)`, n=14, exponents [1,6,7],
`verification_status: verified_exact`, full exact certificate + hash,
pair_class `nontrivial`, λ/β/field, functional version, commit + source
hash. Quarantined legacy example: original record intact +
`_migration: {timestamp, original_index: 0, reason:
legacy_unverified_by_promoter, source_checksum}`.

## 12. τ cohort fingerprint

Smoke D at (9,3,5), rl profile: τ frozen from the generic median with
fingerprint {sampler_version `generic_random_valid_v1`, dataset hash over
the exact sampled arrangements, n_valid=10, sampler seed, identity hash,
dtype float64, normalization convention, **n_numerical_errors=0**}.
Validation enforced: finite, 0<τ≤1, sufficient cohort, no error values —
`CalibrationError` otherwise (tested).

## 13. n≥14 rollout summary

Smoke E: SwapArrangementEnv(n=14, target (6,7), 4 replacement steps,
default τ=None): all transitions valid fixed-cardinality replacements;
raw loss logged per step with `calibrated_potential = 1 − raw` (raw mode);
certification attempted only through the raw-loss gate; nothing written to
any discovery store; 0 numerical errors.

## 14. Remaining limitations (honest)

- The scale-aware Γ tolerance is an engineering policy, not a derived
  forward-error bound (documented as such).
- Stage C is minutes-slow at large n if it ever fires; it is a guard.
- The verified store starts empty post-migration: historical results are
  candidates until re-promoted through the pipeline (a gradual
  reverification job is the intended path; counts in prior reports that
  cited the legacy corpus refer to legacy candidates + their per-item
  exact checks at creation time, not to the new verified store).
- The calibration ablation is smoke-scale (n=12, 2 seeds) and is labeled
  as an implementation test only; no RL-performance conclusion.
- Policy-invariance of shaping now rests on the standard episodic
  assumptions with Φ(terminal)=0 — documented; no claim beyond them.

## Acceptance criteria — all met

NUMERICAL_ERROR never a loss ✓ (propagates; env no-op + named penalty);
failures never in learning/τ/caches ✓ (tested); serious violations go to
arbitrary precision or are rejected ✓; discoveries.json verified-only ✓;
legacy quarantined ✓; τ finite/positive/frozen/fingerprinted ✓; cache keys
carry full identity ✓; terminal handling explicit + tested ✓; proof
inconsistencies fixed ✓; full suite (124) + smokes pass ✓. No discovery
campaign was launched.
