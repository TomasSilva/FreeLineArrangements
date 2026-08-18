# Quadratic-field extension — implementation report (2026-08-18)

## Why

Every known small free-but-not-inductively-free line arrangement requires
irrational coefficients: AKN A(13) over Q(sqrt3) (arXiv:1406.5820, free,
NOT recursively free, exponents (1,6,6)), pentagonal-11 over Q(sqrt5),
dual Hesse-9 over Q(sqrt-3), monomial G(4,4,3)-12 over Q(i), Cuntz–Hoge-27
over Q(zeta5).  The QQ-only pipeline was structurally blind to all of
them — consistent with every discovery so far (including the three
non-supersolvable headliners at n=13/14) being inductively free.  The
search now runs over K = Q(sqrt d), d in {2, 3, 5, -1, -3}.

## Architecture (commits b1cf320, 5aa0a70, 90576ac, 783139e, 49cef17, +fixes)

- **quadfield.py** — canonical `QuadElem` pairs a + b*sqrt(d) (collapse
  invariant: b = 0 results return plain Rational, so QQ data never changes
  representation); `_op_priority` + a registered exact sympy converter
  (sympy would otherwise silently degrade mixed products to Float);
  K-linear algebra by **Weil restriction**: kernel/rank of M = M0 + s*M1
  over K = half the rational kernel/rank of [[M0, d*M1],[M1, M0]] —
  reusing the audited rational `Matrix.nullspace/rank` and the EXISTING
  mod-p elimination (doubled bound), no split-prime number theory.
  Property-tested against `QQ.algebraic_field` + DomainMatrix oracles.
- **arrangement.py** — coordinate coercion admits QuadElem (QQ fast path
  byte-identical, locked by exact goldens in `tests/test_field_backcompat`);
  incidence layer unchanged: canonical equality/hash makes `_structure`
  exact over K by construction.  `derivation_rows` + K branches in
  `derivation_space_dim`/`is_free`.
- **known_arrangements.py** — exact fixtures `akn13(lam)` (with
  `validate_akn13_lattice`: profile (t2,t3,t4,t5) = (21,3,3,3), 30 points,
  b2 = 48, exponents (6,6); degenerate lam rejected) and `dual_hesse()`.
- **certificates.py** — certification over K: kernels via Weil restriction,
  block mod-p prescreen, exact K point evaluation (c in K), field-aware
  verify/JSON with strict grammar (`[a+bs]` tokens; `s` without a declared
  field is an error).  Also fixed a latent QQ hazard: `nsimplify` on exact
  rationals with large denominators could return a DIFFERENT number
  (heuristic float matching); all exact values now pass through untouched.
- **penalized_saito.py v2.5.0** — embeddings (complex128 mandatory for
  d < 0, auto-selected; sqrt(d) -> +sqrt(d) resp. +i*sqrt|d|); Stage C
  rebuilds from the EXACT (a, b) data with mp.sqrt(d) at dps; cache keys
  carry (field, dtype).
- **promotion.py** — schema `discovery-2.1` for K entries (2.0 unchanged,
  no migration); the field tag is load-bearing (validated at promote AND
  load); discovery-id includes the field; `coefficient_field` vs
  `optimization_field` naming split.
- **search** — field-closed singularity candidates (primary K pool,
  embedding-aware scoring) + small O_K grid; Delta-b2 filter unchanged
  (combinatorial); float-snap rationalization QQ-only; K campaigns via
  `run_swap_campaign --field-d D` seeded from validated fixtures + lifted
  K seeds + perturbations; RL arm explicitly QQ-only (guard).

## Ground truths (all in tests; 182 tests green)

| check | result |
|---|---|
| AKN-13 exact certificate at (6,6) | **certified**, c = (315/992)·sqrt(3), 3 s; verify + JSON round-trip |
| AKN-13 inductive freeness | **not_inductively_free** (no line carries the 7 points the deletion condition needs) |
| AKN-13 penalized loss | 0.0 (float64 real embedding); Gamma = 1 at certificate BW vectors; Stage-C agreement |
| dual Hesse at (4,4) over Q(sqrt-3) | certified (c = -1), complex128 loss 0.0, complex BW vectors |
| perturbed-AKN negative control | modp_reject in 0.4 s (block prescreen) |
| swap-recovery gate | greedy over Q(sqrt3) recovers the EXACT AKN lattice from a 1-swap perturbation in ~5 s, certified, non-SS |
| corpus screen | AKN lattice hash absent from all 1800 distinct QQ-corpus lattices |
| promotion | K entry promoted + idempotent + strict-loaded; unsupported d / missing field tag rejected with reasons |

## A structural negative worth reporting

No single-line extension of AKN-13 reaches ANY admissible free n=14 cell:
the achievable Delta-b2 values over its two-point/grid candidate pools are
{8, 9, 10, 11, 13}, while (14,4,9)/(14,5,8)/(14,6,7) require {1, 5, 7}.
This is the combinatorial shadow of the AKN non-recursive-freeness proof
(their elimination of all additions).  Consequence for the campaign: the
Q(sqrt3) cascade moves SIDEWAYS first — swap search at n=13 in the AKN
basin produces new certified K lattices, and those (not AKN itself) are
the lift candidates toward n = 14+.

## First K campaign (running at report time)

`swap_K/cells/n13_d6_6_K3` (MAP-Elites + anneal over Q(sqrt3), AKN-basin
seeds) and `swap_K/cells/n9_d4_4_Km3` (MAP-Elites over Q(sqrt-3)).  Early
results: 2 certified K lattices at (13,6,6); 7 certified at (9,4,4)
including the dual-Hesse lattice (m_max = 3, non-SS) through the full
pipeline.  All certified records re-verify independently from JSON with
correct field tags.

## Honest limitations

- Q(zeta5) (Cuntz–Hoge-27) needs degree 4 — out of scope.
- pentagonal-11 (d=5) and G(4,4,3) (d=-1) fixtures not yet encoded
  (d=5/-1 covered by property tests only) — next increment.
- K certification is slower than QQ (~3 s at n=13 vs ~0.3 s): Weil blocks
  are 2x2 larger; CampaignIO certifies once per lattice, which bounds it.
- K heights are the naive basis-{1, sqrt d} height (documented; for
  d = 5, -3 within 2x of the true O_K height; tie-break only).
- The RL arm stays QQ-only by design (float32 observations).
