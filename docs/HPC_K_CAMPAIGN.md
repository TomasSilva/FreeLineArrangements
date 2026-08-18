# Overnight K-field campaign — runbook

## What it is

14 units/node hunting free line arrangements over quadratic fields at
n = 12..20, seeded by the certified reflection/CEVA fixtures
(`known_arrangements.py`, all five exactly certified + verified):

| fixture | n | field | exponents | SS? |
|---|---|---|---|---|
| Ceva(6) = G(6,6,3) | 18 | Q(sqrt-3) | (1,7,10) | no |
| H3 icosahedral | 15 | Q(sqrt5) | (1,5,9) | no |
| G(4,1,3) | 15 | Q(i) | (1,5,9) | yes |
| G(4,4,3) | 12 | Q(i) | (1,5,6) | no |
| extended Hesse G(3,1,3) | 12 | Q(sqrt-3) | (1,4,7) | yes |
| AKN A(13) | 13 | Q(sqrt3) | (1,6,6) | no (not recursively free) |
| dual Hesse | 9 | Q(sqrt-3) | (1,4,4) | no |

Unit types (see `jobs_K_tierA.txt` / `jobs_K_tierB.txt`):
- `cell N D1 D2 ENGINE SEED FIELD_D` — one cell over Q(sqrt FIELD_D)
  (`run_swap_campaign --field-d`); seeds = registry fixtures at that n +
  field-matched/QQ lift seeds + perturbations; forced-field pools let
  QQ-seeded chains acquire sqrt(d) coordinates.
- `cascade START_N FIELD_D ENGINE SEED` — climbing unit
  (`run_cascade_campaign`): engine slice at level n -> harvest certified
  non-SS distinct lattices -> exact K lifts into n+1 -> those seed the next
  level; repeats to n = 20 or the wall.

Smoke-validated locally: (18,7,10)/Q(sqrt-3) certified 3 lattices (one
non-SS, m_max 7) in 5 min; cascade re-certified AKN in its own basin;
forced d=2 unit certified 8 distinct lattices at (14,6,7).

## Submit (on the cluster)

```
git pull
python -c "import swap_search, quadfield, known_arrangements"   # preflight
qsub -v JOBS_FILE=jobs_K_tierA.txt pbs/step7_swap_K.pbs
qsub -v JOBS_FILE=jobs_K_tierB.txt pbs/step7_swap_K.pbs          # optional 2nd node
```

The PBS file loads `python/3.8.11-intel-2021.3.0` and
`source free_arr/bin/activate`.  Outputs land in
`./results_from_HPC/<jobid>/cells_d<FIELD_D>/n<N>_d<D1>_<D2>/`
(candidates.jsonl, certified.jsonl, certificates/, per-unit logs;
cascade manifests at `cells_d<d>/cascade_manifest_*.json`).  Job-level
stdout appears as `swap_K.o<JOBID>` at job end (`#PBS -j oe`).

## Morning after (local)

```
rsync -a cluster:.../results_from_HPC/ ./results_from_HPC/
python - <<'EOF'                       # merged reference (corpus + fixtures)
import json
ref = {}
h = json.load(open("results_penalized_saito/2026-08-17/swap/reference_hashes/headline.json"))
for cell, info in h.items():
    for k in info.get("hashes", {}):
        ref[k] = {"source": "qq_corpus", "cell": cell}
ref.update(json.load(open("reference_hashes_k_fixtures.json")))
json.dump(ref, open("results_from_HPC/combined_reference.json", "w"), indent=1)
EOF
for d in results_from_HPC/*/cells_d*; do
  python experiments/triage_swap.py --cells-dir "$d" \
    --reference-hashes results_from_HPC/combined_reference.json \
    --out "$d/triage" --top 12 --if-max-n 14
done
```

Headline metric: distinct certified **non-supersolvable** K lattices whose
WL hash is in NEITHER the QQ corpus NOR `reference_hashes_k_fixtures.json`
(rediscovering Ceva(6)/H3/etc. is labeled `known_family`, never "new").
Wording stays `not_found_in_reference_corpus`; literature checks
(Grunbaum/simplicial, reflection restrictions, Ziegler pairs) remain a
human TODO before any stronger claim.  Every reported discovery must
re-verify from JSON (`verify_certificate(certificate_from_json(...))`).

## Notes / limits

- The RL arm is QQ-only by design; all K units use ME/anneal/greedy.
- K certification cost grows with n (Weil blocks are 2x): ~1 min per
  exact check at n = 18; CampaignIO certifies once per lattice and caps
  SS certifications (n >= 17), so exact-check time concentrates on non-SS.
- d = 2 has no known fixture: its unit is exploratory (QQ lift seeds +
  forced sqrt(2) pools) and may produce only QQ-lattice realizations.
- `results_from_HPC/` is gitignored.
