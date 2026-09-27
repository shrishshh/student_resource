# Business Entity Resolution — Amazon ML Challenge 2026 (team TechTitans)

**Final submission: `04_s1_g15`** (leaderboard 0.962). Pipeline: normalisation with learned Indic
transliteration -> DuckDB blocking -> learned pruner -> slim candidates (~5.4 per Source-1 entity on
test) -> pair features -> stage-1 LightGBM (2-fold, out-of-fold) -> one-home odds renormalisation +
expected-F0.5 set selection (gamma 1.5) -> `output/matching_results.tsv` + `output/candidate_pairs.tsv`.

Only the provided training/test data is used: no external data, APIs, geocoding or pretrained models.
Everything learned (transliteration dictionary, pruner, matcher, thresholds) is learned from train.

## 1. Layout after unzipping

Put the challenge's `dataset/` and `utils/` folders next to `code/` and `output/`:

```
<unzipped>/
  code/business_entity_resolution/   <- this folder; run every command from here
  output/                            <- matching_results.tsv, candidate_pairs.tsv (the pipeline rewrites them)
  dataset/train/train_source{1,2,3}.tsv, dataset/train/train_ground_truth.tsv   <- from the challenge
  dataset/test/test_source{1,2,3}.tsv                                           <- from the challenge
  utils/validate_submission.py                                                  <- from the challenge
  Documentation_template.md
```

The dataset is never modified. Without `utils/`, the last step skips validation with a warning.

## 2. Environment

Tested on Windows 11, Intel Core i7-1360P (12 cores / 16 threads), **16 GB RAM, no GPU**, Python 3.13;
~40 GB free disk for `cache/`. Peak process memory ~9 GB (DuckDB is capped at 6 GB and spills to disk).

```bash
# from the unzipped root
python -m venv .venv
.venv/Scripts/python -m pip install -r code/business_entity_resolution/requirements.txt   # Windows
# .venv/bin/python -m pip install -r code/business_entity_resolution/requirements.txt     # Linux / macOS
```

`requirements.txt` pins every package (pandas, numpy, pyarrow, duckdb, rapidfuzz, lightgbm, numba,
scikit-learn, scipy, psutil, pytest and their dependencies).

## 3. One command

```bash
cd code/business_entity_resolution
python -m src.run_all                          # default --variant 04_s1_g15 -> ../../output/*.tsv
python -m src.run_all --dry-run                # list the steps
python -m src.run_all --from-step features     # resume from a step after an interruption
```

## 4. Steps of the final variant, outputs and measured runtime

Runtimes were **measured per step** from our run logs on the laptop above; the full end-to-end run was
not timed in one go (the steps add up to about 4 hours).

| # | Step (`run_all` name) | Command | Writes | Measured time | Peak RAM |
|---|---|---|---|---|---|
| 1 | translit | `python -m src.translit` | `artifacts/translit_dict.json`, `artifacts/indic_state_map.json` | 2.5 min | 2.5 GB |
| 2 | normalize | `python -m src.normalize --split train` / `test` | `cache/{split}_records.parquet` | 18.4 / 9.3 min | 6.2 GB |
| 3 | blocking | `python -m src.blocking generate --split train` / `test` | `cache/{split}_pairs_all_{country}.parquet` | 36.3 / 35.9 min | 8.9 GB |
| 4 | pruner | `python -m src.pruner train`, `score`, `rank`, `grid`, `prune` | `artifacts/pruner/`, `cache/{split}_pairs_pranked_{country}.parquet` | 5.1 + 71.2 + 12.4 + 4.1 + 4.4 min | 9.2 GB |
| 5 | slim | `python -m src.slim apply` | `cache/{split}_candidates.parquet` (tau 0.02, S1 top-15, record top-3) | 0.9 min | < 1 GB |
| 6 | features | `python -m src.features --split train` / `test` | `cache/features/{split}/{country}/part-*.parquet` | 9.9 / 6.7 min | 7.1 GB |
| 7 | stage1 | `python -m src.train --tag v3s1` | `cache/preds_v3s1/`, `artifacts/models_v3s1/` | 27.2 min | 5.1 GB |
| 8 | output | `python -m src.variants --only 04_s1_g15 --out ../../output` | `output/*.tsv` (+ validator `--check-ids`) | ~2 min | 2.1 GB |

`artifacts/slim.json` fixes tau = 0.02; without it `src.slim apply` uses the same default. `cache/` can be
deleted afterwards. Every step is deterministic (seed 42).

Analysis / experiment commands (not needed for the output): `src.eda`, `src.norm_report`,
`src.blocking_report`, `src.pruner report`, `src.slim curve`, `src.decide`, `src.stage2` (stage 2, not in the
final), `src.loco`, `src.pseudo`, `src.testlike`, `src.master`, `src.france_state`, `src.variants` (all
variants), `src.make_package --variant 04_s1_g15 --team TechTitans` (builds the zip).

## 5. Source layout (`src/`)

| File | Purpose |
|---|---|
| `resources.py` | Every hand-written rule list (legal forms, honorifics, markers, abbreviations, states/regions), country-keyed with a generic fallback |
| `translit.py` | Rule-based Indic -> Latin transliteration (9 scripts) + learned token dictionary / native-state map |
| `normalize.py` | Generic cleaning (pyarrow) and name / address fields |
| `blocking.py` | DuckDB blocking keys, candidate pairs, quick scores |
| `pruner.py`, `slim.py` | Learned pruner (LightGBM on cheap features) and slim candidate selection |
| `features.py` | 77 pair features (string similarities, token / IDF overlaps, numbers, competition features) |
| `train.py` | 2-fold out-of-fold LightGBM |
| `decide.py`, `variants.py` | One-home renormalisation, expected-F0.5 optimiser, decision variants / output writing |
| `run_all.py`, `make_package.py` | End-to-end driver and submission zip |
| `data.py`, `io_utils.py`, `metrics.py` | Paths / loaders, the single TSV reader and writer, macro F0.5 |
| other modules | Analysis and experiments (EDA, reports, stage 2, LOCO, pseudo labels, test-like world, audits) |

Records are addressed as `(src, idx)` = source number and 0-based row in that source file. Tests:
`python -m pytest tests -q`.
