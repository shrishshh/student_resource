# Business Entity Resolution — Amazon ML Challenge 2026

Pipeline: normalisation (with learned Indic transliteration) -> DuckDB blocking -> learned pruner ->
slim candidates -> pair features -> two-stage LightGBM (2-fold, out-of-fold) -> one-home odds
renormalisation + expected-F0.5 decision -> `output/matching_results.tsv` + `output/candidate_pairs.tsv`.

Only the provided training/test data is used: no external data, APIs, geocoding or pretrained models.
Everything learned (transliteration dictionary, pruner, matchers, thresholds) is learned from train.

## 1. Hardware and environment

Developed and timed on: Windows 11, Intel Core i7-1360P (12 cores / 16 threads), **16 GB RAM, no GPU**,
~40 GB free disk for the cache. Peak process memory ~9 GB (DuckDB is capped at 6 GB and spills to disk).

```bash
# from student_resource/  (Python 3.13; 3.11+ should work)
python -m venv .venv
.venv/Scripts/python -m pip install -r code/business_entity_resolution/requirements.txt   # Windows
# .venv/bin/python -m pip install -r code/business_entity_resolution/requirements.txt     # Linux / macOS
```

`requirements.txt` pins every package (pandas, numpy, pyarrow, duckdb, rapidfuzz, lightgbm, numba,
scikit-learn, scipy, psutil, pytest and their dependencies).

## 2. Data layout

Put the challenge data under `student_resource/dataset/` exactly as provided (never modified):

```
student_resource/
  dataset/train/train_source{1,2,3}.tsv, dataset/train/train_ground_truth.tsv
  dataset/test/test_source{1,2,3}.tsv
  utils/validate_submission.py
  code/business_entity_resolution/   <- this folder (run every command from here)
```

## 3. One command

```bash
cd code/business_entity_resolution
python -m src.run_all --variant 04_s2_g15           # writes ../../output/{matching_results,candidate_pairs}.tsv
python -m src.run_all --variant 04_s2_g15 --dry-run # list the steps
python -m src.run_all --variant 04_s2_g15 --from-step features   # resume after a failure
```

`--variant` selects the decision configuration (see `src/variants.py`): `04_s2_g15` = stage-2 model,
odds renormalisation, expected-F0.5 optimiser with gamma 1.5 (the submitted configuration).
The last step runs `utils/validate_submission.py --check-ids` on the output.

## 4. Steps, outputs and runtime

| Step (`run_all` name) | Command | Writes | Time* |
|---|---|---|---|
| translit | `python -m src.translit` | `artifacts/translit_dict.json`, `artifacts/indic_state_map.json` | 3 min |
| normalize | `python -m src.normalize --split train` / `test` | `cache/{split}_records.parquet` | 18 / 9 min |
| blocking | `python -m src.blocking generate --split train` / `test` | `cache/{split}_pairs_all_{country}.parquet` (unpruned, scored) | 36 / 36 min |
| pruner | `python -m src.pruner train\|score\|rank\|grid\|prune` | `artifacts/pruner/`, `cache/{split}_pairs_pranked_{country}.parquet` | 95 min |
| slim | `python -m src.slim apply` | `cache/{split}_candidates.parquet` | 5 min |
| features | `python -m src.features --split train` / `test` | `cache/features/{split}/{country}/part-*.parquet` | ~10 / 8 min |
| stage1 | `python -m src.train --tag v3s1` | `cache/preds_v3s1/`, `artifacts/models_v3s1/` | ~40 min |
| stage2f | `python -m src.stage2 --p-tag v3s1` | `cache/stage2/{train,test_m0,test_m1}/` | ~10 min |
| stage2 | `python -m src.train --tag v3s2 --stage2` | `cache/preds_v3s2/`, `artifacts/models_v3s2/` | ~30 min |
| output | `python -m src.variants --only <variant> --out ../../output` | `output/*.tsv` | 5 min |

\*Measured on the laptop above; the full run takes roughly 5-6 hours. `cache/` (~40 GB) can be deleted
afterwards. Every step is deterministic (seed 42).

Analysis / tuning commands (not needed to reproduce the output): `src.eda`, `src.norm_report`,
`src.blocking tune` + `src.blocking_report` (blocking v1), `src.pruner grid`/`report`, `src.slim curve`
(chooses tau into `artifacts/slim.json`), `src.decide --tag <run> [--limited]` (decision search on OOF),
`src.loco` (leave-one-country-out), `src.pseudo`, `src.variants` (all variants), `src.make_package`.

## 5. Source layout (`src/`)

| File | Purpose |
|---|---|
| `resources.py` | Every hand-written rule list (legal forms, honorifics, markers, abbreviations, states/regions), country-keyed with a generic fallback |
| `translit.py` | Rule-based Indic -> Latin transliteration (9 scripts) + learned token dictionary / state map |
| `normalize.py` | Generic cleaning (pyarrow) and name / address fields |
| `blocking.py` | DuckDB blocking keys, candidate pairs, quick scores |
| `pruner.py`, `slim.py` | Learned pruner (LightGBM on cheap features) and slim candidate selection |
| `features.py` | ~77 pair features (string similarities, token/IDF overlaps, numbers, competition features) |
| `train.py` | 2-fold out-of-fold LightGBM (stage 1; `--stage2` adds group-consistency features) |
| `stage2.py` | Stage-2 group-consistency features (DuckDB window functions) |
| `decide.py`, `variants.py` | One-home renormalisation, expected-F0.5 optimiser, decision search / variants |
| `submit.py`, `make_package.py`, `run_all.py` | Submission writing, zip packaging, end-to-end driver |
| `data.py`, `io_utils.py`, `metrics.py` | Paths / loaders, the single TSV reader and writer, macro F0.5 |
| `eda.py`, `text_norm.py`, `norm_report.py`, `blocking_report.py`, `loco.py`, `pseudo.py`, `*_report.py` | Analysis and reports |

Records are addressed as `(src, idx)` = source number and 0-based row in that source file. Tests:
`python -m pytest tests -q`. Reports: `../../reports/`.
