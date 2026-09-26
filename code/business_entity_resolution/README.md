# Business Entity Resolution — Amazon ML Challenge 2026

> Matcher v1: normalisation -> blocking -> pair features -> 2-fold LightGBM -> decision layer ->
> `submissions/01_lgbm_v1/`. See `reports/` for the stage reports and `reports/experiments.md`.

## Environment

Python 3.13, CPU only (tested on Windows 11, 16 GB RAM).

```bash
# from student_resource/
python -m venv .venv
.venv/Scripts/python -m pip install -r code/business_entity_resolution/requirements.txt   # Windows
# .venv/bin/python -m pip install -r code/business_entity_resolution/requirements.txt     # Linux/macOS
```

Data layout expected (never modified): `dataset/train/train_source{1,2,3}.tsv`,
`dataset/train/train_ground_truth.tsv`, `dataset/test/test_source{1,2,3}.tsv`.

## Reproduce (run from `code/business_entity_resolution/`)

| Step | Command | Output | ~Time (16-thread laptop) |
|---|---|---|---|
| 0 | `python -m pytest tests -q` | — | 1 s |
| 1 | `python -m src.translit` | `artifacts/translit_dict.json`, `artifacts/indic_state_map.json` (learned from train only) | 3 min |
| 2 | `python -m src.normalize --split train` / `--split test` | `cache/{split}_records.parquet` | 18 / 9 min |
| 3 | `python -m src.blocking generate --split train` | `cache/train_pairs_all_{country}.parquet` | ~15 min / country |
| 4 | `python -m src.blocking tune` | `artifacts/blocking_budget.json` (chosen k, K) | few min |
| 5 | `python -m src.blocking generate --split test` | `cache/test_pairs_all_{country}.parquet` | |
| 6 | `python -m src.blocking prune --split train` / `--split test` | `cache/{split}_candidates.parquet` | |
| 7 | `python -m src.norm_report`, `python -m src.blocking_report` | `reports/normalization_report.md`, `reports/blocking_report.md` | |
| 8 | `python -m src.features --split train` / `--split test` (`--resume` skips finished countries) | `cache/features/{split}/{country}/part-*.parquet` | ~22 / 20 min |
| 9 | `python -m src.train` | `cache/preds/{train_oof,test}.parquet`, `artifacts/models/` (git-ignored) | 80 min |
| 10 | `python -m src.decide` | `artifacts/decision.json` | 15 min |
| 11 | `python -m src.submit --name 01_lgbm_v1` | `submissions/01_lgbm_v1/`, `reports/matcher_report.md` | 4 min |

Other: `python -m src.eda` (EDA report), `python -m src.make_empty_submission`.

`cache/` is regenerable and git-ignored; `artifacts/` holds everything learned from train.

## Source layout (`src/`)

| File | Purpose |
|---|---|
| `resources.py` | Every hand-written rule list (legal forms, honorifics, markers, abbreviations, states/regions), country-keyed with a generic fallback |
| `translit.py` | Rule-based Indic -> Latin transliteration (9 scripts, one offset table) + learned token dictionary and native-state map |
| `normalize.py` | Generic cleaning (pyarrow) and name / address feature extraction |
| `blocking.py` | DuckDB blocking keys, candidate pairs, quick scores, (k, K) tuning and pruning |
| `norm_report.py`, `blocking_report.py` | Diagnostics reports |
| `data.py` | Paths, loaders, integer ground-truth pairs |
| `features.py`, `train.py`, `decide.py`, `submit.py` | Pair features, 2-fold LightGBM (OOF), decision layer, submission + matcher report |
| `io_utils.py`, `metrics.py` | TSV reader / submission writer; macro F0.5 |
| `eda.py`, `text_norm.py`, `make_empty_submission.py` | EDA and format sanity |

Records are addressed as `(src, idx)` = source number and 0-based row in that source file.


Experiment log: `reports/experiments.md`.
