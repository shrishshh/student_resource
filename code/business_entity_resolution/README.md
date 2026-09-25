# Business Entity Resolution — Amazon ML Challenge 2026

> Stub: the pipeline (blocking -> matching -> output) is not built yet. This file will hold
> the exact end-to-end reproduction steps.

## Environment

Python 3.13 (tested on Windows 11, CPU only).

```bash
# from student_resource/
python -m venv .venv
.venv/Scripts/python -m pip install -r code/business_entity_resolution/requirements.txt   # Windows
# .venv/bin/python -m pip install -r code/business_entity_resolution/requirements.txt     # Linux/macOS
```

Expected data layout (not modified by any script):

```
dataset/train/train_source{1,2,3}.tsv, dataset/train/train_ground_truth.tsv
dataset/test/test_source{1,2,3}.tsv
```

## Source layout (`src/`)

| File | Purpose |
|---|---|
| `io_utils.py` | The single TSV reader used everywhere, split / ground-truth loaders, submission writer |
| `metrics.py` | Entity-level and macro F0.5 (with per-country and singleton breakdowns) |
| `text_norm.py` | Vectorised (pyarrow) name/address normalisation helpers |
| `eda.py` | Exploratory analysis -> `reports/eda_report.md` |
| `make_empty_submission.py` | All-empty submission (format sanity baseline) |

## Commands (from `student_resource/`)

```bash
python -m pytest code/business_entity_resolution/tests -q
python code/business_entity_resolution/src/eda.py                 # ~full EDA report
python code/business_entity_resolution/src/make_empty_submission.py
python utils/validate_submission.py --matching submissions/00_all_empty/matching_results.tsv \
    --candidate submissions/00_all_empty/candidate_pairs.tsv --test-dir dataset/test
```

## Pipeline

TODO: blocking, matching model, thresholding, output generation.
