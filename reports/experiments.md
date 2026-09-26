# Experiments

OOF = out-of-fold macro F0.5 over ALL train S1 (scored with `metrics.py`). "git commit" = the commit the run's code
builds on (the run's own changes land in the next commit, named in "change").

| id | time | git commit | change | OOF all | OOF India | OOF US | LB score |
|---|---|---|---|---|---|---|---|
| 00_all_empty | 2026-09-26 00:40 | 6946727 | every S1 predicted empty (format check) | 0.0558 | 0.0559 | 0.0558 | pending |
| 01_lgbm_v1 | 2026-09-26 12:55 | 29ea76f -> 6643f3c | blocking v1 (k=3, K=10) + 72 pair features + 2-fold LightGBM; odds one-home + expected-F0.5 (gamma 1.0) | 0.97071 | 0.96381 | 0.97532 | pending |
