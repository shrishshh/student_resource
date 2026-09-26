# Experiments

OOF = out-of-fold macro F0.5 over ALL train S1 (scored with `metrics.py`). "git commit" = the commit the run's code
builds on (the run's own changes land in the next commit, named in "change").

| id | time | git commit | change | OOF all | OOF India | OOF US | LB score |
|---|---|---|---|---|---|---|---|
| 00_all_empty | 2026-09-26 00:40 | 6946727 | every S1 predicted empty (format check) | 0.0558 | 0.0559 | 0.0558 | pending |
| 01_lgbm_v1 | 2026-09-26 12:55 | 29ea76f -> 6643f3c | blocking v1 (k=3, K=10) + 72 pair features + 2-fold LightGBM; odds one-home + expected-F0.5 (gamma 1.0) | 0.97071 | 0.96381 | 0.97532 | pending |
| 01b_gamma15 | 2026-09-26 13:32 | a921e7c | LB probe: 01 with expected-F0.5 gamma 1.5 (more conservative); 5,671,235 test matches vs 5,727,038 in 01 | 0.97033 | 0.96352 | 0.97487 | pending |
| 01c_france_empty | 2026-09-26 13:32 | a921e7c | LB probe: 01 with all 259,452 France S1 empty (01 had 13,079 of them empty); candidate_pairs unchanged | 0.97071 (= 01, France not in train) | 0.96381 | 0.97532 | pending |
| 02_prune_v2 | 2026-09-26 20:44 | 5ad2ce7 | blocking v2 (learned pruner) + v1 matcher (limited decision search); decision odds+ef {"gamma": 1.0} | 0.97303 | 0.96668 | 0.97727 | pending |
| 03_stage2 | 2026-09-26 20:44 | 5ad2ce7 | stage-2 LightGBM with group-consistency features on top of 02; decision odds+ef {"gamma": 1.0} | 0.97733 | 0.97267 | 0.98044 | pending |
