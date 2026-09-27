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
| 04_s2_g15 | 2026-09-27 02:59 | e8b7d59 | slim candidates; stage 2 (fold-consistent); odds + ef gamma 1.5; 6,048,656 test matches | 0.97639 | 0.97142 | 0.97970 | pending |
| 04_s2_g20 | 2026-09-27 02:59 | e8b7d59 | slim candidates; stage 2 (fold-consistent); odds + ef gamma 2.0; 5,999,909 test matches | 0.97594 | 0.97080 | 0.97936 | pending |
| 04_s2_g25 | 2026-09-27 02:59 | e8b7d59 | slim candidates; stage 2 (fold-consistent); odds + ef gamma 2.5; 5,963,432 test matches | 0.97544 | 0.97015 | 0.97897 | pending |
| 04_s1_g15 | 2026-09-27 02:59 | e8b7d59 | slim candidates; stage 1; odds + ef gamma 1.5; 5,759,106 test matches | 0.97266 | 0.96606 | 0.97706 | pending |
| 04_s1_g20 | 2026-09-27 02:59 | e8b7d59 | slim candidates; stage 1; odds + ef gamma 2.0; 5,720,848 test matches | 0.97181 | 0.96493 | 0.97640 | pending |
| 04_s2_g15_fr25 | 2026-09-27 02:59 | e8b7d59 | slim candidates; stage 2 (fold-consistent); odds + ef gamma 1.5, France gamma 2.5; 6,036,227 test matches | 0.97639 | 0.97142 | 0.97970 | pending |
| 06_D_g10 | 2026-09-27 16:57 | bb68a48 | stage 1 trained on the test-like world (19% S1 dropped); odds + ef gamma 1.0; test-like world 0.97145; 5,778,970 test matches | 0.97267 | 0.96606 | 0.97709 | pending |
| 06_D_g15 | 2026-09-27 16:57 | bb68a48 | stage 1 trained on the test-like world (19% S1 dropped); odds + ef gamma 1.5; test-like world 0.97094; 5,717,272 test matches | 0.97178 | 0.96503 | 0.97629 | pending |
| 07_D_g10 | 2026-09-27 19:34 | f5c663b | master: best test-like variant = model D, no smoothing (max-if smoothing +0.00004 = tie -> simpler); same content as 06_D_g10; test-like 0.97145; 5,778,970 test matches | 0.97267 | 0.96606 | 0.97709 | pending |
| 07_D_g15 | 2026-09-27 19:34 | f5c663b | master: next more conservative gamma; same content as 06_D_g15; test-like 0.97094; 5,717,272 test matches | 0.97178 | 0.96503 | 0.97629 | pending |
| 08_D_g15_frstate | 2026-09-27 20:38 | db9a136 | 07_D_g15 + France region filled from a city->region table learned from test S1 (449,722 of 500,864 missing filled; France state_match missing 35.7% -> 5.1%); France pairs re-predicted with D, US/India unchanged; 5,717,181 test matches | 0.97178 | 0.96503 | 0.97629 | pending |
