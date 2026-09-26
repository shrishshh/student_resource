# v2 report: LOCO check, learned pruning, matcher v2, stage 2

Out-of-fold macro F0.5 over all train S1 (scored with the `metrics.py` rule):

| run | overall | India | US | singleton | non-singleton |
|---|---|---|---|---|---|
| v1 (01_lgbm_v1) | 0.97071 | 0.96381 | 0.97532 | 0.95219 | 0.97181 |
| v2 (02_prune_v2) | - | - | - | - | - |
| stage 2 (03_stage2) | - | - | - | - | - |

Loss split (train pairs): true pairs never in the candidates / true pairs rejected by the decision / false-positive pairs:

| run | missed by blocking | false negatives | false positives |
|---|---|---|---|
| v1 | 290,922 | 208,272 | 50,186 |

Runtime per part (minutes): B 40.

## Part B. Leave-one-country-out (v1 features, fixed decision odds + ef gamma 1.0)

One LightGBM per source country (v1 params, 30% S1 sample, 10% of it for early stopping), applied to every pair of the other country. Compared with the normal 2-fold OOF, where both countries are in training.

| train on | score on | rounds | LOCO macro F0.5 | normal OOF | drop | singleton | non-singleton | minutes |
|---|---|---|---|---|---|---|---|---|
| India | US | 1899 | 0.95634 | 0.97532 | -0.01898 | 0.8626 | 0.9619 | 26.4 |
| US | India | 1281 | 0.92388 | 0.96381 | -0.03993 | 0.7407 | 0.9347 | 13.9 |

10 features whose gain rank changes most between the India-only and the US-only model:

| feature | rank (India model) | rank (US model) | change | gain (India) | gain (US) |
|---|---|---|---|---|---|
| rank_r_all | 67 | 3 | -64 | 1,115 | 2,867,405 |
| rec_name_has_indic | 25 | 71 | 46 | 67,905 | 0 |
| house_eq | 39 | 10 | -29 | 35,871 | 210,533 |
| addr_ntok_r | 22 | 50 | 28 | 86,563 | 23,285 |
| state_match | 44 | 17 | -27 | 24,097 | 134,336 |
| name_tsr | 49 | 23 | -26 | 15,692 | 97,391 |
| key_addr_number | 45 | 65 | 20 | 23,352 | 2,010 |
| s_gap_best | 41 | 21 | -20 | 28,436 | 112,097 |
| addr_ntok_s | 34 | 53 | 19 | 42,192 | 15,327 |
| addr_cov_s | 17 | 35 | 18 | 118,489 | 52,318 |

Part B runtime 40.3 min, peak RSS 7.0 GiB.


*Part C: not run / failed (see notes).*

## Part D. Matcher on the v2 candidates -> submissions/02_prune_v2/

*train.md: not run / failed (see notes).*

*decide.md: not run / failed (see notes).*

*submit.md: not run / failed (see notes).*

## Part E. Stage-2 model with group-consistency features -> submissions/03_stage2/

*train.md: not run / failed (see notes).*

*decide.md: not run / failed (see notes).*

*submit.md: not run / failed (see notes).*
