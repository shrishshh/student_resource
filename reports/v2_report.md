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

Runtime per part (minutes): B 40, C 98.

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


## Part C. Blocking v2: learned pruning

Pruner: LightGBM (300 trees, 63 leaves, lr 0.1) trained on all 13,246,205 unpruned pairs of a random 3% of train S1 (65,900 S1, 0.0168 positive). Features (28): `key_name_pair`, `key_name_single`, `key_name_phon`, `key_concat_prefix`, `key_name_alt`, `key_addr_pair`, `key_addr_number`, `key_cross`, `nkeys`, `sim_name`, `sim_addr`, `name_tsort`, `name_ratio`, `addr_tsort`, `house_eq`, `house_missing`, `s_addr_empty`, `r_addr_empty`, `r_name_is_web`, `r_name_has_indic`, `r_name_has_marker`, `chain_s`, `chain_r`, `r_tok_in_s1_vocab`, `rank_r`, `rank_s`, `n_cand_s`, `n_cand_r`.

Top pruner features by gain: chain_s (242,529,028,660), rank_r (209,944,819,356), rank_s (158,098,071,320), house_missing (113,723,204,509), n_cand_r (61,311,376,027), chain_r (35,688,955,516), name_tsort (19,447,007,634), n_cand_s (17,167,936,406), sim_addr (12,682,722,332), key_name_phon (8,548,642,239).

Grid (keep if the record's top-k OR the S1's top-K by pruner score; train pairs <= 50,000,000; smallest budget within 0.001 of the best oracle):

| k | K | pairs | pairs India | pairs US | cand/S1 mean | p95 | pair recall | recall S2 | recall S3 | S1 with ALL matches | oracle F0.5 | oracle India | oracle US | *(empty)* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 10 | 28,023,030 | 11,305,275 | 16,717,755 | 12.7 | 20 | 97.65% | 98.00% | 97.32% | 93.13% | 0.99147 | 0.98883 | 0.99323 | **chosen** |
| 2 | 15 | 36,018,965 | 14,443,068 | 21,575,897 | 16.3 | 23 | 97.73% | 98.06% | 97.41% | 93.39% | 0.99164 | 0.98902 | 0.99338 | *(empty)* |
| 2 | 20 | 44,868,857 | 17,941,709 | 26,927,148 | 20.3 | 25 | 97.74% | 98.07% | 97.43% | 93.44% | 0.99168 | 0.98908 | 0.99342 | *(empty)* |
| 3 | 10 | 35,880,400 | 14,467,993 | 21,412,407 | 16.3 | 29 | 97.66% | 98.01% | 97.33% | 93.17% | 0.99150 | 0.98890 | 0.99324 | *(empty)* |
| 3 | 15 | 41,948,045 | 16,837,972 | 25,110,073 | 19.0 | 31 | 97.73% | 98.06% | 97.42% | 93.40% | 0.99165 | 0.98905 | 0.99339 | *(empty)* |
| 3 | 20 | 49,163,532 | 19,684,268 | 29,479,264 | 22.3 | 33 | 97.75% | 98.08% | 97.44% | 93.45% | 0.99169 | 0.98909 | 0.99342 | *(empty)* |
| 5 | 10 | 53,059,649 | 21,338,636 | 31,721,013 | 24.0 | 47 | 97.68% | 98.03% | 97.35% | 93.22% | 0.99155 | 0.98897 | 0.99327 | *(empty)* |
| 5 | 15 | 56,863,537 | 22,815,782 | 34,047,755 | 25.8 | 48 | 97.74% | 98.07% | 97.43% | 93.43% | 0.99168 | 0.98910 | 0.99340 | *(empty)* |
| 5 | 20 | 61,535,501 | 24,670,542 | 36,864,959 | 27.9 | 49 | 97.75% | 98.08% | 97.44% | 93.46% | 0.99170 | 0.98912 | 0.99342 | *(empty)* |

**v1 (quick-score pruning, k=3, K=10): 36,031,868 pairs, oracle 0.98691. v2 chosen k=2, K=10: 28,023,030 pairs, oracle 0.99147.**

| true pairs | v1 candidates | v2 candidates |
|---|---|---|
| all | 7,347,443 (96.19%) | 7,458,833 (97.65%) |
| record address empty | 202,827 (60.18%) | 289,434 (85.88%) |

Of the 134,191 true pairs with an empty record address missed by v1, v2 recovers **86,974**; v2 loses 1,803 true pairs that v1 had (and gains 113,193 in total).

Test candidates (v1 -> v2): France 4,727,989 -> 3,570,014, India 15,376,874 -> 11,510,407, US 12,400,644 -> 9,219,930.

Timing (min): train 5.1, score 71.2, rank 12.4, grid 4.1, prune 4.4; Part C total 97.4 min, peak RSS 9.2 GiB.


## Part D. Matcher on the v2 candidates -> submissions/02_prune_v2/

*train.md: not run / failed (see notes).*

*decide.md: not run / failed (see notes).*

*submit.md: not run / failed (see notes).*

## Part E. Stage-2 model with group-consistency features -> submissions/03_stage2/

*train.md: not run / failed (see notes).*

*decide.md: not run / failed (see notes).*

*submit.md: not run / failed (see notes).*
