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

Runtime per part (minutes): B 40, C 98, D 87.

Notes:

- Part D failed at: === 19:38:19 src.decide --tag v2 --limited. Tail:     search = ("limited: odds + ef (gamma 0.85/1.0/1.2/1.5) and odds + two (t1 0.50-0.70, t2 0.60-0.80)" if a.limited                                                                                                            ^^^^^^^^^ AttributeError: 'int' object has no attribute 'limited' 

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

### Model (LightGBM, 2-fold out-of-fold; run `v2`)

77 features. Params `{'objective': 'binary', 'metric': 'binary_logloss', 'num_leaves': 127, 'learning_rate': 0.05, 'min_data_in_leaf': 200, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': 1.0, 'seed': 42, 'deterministic': True, 'force_col_wise': True, 'num_threads': 16}`, up to 3000 rounds, early stopping 100 (logloss) on a 10% S1 hold-out. Each fold trains on a random 30% of its half's S1s and predicts every pair of the other half; test = mean of both models.

| model | trained on half | rounds | train S1 | train pairs | early-stop S1 |
|---|---|---|---|---|---|
| 0 | 0 | 1679 | 298,355 | 3,785,758 | 32,762 |
| 1 | 1 | 1615 | 298,083 | 3,786,317 | 33,176 |

OOF metrics (model h predicts the other half; raw p before any isotonic step):

| model | country | pairs | AUC | logloss |
|---|---|---|---|---|
| 0 | ALL | 14,006,887 | 0.99912 | 0.02989 |
| 0 | India | 5,650,560 | 0.99882 | 0.03401 |
| 0 | US | 8,356,327 | 0.99928 | 0.02711 |
| 1 | ALL | 14,016,143 | 0.99912 | 0.02996 |
| 1 | India | 5,654,715 | 0.99881 | 0.03407 |
| 1 | US | 8,361,428 | 0.99928 | 0.02717 |

Calibration of OOF p (10 bins). Worst gap 0.0187 -> isotonic not needed.

| bin | pairs | mean p | positive rate | gap |
|---|---|---|---|---|
| 0.0-0.1 | 20,107,767 | 0.0029 | 0.0031 | -0.0002 |
| 0.1-0.2 | 250,838 | 0.1441 | 0.1500 | -0.0059 |
| 0.2-0.3 | 137,597 | 0.2461 | 0.2484 | -0.0023 |
| 0.3-0.4 | 93,558 | 0.3462 | 0.3401 | +0.0060 |
| 0.4-0.5 | 74,833 | 0.4502 | 0.4456 | +0.0046 |
| 0.5-0.6 | 61,103 | 0.5468 | 0.5281 | +0.0187 |
| 0.6-0.7 | 48,511 | 0.6504 | 0.6467 | +0.0037 |
| 0.7-0.8 | 62,701 | 0.7535 | 0.7611 | -0.0076 |
| 0.8-0.9 | 116,959 | 0.8568 | 0.8629 | -0.0062 |
| 0.9-1.0 | 7,069,163 | 0.9965 | 0.9969 | -0.0004 |

Top-30 features by gain, model 0:

| feature | gain |
|---|---|
| pscore | 24,601,264 |
| r_margin_score | 3,524,700 |
| prank_r | 1,629,670 |
| house_logdiff | 1,446,675 |
| nums_jaccard | 465,634 |
| name_cov_r | 320,954 |
| nums_frac_s_in_r | 223,787 |
| addr_cov_r | 197,089 |
| name_len_diff | 157,770 |
| addr_ntok_r | 119,289 |
| addr_tsr | 114,643 |
| s_n_near | 108,415 |
| r_margin_addr | 102,541 |
| name_first_tok | 102,501 |
| prank_s | 93,053 |
| addr_cov_s | 87,028 |
| name_ntok_r | 80,651 |
| addr_partial | 70,801 |
| addr_tsort | 68,994 |
| r_oth_addr | 68,756 |
| chain_s | 66,380 |
| addr_ntok_s | 64,940 |
| name_cov_s | 64,766 |
| r_oth_score | 62,978 |
| sim_addr | 60,179 |
| s_rank_addr | 50,150 |
| name_jw | 49,307 |
| rank_s_all | 47,899 |
| name_partial | 46,550 |
| rank_s | 44,902 |

Top-30 features by gain, model 1:

| feature | gain |
|---|---|
| pscore | 24,625,293 |
| r_margin_score | 3,027,245 |
| prank_r | 1,628,781 |
| house_logdiff | 1,433,792 |
| prank_r_all | 507,880 |
| nums_jaccard | 467,884 |
| name_cov_r | 289,984 |
| nums_frac_s_in_r | 221,848 |
| addr_cov_r | 197,514 |
| name_len_diff | 152,666 |
| addr_ntok_r | 114,122 |
| addr_tsr | 113,991 |
| name_first_tok | 104,912 |
| s_n_near | 103,602 |
| r_margin_addr | 100,701 |
| prank_s | 91,885 |
| addr_cov_s | 84,308 |
| name_ntok_r | 82,504 |
| sim_addr | 73,617 |
| addr_partial | 71,527 |
| name_tsort | 71,039 |
| name_cov_s | 68,680 |
| addr_tsort | 67,155 |
| r_oth_addr | 66,509 |
| addr_ntok_s | 65,795 |
| chain_s | 63,836 |
| r_oth_score | 63,768 |
| s_rank_addr | 52,972 |
| score | 50,516 |
| name_jw | 49,490 |

Training + prediction: 58.3 min, peak RSS 6.7 GiB.


*decide.md: not run / failed (see notes).*

*submit.md: not run / failed (see notes).*

## Part E. Stage-2 model with group-consistency features -> submissions/03_stage2/

*train.md: not run / failed (see notes).*

*decide.md: not run / failed (see notes).*

*submit.md: not run / failed (see notes).*
