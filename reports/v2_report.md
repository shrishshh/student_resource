# v2 report: LOCO check, learned pruning, matcher v2, stage 2

Out-of-fold macro F0.5 over all train S1 (scored with the `metrics.py` rule):

| run | overall | India | US | singleton | non-singleton |
|---|---|---|---|---|---|
| v1 (01_lgbm_v1) | 0.97071 | 0.96381 | 0.97532 | 0.95219 | 0.97181 |
| v2 (02_prune_v2) | 0.97303 | 0.96668 | 0.97727 | 0.95690 | 0.97399 |
| stage 2 (03_stage2) | - | - | - | - | - |

Loss split (train pairs): true pairs never in the candidates / true pairs rejected by the decision / false-positive pairs:

| run | missed by blocking | false negatives | false positives |
|---|---|---|---|
| v1 | 290,922 | 208,272 | 50,186 |
| v2 | 179,532 | 278,457 | 46,302 |

Runtime per part (minutes): B 40, C 98, D 81.

Notes:

- Part D failed at: === 19:38:19 src.decide --tag v2 --limited. Tail:     search = ("limited: odds + ef (gamma 0.85/1.0/1.2/1.5) and odds + two (t1 0.50-0.70, t2 0.60-0.80)" if a.limited                                                                                                            ^^^^^^^^^ AttributeError: 'int' object has no attribute 'limited' 
- Part D first attempt crashed in src.decide after the search (loop variable shadowed the parsed CLI args; same bug in src.submit). Fixed and re-run from the decision step.

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


### Decision layer (run `v2`, tuned on OOF over all train S1)

29 configurations evaluated (limited: odds + ef (gamma 0.85/1.0/1.2/1.5) and odds + two (t1 0.50-0.70, t2 0.60-0.80)). S1s without candidates and true pairs missing from the candidates (179,532) count against recall.

**Winner: D1 = `odds`, D2 = `ef`, params `{"gamma": 1.0}` -> OOF macro F0.5 0.97303** (re-scored with `metrics.macro_f05`: 0.97303).

Best configuration of each (D1, D2) family:

| D1 | D2 | params | OOF F0.5 | India | US | singleton | non-singleton |
|---|---|---|---|---|---|---|---|
| odds | ef | {"gamma": 1.0} | 0.97303 | 0.96668 | 0.97727 | 0.9569 | 0.9740 |
| odds | two | {"t1": 0.55, "t2": 0.7} | 0.97284 | 0.96641 | 0.97713 | 0.9631 | 0.9734 |

All global-threshold and expected-F configurations:

| D1 | D2 | params | OOF F0.5 | India | US | singleton | non-singleton |
|---|---|---|---|---|---|---|---|
| odds | ef | {"gamma": 1.5} | 0.97249 | 0.96616 | 0.97671 | 0.9692 | 0.9727 |
| odds | ef | {"gamma": 1.2} | 0.97289 | 0.96660 | 0.97709 | 0.9630 | 0.9735 |
| odds | ef | {"gamma": 0.85} | 0.97294 | 0.96649 | 0.97724 | 0.9494 | 0.9743 |
| odds | ef | {"gamma": 1.0} | 0.97303 | 0.96668 | 0.97727 | 0.9569 | 0.9740 |

Winner breakdown:

| slice | OOF macro F0.5 |
|---|---|
| overall | 0.97303 |
| India | 0.96668 |
| US | 0.97727 |
| singleton | 0.95690 |
| non_singleton | 0.97399 |

Predicted vs true set size (number of train S1):

| size | true | predicted |
|---|---|---|
| 0 | 123,247 | 130,204 |
| 1 | 119,157 | 164,621 |
| 2 | 375,212 | 417,454 |
| 3 | 530,841 | 540,742 |
| 4 | 484,115 | 459,598 |
| 5 | 321,957 | 287,800 |
| 6-10 | 252,255 | 206,378 |
| >10 | 37 | 24 |

Best configuration per country (vs the global winner on that country):

| country | D1 | D2 | params | best F0.5 | winner F0.5 |
|---|---|---|---|---|---|
| India | odds | ef | {"gamma": 1.0} | 0.96668 | 0.96668 |
| US | odds | ef | {"gamma": 1.0} | 0.97727 | 0.97727 |

Error analysis: 46,302 false-positive pairs, 278,457 false-negative pairs among candidates (+ 179,532 true pairs never in the candidates).

15 random false positives:

| country | S1 id | S1 name | S1 address | S1 name_core | rec id | rec name | rec address | rec name_core | rec name_alt | p | q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| India | S1-505909107 | QQF Action Private Limited | Shrsoli, Akola, Maharashtra, Telhara, C/O Shri S. M. Gawande | qqf action | S2-926716981 | QQF Action L.L.P. | H.NO #80 C/O SHRI S. M. GAWANDE, SHRSOLI, TELHARA, Maharashtra | qqf action | *(empty)* | 0.533 | 0.533 |
| India | S1-295709784 | One International Pvt Ltd | First Floor, 5, 11/12 Sukhrali Enclave, Gurgaon, Haryana | one international | S3-183778779 | One International Limited | No 45 First Floor, 5, 11/12 Sukhrali Enclave, Gurgaon, HR | one international | *(empty)* | 0.958 | 0.958 |
| India | S1-998919983 | Rajasthan Society Corporation | C/O Madhusudan Jena Bijipur Jajpur Odisha, Bijipur, Jajpur, Orissa | rajasthan society | S2-524323029 | Rajasthan S0ciety LLP | C/O MADHUSUDAN JENA BIJIPUR JAJPUR ODISHA 75502, BIJIPUR, Odisha | rajasthan society | *(empty)* | 0.993 | 0.993 |
| India | S1-457579071 | Sky Consultants Limited | Survey No 99P1/P1 Plot 2, Nh 27, Jambudiya Bhayati, Wankaner, Rajkot, Gujarat | sky consultants | S3-886279497 | Sky Consultants Private Limited | Nh 27, Jambudiya Bhayati, Survey No G-99p1/p3 Plot 2, Wankaner, Rajkot, GJ | sky consultants | *(empty)* | 0.946 | 0.946 |
| India | S1-524914784 | Bharatiya Producer Ltd | 9/A Sundarvan Brahamkshapria Society No 1 Paldi, Ahmedabad, Gujarat | bharatiya producer | S2-160652278 | Bharatiya Producer LLP | DOOR NO 52/A SUNDARVAN BRAHAMKSHAPRIA SOCIETY NO 1 PALDI, AHMEDABAD, Gujarat | bharatiya producer | *(empty)* | 0.792 | 0.792 |
| India | S1-457569549 | Hansraj Management LLP | H.No 2-L/29, Nit, Faridabad, Haryana | hansraj management | S3-9480686 | Hansraj Marnameemnt Limited | No 2-L/3, Nit, Faridabad, HR | hansraj marnameemnt | *(empty)* | 0.844 | 0.844 |
| India | S1-222863997 | Vm Spices Private Limited | Office No.701 To 704, Vantage 9 Baner, Maharashtra, Pune, S.No.36/1/1, Haveli | vm spices | S3-644457605 | vm spices limited | S.no.36/1/3, Office No.701 To 704, Vantage 9 Baner, Pune, Haveli, MH | vm spices | *(empty)* | 0.989 | 0.989 |
| US | S1-344367797 | Empire Federation LLC | 230 Willaman Avenue, North Canton, OH | empire federation | S2-597300490 | Empire Federation Ltd | 23 WILLAMAN AVENUE, NORTH CANTON TOWNSHIP, OH | empire federation | *(empty)* | 0.883 | 0.883 |
| US | S1-468604330 | Todyne | 7647 Robben Road, Dixon, CA | todyne | S3-608703378 | Todyne Corp | PMB 5848, California, 7658 Robben Road, Dixon | todyne | *(empty)* | 0.519 | 0.519 |
| US | S1-958880339 | Northwind LLC | 1 1st Avenue, Unit 1B, Glen Burnie, MD | northwind | S2-548580277 | Northwind Ltd | 6 1TH AVE, GLEN BURNIE, MD | northwind | *(empty)* | 0.901 | 0.901 |
| India | S1-801458525 | One Sangh | South Delhi, New Delhi, Delhi, 303D, Jaina Tower-Ii, District Centre, Janak Puri | one sangh | S3-108176324 | One Sangh Pvt Ltd | South Delhi, New Delhi, 3-08D, DL | one sangh | *(empty)* | 0.935 | 0.935 |
| India | S1-933231868 | Bw Solution | Pune, Fl No 9 Vighnaharta Apts Pl No 368/3 Cts 1020/3 Model Colony, Pune, Maharashtra | bw solution | S3-638299622 | Bw Solution LLP | No 68 Fl No 9 Vighnaharta Apts Pl No 368/3 Cts 1020/12 Model Colony, Pune, MH | bw solution | *(empty)* | 0.805 | 0.805 |
| US | S1-169606115 | Brenda S. Philbeck, Esq. LP | 8338 Traford Lane, Unit STE 8338, Fairfax County, VA | brenda s philbeck esq | S3-432305266 | Brenda S. Philbeck, Esq. Inc | 008342 Traford Lane, Unit STE 8338, Springfield, Virginia | brenda s philbeck esq | *(empty)* | 0.827 | 0.827 |
| US | S1-300493908 | Urology Associates Inc | 20 B Maple Street, Maynard, MA | urology associates | S3-470101575 | Ur0logy LLC Associates | 4 B Maple Street, Maynard, Massachusetts | urology associates | *(empty)* | 0.864 | 0.864 |
| US | S1-4687146 | Aggie's Sterling Recruiting | 4886 Oak Road, Bluffton, IN | aggies sterling recruiting | S3-734756088 | Aggie's Sterling Recruiting Ltd | 488 Oak Road, Bluffton, Indiana | aggies sterling recruiting | *(empty)* | 0.814 | 0.814 |

15 random false negatives (true pairs in the candidates that were rejected):

| country | S1 id | S1 name | S1 address | S1 name_core | rec id | rec name | rec address | rec name_core | rec name_alt | p | q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| India | S1-710661804 | Coimbatore Chain Private Limited | No: 132& 133, 8Th Street, Gandhipuram, Coimbatore, Tamil Nadu | coimbatore chain | S3-413242670 | Coimbatore Private Limited Service | 8Th Street, Gandhipuram, No: 13& 133, தமிழ்நாடு, Coimbatore | coimbatore service | *(empty)* | 0.203 | 0.170 |
| India | S1-184965831 | Mumbai Media Private Limited | 158/4877, Takshashila Chs, Kannamwar Nagar-1, Vikhroli (East), Mumbai, Mumbai City, Maharashtra | mumbai media | S3-377719070 | Mumbai Media Private Ltd | *(empty)* | mumbai media | *(empty)* | 0.056 | 0.032 |
| India | S1-657971052 | Thane Medical Private Limited | Office No.9, 2 Floor S-2, Imperial Height, Thane, Maharashtra | thane medical | S3-187805539 | Thane Thane Medical Limited Private (ID: 11219) | *(empty)* | thane medical | *(empty)* | 0.143 | 0.079 |
| US | S1-616231614 | Classic Immunopharma Inc | 17105 Hwy 24, Wamego, KS | classic immunopharma | S2-629948450 | Classic Immun0pharma | *(empty)* | classic immunopharma | *(empty)* | 0.456 | 0.293 |
| US | S1-272572745 | Bhuiyan Lifestyle Inc | 315 Barley Drive, Newark, DE | bhuiyan lifestyle | S3-403275056 | Bhuiyan Lifestyle Incorporated | Delaware, Newark, 15 Barley Dr | bhuiyan lifestyle | *(empty)* | 0.650 | 0.650 |
| US | S1-443448243 | Mendoza Worldwide LLC | 314 Williams Street, Troy, AL | mendoza worldwide | S2-912215527 | Mendoza Mendoza LLC Worldwide | 312 WILLIAMS STREET, TROY, AL | mendoza worldwide | *(empty)* | 0.507 | 0.507 |
| US | S1-366784539 | Eshelman Care LLC | 1730 Foors Lane, Owensboro, KY | eshelman care | S3-783740000 | Eshelman Care Llc | 1733 Foors Ln, Owensboro, Kentucky | eshelman care | *(empty)* | 0.130 | 0.130 |
| US | S1-607510162 | Quintero Hartford LLC | Waldorf, MD, 4846 Magpie Lane | quintero hartford | S2-867344532 | (LLC) Quintero Hartford | 4847 MAGPIE LN, WALDORF, MD | quintero hartford | *(empty)* | 0.335 | 0.335 |
| India | S1-679220964 | Ace Public School Private Limited | Opp.Police Station Chaoni Nagpur, Nagpur, Maharashtra | ace public school | S3-901639574 | Ace School Private Limited  Center | *(empty)* | ace school center | *(empty)* | 0.026 | 0.012 |
| India | S1-40767240 | Alpha Management Private Limited | Coimbatore, Tamil Nadu, 2 Ramalinga Nagariii Cross Road K K Pudur, Coimbatore | alpha management | S3-930060291 | Jaxjaxf1ux | 2 Ramalinga Nagariii Cross Road K K Pudur, Coimbatore, TN | jaxjaxflux | *(empty)* | 0.309 | 0.183 |
| India | S1-147008433 | Bajrangi & Partners | H No 40-448-503, Vittal Nagar Teja Deluxe Apartments, Kurnool, Andhra Pradesh | bajrangi partners | S3-30848848 | Mr Bajrangi-& | H No 40-448-50, Kurnool, Munagalapadu, ఆంధ్రప్రదేశ్ | bajrangi | *(empty)* | 0.628 | 0.628 |
| US | S1-413499032 | Fiix Chiron Inc | 1718 Underpass Way, Unit LOFT 211, Hagerstown, MD | fiix chiron | S3-181801218 | Fiix Inc Service | *(empty)* | fiix service | *(empty)* | 0.021 | 0.015 |
| US | S1-692013363 | Glenwood Urgent Care Specialists Inc. | 26573 Noyes Avenue, Glenwood, IA | glenwood urgent care specialists | S2-890246451 | GLENWOOD URGENT CARE SPECIALISTS [INCORPORATED] | 26570 NOYES AVENUE, GLENWOOD, IA | glenwood urgent care specialists | *(empty)* | 0.011 | 0.011 |
| US | S1-596017841 | Community Rapid Center | 3949 Lyndale Avenue, Minneapolis, MN | community rapid center | S2-250807387 | COMMUNITY-RAPID | *(empty)* | community rapid | *(empty)* | 0.364 | 0.266 |
| US | S1-252953979 | Granite | N70W17366 Fawn Avenue, Village Of Menomonee Falls, WI | granite | S3-53080418 | Granite | Menomonee Falls, N70w1736 Fawn Ave, Wisconsin | granite | *(empty)* | 0.619 | 0.619 |

Decision search time 2.5 min, peak RSS 7.8 GiB.


### Test submission (run `v2`)

`submissions/02_prune_v2/` with decision `odds` + `ef` `{"gamma": 1.0}`: 5,808,561 matched pairs over 1,732,544 test S1; candidate_pairs.tsv lists all 24,300,351 scored pairs.

Validator (`--check-ids`):

```
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (94053 empty, 1638491 non-empty).
  candidate_pairs.tsv: 1732544 rows (116 empty, 1732428 non-empty).

PASS — no blocking issues found. Safe to submit.
```

Sanity, test vs train OOF (same decision config):

| split | country | S1 | % S1 predicted empty | mean predicted set size | mean q of accepted pairs |
|---|---|---|---|---|---|
| train OOF | India | 883,188 | 5.86% | 3.252 | 0.9900 |
| train OOF | US | 1,323,633 | 5.93% | 3.290 | 0.9945 |
| test | France | 259,452 | 4.85% | 3.438 | 0.9940 |
| test | India | 809,986 | 5.57% | 3.310 | 0.9891 |
| test | US | 663,106 | 5.48% | 3.371 | 0.9932 |

### France: 15 random test S1 (accepted matches, then top-3 rejected candidates)

| S1 id | name | address | role | record id | q |
|---|---|---|---|---|---|
| S1-13985225 | Lille Loisirs SASU | 144 Rue Roger Salengro, Lille, Hauts-de-France | S1 | *(empty)* | *(empty)* |
| *(empty)* | Lille Loisirs | 144 R Roger Salengro, Lille | MATCH | S3-227686224 | 1.000 |
| *(empty)* | Lille-Loisirs | 144 RUE ROGER SALENGRO, LILLE, Hauts-de-France | MATCH | S2-137268766 | 1.000 |
| *(empty)* | LL | 144 Rue Roger Salengro, Lille | MATCH | S3-849615471 | 0.991 |
| *(empty)* | Groupe Lille SASU | 144 RUE ROGER SALENGRO, Hauts-de-France, Lille | MATCH | S2-163717074 | 0.881 |
| *(empty)* | arts & frères france eurl | 144 R. Roger Salengro, Lille | rejected | S3-918404853 | 0.002 |
| *(empty)* | Amicale des Pompiers SA | 144 R Roger Salengro, Lille | rejected | S3-397639182 | 0.001 |
| *(empty)* | Amicale Pompiers des (Distribution) | 144 R ROGER SALENGRO, LILLE, Hauts-de-France | rejected | S2-101372673 | 0.000 |
| S1-746566353 | Deleves Jeunes | 2 Rue Pierre et Marie Curie, Nantes, Pays de la Loire | S1 | *(empty)* | *(empty)* |
| *(empty)* | De1eves Jeunes | No 2 Rue Pirre Et Marie Curie, Nantes, Loire-Atlantique | MATCH | S3-91367773 | 1.000 |
| *(empty)* | Deleves Jeunes | Loire-Atlantique, No 2 Rue Pierre Et Marie Cure, Nantes | MATCH | S3-587161508 | 1.000 |
| *(empty)* | DELEVES JEUNES | #2 RUE PIERRE ET MAROE CURIE, NANTES, Loire-Atlantique | MATCH | S2-678345630 | 1.000 |
| *(empty)* | Deleves Jeunes SA | 3 Rue Pierre Et Marie Curie, Nantes, Pays de la Loire | MATCH | S3-412821487 | 0.939 |
| *(empty)* | Deleves France  Jeunes | N°3 RUE PIERRE ET MARIE CURIE, NANTES, Loire-Atlantique | rejected | S2-239155518 | 0.103 |
| *(empty)* | Deleves Jeunes [SARL] | *(empty)* | rejected | S3-254901695 | 0.032 |
| *(empty)* | Deleves Jcnoes | *(empty)* | rejected | S3-4604854 | 0.026 |
| S1-115610473 | Boules Club SAS | 11 Bis Chemin de la Violerie, Nantes, Pays de la Loire | S1 | *(empty)* | *(empty)* |
| *(empty)* | SAS Boules Club | Loire-Atlantique, Nantes, 11 Bis Chemin De La Violerie | MATCH | S3-769783373 | 1.000 |
| *(empty)* | Boules Club | Loire-Atlantique, Nantes, 11 Bis Ch De La Violerie | MATCH | S3-124509233 | 1.000 |
| *(empty)* | SAS Boules Club | Loire-Atlantique, Nantes, 11 Bis Chemin De La Violerie | MATCH | S3-719394688 | 1.000 |
| *(empty)* | BOULES CLUB SAS | 11 BIS CH. DE LA VILOERIE, NANTES, Loire-Atlantique | MATCH | S2-118722559 | 1.000 |
| *(empty)* | Boules Sante SAS | NANTES, 24 BIS CHEMIN DE LA VIOLERIE | rejected | S2-91384621 | 0.012 |
| *(empty)* | S.A.S Boules Sportive | 24 Bis Ch. De La Violerie, Nantes, Pays de la Loire | rejected | S3-48281074 | 0.003 |
| *(empty)* | Frequence Club EURL | 18 Bis Ch. De La Violerie, Nantes, Pays de la Loire | rejected | S3-670712087 | 0.000 |
| S1-50859199 | Universitaire Amis SAS | 117 Rue de la Pelouse de Douet, Bordeaux, Nouvelle-Aquitaine | S1 | *(empty)* | *(empty)* |
| *(empty)* | Universitaire Comite SAS | 117 RUE DE LA PELOUSE DE DOUET, BORDEAUX, Gironde | MATCH | S2-146819225 | 0.998 |
| *(empty)* | Universitaire Union | 138 R DE LA PELOUSE DE DOUET, BORDEAUX | rejected | S2-862288807 | 0.002 |
| *(empty)* | Robotique Foyer SASU | 117 R De La Pelouse De Douet, Bordeaux, Gironde | rejected | S3-865273917 | 0.001 |
| *(empty)* | Lunion Amis SARL | 83 R De La Pelouse De Douet, Bordeaux, Nouvelle-Aquitaine | rejected | S3-340559672 | 0.000 |
| S1-989309417 | Bordeaux Club SARL | 18 Rue Dubessan, Bordeaux, Nouvelle-Aquitaine | S1 | *(empty)* | *(empty)* |
| *(empty)* | BORDEAUX ÇLUB SARL | 18 R DUBESSAN, BORDEAUX | MATCH | S2-173366715 | 1.000 |
| *(empty)* | Bordeaux Crllnb SARL | 18 R. Dubessan, Nouvelle-Aquitaine, Bordeaux | MATCH | S3-398746343 | 1.000 |
| *(empty)* | Bordeaux  Club Distribution SARL | 25 Rue Dubessan, Bordeaux, Nouvelle-Aquitaine | rejected | S3-266743956 | 0.001 |
| *(empty)* | SARL Taxi Club Participations | 11 Rue Dubessan, Bordeaux | rejected | S3-533203030 | 0.000 |
| *(empty)* | MSSP | 4 R. DUBESSAN, Bordeaux, Gironde | rejected | S2-910581354 | 0.000 |
| S1-702936992 | Union du Faubourgs | 142 Rue Raspail, Lille, Hauts-de-France | S1 | *(empty)* | *(empty)* |
| *(empty)* | Faubourgs du Union (Groupe) | 151 R. RASPAIL, Lille, Hauts-de-France | rejected | S2-443499532 | 0.000 |
| *(empty)* | Union Du Fâubourgs International | 151 Rue Raspail, Lille, Hauts-de-France | rejected | S3-487191250 | 0.000 |
| *(empty)* | Ets Faubourgs SARL France | *(empty)* | rejected | S3-8725 | 0.000 |
| S1-774223557 | Daffaires Sportive SARL | 11 Rue Clément Ader, Tourcoing, Hauts-de-France | S1 | *(empty)* | *(empty)* |
| *(empty)* | (Sarl) Daffaires Sportive | 11 Rue Clement Ader, Tourcoing, Hauts-de-France | MATCH | S3-165106053 | 1.000 |
| *(empty)* | SARL Daffaires Spôrtive | Nord, 11 RUE CLÉMENT ADER, Tourcoing | MATCH | S2-934678039 | 1.000 |
| *(empty)* | Daffaires Sportive SASU | Hauts-de-France, 13 Rue Clément Ader, Tourcoing | MATCH | S3-17866614 | 0.782 |
| *(empty)* | Daffaires & | *(empty)* | rejected | S3-894744964 | 0.067 |
| *(empty)* | Daffaires SAS Développement | *(empty)* | rejected | S2-143789536 | 0.041 |
| *(empty)* | ZA | R CLÉMENT ADER, Tourcoing | rejected | S2-220366321 | 0.001 |
| S1-983257243 | Litalie Groupement (France) EURL | 16 bis Rue de la Joselière, Pornic, Pays de la Loire | S1 | *(empty)* | *(empty)* |
| *(empty)* | Litalie Groupement (France) | Pornic, 16 Bis R De La Joselière | MATCH | S3-623725439 | 1.000 |
| *(empty)* | EURL Litalie Groupement (Frànce) | 16 B RUE DE LA JOSELIÈRE, Pornic, Pays de la Loire | MATCH | S2-972442071 | 1.000 |
| *(empty)* | EURL Litalie Groupement (France) | 16 Bis Rue De La Joseliere, Pornic | MATCH | S3-733576441 | 1.000 |
| *(empty)* | Litalie Club (France) EURL | 16 Bis Rue De La Joselière, Pornic | MATCH | S3-490745586 | 0.994 |
| *(empty)* | Keloecto | 16 bis Rue de la Joselière, Pornic, Pays de la Loire | rejected | S3-303039848 | 0.659 |
| *(empty)* | LITALIE GROUPEMENT (FRANCE) S.N.C. | 27B R DE LA JOSELIÈRE, PORNIC | rejected | S2-102932901 | 0.007 |
| *(empty)* | Pornic Groupe France SARL | 67 Bis Rue De La Joseliere, Pornic, Loire-Atlantique | rejected | S3-945611014 | 0.000 |
| S1-772337801 | Club du Beyond | 13 Impasse Emile Lanusse Cazaux, Nouvelle-Aquitaine, La Teste-de-Buch | S1 | *(empty)* | *(empty)* |
| *(empty)* | Club Beyond  du | 13 IMP. EMILE LANUSSE CAZAUX, LA TESTE-DE-BUCH, Gironde | MATCH | S2-42389812 | 1.000 |
| *(empty)* | Club du Beyond SCI | 13 IMP. EMILE LANUSSE CAZAUX, LA TESTE DE BUCH, Gironde | MATCH | S2-17963325 | 1.000 |
| *(empty)* | Club du [Beyond] | LA TESTE-DE-BUCH, 13 IMP EMILE LANUSSE CAZAUX, Gironde | MATCH | S2-715440609 | 1.000 |
| *(empty)* | CLUB DU BEYOND + ASSOCIÉS | LA TESTE-DE-BUCH, Gironde, 13 IMPASSE EMILE LANUSSE CAZAUX | MATCH | S2-886918475 | 0.970 |
| *(empty)* | Club du Beyond S.A. | 20 Impasse Emile Lanusse Cazaux, La Teste-de-buch | rejected | S3-624683454 | 0.047 |
| *(empty)* | CYN Ecole S.A.R.L. | Gironde, LA TESTE-DE-BUCH, 9 IMPASSE ENILE LANUSSE CAZAUX | rejected | S2-919554451 | 0.000 |
| *(empty)* | CN SARL  Développement | 18 Impasse Emile Lanusse Cazaux, La Teste-de-buch | rejected | S3-648956107 | 0.000 |
| S1-200342034 | Karate Lycée SAS | 15 Rue des Chambelles, Nantes, Pays de la Loire | S1 | *(empty)* | *(empty)* |
| *(empty)* | Karate-Lycée SAS | 15 R. DES CHAMBELLES, NANTES | MATCH | S2-313652548 | 1.000 |
| *(empty)* | Karate Lycée SAS | 15 R. DES CHAMBELLES, NANTES | MATCH | S2-57598555 | 1.000 |
| *(empty)* | Karate Lyfee EURL | *(empty)* | rejected | S2-972980486 | 0.207 |
| *(empty)* | KARATE LYCÉE SA | (18) Rue Des Chambelles, Nantes | rejected | S3-7005314 | 0.036 |
| *(empty)* | SAS Karate Lycée Développement | (18) Rue Des Chambelles, Nantes | rejected | S3-276350007 | 0.001 |
| S1-507990218 | Scorpio Club | 32 Rue du Corsaire, Dunkerque, Hauts-de-France | S1 | *(empty)* | *(empty)* |
| *(empty)* | Scorpio Club | 32 Rue Du Corsaire, Nord, Dunkerque | MATCH | S3-783467101 | 1.000 |
| *(empty)* | Scorpio CLUB | 32 R. Du Corsaire, Dunkerque, Nord | MATCH | S3-284097524 | 1.000 |
| *(empty)* | Scorpio Club SA | 32 RUE DU CORSAIRE, DUNKERQUE, Hauts-de-France | MATCH | S2-594979572 | 1.000 |
| *(empty)* | Scorpio Club SARL | 32 R. Du Corsaire, Dunkerque, Nord | MATCH | S3-604351882 | 1.000 |
| *(empty)* | Scorpio Sportive | 32 Rue Du Corsaire, Dunkerque, Nord | MATCH | S3-142231258 | 1.000 |
| *(empty)* | Scorpio Cb | 32 Rue Du Corsaire, Dunkerque, Nord | MATCH | S3-795342602 | 0.999 |
| *(empty)* | Scorpio Et  Fils | 32 Rue Du Corsaire, Dunkerque, Nord | MATCH | S3-507677700 | 0.999 |
| *(empty)* | Scorpio Conseil | 43 R DU CORSAIRE, DUNKERQUE | rejected | S2-698869479 | 0.004 |
| *(empty)* | SC0RPIO CLUB PARTICIPATIONS | 43 R DU CIRSAIRE, Dunkerque | rejected | S2-592001586 | 0.000 |
| *(empty)* | @lochcri | 24 Rue Du Corsaire, Dunkerque, Nord | rejected | S3-650727856 | 0.000 |
| S1-288686294 | Reseau & Frères SARL | 85 Rue Croix de Seguey, Bordeaux, Nouvelle-Aquitaine | S1 | *(empty)* | *(empty)* |
| *(empty)* | reseau & frères sarl | 85 Rue Croix De Seguey, Bordeaux, Gironde | MATCH | S3-748990621 | 1.000 |
| *(empty)* | Reseau & | 85 Rue Croix De Seguey, Bordeaux, Gironde | MATCH | S3-273137125 | 1.000 |
| *(empty)* | Reseau & Frères SARL | Nº 85 R. CROIX DE SEGUEY, BORDEAUX | MATCH | S2-904891446 | 1.000 |
| *(empty)* | Reseau & Frèrës SARL | 85 R. Crsix De Seguey, Bordeaux, Gironde | MATCH | S3-830040073 | 1.000 |
| *(empty)* | France Reseau & SARL | 85 Rue Croix De Seguey, Bordeaux, Gironde | MATCH | S3-732341307 | 0.999 |
| *(empty)* | Yn Coiffure | 85 Rue Croix De Seguey, Bordeaux, Gironde | rejected | S3-93427129 | 0.299 |
| *(empty)* | Chambre Service EURL | 85 Rue Croix De Seguey, Bordeaux, Gironde | rejected | S3-173149925 | 0.017 |
| *(empty)* | CHAMBRE SANTE DÉVELOPPEMENT | BORDEAUX, Nº 85 R CROIX DE SEGUEY | rejected | S2-85740888 | 0.001 |
| S1-549612175 | IJE Culturelle SARL | 70 Rue Champailler, Calais, Hauts-de-France | S1 | *(empty)* | *(empty)* |
| *(empty)* | IJE Culturelle SARL | 70 R CHAMPAILLER, CALAIS | MATCH | S2-776891879 | 1.000 |
| *(empty)* | IJE CULTURELLE SARL | CALAIS, 70 RUE CHAMPAILLER | MATCH | S2-154038975 | 1.000 |
| *(empty)* | IJE Culturelle SARL | 70 Rue Champailler, Calais, Hauts-de-France | MATCH | S3-861937670 | 1.000 |
| *(empty)* | IJE Çulturelle SARL | 70 Rue Champailler, Calais, Hauts-de-France | MATCH | S3-618325488 | 1.000 |
| *(empty)* | IJE CLUB SARL | 70 R. CHAMPAELLER, CALAIS | MATCH | S2-333551947 | 0.999 |
| *(empty)* | SARL IJE Ecole | 70 R. CHAMPIALLER, Calais | MATCH | S2-676500959 | 0.999 |
| *(empty)* | IJE Culturelle SA | *(empty)* | rejected | S3-216040750 | 0.285 |
| *(empty)* | Reine Culture Groupe SARL | 19 R. Champailler, Calais, Hauts-de-France | rejected | S3-184356212 | 0.000 |
| *(empty)* | cultuelleunionsarl.com | 58 R. CHAMPAILLER, CALAIS, Pas-de-Calais | rejected | S2-505376124 | 0.000 |
| S1-387739123 | Bourse Service SAS | 153, rue David Johnston, Bordeaux, Nouvelle-Aquitaine | S1 | *(empty)* | *(empty)* |
| *(empty)* | bourse service sas | 153, Rue David Johnston, Bordeaux | MATCH | S3-174546589 | 1.000 |
| *(empty)* | Bourse Service SCI | Gironde, Bordeaux, 71, 153, Rue David Johnston | MATCH | S3-6058603 | 0.996 |
| *(empty)* | SOLUMBRACALO | 153 R DAVID JOHNSTON, Bordeaux, Nouvelle-Aquitaine | rejected | S2-696587485 | 0.048 |
| *(empty)* | Novitavoarc | 153, RUE DAVID JOHNSTON, Nouvelle-Aquitaine, Bordeaux | rejected | S2-806820964 | 0.039 |
| *(empty)* | BOURSE SERVICE PARTICIPATIONS SAS | 71, 153, RUE DAVID JOHNSTON, BORDEAUX | rejected | S2-69727110 | 0.019 |
| S1-45027199 | Centre Hospitalier Sainte Dame | 24 Rue Castel, Lille, Hauts-de-France | S1 | *(empty)* | *(empty)* |
| *(empty)* | CENTRE HOSPITALIER SÀINTE DAME | 24 RUE CASTEL, LILLE, Nord | MATCH | S2-291576111 | 1.000 |
| *(empty)* | Centre Hospitalier  Sainte Dame | 24 Rue Castel, Lille, Hauts-de-France | MATCH | S3-719764905 | 1.000 |
| *(empty)* | Centre Hospitalier Sainte Dame | 24 RUE CASTEL, LILLE, Nord | MATCH | S2-480952141 | 1.000 |
| *(empty)* | CENTRE HOSPITALIER SAINTE DLER | 24 R. CASTEL, LILLE, Nord | MATCH | S2-858025055 | 0.997 |
| *(empty)* | Établissements Falcon [Développement] | LILLE, 33 RUE CASTEL | rejected | S2-141930164 | 0.000 |
| *(empty)* | Genetique (France) Maison SARL | 37 R Castel, Lille | rejected | S3-851239493 | 0.000 |
| *(empty)* | Nylaevosynio | 7 Rue de la Ferme Castel, Lille, Hauts-de-France | rejected | S3-92625251 | 0.000 |

Submission step 3.3 min.


## Part E. Stage-2 model with group-consistency features -> submissions/03_stage2/

*train.md: not run / failed (see notes).*

*decide.md: not run / failed (see notes).*

*submit.md: not run / failed (see notes).*
