# Matcher report (v1)

Decision: **odds + ef {"gamma": 1.0}**, OOF macro F0.5 **0.97071** (India 0.96381, US 0.97532, singleton 0.95219, non_singleton 0.97181).

## 1. Pair features

72 features per candidate pair (country is not a feature): `name_tsr`, `name_tsort`, `name_ratio`, `name_partial`, `name_jw`, `name_alt_tsr`, `concat_ratio`, `phon_tsr`, `addr_tsr`, `addr_tsort`, `addr_ratio`, `addr_partial`, `name_jaccard`, `name_cov_s`, `name_cov_r`, `name_first_tok`, `name_rarest_in_r`, `name_exact`, `concat_contain`, `addr_cov_s`, `addr_cov_r`, `nums_jaccard`, `nums_frac_s_in_r`, `house_eq`, `house_logdiff`, `house_prefix`, `house_missing`, `name_ntok_s`, `name_ntok_r`, `addr_ntok_s`, `addr_ntok_r`, `name_len_diff`, `state_match`, `rec_src_s3`, `rec_addr_empty`, `rec_name_has_indic`, `rec_name_is_web`, `rec_name_has_marker`, `rec_addr_has_landmark`, `rec_addr_has_pobox`, `chain_s`, `chain_r`, `r_tok_in_s1_vocab`, `key_name_pair`, `key_name_single`, `key_name_phon`, `key_concat_prefix`, `key_name_alt`, `key_addr_pair`, `key_addr_number`, `key_cross`, `nkeys`, `score`, `sim_name`, `sim_addr`, `rank_r`, `rank_s`, `rank_r_all`, `rank_s_all`, `n_cand_s1`, `n_cand_rec`, `r_n_name90`, `s_rank_name`, `s_rank_addr`, `s_gap_best`, `s_n_near`, `r_oth_name`, `r_margin_name`, `r_oth_addr`, `r_margin_addr`, `r_oth_score`, `r_margin_score`.

| split | country | pairs | minutes |
|---|---|---|---|
| train | India | 14,526,330 | ~12 (built in a first run that hit a Windows file-lock error while deleting its temp file; `--resume` skipped it afterwards) |
| train | US | 21,505,538 | 10.1 |
| test | France | 4,727,989 | 1.9 |
| test | India | 15,376,874 | 11.9 |
| test | US | 12,400,644 | 5.7 |
| train | total | *(empty)* | ~22 (US 10.1 in the resumed run; peak RSS 7.8 GiB) |
| test | total | *(empty)* | 19.5 (peak RSS 7.1 GiB) |

## 2. Model (LightGBM, 2-fold out-of-fold)

72 features. Params `{'objective': 'binary', 'metric': 'binary_logloss', 'num_leaves': 127, 'learning_rate': 0.05, 'min_data_in_leaf': 200, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': 1.0, 'seed': 42, 'deterministic': True, 'force_col_wise': True, 'num_threads': 16}`, up to 3000 rounds, early stopping 100 (logloss) on a 10% S1 hold-out. Each fold trains on a random 30% of its half's S1s and predicts every pair of the other half; test = mean of both models.

| model | trained on half | rounds | train S1 | train pairs | early-stop S1 |
|---|---|---|---|---|---|
| 0 | 0 | 1654 | 298,355 | 4,869,402 | 32,762 |
| 1 | 1 | 1837 | 298,083 | 4,872,668 | 33,176 |

OOF metrics (model h predicts the other half; raw p before any isotonic step):

| model | country | pairs | AUC | logloss |
|---|---|---|---|---|
| 0 | ALL | 18,010,146 | 0.99972 | 0.01684 |
| 0 | India | 7,261,359 | 0.99963 | 0.01916 |
| 0 | US | 10,748,787 | 0.99977 | 0.01528 |
| 1 | ALL | 18,021,722 | 0.99972 | 0.01683 |
| 1 | India | 7,264,971 | 0.99963 | 0.01911 |
| 1 | US | 10,756,751 | 0.99976 | 0.01528 |

Calibration of OOF p (10 bins). Worst gap 0.0192 -> isotonic not needed.

| bin | pairs | mean p | positive rate | gap |
|---|---|---|---|---|
| 0.0-0.1 | 28,324,292 | 0.0009 | 0.0010 | -0.0001 |
| 0.1-0.2 | 147,316 | 0.1456 | 0.1648 | -0.0192 |
| 0.2-0.3 | 100,775 | 0.2471 | 0.2590 | -0.0118 |
| 0.3-0.4 | 76,770 | 0.3470 | 0.3472 | -0.0002 |
| 0.4-0.5 | 67,182 | 0.4513 | 0.4490 | +0.0023 |
| 0.5-0.6 | 58,223 | 0.5466 | 0.5284 | +0.0182 |
| 0.6-0.7 | 48,406 | 0.6508 | 0.6343 | +0.0165 |
| 0.7-0.8 | 62,412 | 0.7532 | 0.7429 | +0.0103 |
| 0.8-0.9 | 115,980 | 0.8571 | 0.8537 | +0.0035 |
| 0.9-1.0 | 7,030,512 | 0.9964 | 0.9965 | -0.0001 |

Top-30 features by gain, model 0:

| feature | gain |
|---|---|
| r_margin_score | 22,707,086 |
| rank_r | 5,569,654 |
| house_logdiff | 2,578,133 |
| nums_frac_s_in_r | 1,421,552 |
| nums_jaccard | 1,182,183 |
| name_cov_r | 1,043,956 |
| addr_cov_r | 721,319 |
| score | 466,653 |
| name_len_diff | 460,508 |
| r_margin_addr | 238,420 |
| sim_addr | 236,265 |
| nkeys | 191,040 |
| addr_ntok_r | 190,489 |
| addr_tsr | 166,043 |
| r_oth_score | 139,903 |
| addr_tsort | 136,367 |
| chain_s | 129,813 |
| house_eq | 114,677 |
| name_first_tok | 108,721 |
| house_missing | 105,753 |
| r_tok_in_s1_vocab | 105,741 |
| addr_cov_s | 102,112 |
| name_ntok_r | 97,789 |
| addr_partial | 96,046 |
| name_jw | 94,855 |
| name_cov_s | 93,478 |
| name_tsort | 93,422 |
| s_rank_addr | 81,661 |
| chain_r | 81,286 |
| s_gap_best | 80,552 |

Top-30 features by gain, model 1:

| feature | gain |
|---|---|
| r_margin_score | 22,732,585 |
| rank_r | 5,577,615 |
| house_logdiff | 3,027,872 |
| nums_jaccard | 1,169,768 |
| name_cov_r | 1,074,030 |
| nums_frac_s_in_r | 1,066,066 |
| addr_cov_r | 590,789 |
| score | 475,752 |
| name_len_diff | 469,661 |
| r_margin_addr | 244,472 |
| sim_addr | 243,344 |
| nkeys | 204,668 |
| addr_tsr | 175,016 |
| addr_ntok_r | 154,608 |
| addr_tsort | 139,108 |
| chain_s | 136,480 |
| r_oth_score | 121,731 |
| name_first_tok | 117,411 |
| name_tsort | 109,242 |
| addr_partial | 107,733 |
| r_tok_in_s1_vocab | 105,218 |
| house_missing | 102,648 |
| addr_cov_s | 98,259 |
| rank_s | 97,952 |
| name_jw | 96,066 |
| name_cov_s | 95,848 |
| name_ntok_r | 93,713 |
| addr_ntok_s | 91,313 |
| chain_r | 85,687 |
| s_rank_addr | 85,586 |

Training + prediction: 80.0 min, peak RSS 5.9 GiB.

## 3. Decision layer (tuned on OOF over all train S1)

297 configurations evaluated (3 D1 x (13 thresholds + 81 two-threshold pairs + 5 expected-F gammas)). S1s without candidates and true pairs missing from the candidates (290,922) count against recall.

**Winner: D1 = `odds`, D2 = `ef`, params `{"gamma": 1.0}` -> OOF macro F0.5 0.97071** (re-scored with `metrics.macro_f05`: 0.97071).

Best configuration of each (D1, D2) family:

| D1 | D2 | params | OOF F0.5 | India | US | singleton | non-singleton |
|---|---|---|---|---|---|---|---|
| odds | ef | {"gamma": 1.0} | 0.97071 | 0.96381 | 0.97532 | 0.9522 | 0.9718 |
| argmax | ef | {"gamma": 1.2} | 0.97060 | 0.96376 | 0.97516 | 0.9577 | 0.9714 |
| none | ef | {"gamma": 1.2} | 0.97052 | 0.96365 | 0.97510 | 0.9562 | 0.9714 |
| odds | two | {"t1": 0.6, "t2": 0.7} | 0.97050 | 0.96358 | 0.97513 | 0.9638 | 0.9709 |
| odds | thr | {"t": 0.65} | 0.97045 | 0.96353 | 0.97507 | 0.9677 | 0.9706 |
| argmax | two | {"t1": 0.6, "t2": 0.7} | 0.97041 | 0.96347 | 0.97504 | 0.9628 | 0.9709 |
| argmax | thr | {"t": 0.65} | 0.97035 | 0.96343 | 0.97496 | 0.9670 | 0.9705 |
| none | two | {"t1": 0.6, "t2": 0.7} | 0.97032 | 0.96334 | 0.97498 | 0.9621 | 0.9708 |
| none | thr | {"t": 0.7} | 0.97025 | 0.96328 | 0.97490 | 0.9710 | 0.9702 |

All global-threshold and expected-F configurations:

| D1 | D2 | params | OOF F0.5 | India | US | singleton | non-singleton |
|---|---|---|---|---|---|---|---|
| argmax | ef | {"gamma": 0.7} | 0.96988 | 0.96291 | 0.97453 | 0.9280 | 0.9724 |
| argmax | ef | {"gamma": 1.5} | 0.97028 | 0.96347 | 0.97483 | 0.9647 | 0.9706 |
| argmax | ef | {"gamma": 0.85} | 0.97038 | 0.96346 | 0.97500 | 0.9406 | 0.9721 |
| argmax | ef | {"gamma": 1.0} | 0.97057 | 0.96366 | 0.97518 | 0.9494 | 0.9718 |
| argmax | ef | {"gamma": 1.2} | 0.97060 | 0.96376 | 0.97516 | 0.9577 | 0.9714 |
| argmax | thr | {"t": 0.3} | 0.96366 | 0.95618 | 0.96865 | 0.9173 | 0.9664 |
| argmax | thr | {"t": 0.9} | 0.96523 | 0.95654 | 0.97103 | 0.9881 | 0.9639 |
| argmax | thr | {"t": 0.35} | 0.96545 | 0.95796 | 0.97044 | 0.9281 | 0.9677 |
| argmax | thr | {"t": 0.4} | 0.96687 | 0.95942 | 0.97184 | 0.9364 | 0.9687 |
| argmax | thr | {"t": 0.45} | 0.96793 | 0.96055 | 0.97284 | 0.9440 | 0.9693 |
| argmax | thr | {"t": 0.85} | 0.96799 | 0.96035 | 0.97309 | 0.9834 | 0.9671 |
| argmax | thr | {"t": 0.5} | 0.96887 | 0.96165 | 0.97369 | 0.9510 | 0.9699 |
| argmax | thr | {"t": 0.8} | 0.96938 | 0.96220 | 0.97417 | 0.9795 | 0.9688 |
| argmax | thr | {"t": 0.55} | 0.96968 | 0.96260 | 0.97441 | 0.9576 | 0.9704 |
| argmax | thr | {"t": 0.75} | 0.97003 | 0.96304 | 0.97470 | 0.9754 | 0.9697 |
| argmax | thr | {"t": 0.6} | 0.97014 | 0.96318 | 0.97479 | 0.9628 | 0.9706 |
| argmax | thr | {"t": 0.7} | 0.97032 | 0.96339 | 0.97495 | 0.9714 | 0.9703 |
| argmax | thr | {"t": 0.65} | 0.97035 | 0.96343 | 0.97496 | 0.9670 | 0.9705 |
| none | ef | {"gamma": 0.7} | 0.96943 | 0.96243 | 0.97409 | 0.9186 | 0.9724 |
| none | ef | {"gamma": 0.85} | 0.97008 | 0.96314 | 0.97472 | 0.9346 | 0.9722 |
| none | ef | {"gamma": 1.5} | 0.97024 | 0.96340 | 0.97480 | 0.9639 | 0.9706 |
| none | ef | {"gamma": 1.0} | 0.97042 | 0.96348 | 0.97506 | 0.9464 | 0.9718 |
| none | ef | {"gamma": 1.2} | 0.97052 | 0.96365 | 0.97510 | 0.9562 | 0.9714 |
| none | thr | {"t": 0.3} | 0.96168 | 0.95437 | 0.96655 | 0.9067 | 0.9649 |
| none | thr | {"t": 0.35} | 0.96412 | 0.95668 | 0.96908 | 0.9207 | 0.9667 |
| none | thr | {"t": 0.9} | 0.96522 | 0.95651 | 0.97103 | 0.9880 | 0.9639 |
| none | thr | {"t": 0.4} | 0.96588 | 0.95844 | 0.97085 | 0.9307 | 0.9680 |
| none | thr | {"t": 0.45} | 0.96723 | 0.95982 | 0.97217 | 0.9400 | 0.9688 |
| none | thr | {"t": 0.85} | 0.96797 | 0.96031 | 0.97308 | 0.9832 | 0.9671 |
| none | thr | {"t": 0.5} | 0.96849 | 0.96121 | 0.97335 | 0.9489 | 0.9696 |
| none | thr | {"t": 0.8} | 0.96934 | 0.96214 | 0.97415 | 0.9792 | 0.9688 |
| none | thr | {"t": 0.55} | 0.96948 | 0.96234 | 0.97425 | 0.9565 | 0.9703 |
| none | thr | {"t": 0.75} | 0.96998 | 0.96294 | 0.97467 | 0.9751 | 0.9697 |
| none | thr | {"t": 0.6} | 0.97001 | 0.96299 | 0.97470 | 0.9621 | 0.9705 |
| none | thr | {"t": 0.65} | 0.97025 | 0.96328 | 0.97490 | 0.9665 | 0.9705 |
| none | thr | {"t": 0.7} | 0.97025 | 0.96328 | 0.97490 | 0.9710 | 0.9702 |
| odds | ef | {"gamma": 0.7} | 0.97007 | 0.96308 | 0.97474 | 0.9300 | 0.9724 |
| odds | ef | {"gamma": 1.5} | 0.97033 | 0.96352 | 0.97487 | 0.9656 | 0.9706 |
| odds | ef | {"gamma": 0.85} | 0.97060 | 0.96366 | 0.97522 | 0.9442 | 0.9722 |
| odds | ef | {"gamma": 1.2} | 0.97068 | 0.96385 | 0.97524 | 0.9594 | 0.9714 |
| odds | ef | {"gamma": 1.0} | 0.97071 | 0.96381 | 0.97532 | 0.9522 | 0.9718 |
| odds | thr | {"t": 0.3} | 0.96431 | 0.95660 | 0.96945 | 0.9188 | 0.9670 |
| odds | thr | {"t": 0.9} | 0.96523 | 0.95654 | 0.97103 | 0.9883 | 0.9639 |
| odds | thr | {"t": 0.35} | 0.96633 | 0.95863 | 0.97146 | 0.9315 | 0.9684 |
| odds | thr | {"t": 0.4} | 0.96777 | 0.96014 | 0.97285 | 0.9406 | 0.9694 |
| odds | thr | {"t": 0.85} | 0.96801 | 0.96037 | 0.97310 | 0.9837 | 0.9671 |
| odds | thr | {"t": 0.45} | 0.96874 | 0.96123 | 0.97374 | 0.9479 | 0.9700 |
| odds | thr | {"t": 0.8} | 0.96940 | 0.96223 | 0.97419 | 0.9798 | 0.9688 |
| odds | thr | {"t": 0.5} | 0.96947 | 0.96218 | 0.97434 | 0.9539 | 0.9704 |
| odds | thr | {"t": 0.55} | 0.97000 | 0.96288 | 0.97475 | 0.9593 | 0.9706 |
| odds | thr | {"t": 0.75} | 0.97007 | 0.96308 | 0.97474 | 0.9758 | 0.9697 |
| odds | thr | {"t": 0.6} | 0.97033 | 0.96335 | 0.97498 | 0.9638 | 0.9707 |
| odds | thr | {"t": 0.7} | 0.97039 | 0.96346 | 0.97501 | 0.9718 | 0.9703 |
| odds | thr | {"t": 0.65} | 0.97045 | 0.96353 | 0.97507 | 0.9677 | 0.9706 |

Winner breakdown:

| slice | OOF macro F0.5 |
|---|---|
| overall | 0.97071 |
| India | 0.96381 |
| US | 0.97532 |
| singleton | 0.95219 |
| non_singleton | 0.97181 |

Predicted vs true set size (number of train S1):

| size | true | predicted |
|---|---|---|
| 0 | 123,247 | 130,032 |
| 1 | 119,157 | 173,501 |
| 2 | 375,212 | 421,380 |
| 3 | 530,841 | 537,962 |
| 4 | 484,115 | 454,489 |
| 5 | 321,957 | 284,655 |
| 6-10 | 252,255 | 204,771 |
| >10 | 37 | 31 |

Best configuration per country (vs the global winner on that country):

| country | D1 | D2 | params | best F0.5 | winner F0.5 |
|---|---|---|---|---|---|
| India | odds | ef | {"gamma": 1.2} | 0.96385 | 0.96381 |
| US | odds | ef | {"gamma": 1.0} | 0.97532 | 0.97532 |

Error analysis: 50,186 false-positive pairs, 208,272 false-negative pairs among candidates (+ 290,922 true pairs never in the candidates).

15 random false positives:

| country | S1 id | S1 name | S1 address | S1 name_core | rec id | rec name | rec address | rec name_core | rec name_alt | p | q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| India | S1-899007667 | Parmar Project Private Limited | S-2, F Block International Trade Tower, Nehru Place, New Delhi, South Delhi, Delhi | parmar project | S3-245817110 | Project Parmar LLP | S-4, F Block International Trade Tower, Nehru Place, South Delhi, New Delhi, DL | project parmar | *(empty)* | 0.800 | 0.800 |
| India | S1-274643464 | Devi Vidyalaya Private Limited | S.No.12/1/1, Jaykamal, Morya Park, Lane No-1, Pimple Gurav, Pune, Maharashtra | devi vidyalaya | S3-565081155 | Devi Vidyalaya Limited | No. 436 S.no.12/1/14, Pune, MH | devi vidyalaya | *(empty)* | 0.998 | 0.998 |
| India | S1-243024312 | Sd Exports Private Limited | Sohagpur, Main Market Lodge, R2 No 5621, Hoshangabad, Madhya Pradesh | sd exports | S3-47379842 | L.L.P. Sd Exports | No 28 Main Market Lodme, R2 No 5621, Sohagpur, Hoshangabad, मध्य प्रदेश | sd exports | *(empty)* | 0.981 | 0.981 |
| India | S1-772915646 | Maa Seva Samiti | Unit No.19, H.No774, Gala18-19, Bldg. F-8 Bhumi World, Pimplas Village, Bhiwandi, Bhiwandi, Thane, Maharashtra | maa seva samiti | S3-400153202 | Maa Seva Samiti Private Limited | Bhiwandi, Thane, MH, H.no774, Gala18-19, Bldg. F-8 Bhumi World, Pimplas Village, Bhiwandi, Unit No.23 | maa seva samiti | *(empty)* | 0.968 | 0.968 |
| India | S1-731281018 | Products Aka Sugar Private Limited | H/O Sangram Keshari Rath Kantabada, Bharatpur, Khandagiri, Bhubaneswar, Khordha, Orissa | products sugar | S3-530081777 | Products Aka Sugar Limited | ଓଡ଼ିଶା, #76 H/o Sangram Keshari Rath Kantabada, Bharatpur, Khandagiri, Bhubaneswar | products sugar | sugar | 0.993 | 0.993 |
| US | S1-74337047 | Zimmerman Gen LLC | 77 Arrowhead Avenue, Riverhead, NY | zimmerman gen | S2-697039621 | Zimmerman Diagnostics | 77 ARROWHEAD AVE, RIVERHEAD, NY | zimmerman diagnostics | *(empty)* | 0.975 | 0.975 |
| US | S1-24080878 | Downtown Cafe | City Of Muskego, Unit Apartment B2, W182S8450 Racine Avenue, WI | downtown cafe | S3-868694903 | The D0wntown Cafe Ltd | W182s8463- Racine Avenue, # Apartment B2, Muskeg CITY, Wisconsin | downtown cafe | *(empty)* | 0.800 | 0.800 |
| US | S1-407988798 | E 2 L Biotech Co | 151 Marshall Avenue, Pemberville, OH | e 2 biotech | S2-493822312 | E 2 L Bíotech Ltd | 16 MARSHALL AVE, PEMBERVILLE, OH | e 2 biotech | *(empty)* | 0.894 | 0.894 |
| India | S1-556943096 | Premier Media Private Limited | Ashanagar, P.O. Sahsarai, Patna Nalanda, Patna, Bihar | premier media | S3-589948822 | प्रीमियर मीडिया लिमिटेड | Patna, Patna Nalanda, BR, No 65 Ashanaagr, P.o. Sahsarai | premier media | *(empty)* | 0.911 | 0.911 |
| India | S1-909803370 | Services Personal Solutions of Chennai | Chennai, No-20/12, 8Th Street, Anna Nagar East, M-Block, Chennai, Tamil Nadu | services personal solutions chennai | S3-364238932 | of Personal Technologies Services Chennai | #70 M-block, தமிழ்நாடு, No-20/12, 8Th Street, Anna Nagar East, Chennai | personal technologies services chennai | *(empty)* | 0.945 | 0.945 |
| US | S1-82147595 | Community Fellowship Inc | WV, Rainelle, Unit 10, 362 Pennsylvania Avenue | community fellowship | S2-535734048 | Community Fellowship LLC | 369 1/2 PENNSYLVANIA AVENUE, WV, RAINELE | community fellowship | *(empty)* | 0.831 | 0.831 |
| US | S1-907976608 | Surgical Associates Inc | 3620 Forestdale Avenue, Prince William County, VA | surgical associates | S3-563608252 | Co Surgical Assocates | 362 Forestdale Ave, PO Box 9059, Prince William County, Virginia | surgical assocates | *(empty)* | 0.795 | 0.795 |
| US | S1-483512245 | Great Prime Robotics | Newport, 255 Legacy Lane, NC | great prime robotics | S3-691225019 | Great Prime Robotics Ltd | 59 Legacy Ln, Newport, North Carolina | great prime robotics | *(empty)* | 0.526 | 0.526 |
| US | S1-223158005 | Raf's Ace Insurance PLLC | 3554 Old Woods Mill Road, Gloucester County, VA | rafs ace insurance | S3-425661248 | Raf's Ace  Insurance Inc | *(empty)* | rafs ace insurance | *(empty)* | 0.985 | 0.985 |
| US | S1-704461304 | Veterans Crystal Program P.C. | VA, 1181 Duncan Drive, York County | veterans crystal program | S2-887828234 | ... veterans crysta1 program ltd | WILLIAMSBURG, 118 DUNCAN DR, VA | veterans crystal program | *(empty)* | 0.886 | 0.886 |

15 random false negatives (true pairs in the candidates that were rejected):

| country | S1 id | S1 name | S1 address | S1 name_core | rec id | rec name | rec address | rec name_core | rec name_alt | p | q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| India | S1-311195279 | Amaze Holdings Associates | N 301, 3Rd Floor, North Block Manipal Centre, Bangalore, Karnataka | amaze holdings associates | S3-568527625 | Amaze Holdings | N 601, <NULL>, Bangalore, KA | amaze holdings | *(empty)* | 0.446 | 0.445 |
| India | S1-606942885 | Global International Private Limited | Rawalwasia Houseold Mandi Road Hisar, Haryana, Central Delhi, Delhi | global international | S2-822075565 | ग्लोबल इंटरनेशनल प्राइवेट लिमिटेड | DOOR NO 52 RAWALWASIA HOUSEOLD MANDI ROAD HISAR, HARYANA, CENTRAL DELHI, दिल्ली | global international | *(empty)* | 0.702 | 0.702 |
| India | S1-466620919 | OS Marketing Private Limited | 711(3), Floor-Grd, Salamati Hill, Hemant Manjarekar Marg, Sardar Ngr 3 Sion, Koliwada, Sion(E), Mumbai, Mumbai City, Maharashtra | os marketing | S3-233041571 | OS Marketing Private Limited | *(empty)* | os marketing | *(empty)* | 0.249 | 0.233 |
| India | S1-213867395 | UQV Projects Private Limited | Sco 9 Peer Muchhala, Zirakpur, Mohali, Punjab | uqv projects | S3-56938600 | UQV Projects Private-Limited | *(empty)* | uqv projects | *(empty)* | 0.480 | 0.334 |
| US | S1-87628486 | Foot & Ankle Health PC | NC, Charlotte, Unit E, 5508 Keyway Boulevard | foot ankle health | S2-275439525 | Foot  & Ankle Health | 5491  KEYWAY BOULEVARD, CHARLOTE, NC | foot ankle health | *(empty)* | 0.677 | 0.677 |
| US | S1-809780269 | Reliable Wireless Brands | 5400 Auburn Boulevard, Unit 326, Sacramento, CA | reliable wireless brands | S3-15511094 | reliable wireless brands | *(empty)* | reliable wireless brands | *(empty)* | 0.420 | 0.420 |
| US | S1-853359943 | Primary Care Group | 10701 99th Avenue, Unit 174, Peoria, AZ | primary care group | S2-810950752 | Primary Care Group | 10701-10703 99TH AVENUE, PEORIA, AZ | primary care group | *(empty)* | 0.676 | 0.676 |
| India | S1-518648949 | Cig Brothers Private Limited | Sriramnagar, First, Floor, Unit F1, Navalur, Kanchipuram, Chinglepet, Tamil Nadu, No.2 | cig brothers | S3-647543776 | Cig Cig Brothers Private | H.no 6, Chinglepet, Kanchipuram, TN | cig brothers | *(empty)* | 0.150 | 0.150 |
| India | S1-944283607 | Alfa Realty Public Limited | New Dtc Corporation 3Rd, Floor Office 308 Haripura, Surat City, Surat, Gujarat | alfa realty public | S2-354881368 | Drexdrex | NEW DTC CORPORFTION 3RD, SURAT, Gujarat | drexdrex | *(empty)* | 0.441 | 0.441 |
| India | S1-816930701 | Spider (India) Nirman Ltd | No.6 1St Floor 3Rd Cross 11Th Main Maruthinagar Kamakshipalya, Bengaluru, Bangalore, Karnataka | spider india nirman | S2-670230599 | Spider (India) Nirman | NO.3 1ST FLOOR 3RD CROSS 11TH MAIN MARUTHINAGAR KAMAKSHIPALYA, BENGALURU, Karnataka | spider india nirman | *(empty)* | 0.678 | 0.678 |
| India | S1-76455069 | Blue Power Private Limited | 221-222, 2Nd Floor Indraprasth Tower 6, M.G. Road, Indore, Madhya Pradesh | blue power | S3-859883555 | Blue Power Limited Sérvices | 26 221-222, Indore, मध्य प्रदेश | blue power services | *(empty)* | 0.617 | 0.617 |
| US | S1-642583475 | Gabbi Davis Associates Inc. | 40014 Matt-neal Road, Norwood, Unit 12, NC | gabbi davis associates | S3-702172603 | davis, gabbi associates inc. | 40013 Matt-Neal Road, Norwod CITY, North Carolina, # 12 | davis gabbi associates | *(empty)* | 0.630 | 0.630 |
| US | S1-568961027 | Atlantic Innovation, LLC | WV, Buckhannon, 169 Pocahontas Street | atlantic innovation | S2-366057335 | Atlantic Ínnovation, LLC | *(empty)* | atlantic innovation | *(empty)* | 0.468 | 0.327 |
| US | S1-187211613 | Federal Vanguard Life LLC | 529 Venture Boulevard, Leander, TX | federal vanguard life | S3-778184732 | Federal Vanguard Life [LLC] | 528 Venture Blvd, Leander, Texas | federal vanguard life | *(empty)* | 0.146 | 0.146 |
| US | S1-458470932 | KZ Legato LLC | 31 Greenpoint Avenue, Brooklyn, NY | kz legato | S3-750614626 | KZ Liaddo  LLC | *(empty)* | kz liaddo | *(empty)* | 0.658 | 0.656 |

Decision search time 14.3 min, peak RSS 7.8 GiB.

## 4. Test submission

`submissions/01_lgbm_v1/` with decision `odds` + `ef` `{"gamma": 1.0}`: 5,727,038 matched pairs over 1,732,544 test S1; candidate_pairs.tsv lists all 32,505,507 scored pairs.

Validator (`--check-ids`):

```
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (95870 empty, 1636674 non-empty).
  candidate_pairs.tsv: 1732544 rows (116 empty, 1732428 non-empty).

PASS — no blocking issues found. Safe to submit.
```

Sanity, test vs train OOF (same decision config):

| split | country | S1 | % S1 predicted empty | mean predicted set size | mean q of accepted pairs |
|---|---|---|---|---|---|
| train OOF | India | 883,188 | 5.84% | 3.232 | 0.9898 |
| train OOF | US | 1,323,633 | 5.92% | 3.275 | 0.9944 |
| test | France | 259,452 | 5.04% | 3.339 | 0.9944 |
| test | India | 809,986 | 5.63% | 3.271 | 0.9885 |
| test | US | 663,106 | 5.61% | 3.335 | 0.9934 |

### France: 15 random test S1 (accepted matches, then top-3 rejected candidates)

| S1 id | name | address | role | record id | q |
|---|---|---|---|---|---|
| S1-13985225 | Lille Loisirs SASU | 144 Rue Roger Salengro, Lille, Hauts-de-France | S1 | *(empty)* | *(empty)* |
| *(empty)* | Lille Loisirs | 144 R Roger Salengro, Lille | MATCH | S3-227686224 | 1.000 |
| *(empty)* | Lille-Loisirs | 144 RUE ROGER SALENGRO, LILLE, Hauts-de-France | MATCH | S2-137268766 | 1.000 |
| *(empty)* | LL | 144 Rue Roger Salengro, Lille | MATCH | S3-849615471 | 0.994 |
| *(empty)* | Groupe Lille SASU | 144 RUE ROGER SALENGRO, Hauts-de-France, Lille | MATCH | S2-163717074 | 0.941 |
| *(empty)* | arts & frères france eurl | 144 R. Roger Salengro, Lille | rejected | S3-918404853 | 0.003 |
| *(empty)* | Amicale des Pompiers SA | 144 R Roger Salengro, Lille | rejected | S3-397639182 | 0.002 |
| *(empty)* | Amicale Pompiers des (Distribution) | 144 R ROGER SALENGRO, LILLE, Hauts-de-France | rejected | S2-101372673 | 0.000 |
| S1-746566353 | Deleves Jeunes | 2 Rue Pierre et Marie Curie, Nantes, Pays de la Loire | S1 | *(empty)* | *(empty)* |
| *(empty)* | DELEVES JEUNES | #2 RUE PIERRE ET MAROE CURIE, NANTES, Loire-Atlantique | MATCH | S2-678345630 | 1.000 |
| *(empty)* | De1eves Jeunes | No 2 Rue Pirre Et Marie Curie, Nantes, Loire-Atlantique | MATCH | S3-91367773 | 1.000 |
| *(empty)* | Deleves Jeunes | Loire-Atlantique, No 2 Rue Pierre Et Marie Cure, Nantes | MATCH | S3-587161508 | 1.000 |
| *(empty)* | Deleves Jeunes SA | 3 Rue Pierre Et Marie Curie, Nantes, Pays de la Loire | MATCH | S3-412821487 | 0.925 |
| *(empty)* | Deleves Groupe Jéunes | Pays de la Loire, 3 Rue Pierre Et Marie Curie, Nantes | rejected | S3-578161022 | 0.062 |
| *(empty)* | Deleves Jcnoes | *(empty)* | rejected | S3-4604854 | 0.026 |
| *(empty)* | Deleves France  Jeunes | N°3 RUE PIERRE ET MARIE CURIE, NANTES, Loire-Atlantique | rejected | S2-239155518 | 0.010 |
| S1-115610473 | Boules Club SAS | 11 Bis Chemin de la Violerie, Nantes, Pays de la Loire | S1 | *(empty)* | *(empty)* |
| *(empty)* | SAS Boules Club | Loire-Atlantique, Nantes, 11 Bis Chemin De La Violerie | MATCH | S3-769783373 | 1.000 |
| *(empty)* | Boules Club | Loire-Atlantique, Nantes, 11 Bis Ch De La Violerie | MATCH | S3-124509233 | 1.000 |
| *(empty)* | SAS Boules Club | Loire-Atlantique, Nantes, 11 Bis Chemin De La Violerie | MATCH | S3-719394688 | 1.000 |
| *(empty)* | BOULES CLUB SAS | 11 BIS CH. DE LA VILOERIE, NANTES, Loire-Atlantique | MATCH | S2-118722559 | 1.000 |
| *(empty)* | Boules Sante SAS | NANTES, 24 BIS CHEMIN DE LA VIOLERIE | rejected | S2-91384621 | 0.008 |
| *(empty)* | S.A.S Boules Sportive | 24 Bis Ch. De La Violerie, Nantes, Pays de la Loire | rejected | S3-48281074 | 0.004 |
| *(empty)* | Nantes Club  SARL | 21 CHEMIN DE LA VIOLERIE, NANTES, Loire-Atlantique | rejected | S2-493249213 | 0.000 |
| S1-50859199 | Universitaire Amis SAS | 117 Rue de la Pelouse de Douet, Bordeaux, Nouvelle-Aquitaine | S1 | *(empty)* | *(empty)* |
| *(empty)* | Universitaire Comite SAS | 117 RUE DE LA PELOUSE DE DOUET, BORDEAUX, Gironde | MATCH | S2-146819225 | 0.998 |
| *(empty)* | Universitaire Union | 138 R DE LA PELOUSE DE DOUET, BORDEAUX | rejected | S2-862288807 | 0.002 |
| *(empty)* | Lunion Amis SARL | 83 R De La Pelouse De Douet, Bordeaux, Nouvelle-Aquitaine | rejected | S3-340559672 | 0.000 |
| *(empty)* | Universitaire Caveau Groupe SAS + (Associés) | LEGE-CAP-FERRET, 133 AV. DE BORDEAUX CAP FERRET, Gironde | rejected | S2-42959048 | 0.000 |
| S1-989309417 | Bordeaux Club SARL | 18 Rue Dubessan, Bordeaux, Nouvelle-Aquitaine | S1 | *(empty)* | *(empty)* |
| *(empty)* | BORDEAUX ÇLUB SARL | 18 R DUBESSAN, BORDEAUX | MATCH | S2-173366715 | 1.000 |
| *(empty)* | Bordeaux Crllnb SARL | 18 R. Dubessan, Nouvelle-Aquitaine, Bordeaux | MATCH | S3-398746343 | 1.000 |
| *(empty)* | Bordeaux  Club Distribution SARL | 25 Rue Dubessan, Bordeaux, Nouvelle-Aquitaine | rejected | S3-266743956 | 0.001 |
| *(empty)* | Bordeaux  Parents Holding | 10 Bis R. Dubessan, Bordeaux, Gironde | rejected | S3-393344658 | 0.000 |
| *(empty)* | Bordeaux Parents SARL | 10 BIS RUE DUBESSAN, BORDEAUX | rejected | S2-117548320 | 0.000 |
| S1-702936992 | Union du Faubourgs | 142 Rue Raspail, Lille, Hauts-de-France | S1 | *(empty)* | *(empty)* |
| *(empty)* | Faubourgs du Union (Groupe) | 151 R. RASPAIL, Lille, Hauts-de-France | rejected | S2-443499532 | 0.001 |
| *(empty)* | Union Du Fâubourgs International | 151 Rue Raspail, Lille, Hauts-de-France | rejected | S3-487191250 | 0.000 |
| *(empty)* | Comité du Faubourgs Distribution | 17 R. À FIENS, LILLE | rejected | S2-343460479 | 0.000 |
| S1-774223557 | Daffaires Sportive SARL | 11 Rue Clément Ader, Tourcoing, Hauts-de-France | S1 | *(empty)* | *(empty)* |
| *(empty)* | (Sarl) Daffaires Sportive | 11 Rue Clement Ader, Tourcoing, Hauts-de-France | MATCH | S3-165106053 | 1.000 |
| *(empty)* | SARL Daffaires Spôrtive | Nord, 11 RUE CLÉMENT ADER, Tourcoing | MATCH | S2-934678039 | 1.000 |
| *(empty)* | Daffaires Sportive SASU | Hauts-de-France, 13 Rue Clément Ader, Tourcoing | rejected | S3-17866614 | 0.328 |
| *(empty)* | Daffaires & | *(empty)* | rejected | S3-894744964 | 0.198 |
| *(empty)* | Daffaires Sportive Participations SARL | 13 R. Clément Adtr, Tourcoing, Hauts-de-France | rejected | S3-504568075 | 0.001 |
| S1-983257243 | Litalie Groupement (France) EURL | 16 bis Rue de la Joselière, Pornic, Pays de la Loire | S1 | *(empty)* | *(empty)* |
| *(empty)* | EURL Litalie Groupement (France) | 16 Bis Rue De La Joseliere, Pornic | MATCH | S3-733576441 | 1.000 |
| *(empty)* | Litalie Groupement (France) | Pornic, 16 Bis R De La Joselière | MATCH | S3-623725439 | 1.000 |
| *(empty)* | EURL Litalie Groupement (Frànce) | 16 B RUE DE LA JOSELIÈRE, Pornic, Pays de la Loire | MATCH | S2-972442071 | 1.000 |
| *(empty)* | Litalie Club (France) EURL | 16 Bis Rue De La Joselière, Pornic | MATCH | S3-490745586 | 0.997 |
| *(empty)* | LITALIE GROUPEMENT (FRANCE) S.N.C. | 27B R DE LA JOSELIÈRE, PORNIC | rejected | S2-102932901 | 0.039 |
| *(empty)* | Pornic Groupe France SARL | 67 Bis Rue De La Joseliere, Pornic, Loire-Atlantique | rejected | S3-945611014 | 0.000 |
| *(empty)* | Doncologie Maison EURL | 78 BIS R DE LA JOSELIÈRE, PORNIC, Pays de la Loire | rejected | S2-71475176 | 0.000 |
| S1-772337801 | Club du Beyond | 13 Impasse Emile Lanusse Cazaux, Nouvelle-Aquitaine, La Teste-de-Buch | S1 | *(empty)* | *(empty)* |
| *(empty)* | Club Beyond  du | 13 IMP. EMILE LANUSSE CAZAUX, LA TESTE-DE-BUCH, Gironde | MATCH | S2-42389812 | 1.000 |
| *(empty)* | Club du Beyond SCI | 13 IMP. EMILE LANUSSE CAZAUX, LA TESTE DE BUCH, Gironde | MATCH | S2-17963325 | 1.000 |
| *(empty)* | Club du [Beyond] | LA TESTE-DE-BUCH, 13 IMP EMILE LANUSSE CAZAUX, Gironde | MATCH | S2-715440609 | 1.000 |
| *(empty)* | CLUB DU BEYOND + ASSOCIÉS | LA TESTE-DE-BUCH, Gironde, 13 IMPASSE EMILE LANUSSE CAZAUX | MATCH | S2-886918475 | 0.978 |
| *(empty)* | Club du Beyond S.A. | 20 Impasse Emile Lanusse Cazaux, La Teste-de-buch | rejected | S3-624683454 | 0.054 |
| *(empty)* | CYN Ecole S.A.R.L. | Gironde, LA TESTE-DE-BUCH, 9 IMPASSE ENILE LANUSSE CAZAUX | rejected | S2-919554451 | 0.000 |
| *(empty)* | Ck Club SAS | 21 R. Emile Lanusse, La Teste De Buch, Nouvelle-Aquitaine | rejected | S3-505014404 | 0.000 |
| S1-200342034 | Karate Lycée SAS | 15 Rue des Chambelles, Nantes, Pays de la Loire | S1 | *(empty)* | *(empty)* |
| *(empty)* | Karate Lycée SAS | 15 R. DES CHAMBELLES, NANTES | MATCH | S2-57598555 | 1.000 |
| *(empty)* | Karate-Lycée SAS | 15 R. DES CHAMBELLES, NANTES | MATCH | S2-313652548 | 1.000 |
| *(empty)* | Karate Lyfee EURL | *(empty)* | rejected | S2-972980486 | 0.189 |
| *(empty)* | KARATE LYCÉE SA | (18) Rue Des Chambelles, Nantes | rejected | S3-7005314 | 0.071 |
| *(empty)* | Mirakelo+ | 41 Rue des Chambelles, Nantes, Pays de la Loire | rejected | S3-917245749 | 0.027 |
| S1-507990218 | Scorpio Club | 32 Rue du Corsaire, Dunkerque, Hauts-de-France | S1 | *(empty)* | *(empty)* |
| *(empty)* | Scorpio Club | 32 Rue Du Corsaire, Nord, Dunkerque | MATCH | S3-783467101 | 1.000 |
| *(empty)* | Scorpio Club SA | 32 RUE DU CORSAIRE, DUNKERQUE, Hauts-de-France | MATCH | S2-594979572 | 1.000 |
| *(empty)* | Scorpio Club SARL | 32 R. Du Corsaire, Dunkerque, Nord | MATCH | S3-604351882 | 1.000 |
| *(empty)* | Scorpio CLUB | 32 R. Du Corsaire, Dunkerque, Nord | MATCH | S3-284097524 | 1.000 |
| *(empty)* | Scorpio Sportive | 32 Rue Du Corsaire, Dunkerque, Nord | MATCH | S3-142231258 | 0.999 |
| *(empty)* | Scorpio Et  Fils | 32 Rue Du Corsaire, Dunkerque, Nord | MATCH | S3-507677700 | 0.999 |
| *(empty)* | Scorpio Cb | 32 Rue Du Corsaire, Dunkerque, Nord | MATCH | S3-795342602 | 0.999 |
| *(empty)* | Scorpio Conseil | 43 R DU CORSAIRE, DUNKERQUE | rejected | S2-698869479 | 0.003 |
| *(empty)* | SC0RPIO CLUB PARTICIPATIONS | 43 R DU CIRSAIRE, Dunkerque | rejected | S2-592001586 | 0.000 |
| *(empty)* | Loch Club Cri [SAS] | 29 R Du Corsaire, Dunkerque | rejected | S3-956945555 | 0.000 |
| S1-288686294 | Reseau & Frères SARL | 85 Rue Croix de Seguey, Bordeaux, Nouvelle-Aquitaine | S1 | *(empty)* | *(empty)* |
| *(empty)* | Reseau & | 85 Rue Croix De Seguey, Bordeaux, Gironde | MATCH | S3-273137125 | 1.000 |
| *(empty)* | reseau & frères sarl | 85 Rue Croix De Seguey, Bordeaux, Gironde | MATCH | S3-748990621 | 1.000 |
| *(empty)* | Reseau & Frères SARL | Nº 85 R. CROIX DE SEGUEY, BORDEAUX | MATCH | S2-904891446 | 1.000 |
| *(empty)* | Reseau & Frèrës SARL | 85 R. Crsix De Seguey, Bordeaux, Gironde | MATCH | S3-830040073 | 1.000 |
| *(empty)* | France Reseau & SARL | 85 Rue Croix De Seguey, Bordeaux, Gironde | MATCH | S3-732341307 | 0.999 |
| *(empty)* | Yn Coiffure | 85 Rue Croix De Seguey, Bordeaux, Gironde | rejected | S3-93427129 | 0.388 |
| *(empty)* | Sarl Reseau & Frères Distribution | 90 Rue Croix De Seguey, Bordeaux, Gironde | rejected | S3-113723764 | 0.000 |
| *(empty)* | Reseau  & Frères Groupe EURL | BORDEAUX, Nouvelle-Aquitaine, 17 RUE HENRY DEFFSÈ | rejected | S2-244662774 | 0.000 |
| S1-549612175 | IJE Culturelle SARL | 70 Rue Champailler, Calais, Hauts-de-France | S1 | *(empty)* | *(empty)* |
| *(empty)* | IJE Culturelle SARL | 70 R CHAMPAILLER, CALAIS | MATCH | S2-776891879 | 1.000 |
| *(empty)* | IJE CULTURELLE SARL | CALAIS, 70 RUE CHAMPAILLER | MATCH | S2-154038975 | 1.000 |
| *(empty)* | IJE Culturelle SARL | 70 Rue Champailler, Calais, Hauts-de-France | MATCH | S3-861937670 | 1.000 |
| *(empty)* | IJE Çulturelle SARL | 70 Rue Champailler, Calais, Hauts-de-France | MATCH | S3-618325488 | 1.000 |
| *(empty)* | SARL IJE Ecole | 70 R. CHAMPIALLER, Calais | MATCH | S2-676500959 | 0.999 |
| *(empty)* | IJE CLUB SARL | 70 R. CHAMPAELLER, CALAIS | MATCH | S2-333551947 | 0.990 |
| *(empty)* | IJE Culturelle SA | *(empty)* | rejected | S3-216040750 | 0.383 |
| *(empty)* | Reine Culture Groupe SARL | 19 R. Champailler, Calais, Hauts-de-France | rejected | S3-184356212 | 0.000 |
| *(empty)* | Cultuelle  Primaire SARL | 58 R Champailler, Calais | rejected | S3-33968578 | 0.000 |
| S1-387739123 | Bourse Service SAS | 153, rue David Johnston, Bordeaux, Nouvelle-Aquitaine | S1 | *(empty)* | *(empty)* |
| *(empty)* | bourse service sas | 153, Rue David Johnston, Bordeaux | MATCH | S3-174546589 | 1.000 |
| *(empty)* | Bourse Service SCI | Gironde, Bordeaux, 71, 153, Rue David Johnston | MATCH | S3-6058603 | 0.989 |
| *(empty)* | SOLUMBRACALO | 153 R DAVID JOHNSTON, Bordeaux, Nouvelle-Aquitaine | rejected | S2-696587485 | 0.067 |
| *(empty)* | BOURSE SERVICE PARTICIPATIONS SAS | 71, 153, RUE DAVID JOHNSTON, BORDEAUX | rejected | S2-69727110 | 0.047 |
| *(empty)* | Parti SAS Groupe | 153 R. David Johnston, Bordeaux | rejected | S3-254943870 | 0.013 |
| S1-45027199 | Centre Hospitalier Sainte Dame | 24 Rue Castel, Lille, Hauts-de-France | S1 | *(empty)* | *(empty)* |
| *(empty)* | Centre Hospitalier  Sainte Dame | 24 Rue Castel, Lille, Hauts-de-France | MATCH | S3-719764905 | 1.000 |
| *(empty)* | CENTRE HOSPITALIER SÀINTE DAME | 24 RUE CASTEL, LILLE, Nord | MATCH | S2-291576111 | 1.000 |
| *(empty)* | Centre Hospitalier Sainte Dame | 24 RUE CASTEL, LILLE, Nord | MATCH | S2-480952141 | 1.000 |
| *(empty)* | CENTRE HOSPITALIER SAINTE DLER | 24 R. CASTEL, LILLE, Nord | MATCH | S2-858025055 | 0.999 |
| *(empty)* | Secours Groupe Culturel | 43 R. CASTEL, LILLE, Hauts-de-France | rejected | S2-221848162 | 0.000 |
| *(empty)* | Établissements Falcon [Développement] | LILLE, 33 RUE CASTEL | rejected | S2-141930164 | 0.000 |
| *(empty)* | SARL Academie Patrimoine | 28 RUE DE LA FEMRE CASTEL, Lille | rejected | S2-181562688 | 0.000 |

Submission step 4.2 min.
