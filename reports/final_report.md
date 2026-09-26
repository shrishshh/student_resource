# Final report (Task 5): slim candidates, fold-consistent stage 2, variants, package

**Chosen slim candidates: tau = 0.02** (S1 top-15, record top-3): 9,926,200 train / 9,412,340 test pairs, candidates per S1 4.5 train / 5.4 test (p95 8 / 9), pair recall 96.66%, oracle 0.98822, proxy loss 0.00059.

Runtime per part (minutes): A 4, A_apply 1, B 67, C 12, D 1, E 11.

## Part A. Slim candidates

Keep a pair if pruner p >= tau, and it is in the S1's top 15 and the record's top 3 by pruner p. Proxy loss = 03's OOF macro F0.5 (0.97733) minus the score after dropping the 03-accepted pairs the slim set removes. Rule: fewest train pairs with proxy loss <= 0.0007.

| tau | train pairs | test pairs | train cand/S1 mean | p50 | p95 | max | test cand/S1 mean | p50 | p95 | max | pair recall | oracle F0.5 | oracle India | oracle US | proxy loss | *(empty)* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.002 | 11,566,862 | 11,329,044 | 5.2 | 5 | 9 | 15 | 6.5 | 6 | 11 | 15 | 96.88% | 0.98898 | 0.98610 | 0.99090 | 0.00015 | *(empty)* |
| 0.005 | 10,889,839 | 10,561,807 | 4.9 | 5 | 8 | 15 | 6.1 | 6 | 10 | 15 | 96.84% | 0.98883 | 0.98586 | 0.99080 | 0.00023 | *(empty)* |
| 0.01 | 10,462,151 | 10,063,047 | 4.7 | 5 | 8 | 15 | 5.8 | 6 | 10 | 15 | 96.78% | 0.98863 | 0.98557 | 0.99067 | 0.00035 | *(empty)* |
| 0.02 | 9,926,200 | 9,412,340 | 4.5 | 4 | 8 | 15 | 5.4 | 5 | 9 | 15 | 96.66% | 0.98822 | 0.98505 | 0.99033 | 0.00059 | **chosen** |
| 0.05 | 9,140,960 | 8,347,338 | 4.1 | 4 | 7 | 15 | 4.8 | 5 | 8 | 15 | 96.32% | 0.98700 | 0.98372 | 0.98918 | 0.00140 | *(empty)* |

v2 (02/03) used 28,023,030 train / 24,300,351 test pairs (~12.7 / 14.0 per S1), oracle 0.99147. Curve computed in 4.1 min.


## Part B. Retrain on the slim candidates (fold-consistent stage-2 test inputs)

OOF macro F0.5 over all train S1 with the decision chosen by the limited search (all runs picked odds + ef gamma 1.0), and the loss split in train pairs:

| run | OOF | India | US | singleton | non-singleton | missed by blocking | false negatives | false positives |
|---|---|---|---|---|---|---|---|---|
| 02_prune_v2 (stage 1, v2 cands) | 0.97303 | 0.96668 | 0.97727 | 0.9569 | 0.9740 | 179,532 | 278,457 | 46,302 |
| 03_stage2 (stage 2, v2 cands) | 0.97733 | 0.97267 | 0.98044 | 0.9692 | 0.9778 | 179,532 | 219,144 | 27,795 |
| slim stage 1 (v3s1) | 0.97316 | 0.96669 | 0.97748 | 0.9599 | 0.9739 | 254,909 | 200,791 | 43,476 |
| slim stage 2 (v3s2) | 0.97662 | 0.97174 | 0.97988 | 0.9676 | 0.9772 | 254,909 | 154,422 | 27,580 |

Mean predicted set size per S1 (odds + ef, gamma 1.0): train OOF vs test. 03 used the mean of both stage-1 models' test p for every stage-2 model; the slim stage 2 applies each stage-2 model to test features built from the stage-1 model whose p it was trained on.

| run | train OOF France | train OOF India | train OOF US | test France | test India | test US |
|---|---|---|---|---|---|---|
| 02_prune_v2 (stage 1, v2 cands) | - | 3.252 | 3.290 | 3.438 | 3.310 | 3.371 |
| 03_stage2 (stage 2, v2 cands) | - | 3.272 | 3.307 | 3.601 | 3.440 | 3.661 |
| slim stage 1 (v3s1) | - | 3.249 | 3.291 | 3.437 | 3.313 | 3.377 |
| slim stage 2 (v3s2) | - | 3.264 | 3.305 | 3.610 | 3.427 | 3.632 |

### Stage 1 (slim)

### Model (LightGBM, 2-fold out-of-fold; run `v3s1`)

77 features. Params `{'objective': 'binary', 'metric': 'binary_logloss', 'num_leaves': 127, 'learning_rate': 0.05, 'min_data_in_leaf': 200, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': 1.0, 'seed': 42, 'deterministic': True, 'force_col_wise': True, 'num_threads': 16}`, up to 3000 rounds, early stopping 100 (logloss) on a 10% S1 hold-out. Each fold trains on a random 30% of its half's S1s and predicts every pair of the other half; test = mean of both models.

| model | trained on half | rounds | train S1 | train pairs | early-stop S1 |
|---|---|---|---|---|---|
| 0 | 0 | 1593 | 298,355 | 1,340,697 | 32,762 |
| 1 | 1 | 1553 | 298,083 | 1,341,560 | 33,176 |

OOF metrics (model h predicts the other half; raw p before any isotonic step):

| model | country | pairs | AUC | logloss |
|---|---|---|---|---|
| 0 | ALL | 4,964,513 | 0.99738 | 0.05571 |
| 0 | India | 1,971,521 | 0.99644 | 0.06478 |
| 0 | US | 2,992,992 | 0.99786 | 0.04973 |
| 1 | ALL | 4,961,687 | 0.99737 | 0.05578 |
| 1 | India | 1,967,810 | 0.99646 | 0.06452 |
| 1 | US | 2,993,877 | 0.99783 | 0.05004 |

Calibration of OOF p (10 bins). Worst gap 0.0185 -> isotonic not needed.

| bin | pairs | mean p | positive rate | gap |
|---|---|---|---|---|
| 0.0-0.1 | 2,177,820 | 0.0097 | 0.0097 | -0.0000 |
| 0.1-0.2 | 149,212 | 0.1465 | 0.1587 | -0.0122 |
| 0.2-0.3 | 106,411 | 0.2475 | 0.2545 | -0.0070 |
| 0.3-0.4 | 80,015 | 0.3471 | 0.3460 | +0.0011 |
| 0.4-0.5 | 66,812 | 0.4500 | 0.4478 | +0.0023 |
| 0.5-0.6 | 55,840 | 0.5471 | 0.5329 | +0.0142 |
| 0.6-0.7 | 44,772 | 0.6500 | 0.6316 | +0.0185 |
| 0.7-0.8 | 54,524 | 0.7530 | 0.7477 | +0.0053 |
| 0.8-0.9 | 98,386 | 0.8570 | 0.8568 | +0.0002 |
| 0.9-1.0 | 7,092,408 | 0.9969 | 0.9970 | -0.0001 |

Top-30 features by gain, model 0:

| feature | gain |
|---|---|
| pscore | 6,627,565 |
| nums_jaccard | 760,213 |
| house_logdiff | 638,379 |
| nums_frac_s_in_r | 464,190 |
| r_margin_score | 417,759 |
| name_cov_r | 298,463 |
| addr_cov_r | 257,803 |
| sim_addr | 197,009 |
| n_cand_s1 | 125,804 |
| addr_ntok_r | 123,820 |
| name_len_diff | 113,672 |
| addr_tsr | 100,412 |
| name_first_tok | 100,292 |
| r_margin_addr | 95,055 |
| addr_cov_s | 94,143 |
| addr_tsort | 73,233 |
| rank_r_all | 67,571 |
| rank_s_all | 67,166 |
| addr_partial | 67,106 |
| name_ntok_r | 66,975 |
| s_n_near | 59,337 |
| name_cov_s | 58,642 |
| prank_s | 58,023 |
| s_rank_addr | 52,451 |
| chain_s | 50,720 |
| prank_s_all | 49,837 |
| r_tok_in_s1_vocab | 49,380 |
| addr_ntok_s | 47,712 |
| addr_ratio | 46,895 |
| house_prefix | 43,883 |

Top-30 features by gain, model 1:

| feature | gain |
|---|---|
| pscore | 6,622,617 |
| nums_jaccard | 761,832 |
| house_logdiff | 637,574 |
| nums_frac_s_in_r | 470,548 |
| r_margin_score | 415,316 |
| name_cov_r | 301,066 |
| addr_cov_r | 267,373 |
| sim_addr | 196,907 |
| n_cand_s1 | 122,239 |
| addr_ntok_r | 111,326 |
| name_first_tok | 102,077 |
| addr_tsr | 98,220 |
| name_len_diff | 89,469 |
| r_margin_addr | 89,145 |
| addr_cov_s | 88,217 |
| addr_tsort | 73,682 |
| addr_partial | 69,261 |
| rank_s_all | 68,546 |
| name_ntok_r | 66,610 |
| rank_r_all | 60,223 |
| name_cov_s | 59,550 |
| s_n_near | 59,212 |
| name_tsort | 59,007 |
| addr_ntok_s | 57,424 |
| prank_s | 56,651 |
| r_tok_in_s1_vocab | 50,906 |
| s_rank_addr | 50,837 |
| chain_s | 50,038 |
| prank_r | 49,525 |
| prank_s_all | 49,017 |

Training + prediction: 27.2 min, peak RSS 5.1 GiB.


### Decision layer (run `v3s1`, tuned on OOF over all train S1)

29 configurations evaluated (limited: odds + ef (gamma 0.85/1.0/1.2/1.5) and odds + two (t1 0.50-0.70, t2 0.60-0.80)). S1s without candidates and true pairs missing from the candidates (254,909) count against recall.

**Winner: D1 = `odds`, D2 = `ef`, params `{"gamma": 1.0}` -> OOF macro F0.5 0.97316** (re-scored with `metrics.macro_f05`: 0.97316).

Best configuration of each (D1, D2) family:

| D1 | D2 | params | OOF F0.5 | India | US | singleton | non-singleton |
|---|---|---|---|---|---|---|---|
| odds | ef | {"gamma": 1.0} | 0.97316 | 0.96669 | 0.97748 | 0.9599 | 0.9739 |
| odds | two | {"t1": 0.55, "t2": 0.7} | 0.97301 | 0.96641 | 0.97741 | 0.9662 | 0.9734 |

All global-threshold and expected-F configurations:

| D1 | D2 | params | OOF F0.5 | India | US | singleton | non-singleton |
|---|---|---|---|---|---|---|---|
| odds | ef | {"gamma": 1.5} | 0.97266 | 0.96606 | 0.97706 | 0.9714 | 0.9727 |
| odds | ef | {"gamma": 1.2} | 0.97304 | 0.96654 | 0.97738 | 0.9656 | 0.9735 |
| odds | ef | {"gamma": 0.85} | 0.97309 | 0.96657 | 0.97743 | 0.9533 | 0.9743 |
| odds | ef | {"gamma": 1.0} | 0.97316 | 0.96669 | 0.97748 | 0.9599 | 0.9739 |

Winner breakdown:

| slice | OOF macro F0.5 |
|---|---|
| overall | 0.97316 |
| India | 0.96669 |
| US | 0.97748 |
| singleton | 0.95988 |
| non_singleton | 0.97395 |

Predicted vs true set size (number of train S1):

| size | true | predicted |
|---|---|---|
| 0 | 123,247 | 131,370 |
| 1 | 119,157 | 164,519 |
| 2 | 375,212 | 417,402 |
| 3 | 530,841 | 539,412 |
| 4 | 484,115 | 458,953 |
| 5 | 321,957 | 287,950 |
| 6-10 | 252,255 | 207,191 |
| >10 | 37 | 24 |

Best configuration per country (vs the global winner on that country):

| country | D1 | D2 | params | best F0.5 | winner F0.5 |
|---|---|---|---|---|---|
| India | odds | ef | {"gamma": 1.0} | 0.96669 | 0.96669 |
| US | odds | ef | {"gamma": 1.0} | 0.97748 | 0.97748 |

Error analysis: 43,476 false-positive pairs, 200,791 false-negative pairs among candidates (+ 254,909 true pairs never in the candidates).

15 random false positives:

| country | S1 id | S1 name | S1 address | S1 name_core | rec id | rec name | rec address | rec name_core | rec name_alt | p | q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| India | S1-795508335 | Data Services Private Limited | Ps Tower, Gautam Buddha Nagar, Noida, Uttar Pradesh, Third Floor, Kh. No. 227 | data services | S2-346364229 | Services Data Ltd | NO 303 THIRD FLOOR, LUCKNOW, Uttar Pradesh | services data | *(empty)* | 0.793 | 0.793 |
| India | S1-139033658 | Vishwa Mills Private Limited | # 800 K N 3013/800, Bangalore, Karnataka, 2Nd Sector Hsr Layout, Bangalore | vishwa mills | S2-80938262 | Vishwa Limited Services | # 809 K N 3013/800, 2ND SECTOR HSR LAYOUT, BANGALORE, ಕರ್ನಾಟಕ | vishwa services | *(empty)* | 0.603 | 0.603 |
| India | S1-185868646 | Vrindavan Trust | No.306, 3Rd Floor, Embassy Chambers, Grant/Vittal Mallya Road, Bangalore North, Bangalore, Karnataka | vrindavan trust | S3-436722477 | Vrindavan Trust Private Limited | KA, 3Rd Floor, Embassy Chambers, Grant/vittal Mallya Road, Bangalore North, 311, Bangalore | vrindavan trust | *(empty)* | 0.845 | 0.845 |
| India | S1-805052319 | Arihant Logistics Private Limited | Unit No.602, B Wing, 6Th Flr, Plot 834/1, 834/2, 822/1 Off.Link Rd, Village Ambivali, Sahakar Ngr, Andheri W, Mumbai, Mumbai City, Maharashtra | arihant logistics | S3-958134866 | अरिहंत लॉजिस्टिक्स प्राइवेट लिमिटेड | 2-602-B, Mumbai, Mumbai City, महाराष्ट्र | arihant logistics | *(empty)* | 0.998 | 0.998 |
| India | S1-927643101 | Diamond & Sons Pvt Ltd | 1St-Fr, 2/1/1 Bholanath Nandy Lane Lp-463/3, Howrah, West Bengal | diamond sons | S2-807638062 | Diamond & Sons Public Limited | West Bengal, 2ST-FR, 2/1/1 BHOLANATH NANDY LANE LP-463/3, HOWRAH | diamond sons public | *(empty)* | 0.919 | 0.919 |
| India | S1-895652216 | Taransh Infosolutions Private Limited | C/O Rajbala, Uklana Road, Village Prabhuwala, Hisar, Haryana | taransh infosolutions | S3-75531817 | Taranshyn Ifosolutions Private Limited | Block A-233 C/o Rajbala, Uklana Road, Village Prabhuwala, Hisar, HR | taransh yn ifo solutions | *(empty)* | 0.901 | 0.901 |
| India | S1-161005351 | Sarvodaya Nursing Home | 6/4/28/4/2. Arundeleptguntur Guntur, Andhra Pradesh, Krishna, Andhra Pradesh | sarvodaya nursing home | S3-893712451 | Mr Sarvodaya  Nursing Home | *(empty)* | sarvodaya nursing home | *(empty)* | 0.672 | 0.464 |
| India | S1-532625731 | Incon Buildcon Private Limited | #5-7-38/1 Maqdoom Nagar, Sangareddy, Medak, Telangana | incon buildcon | S2-240731724 | incon buildcon overseas private limited | #1-5-7-38/5 MAQDOOM NAGAR, SANGAREDDY, Telangana | incon buildcon overseas | *(empty)* | 0.947 | 0.947 |
| US | S1-44802957 | Vision Health LLC | 44 Water Plant Road, Bigelow, AR | vision health | S2-331037355 | Inc Vision Health | 45 WATER PLANT RD, BIGELOW, AR | vision health | *(empty)* | 0.828 | 0.828 |
| US | S1-569734153 | Dehlia T. Rosales, DMD, P.C. PC | MD, Lanham, 8511 Red Wing Lane | dehlia t rosales dmd | S3-826027301 | Dehlia T. Rasles, DMDD, P.C. PC | 8511 Red Wing Lane, Lanham, Maryland | dehlia t rasles dmdd | *(empty)* | 0.993 | 0.993 |
| US | S1-759273783 | All Pizza LLC | 17 Mississippi Avenue, Unit 5, Washington, DC | all pizza | S2-944849024 | ALL PIZZA CO. | 18 Mississippi Avenue, WASHINGTON, DC | all pizza | *(empty)* | 0.852 | 0.852 |
| India | S1-512755630 | Itl Financial Pvt. Ltd. | Plot No.Ad-15, Room No.D6, Sai Kripa Soc, S.V.P.Nagar, Mumbai, Maharashtra | itl financial | S3-644873063 | Itl Financial [Pvt.] | *(empty)* | itl financial | *(empty)* | 0.995 | 0.995 |
| US | S1-923188258 | Rocky Disciplined | 12104 Wilmont Turn, MD, Bowie | rocky disciplined | S3-535467632 | Rocky Disciplined LLC | 12117 Wilmont Turn, Bowie, Maryland | rocky disciplined | *(empty)* | 0.680 | 0.680 |
| US | S1-540073751 | Urban Diner | Minatare, Unit 5, 507 Ave A, NE | urban diner | S2-241582754 | Urban Diner LLC | 509 AVE A, MINATARE, NE | urban diner | *(empty)* | 0.822 | 0.822 |
| US | S1-634050908 | Gurule Capital Vine LLC | 906 Park Heights Drive, Richmond, TX | gurule capital vine | S2-168056327 | Gurule LLC  Capital Capital | 911 PARK HEIGHTS DR, RICHMOND, TX | gurule capital | *(empty)* | 0.804 | 0.804 |

15 random false negatives (true pairs in the candidates that were rejected):

| country | S1 id | S1 name | S1 address | S1 name_core | rec id | rec name | rec address | rec name_core | rec name_alt | p | q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| India | S1-210255804 | Tea Trading | C1, 2Nd Street, Burma Colony Perungudi, Chennai, Tamil Nadu | tea trading | S3-389319146 | Tea Trading | *(empty)* | tea trading | *(empty)* | 0.395 | 0.270 |
| US | S1-561231635 | Boulware Liberty Digital | 9362 Little Hominy Ridge Road, Florence, IN | boulware liberty digital | S2-626271640 | Boulware Liberty | *(empty)* | boulware liberty | *(empty)* | 0.229 | 0.164 |
| US | S1-132778268 | Erena F. Patterson, CPA | Tuscaloosa, 1802 4th Avenue, AL | erena f patterson cpa | S2-431306268 | ERENA F. PÁTTERSON, CPA | *(empty)* | erena f patterson cpa | *(empty)* | 0.257 | 0.143 |
| US | S1-332264019 | Central Materials Concepts | 38 Brookhaven Boulevard, NY, Brookhaven | central materials concepts | S3-46247898 | Central Materials Concepts Ltd | 37 Brookhaven Blvd, Port Jefferson Station, New York | central materials concepts | *(empty)* | 0.749 | 0.749 |
| US | S1-555838106 | Ace Investors LLC | 1334 Minter School Road, Sanford, NC | ace investors | S3-759964274 | Ace  Investors LLC | *(empty)* | ace investors | *(empty)* | 0.237 | 0.174 |
| India | S1-494026007 | RI Estates | Vpo Mansurwal Dona Jalandhar Road, Kapurthala, Punjab | ri estates | S3-276383197 | RI  Estates (ID: 67415) | *(empty)* | ri estates | *(empty)* | 0.320 | 0.165 |
| India | S1-425943824 | Ed International School | C/O Suraj Singh Thakur, Veer Sawarkar Ward, Bina, Sagar, Madhya Pradesh | ed international school | S2-310771528 | ED INTERNATIONAL 5CHOOL CORP | #26 C/O SURAJ SINGH THAKUR, VEER SAWARKAR WARD, BINA, Madhya Pradesh | ed international school | *(empty)* | 0.302 | 0.302 |
| India | S1-893165260 | Jyoti Developers Limited | Sanjib Sarani Durgapurburdwan, Burdwan, Medinipur, West Bengal | jyoti developers | S3-734067522 | Jyoti Developers Limited | *(empty)* | jyoti developers | *(empty)* | 0.573 | 0.324 |
| India | S1-280316572 | Sohum India LLP | Unit No 17A, 12Th Floor, Sushma Infinium, Rajpura, Mohali, Punjab | sohum india | S2-101527207 | Sohum LLP Center | DOOR NO 8-590 UNIT NO 17A, 12TH FLOOR, SUSHMA INFINIUM, RAJPURA, Punjab | sohum center | *(empty)* | 0.706 | 0.706 |
| US | S1-70887702 | Nguyen Osisko LLC | 39129 Woodside Drive, Erskine, MN | nguyen osisko | S3-2950969 | Nguyen Osisko LLC | 39128 Woodside Dr, Erskine, Minnesota | nguyen osisko | *(empty)* | 0.705 | 0.705 |
| US | S1-686590629 | Arts Project IV | 111 Pine Cone Road, Wilmington, NC | arts project iv | S3-440475547 | Arts Project  IV | *(empty)* | arts project iv | *(empty)* | 0.501 | 0.372 |
| US | S1-203567713 | Pediatric Dentistry Metropolitan Clinic | 700 7th Street, Waseca, MN | pediatric dentistry metropolitan clinic | S3-939430095 | Pediatric Dentistry Metropolitan-Clinic | *(empty)* | pediatric dentistry metropolitan clinic | *(empty)* | 0.520 | 0.330 |
| US | S1-971983480 | Starnes Japan Inc | 7211 History Lane, Hanover County, VA | starnes japan | S3-128703796 | 5tarnes  Jp Inc | Virginia, 7198 History Ln, Hanover County | starnes jp | *(empty)* | 0.041 | 0.041 |
| US | S1-518997162 | Dynamic Engineering Concepts LLC | 4708 Key Avenue, Sioux Falls, SD | dynamic engineering concepts | S3-895341026 | Dynamic Engineering  Concepts | *(empty)* | dynamic engineering concepts | *(empty)* | 0.185 | 0.150 |
| US | S1-694647687 | Prairie Fannie, LLC | Rochester, 155 Versailles Road, NY | prairie fannie | S2-362980770 | Prairie Fannie,  LLC | 152 VERSAILLES ROAD, ROCHESTER, NY | prairie fannie | *(empty)* | 0.392 | 0.392 |

Decision search time 2.2 min, peak RSS 6.1 GiB.


### Stage 2 (slim)

### Model (LightGBM, 2-fold out-of-fold, stage 2; run `v3s2`)

98 features. Params `{'objective': 'binary', 'metric': 'binary_logloss', 'num_leaves': 127, 'learning_rate': 0.05, 'min_data_in_leaf': 200, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': 1.0, 'seed': 42, 'deterministic': True, 'force_col_wise': True, 'num_threads': 16}`, up to 3000 rounds, early stopping 100 (logloss) on a 10% S1 hold-out. Each fold trains on a random 30% of its half's S1s and predicts every pair of the other half; test = mean of both models.

| model | trained on half | rounds | train S1 | train pairs | early-stop S1 |
|---|---|---|---|---|---|
| 0 | 0 | 677 | 298,355 | 1,340,697 | 32,762 |
| 1 | 1 | 506 | 298,083 | 1,341,560 | 33,176 |

OOF metrics (model h predicts the other half; raw p before any isotonic step):

| model | country | pairs | AUC | logloss |
|---|---|---|---|---|
| 0 | ALL | 4,964,513 | 0.99846 | 0.04193 |
| 0 | India | 1,971,521 | 0.99819 | 0.04545 |
| 0 | US | 2,992,992 | 0.99861 | 0.03960 |
| 1 | ALL | 4,961,687 | 0.99847 | 0.04169 |
| 1 | India | 1,967,810 | 0.99821 | 0.04504 |
| 1 | US | 2,993,877 | 0.99861 | 0.03949 |

Calibration of OOF p (10 bins). Worst gap 0.0234 -> isotonic not needed.

| bin | pairs | mean p | positive rate | gap |
|---|---|---|---|---|
| 0.0-0.1 | 2,254,925 | 0.0066 | 0.0075 | -0.0010 |
| 0.1-0.2 | 119,840 | 0.1474 | 0.1599 | -0.0126 |
| 0.2-0.3 | 90,997 | 0.2478 | 0.2542 | -0.0064 |
| 0.3-0.4 | 69,371 | 0.3464 | 0.3423 | +0.0041 |
| 0.4-0.5 | 53,916 | 0.4500 | 0.4438 | +0.0062 |
| 0.5-0.6 | 43,102 | 0.5456 | 0.5222 | +0.0234 |
| 0.6-0.7 | 27,611 | 0.6484 | 0.6250 | +0.0234 |
| 0.7-0.8 | 29,309 | 0.7527 | 0.7401 | +0.0126 |
| 0.8-0.9 | 49,857 | 0.8566 | 0.8493 | +0.0073 |
| 0.9-1.0 | 7,187,272 | 0.9982 | 0.9980 | +0.0002 |

Top-30 features by gain, model 0:

| feature | gain |
|---|---|
| p1 | 8,543,518 |
| s_gap_maxp | 2,292,938 |
| s_cnt_p50 | 283,533 |
| pscore | 147,715 |
| s_sum_p | 85,696 |
| s_pshare_r_house | 81,132 |
| s_oth_house_maxp | 64,427 |
| s_oth_house_sump | 40,969 |
| r_oth_maxp | 33,293 |
| house_logdiff | 31,503 |
| r_cnt_p10 | 30,927 |
| s_oth_name_maxp | 19,199 |
| rank_s_all | 19,038 |
| n_cand_s1 | 18,052 |
| addr_tsort | 12,320 |
| chain_s | 11,907 |
| s_n_near | 11,656 |
| addr_cov_s | 11,528 |
| s_oth_name_sump | 11,456 |
| addr_cov_r | 11,276 |
| addr_ratio | 10,913 |
| prank_s | 10,426 |
| addr_ntok_s | 10,400 |
| s_rank_addr | 9,813 |
| addr_partial | 9,672 |
| score | 9,629 |
| s_gap_best | 9,290 |
| s_oth_addr_maxp | 8,922 |
| r_margin_addr | 8,368 |
| prank_s_all | 7,812 |

Top-30 features by gain, model 1:

| feature | gain |
|---|---|
| p1 | 8,538,205 |
| s_gap_maxp | 2,291,010 |
| s_cnt_p50 | 287,403 |
| pscore | 143,766 |
| s_sum_p | 80,814 |
| s_pshare_r_house | 73,757 |
| s_oth_house_maxp | 61,908 |
| s_oth_house_sump | 43,905 |
| r_oth_maxp | 31,044 |
| house_logdiff | 30,614 |
| r_cnt_p10 | 30,049 |
| n_cand_s1 | 18,699 |
| s_oth_name_maxp | 15,328 |
| rank_s_all | 14,390 |
| prank_s | 12,753 |
| s_oth_name_sump | 9,886 |
| addr_cov_r | 9,864 |
| s_n_near | 9,723 |
| chain_s | 9,596 |
| addr_tsort | 9,386 |
| s_oth_house_n | 8,861 |
| addr_partial | 8,824 |
| addr_cov_s | 8,786 |
| addr_ratio | 8,714 |
| s_gap_best | 8,127 |
| s_rank_addr | 8,112 |
| r_margin_addr | 7,999 |
| addr_ntok_s | 7,669 |
| s_oth_addr_maxp | 7,632 |
| score | 7,458 |

Training + prediction: 10.4 min, peak RSS 7.0 GiB.


### Decision layer (run `v3s2`, tuned on OOF over all train S1)

29 configurations evaluated (limited: odds + ef (gamma 0.85/1.0/1.2/1.5) and odds + two (t1 0.50-0.70, t2 0.60-0.80)). S1s without candidates and true pairs missing from the candidates (254,909) count against recall.

**Winner: D1 = `odds`, D2 = `ef`, params `{"gamma": 1.0}` -> OOF macro F0.5 0.97662** (re-scored with `metrics.macro_f05`: 0.97662).

Best configuration of each (D1, D2) family:

| D1 | D2 | params | OOF F0.5 | India | US | singleton | non-singleton |
|---|---|---|---|---|---|---|---|
| odds | ef | {"gamma": 1.0} | 0.97662 | 0.97174 | 0.97988 | 0.9676 | 0.9772 |
| odds | two | {"t1": 0.5, "t2": 0.7} | 0.97660 | 0.97169 | 0.97987 | 0.9692 | 0.9770 |

All global-threshold and expected-F configurations:

| D1 | D2 | params | OOF F0.5 | India | US | singleton | non-singleton |
|---|---|---|---|---|---|---|---|
| odds | ef | {"gamma": 1.5} | 0.97639 | 0.97142 | 0.97970 | 0.9758 | 0.9764 |
| odds | ef | {"gamma": 1.2} | 0.97657 | 0.97165 | 0.97985 | 0.9719 | 0.9768 |
| odds | ef | {"gamma": 0.85} | 0.97657 | 0.97166 | 0.97985 | 0.9630 | 0.9774 |
| odds | ef | {"gamma": 1.0} | 0.97662 | 0.97174 | 0.97988 | 0.9676 | 0.9772 |

Winner breakdown:

| slice | OOF macro F0.5 |
|---|---|
| overall | 0.97662 |
| India | 0.97174 |
| US | 0.97988 |
| singleton | 0.96760 |
| non_singleton | 0.97716 |

Predicted vs true set size (number of train S1):

| size | true | predicted |
|---|---|---|
| 0 | 123,247 | 131,889 |
| 1 | 119,157 | 157,687 |
| 2 | 375,212 | 415,891 |
| 3 | 530,841 | 539,693 |
| 4 | 484,115 | 461,104 |
| 5 | 321,957 | 290,737 |
| 6-10 | 252,255 | 209,796 |
| >10 | 37 | 24 |

Best configuration per country (vs the global winner on that country):

| country | D1 | D2 | params | best F0.5 | winner F0.5 |
|---|---|---|---|---|---|
| India | odds | ef | {"gamma": 1.0} | 0.97174 | 0.97174 |
| US | odds | ef | {"gamma": 1.0} | 0.97988 | 0.97988 |

Error analysis: 27,580 false-positive pairs, 154,422 false-negative pairs among candidates (+ 254,909 true pairs never in the candidates).

15 random false positives:

| country | S1 id | S1 name | S1 address | S1 name_core | rec id | rec name | rec address | rec name_core | rec name_alt | p | q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| India | S1-288773215 | White Investment Private Limited | 82/D, Dr. Suresh Chandra, Banerjee Road 4Th Floor, Beleghata, Kolkata, Howrah, West Bengal | white investment | S2-963740220 | Walker & Có | 82/D, DR. SURESH CHANDRA, BANERJEE ROAD 4TH FLOOR, BELEGHATA, KOLKATA, West Bengal | walker | *(empty)* | 0.844 | 0.844 |
| India | S1-696463647 | NQ Fintech Private Limited | Office No.-1013, B Wing, Samartha Aishwariya, Mumbai, Maharashtra | nq fintech | S2-954001117 | NQZ FINTECH PRIVATE | OFFICE NO., B WING, SAMARTHA AISHWARIYA, MUMBAI, महाराष्ट्र | nqz fintech | *(empty)* | 0.832 | 0.832 |
| India | S1-30898429 | Kolkata India Pvt Ltd | 33, Tollygunge Circular, Road, Floor- Ground, Kolkata, Kolkata, Howrah, West Bengal | kolkata india | S3-416700767 | Kolkata (india) Pvt (Ltd.) | 80, Canal Circular Road, Kolkata, Kolkata, Howrah, WB | kolkata india | *(empty)* | 0.712 | 0.712 |
| India | S1-394988424 | Pan Forum Private Limited | Lucknow, C/O Suresh Village Mashira Ratan Post Mall Malihabad Lucknow Up, Lucknow, Uttar Pradesh | pan forum | S2-365746573 | Pan Movie  Private | C/O SURESH VILLAGE MASHIRA RATAN POST MALL MALIHABAD LUCKNOW UP, LUCKNOW, उत्तर प्रदेश | pan movie | *(empty)* | 0.968 | 0.968 |
| India | S1-670262471 | Anand Agro | C/O Safiya, 5/210 Chakkala House, Po Kadampuzha, Malappuram, Malappuram, Malapuram, Kerala | anand agro | S2-441506161 | ആനന്ദ് അഗ്രോ എൽഎൽപി | Kerala, MALAPURAM, 5/210 CHAKKALA HOUSE, PO KADAMPUZHA, MALAPPURAM, MALAPPURAM, NO 70 C/O SAFIYA | anand agro | *(empty)* | 0.940 | 0.940 |
| India | S1-11802222 | Sai Developers (India)-Poonamallee | No.1601, Poonamallee, Casagrand Crescendo, Mel Ayanambakkam, Mogappair, Tamil Nadu, Tiruvallur | sai developers india poonamallee | S2-32911204 | Sai Developers (India)-Poonamallee Private Limited | NO , CASAGRAND CRESCENDO, MEL AYANAMBAKKAM, MOGAPPAIR, POONAMALLEE, தமிழ்நாடு | sai developers india poonamallee | *(empty)* | 1.000 | 1.000 |
| US | S1-928013163 | Winters Anadolu LLC | 1296 Rising Sun Place, Pueblo West, CO | winters anadolu | S2-118295814 | Winters Anddu Ltd | 1297 RISING SUN PLACE, PUEBLO WEST, CO | winters anddu | *(empty)* | 0.775 | 0.775 |
| US | S1-432773307 | Tadaex Industries, LLC | 4267 Shopping Lane, Simi Valley, CA | tadaex industries | S2-758955449 | Tadaex Industries, Co | SIMI VALLEY, 427 SHOPPING LN, CA | tadaex industries | *(empty)* | 0.924 | 0.924 |
| India | S1-459663247 | Vulcan Public School | Orchid, 271, Ruchi Life Scapes, Hoshangabad Road, Jatkhedi, Bhopal, Madhya Pradesh | vulcan public school | S3-61688199 | Vulcan Public School Pvt Ltd | Door No 016 Orchid, Bhopal, Kalapani, मध्य प्रदेश | vulcan public school | *(empty)* | 0.796 | 0.796 |
| India | S1-132184099 | Basa Communications Private Limited | No.567, Beml Embcs Layout 5Th Stage, Rr Nagar, Bangalore, Karnataka | basa communications | S3-360673963 | Dr Basa Communications (Llp) | No.5-69, Beml Embcs Layout 5Th Stage, Rr Nagar, Bangalore, KA | basa communications | *(empty)* | 0.939 | 0.939 |
| India | S1-741564318 | Sai Investment Private Limited | Sy No. 353, Near Chinna, Krishna Bricks, Basuragadi, Tirumalagiri, Hyderabad, Telangana | sai investment | S3-302087709 | Sai Investment Pvt | Krishna, ఆంధ్రప్రదేశ్ | sai investment | *(empty)* | 0.903 | 0.903 |
| US | S1-267398201 | Victor Pharmaceuticals L.L.C. | 188 Wolfsnare Lane, Morrisville, NC | victor pharmaceuticals | S2-896304535 | Victor Industries L.L.C. | 188 WOLFSNARE LANE, MORRISVILLE, NC | victor industries | *(empty)* | 0.807 | 0.807 |
| US | S1-115608374 | Phillips and Brown Nextera, LLC | 40 334, Bleeker, AL | phillips brown nextera | S3-776460479 | Phillips and Brown Nextera, Inc | 41 334, SALM, Alabama | phillips brown nextera | *(empty)* | 0.987 | 0.987 |
| US | S1-902553151 | Vision Medicine | 6943 20th Place, Tulsa, OK | vision medicine | S3-430697341 | Vision Medicine Llc | 93th Pl, Tulsa, Oklahoma | vision medicine | *(empty)* | 0.976 | 0.976 |
| US | S1-626694631 | Electra Rockwell Championsgate Inc | 130 Roosevelt Street, Marion, KS | electra rockwell championsgate | S3-397488848 | Electra Rockwell | *(empty)* | electra rockwell | *(empty)* | 0.998 | 0.998 |

15 random false negatives (true pairs in the candidates that were rejected):

| country | S1 id | S1 name | S1 address | S1 name_core | rec id | rec name | rec address | rec name_core | rec name_alt | p | q |
|---|---|---|---|---|---|---|---|---|---|---|---|
| India | S1-629048354 | Techno Ratna Private Limited | House No. 3482 Golande, Ahed Buldana Buldan, Buldana, Buldhana, Maharashtra | techno ratna | S2-807441348 | TECHNO RATNA PRIVATE LÍMITED | #471 HOUSE NO. 3482 GOLANDE, AHED BULDANA BULDAN, BULDANA, Maharashtra | techno ratna | *(empty)* | 0.739 | 0.739 |
| India | S1-959511952 | Kridha Home Corporation | Rs409/If203- Sri Ram Resi, Opp Sbi, Konthamuru, Raja, Korukonda, East Godavari, Andhra Pradesh | kridha home | S3-30764979 | Sri #kridhahome | Rs4e9/if203- Sri Ram Resi, Opp Sbi, Konthamuru, Raja, East Godavari, AP | kridha home | *(empty)* | 0.599 | 0.599 |
| India | S1-168130358 | Kritin Wear Pvt. Ltd. | D4/21, The Westerlies Sector 108 Gurugram, Railway Road, Gurgaon, Haryana | kritin wear | S3-991409926 | Kritin Wear Pvt. Limited | D2/21, Gurgaon, HR | kritin wear | *(empty)* | 0.673 | 0.673 |
| India | S1-973428925 | Jaipur Builders Private Limited | 80, Kajod Bhawan, Hari Marg, Somnath Mahadev Mandir Ke Pass, Civil Li, Nes, Jaipur, Rajasthan | jaipur builders | S2-488274761 | Jaipur Bui1ders Private | *(empty)* | jaipur builders | *(empty)* | 0.183 | 0.139 |
| US | S1-158945274 | Brown Silver Nexus | 4519 Kernersville Road, Winston-salem, NC | brown silver nexus | S2-707189944 | Deltafayeavi | KERNERSVILLE, NC, 4519 KERNERSVILLE RD | deltafayeavi | *(empty)* | 0.702 | 0.702 |
| US | S1-343814906 | Wells Chesapeake | 312 44th Street, Hibbing, MN | wells chesapeake | S2-272493393 | Wells  Chesapeake | *(empty)* | wells chesapeake | *(empty)* | 0.514 | 0.403 |
| US | S1-959486030 | Summit Choice Crest | 1921 Humboldt Avenue, Minneapolis, MN | summit choice crest | S2-81526405 | Summit Choice | *(empty)* | summit choice | *(empty)* | 0.251 | 0.185 |
| US | S1-529889117 | JLQ Automotive LLC | 6544 Boston Avenue, Portland, OR | jlq automotive | S2-837205377 | JLQ Automotive | *(empty)* | jlq automotive | *(empty)* | 0.462 | 0.300 |
| US | S1-50536397 | Velazquez Worldwide Inc | 1350 Main Street, Manchester, OK | velazquez worldwide | S2-555961760 | Velazquez Inc Worldwide | *(empty)* | velazquez worldwide | *(empty)* | 0.537 | 0.449 |
| US | S1-592146044 | Trusted Platinum Jersey LLC | 120 Fifth Avenue, Clifton, IL | trusted platinum jersey | S2-41617606 | Trusted Platinum Jersey | 5th Ave, CLIFTON CIITY, IL | trusted platinum jersey | *(empty)* | 0.670 | 0.670 |
| US | S1-111364292 | Empire Renewables | 15320 205th Avenue, Bonney Lake, WA | empire renewables | S3-207300172 | Empire Renewables Ltd | *(empty)* | empire renewables | *(empty)* | 0.337 | 0.212 |
| US | S1-587253897 | Mick Municipals Inc. | 5143 Prairie Clover Trail, Tucson, AZ | mick municipals | S3-664219861 | Mick Municipals Inc | 539 Prairie Clover Trl, Tucson, Arizona | mick municipals | *(empty)* | 0.588 | 0.588 |
| US | S1-860326654 | Horizon Chiron, Inc | 1631 Shade Road, Akron, OH | horizon chiron | S2-308517255 | HORIZON CHIRON,  INCORPORATED | *(empty)* | horizon chiron | *(empty)* | 0.294 | 0.194 |
| US | S1-424095229 | ACM Holdings | 3 Winthrop Park, Malden, MA | acm holdings | S2-794299154 | ACM Holdings | MALDEN, 1 WINTHROP PARK, MA | acm holdings | *(empty)* | 0.641 | 0.641 |
| US | S1-738841324 | Eddi's Value Productions LLC | Unit C4, 80 Smith Street, MA, Lowell | eddis value productions | S3-925881041 | Eddi's Value Productions [LLC] | *(empty)* | eddis value productions | *(empty)* | 0.601 | 0.505 |

Decision search time 1.9 min, peak RSS 6.7 GiB.


## Part C. Submission variants (decision layer only)

All: odds one-home + expected-F0.5 on the slim candidates; the same candidate_pairs.tsv. OOF = macro F0.5 over all train S1 (France overrides have no train effect).

| variant | stage | gamma | override | OOF | OOF India | OOF US | test matches | mean set size France | mean set size India | mean set size US | validator --check-ids |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 04_s2_g15 | s2 | 1.5 | - | 0.97639 | 0.97142 | 0.97970 | 6,048,656 | 3.568 | 3.388 | 3.587 | PASS |
| 04_s2_g20 | s2 | 2.0 | - | 0.97594 | 0.97080 | 0.97936 | 5,999,909 | 3.540 | 3.362 | 3.556 | PASS |
| 04_s2_g25 | s2 | 2.5 | - | 0.97544 | 0.97015 | 0.97897 | 5,963,432 | 3.520 | 3.343 | 3.533 | PASS |
| 04_s1_g15 | s1 | 1.5 | - | 0.97266 | 0.96606 | 0.97706 | 5,759,106 | 3.411 | 3.275 | 3.350 | PASS |
| 04_s1_g20 | s1 | 2.0 | - | 0.97181 | 0.96493 | 0.97640 | 5,720,848 | 3.395 | 3.248 | 3.332 | PASS |
| 04_s2_g15_fr25 | s2 | 1.5 | {"France": 2.5} | 0.97639 | 0.97142 | 0.97970 | 6,036,227 | 3.520 | 3.388 | 3.587 | PASS |


## Part D. Final package

Dry run: `python -m src.make_package --variant 04_s2_g15 --team TEAM` -> `TEAM_submission.zip`

```
uncompressed_bytes  compressed_bytes  path
  100,389,819     43,120,094  output/matching_results.tsv
  143,654,883     61,430,770  output/candidate_pairs.tsv
          129            110  code/business_entity_resolution/src/__init__.py
       23,689          7,374  code/business_entity_resolution/src/blocking.py
       16,114          4,999  code/business_entity_resolution/src/blocking_report.py
        4,646          1,978  code/business_entity_resolution/src/data.py
       15,795          5,587  code/business_entity_resolution/src/decide.py
       54,530         18,029  code/business_entity_resolution/src/eda.py
       17,107          5,783  code/business_entity_resolution/src/features.py
        5,638          2,108  code/business_entity_resolution/src/final_report.py
        3,840          1,725  code/business_entity_resolution/src/io_utils.py
        6,518          2,628  code/business_entity_resolution/src/loco.py
        1,069            538  code/business_entity_resolution/src/make_empty_submission.py
        2,342            994  code/business_entity_resolution/src/make_package.py
        2,819          1,094  code/business_entity_resolution/src/metrics.py
        8,968          3,260  code/business_entity_resolution/src/norm_report.py
       16,362          5,960  code/business_entity_resolution/src/normalize.py
        4,407          1,702  code/business_entity_resolution/src/probes.py
       19,900          6,325  code/business_entity_resolution/src/pruner.py
        6,370          2,411  code/business_entity_resolution/src/pseudo.py
        9,407          4,024  code/business_entity_resolution/src/resources.py
        3,476          1,454  code/business_entity_resolution/src/run_all.py
       10,612          3,793  code/business_entity_resolution/src/slim.py
        8,704          3,132  code/business_entity_resolution/src/stage2.py
        9,512          3,481  code/business_entity_resolution/src/submit.py
        2,750          1,300  code/business_entity_resolution/src/text_norm.py
       12,130          4,549  code/business_entity_resolution/src/train.py
       15,342          5,389  code/business_entity_resolution/src/translit.py
        3,496          1,446  code/business_entity_resolution/src/v2_report.py
        7,339          2,778  code/business_entity_resolution/src/variants.py
        5,543          2,419  code/business_entity_resolution/README.md
          415            264  code/business_entity_resolution/requirements.txt
       11,928          5,220  Documentation_template.md
TEAM_submission.zip: 99.8 MiB
```

Pipeline plan (`python -m src.run_all --variant 04_s2_g15 --dry-run`):

```
[run_all] translit: src.translit
[run_all] normalize: src.normalize --split train
[run_all] normalize: src.normalize --split test
[run_all] blocking: src.blocking generate --split train
[run_all] blocking: src.blocking generate --split test
[run_all] pruner: src.pruner train
[run_all] pruner: src.pruner score
[run_all] pruner: src.pruner rank
[run_all] pruner: src.pruner grid
[run_all] pruner: src.pruner prune
[run_all] slim: src.slim apply
[run_all] features: src.features --split train
[run_all] features: src.features --split test
[run_all] stage1: src.train --tag v3s1
[run_all] stage2f: src.stage2 --p-tag v3s1
[run_all] stage2: src.train --tag v3s2 --stage2
[run_all] output: src.variants --only 04_s2_g15 --out C:\Users\Shrish Singh Sourya\Downloads\6ab10eb3b23ba_student_resource\student_resource\output
```


## Part E. Pseudo-labelling check (target country as a stand-in for France)

Stage-1 slim features. Source US: 30% S1 sample (1,613,548 pairs). Pseudo labels on the pairs of a random 30% of India's S1: p >= 0.97 -> 1, p <= 0.03 -> 0, others dropped (979,287 pairs, 0.8293 positive, 98.27% agree with the true labels). Scored on ALL India pairs with the true labels, decision odds + ef gamma 1.0.

| model | rounds | India macro F0.5 | singleton | non-singleton |
|---|---|---|---|---|
| US only | 1378 | 0.93715 | 0.8091 | 0.9447 |
| US + pseudo-labelled India | 1309 | 0.93757 | 0.8145 | 0.9449 |

Change: +0.00042. Runtime 11.3 min, peak RSS 5.5 GiB. No submission was changed.

