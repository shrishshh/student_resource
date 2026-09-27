# Master task: diagnostics, collective smoothing, final 07 submissions

## 1a. Data audit (train)

(i) Train-test overlap (test rows whose values exactly equal a train row of the same source):

| source | test rows | raw (name, address) | normalised (name_core, addr_core) |
|---|---|---|---|
| S1 | 1,732,544 | 0.000% | 0.000% |
| S2 | 4,887,273 | 0.000% | 0.313% |
| S3 | 5,082,316 | 0.000% | 0.362% |

(ii) Link tokens removed by our cleaning (all sources; same value + same country):

| token | records | same-value pairs | precision (same S1 entity) | coverage (multi-record entities) |
|---|---|---|---|---|
| phone (>=7 digits) | 505,946 | 851,436 | 0.4648 | 4.053% |
| (ID: n) tag | 30,924 | 2,581 | 0.0000 | 0.000% |
| #tag | 33,124 | 411,625 | 0.0058 | 0.026% |
| @handle | 32,079 | 28,100 | 0.0037 | 0.005% |
| web domain | 412,008 | 899,943 | 0.0213 | 0.854% |

(iii) Strict record-to-record links within the same source (S2 or S3) and country; coverage = share of same-entity same-source record pairs captured; last column = 04_s1 (gamma 1.5) OOF false negatives linked to a record accepted for the correct S1 (of 236,521 false negatives; 254,909 further true pairs are not in the candidates):

| rule | linked pairs | precision | coverage | linked false negatives |
|---|---|---|---|---|
| name_core+addr_core | 1,009,109 | 0.9985 | 17.792% | 5,355 |
| raw name | 8,482,809 | 0.0312 | 4.671% | 8,066 |
| raw address | 1,320,331 | 0.7318 | 17.062% | 13,045 |
| name_core, both addr empty | 73,953 | 0.0773 | 0.101% | 23 |

## 1b. Adversarial validation (train vs test pair features)

**US+India** test pairs vs train pairs: AUC **0.7705** (1,000,000 test pairs). Top 15 features:

| feature | gain | train mean | test mean |
|---|---|---|---|
| chain_s | 494,596 | 20.5557 | 14.6026 |
| n_cand_s1 | 382,479 | 5.3044 | 6.0872 |
| addr_ntok_s | 115,490 | 7.3588 | 8.2963 |
| pscore | 107,268 | 0.7464 | 0.6767 |
| rank_s_all | 90,921 | 16.6207 | 15.2553 |
| prank_s | 76,300 | 3.1555 | 3.5456 |
| house_logdiff | 72,385 | 1.2533 | 1.4631 |
| prank_s_all | 59,948 | 3.1775 | 3.5708 |
| chain_r | 53,662 | 14.0976 | 10.0893 |
| s_rank_addr | 49,843 | 2.3620 | 2.7743 |
| name_ntok_s | 48,037 | 2.5149 | 2.4350 |
| rank_s | 41,107 | 3.1534 | 3.5450 |
| name_cov_r | 41,077 | 0.8365 | 0.8258 |
| key_name_pair | 35,656 | 0.6998 | 0.6807 |
| s_n_near | 33,101 | 3.7540 | 4.2557 |

**France** test pairs vs train pairs: AUC **0.9898** (1,000,000 test pairs). Top 15 features:

| feature | gain | train mean | test mean |
|---|---|---|---|
| state_match | 1,151,763 | 0.8648 | 0.2833 |
| n_cand_s1 | 1,074,399 | 5.3044 | 7.3329 |
| addr_ntok_s | 886,884 | 7.3588 | 5.7381 |
| addr_tsort | 650,549 | 83.6034 | 91.1072 |
| name_ntok_s | 495,253 | 2.5149 | 2.2000 |
| addr_tsr | 477,751 | 89.8077 | 92.5480 |
| chain_s | 358,872 | 20.5557 | 18.8972 |
| addr_ntok_r | 294,472 | 6.4278 | 5.4977 |
| nkeys | 293,541 | 5.6357 | 4.6373 |
| house_logdiff | 254,258 | 1.2533 | 0.8772 |
| key_addr_pair | 195,786 | 0.7729 | 0.7125 |
| nums_jaccard | 187,807 | 0.5965 | 0.5691 |
| addr_ratio | 185,353 | 78.7381 | 86.0395 |
| name_len_diff | 183,576 | 2.2014 | 3.2299 |
| addr_cov_r | 179,122 | 0.7911 | 0.7923 |

## 2. Variants in the normal (a) and test-like (b) worlds (odds + ef)

Collective smoothing groups records of the same S1 and source linked by rules with train precision >= 0.98: ['name_core+addr_core'].

| model | smoothing | (a) g 1.0 | (a) g 1.5 | (a) g 2.0 | (a) best | (b) g 1.0 | (b) g 1.5 | (b) g 2.0 | (b) best |
|---|---|---|---|---|---|---|---|---|---|
| 04_s1 | none | 0.97316 | 0.97266 | 0.97181 | 0.97316 (g 1.0) | 0.97068 | 0.97083 | 0.97036 | 0.97083 (g 1.5) |
| 04_s1 | mean | 0.97318 | 0.97267 | 0.97180 | 0.97318 (g 1.0) | 0.97071 | 0.97085 | 0.97036 | 0.97085 (g 1.5) |
| 04_s1 | median | 0.97318 | 0.97268 | 0.97181 | 0.97318 (g 1.0) | 0.97071 | 0.97086 | 0.97037 | 0.97086 (g 1.5) |
| 04_s1 | maxif | 0.97321 | 0.97274 | 0.97192 | 0.97321 (g 1.0) | 0.97074 | 0.97091 | 0.97046 | 0.97091 (g 1.5) |
| D | none | 0.97267 | 0.97178 | 0.97065 | 0.97267 (g 1.0) | 0.97145 | 0.97094 | 0.97002 | 0.97145 (g 1.0) |
| D | mean | 0.97269 | 0.97179 | 0.97064 | 0.97269 (g 1.0) | 0.97147 | 0.97095 | 0.97001 | 0.97147 (g 1.0) |
| D | median | 0.97270 | 0.97180 | 0.97065 | 0.97270 (g 1.0) | 0.97147 | 0.97096 | 0.97002 | 0.97147 (g 1.0) |
| D | maxif | 0.97272 | 0.97186 | 0.97076 | 0.97272 (g 1.0) | 0.97149 | 0.97101 | 0.97012 | 0.97149 (g 1.0) |
| D2 | none | 0.97177 | 0.97112 | 0.97009 | 0.97177 (g 1.0) | 0.97070 | 0.97033 | 0.96945 | 0.97070 (g 1.0) |
| D2 | mean | 0.97180 | 0.97113 | 0.97008 | 0.97180 (g 1.0) | 0.97073 | 0.97034 | 0.96944 | 0.97073 (g 1.0) |
| D2 | median | 0.97180 | 0.97113 | 0.97009 | 0.97180 (g 1.0) | 0.97073 | 0.97035 | 0.96946 | 0.97073 (g 1.0) |
| D2 | maxif | 0.97182 | 0.97118 | 0.97018 | 0.97182 (g 1.0) | 0.97075 | 0.97039 | 0.96954 | 0.97075 (g 1.0) |

## 3. Final submissions

| submission | test matches | mean set size France | mean set size India | mean set size US | validator |
|---|---|---|---|---|---|
| 07_D_g10 | 5,778,970 | 3.405 | 3.294 | 3.359 | PASS |
| 07_D_g15 | 5,717,272 | 3.375 | 3.252 | 3.329 | PASS |

## Findings summary

- **1a**: no train-test overlap (raw 0.000-0.0004%, normalised <= 0.36% per source), so no overlap rule. None of the removed link tokens is a usable key (phone-like runs 46% precise, domains 2%, #tags / @handles / ID tags ~0%). Of the strict same-source rules only identical name_core + addr_core qualifies (99.85% precise, 17.8% of same-entity same-source pairs); it links only 5,355 of 236,521 false negatives of 04_s1 (gamma 1.5) to an accepted true record, which is why collective smoothing adds at most +0.00008.
- **1b**: US+India test pairs are separable from train pairs with AUC 0.77, driven by candidate/S1-set structure (chain_s 20.6 -> 14.6, n_cand_s1 5.3 -> 6.1, pscore 0.75 -> 0.68), consistent with the test-like world. France is almost perfectly separable (AUC 0.99), led by **state_match: 36% of France pairs have no state on one side (5% in train)** because ~35% of France S2/S3 addresses carry only street + city (e.g. "31 AV. GILARD, NANTES"). A city -> region table learned unsupervised from the test S1 addresses (which always carry city and region) would fill most of these; not done here because it cannot be validated before the deadline.
- **2b**: removing the top-5 shifted features (D2) lost 0.0009 in the normal world and 0.0008 in the test-like world vs D, so it was not selected.
