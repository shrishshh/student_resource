# Final report (Task 5): slim candidates, fold-consistent stage 2, variants, package

**Chosen slim candidates: tau = 0.02** (S1 top-15, record top-3): 9,926,200 train / 9,412,340 test pairs, candidates per S1 4.5 train / 5.4 test (p95 8 / 9), pair recall 96.66%, oracle 0.98822, proxy loss 0.00059.

Runtime per part (minutes): A 4, A_apply 1.

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

Mean predicted set size per S1 (odds + ef, gamma 1.0): train OOF vs test. 03 used the mean of both stage-1 models' test p for every stage-2 model; the slim stage 2 applies each stage-2 model to test features built from the stage-1 model whose p it was trained on.

| run | train OOF France | train OOF India | train OOF US | test France | test India | test US |
|---|---|---|---|---|---|---|
| 02_prune_v2 (stage 1, v2 cands) | - | 3.252 | 3.290 | 3.438 | 3.310 | 3.371 |
| 03_stage2 (stage 2, v2 cands) | - | 3.272 | 3.307 | 3.601 | 3.440 | 3.661 |

### Stage 1 (slim)

*v3s1/train.md: not run / failed (see notes).*

*v3s1/decide.md: not run / failed (see notes).*

### Stage 2 (slim)

*v3s2/train.md: not run / failed (see notes).*

*v3s2/decide.md: not run / failed (see notes).*

*variants.md: not run / failed (see notes).*

## Part D. Final package

*package.md: not run / failed (see notes).*

*pseudo.md: not run / failed (see notes).*
