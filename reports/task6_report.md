# Task 6: record-side competition features vs the train -> test gap

Models: **04_s1** (existing slim stage 1), **N** (without 12 record-side competition features: `r_margin_score`, `r_margin_name`, `r_margin_addr`, `r_oth_score`, `r_oth_name`, `r_oth_addr`, `r_n_name90`, `rank_r`, `rank_r_all`, `n_cand_rec`, `prank_r`, `prank_r_all`; `pscore` is kept although the pruner saw pre-pruning record ranks/counts), **D** (trained on the test-like world). Test-like world: 19% of each country's train S1 dropped (seed 42; 419,090 S1) with their pairs; quick-score ranks, pruner scores and ranks, slim selection, candidate counts, competition features and chain sizes rebuilt (pruner model and blocking keys reused). Each fold model scores only the half it did not train on. Decision: odds + ef, gamma in [1.0, 1.5, 2.0, 2.5, 3.0]; R = drop an accepted record whose house number differs from the S1's and is shared by another candidate of that S1.

| model | world | best gamma | macro F0.5 | India | US | singleton | non-singleton | pred set size | true set size | with R (same gamma) | best with R |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 04_s1 | (a) normal OOF | 1.0 | 0.97316 | 0.96669 | 0.97748 | 0.9599 | 0.9739 | 3.274 | 3.461 | 0.92789 (-0.04527) | 0.92789 (g 1.0) |
| N | (a) normal OOF | 1.0 | 0.97222 | 0.96570 | 0.97657 | 0.9615 | 0.9729 | 3.263 | 3.461 | 0.92694 (-0.04528) | 0.92694 (g 1.0) |
| D | (a) normal OOF | 1.0 | 0.97267 | 0.96606 | 0.97709 | 0.9632 | 0.9732 | 3.263 | 3.461 | 0.92744 (-0.04523) | 0.92744 (g 1.0) |
| 04_s1 | (b) test-like | 1.5 | 0.97083 | 0.96403 | 0.97538 | 0.9569 | 0.9717 | 3.269 | 3.461 | 0.92586 (-0.04498) | 0.92586 (g 1.5) |
| N | (b) test-like | 1.0 | 0.97062 | 0.96402 | 0.97503 | 0.9473 | 0.9720 | 3.275 | 3.461 | 0.92556 (-0.04507) | 0.92556 (g 1.0) |
| D | (b) test-like | 1.0 | 0.97145 | 0.96475 | 0.97591 | 0.9517 | 0.9726 | 3.274 | 3.461 | 0.92631 (-0.04514) | 0.92631 (g 1.0) |

Macro F0.5 per gamma:

| model | world | gamma 1.0 | gamma 1.5 | gamma 2.0 | gamma 2.5 | gamma 3.0 |
|---|---|---|---|---|---|---|
| 04_s1 | a | 0.97316 | 0.97266 | 0.97181 | 0.97097 | 0.97008 |
| N | a | 0.97222 | 0.97134 | 0.97018 | 0.96907 | 0.96803 |
| D | a | 0.97267 | 0.97178 | 0.97065 | 0.96948 | 0.96835 |
| 04_s1 | b | 0.97068 | 0.97083 | 0.97036 | 0.96974 | 0.96904 |
| N | b | 0.97062 | 0.97025 | 0.96936 | 0.96840 | 0.96744 |
| D | b | 0.97145 | 0.97094 | 0.97002 | 0.96898 | 0.96797 |

Test submissions (odds + ef, slim candidates; candidate_pairs.tsv identical to 04_s1_g15):

| submission | test matches | mean set size France | mean set size India | mean set size US | validator --check-ids |
|---|---|---|---|---|---|
| 04_s1_g15 (reference) | 5,759,106 | 3.411 | 3.275 | 3.350 | PASS |
| 06_D_g10 | 5,778,970 | 3.405 | 3.294 | 3.359 | PASS |
| 06_D_g15 | 5,717,272 | 3.375 | 3.252 | 3.329 | PASS |

Runtime (minutes): train N 52, build test-like world 72, train D 22, evaluate 24, submissions 3.
