# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** TEAM  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
A four-stage entity-resolution pipeline built only from the provided data: country-aware normalisation with a
learned Indic-to-Latin transliteration dictionary; high-recall DuckDB blocking followed by a *learned* pruner
that keeps a slim candidate set (~{{CANDS_TEST}} candidates per Source-1 entity on test); a two-stage LightGBM
matcher whose second stage sees group-consistency features (what the other candidates of the same S1 / record
look like); and a decision layer that makes each S2/S3 record choose at most one home (odds renormalisation)
and picks each S1's match set with an expected-F0.5 optimiser. Out-of-fold macro F0.5 on train:
**{{OOF_FINAL}}** (stage 2), validated by 2-fold out-of-fold predictions by S1 and a leave-one-country-out check.

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from our EDA (`reports/eda_report.md`) that shaped the design:
- Train: 2.21M S1, 5.03M S2, 5.29M S3 (US 60%, India 40%); test 1.73M S1 adds **France (15% of test S1)**,
  unseen in train. Every S2/S3 record matches at most one S1, never across countries; 5.6% of S1 are
  singletons; mean 3.46 matches per S1; ~26% of S2/S3 records are orphans; test has 5.75 S2/S3 per S1 vs 4.68 in train.
- S2/S3 are not deduplicated (81% of non-singletons have >= 2 matches from the same source).
- ~18% of Indian noisy names are written in one of 9 Indic scripts (phonetic transliterations); Indian states
  appear as native script or codes (MH, TN); US states as names where S1 uses codes; France uses regions in S1
  and departments (Nord, Gironde, ...) in S2/S3.
- Names repeat heavily across S1 (43-54% of S1 share their stripped name with another S1 in the same country;
  identical name + country is only 4.2% precise), so the address decides those; co-located businesses share
  addresses (identical address is 82% precise), so the name decides those.
- Noise: web/handle forms, honorifics (M/s, Shri, Dr), "JUNK formerly known as / DBA REAL NAME", appended
  phone numbers and #tags, leet digits (5ervices), random accents, typos; addresses with abbreviations,
  state code/name swaps, "null" components, '#' insertions, jittered house numbers, reordered/dropped parts,
  ~4.4% empty addresses; postal codes almost never present.

### 2.2 Solution Strategy
**Approach Type:** Hybrid: rule-based normalisation + learned blocking/pruning + two-stage gradient-boosted
pairwise classifier + set-level decision optimisation.  
**Core Innovation:** (1) a transliteration dictionary learned from aligned train pairs (Indic name q10
similarity 10 -> 84); (2) a learned pruner that ranks all ~440M blocked pairs with cheap features, recovering
the empty-address matches that score-based pruning dropped; (3) stage-2 group-consistency features plus a
one-home, expected-F0.5 decision layer tuned directly on the leaderboard metric.

---

## 3. Candidate Generation (Blocking)
- **Normalisation first** (`src/normalize.py`): mojibake repair, NFKC, Indic transliteration (dictionary first,
  rule-based fallback over one shared Brahmic offset table), accent folding, ligatures, dotted initials
  (l.l.c. -> llc), French elisions; names -> `name_core` (honorifics, legal forms of all countries, web parts,
  stop words, phone numbers, #tags removed; glued words segmented against the S1 vocabulary), `name_alt` (the
  real name after a DBA / formerly marker), phonetic keys; addresses -> country-keyed abbreviation expansion,
  canonical state/region code (native-script states mapped by a learned table), house number, numbers.
- **Blocking keys used** (DuckDB, per country, hashed): pairs of the 3 rarest name tokens; rare single name
  tokens (S1 df <= 50); phonetic pair of the 2 rarest tokens; 8-char name prefix; the same keys from `name_alt`;
  pairs of the 3 rarest address tokens; (house number, rarest address token); (rarest name token, rarest
  address token). S2/S3 keys use only tokens present in the country's S1 vocabulary; blocks with
  n_S1 x n_S2S3 > 20,000 are skipped. Recall before pruning: 97.1% (India) / 98.3% (US) of true pairs.
- **Learned pruner** (`src/pruner.py`): LightGBM (300 trees, 63 leaves) trained on all unpruned pairs of 3% of
  train S1 with 28 cheap features (key-type bits, quick similarities, house-number agreement, empty-address
  flags, chain sizes, pre-pruning ranks and counts). **Slim candidates** (`src/slim.py`): keep a pair if pruner
  p >= {{TAU}} and it is in the S1's top 15 and the record's top 3; tau chosen as the smallest set whose proxy
  loss on our best model's decisions is <= 0.0007.
- **Candidate pairs generated:** {{TRAIN_PAIRS}} train / {{TEST_PAIRS}} test; **{{CANDS_TRAIN}} / {{CANDS_TEST}}
  candidates per S1** (mean; p95 {{P95_TRAIN}} / {{P95_TEST}}); pair recall {{RECALL}}; reduction ratio > 0.99999
  vs all same-country pairs.
- **How you ensured true matches were not lost:** several independent key families (name, phonetic, prefix,
  alternate name, address, house number, name x address); the pruner learns to keep matches whose address is
  empty (86,974 of the 134,191 empty-address matches missed by score pruning were recovered); every budget was
  chosen on train by the oracle macro F0.5 (predicting ground truth ∩ candidates): {{ORACLE}}.

---

## 4. Matching Model

**Features used** (~77 per pair, `src/features.py`; country is never a feature):
- Name features: token_set / token_sort / ratio / partial ratio, Jaro-Winkler on `name_core`; similarity to the
  record's `name_alt`; concatenated-name ratio and containment; phonetic token_set ratio; exact match; token
  Jaccard; IDF-weighted coverage in both directions; first-token and rarest-token agreement; token counts;
  length difference.
- Address features: token_set / token_sort / ratio / partial ratio of `addr_core`; IDF-weighted coverage both
  ways; state match (same / different / missing); house number equal / log-difference / prefix / missing;
  number-set Jaccard and coverage; token counts.
- Other: record flags (source, empty address, Indic name, web name, trade-name marker, landmark, PO box);
  frequency features (S1 chain size, how many S1 share the record's name, share of the record's tokens found
  in the S1 vocabulary = orphan signal); blocking key bits, quick scores, pruner score and ranks; competition
  features (DuckDB windows: best other S1 of the record and margins, number of near-duplicates, rank of the
  pair inside the S1 by name / address). **Stage 2** (`src/stage2.py`) adds, from stage-1 probabilities:
  rank/gap/sum/count of p inside the S1, how many of the S1's other candidates share the record's house number /
  name / address and their summed and max p, whether the record is at the S1's p-weighted house number, the
  record's best other S1 and margin, and split-wide group sizes of (name, house number) and (name, address).

**Model type:** LightGBM binary classifiers (num_leaves 127, learning rate 0.05, min_data_in_leaf 200,
feature/bagging fraction 0.8, lambda_l2 1, early stopping on logloss). Train S1 are split into two halves by a
hash of the S1 id; each fold model trains on 30% of one half's S1 and predicts the other half, giving
out-of-fold probabilities for every train pair; test = mean of the two models. Stage 2 is trained the same
way on the out-of-fold stage-1 p; on test each stage-2 model sees features built from the stage-1 model whose
probabilities it was trained on (fold-consistent).  
**Threshold selection method:** decision layer tuned on the out-of-fold predictions of all train S1 with the
exact metric: (1) one-home: each record's candidate probabilities are renormalised as odds / (1 + sum of odds)
so a record cannot be given to several S1; (2) per S1, an expected-F0.5 optimiser (numba) picks the set size j
maximising E[F0.5] under Poisson-binomial models of true positives and missed matches, on q^gamma. Leaderboard
probes showed the test set rewards more conservative decisions, so the submission uses gamma = 1.5.

---

## 5. Results & Error Analysis

{{RESULTS_TABLE}}

- **F_0.5 Score (macro):** OOF {{OOF_FINAL}} (stage 2, slim candidates); leave-one-country-out stand-in for an
  unseen country: US -> India 0.924, India -> US 0.956 (vs 0.964 / 0.975 when both are in training).
- **Common false positives (wrong merges):** sibling businesses at the same address with a similar generic name
  (chains, "X Services" vs "X Consultants"), and records whose house number was jittered to a neighbour.
- **Common false negatives (missed matches):** records with empty or junk names that only share an address
  with many candidates, heavily abbreviated names (initials only), and matches lost in blocking when both name
  and address are noisy.

---

## 6. Conclusion
Careful normalisation (especially learned transliteration) and a learned, recall-oriented pruner mattered most:
they raised the achievable ceiling while shrinking candidate sets; the two-stage matcher and the set-level
decision layer then turned that recall into precision-weighted F0.5. Treating France as an unseen country was
handled by never using country as a feature and by country-keyed rule tables with a generic fallback.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/`: `src/` (all code), `README.md` (exact commands, hardware, runtime),
`requirements.txt` (pinned). Entry point: `python -m src.run_all --variant 04_s2_g15` from
`code/business_entity_resolution/` regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv`
(~5-6 h on a 16 GB RAM / 16-thread laptop, no GPU).

**Compliance:** only the provided train/test files are used; no external data, APIs, geocoding or pretrained
models. All rule lists are hand-written in `src/resources.py`; everything learned is learned from train (or
unsupervised from a split's own records). Libraries and licenses: LightGBM (MIT), DuckDB (MIT), rapidfuzz (MIT),
numba (BSD-2), pandas / numpy / scikit-learn / scipy (BSD-3), pyarrow (Apache-2.0), psutil (BSD-3). The final
models are LightGBM tree ensembles (well under the 8B-parameter limit).

### B. Additional Results
Stage reports: `reports/eda_report.md`, `normalization_report.md`, `blocking_report.md`, `matcher_report.md`,
`v2_report.md`, `final_report.md`; experiment log `reports/experiments.md`.

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
