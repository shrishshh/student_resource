# France region fill -> 08_D_g15_frstate

City -> region table from test S1 France addresses (digit-free components, >= 20 occurrences, one region only): **25 cities** (hdf 13, naq 5, pdl 7).

France S2/S3 records without a region: 500,864; **filled: 449,722** (89.8%).

state_match on the France test pairs:

| *(empty)* | -1 (missing) | 0 (different) | 1 (same) |
|---|---|---|---|
| before | 584,865 (35.73%) | 5,714 (0.35%) | 1,046,455 (63.92%) |
| after | 82,766 (5.06%) | 8,396 (0.51%) | 1,545,872 (94.43%) |

Decision odds + ef gamma 1.5 with model D (mean of both fold models); US and India rows identical to 07_D_g15: **True**.

| France | % S1 predicted empty | mean set size | France matches |
|---|---|---|---|
| 07_D_g15 | 5.07% | 3.375 | 875,611 |
| 08_D_g15_frstate | 5.08% | 3.374 | 875,520 |

Total test matches: 5,717,272 (07_D_g15) -> **5,717,181** (08_D_g15_frstate). Validator (--check-ids): PASS. Runtime 3.2 min.

Sample of the city table: av willy brandt -> hdf, bordeaux -> naq, calais -> hdf, dunkerque -> hdf, hellemmes lille -> hdf, iut c -> hdf, la baule escoublac -> pdl, la teste de buch -> naq, le clion -> pdl, lege cap ferret -> naq, lille -> hdf, lomme -> hdf, maison de la vie associative -> hdf, merignac -> naq, nantes -> pdl, pessac -> naq, pornic -> pdl, roubaix -> hdf, rpt de leurope -> hdf, rue monge -> hdf, saint herblain -> pdl, saint nazaire -> pdl, sainte marie -> pdl, terre plein du jeu de mail -> hdf, tourcoing -> hdf.
