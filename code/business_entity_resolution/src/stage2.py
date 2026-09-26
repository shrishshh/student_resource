"""Stage-2 group-consistency features -> cache/stage2/{split}/{country}.parquet.

Usage (from code/business_entity_resolution/, after the stage-1 run ``v2``):
    python -m src.stage2 --p-tag v2

For every candidate pair (s, r), with p = stage-1 probability (out-of-fold for train,
mean of both fold models for test), DuckDB window functions / group-bys compute:

* S1 side: rank of p among s's candidates, gap to s's max p, sum of p, count of p > 0.5.
* S1 side, shared attributes: among s's OTHER candidates, how many share r's house_num /
  name_core / addr_core, and the sum and max of their p; whether r's house_num is the
  p-weighted most common house_num among s's candidates; share of s's total p at r's house_num.
* Record side: max p over r's OTHER candidate S1s, this p's margin over it, count of r's
  candidates with p > 0.1.
* Group sizes over the whole split: S2/S3 records sharing r's (name_core, house_num) and
  (name_core, addr_core); S2/S3 records sharing s's (name_core, house_num).

Rows are written in the stage-1 feature order (s1, rank_s) with one row group per feature
part, so :func:`augment` can attach them to a part without a join (keys are asserted).
"""

from __future__ import annotations

import argparse
import re
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .blocking import connect, country_list
from .data import CACHE, log, peak_rss, run_paths
from .features import CHUNK

OUT = CACHE / "stage2"


def _others(key: str, name: str) -> str:
    """SQL: count / sum p / max p over s's OTHER candidates with the same ``key`` as r."""
    w = f"PARTITION BY s1, {key}"
    wo = f"{w} ORDER BY p DESC, src, idx"
    null = f"{key} IS NULL OR {key} = ''"
    return f"""
        CASE WHEN {null} THEN NULL ELSE count(*) OVER ({w}) - 1 END AS s_oth_{name}_n,
        CASE WHEN {null} THEN NULL ELSE sum(p) OVER ({w}) - p END AS s_oth_{name}_sump,
        CASE WHEN {null} OR count(*) OVER ({w}) = 1 THEN NULL
             WHEN row_number() OVER ({wo}) = 1
                  THEN nth_value(p, 2) OVER ({wo} ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
             ELSE max(p) OVER ({w}) END AS s_oth_{name}_maxp"""


def build(split: str, p_tag: str, variant: str | None = None, p_col: str = "p") -> None:
    """Stage-2 features for every candidate pair of a split.

    Train uses the out-of-fold stage-1 p. For test, ``variant`` = ``test_m0`` / ``test_m1``
    builds the features from one stage-1 model's test p (column ``p_m0`` / ``p_m1``), so that
    each stage-2 model sees test inputs built like its training inputs (fold-consistent).
    """
    t0 = time.time()
    rp = run_paths(p_tag)
    p_file = rp["preds"] / ("train_oof.parquet" if split == "train" else "test.parquet")
    rec = (CACHE / f"{split}_records.parquet").as_posix()
    cand = (CACHE / f"{split}_candidates.parquet").as_posix()
    out_dir = OUT / (variant or split)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    con = connect()
    # group sizes over the whole split (S2/S3 records)
    con.execute(f"""
        CREATE TABLE g_nh AS SELECT name_core, house_num, count(*) AS n FROM read_parquet('{rec}')
            WHERE src <> 1 AND house_num IS NOT NULL GROUP BY ALL;
        CREATE TABLE g_na AS SELECT name_core, addr_core, count(*) AS n FROM read_parquet('{rec}')
            WHERE src <> 1 AND addr_core <> '' GROUP BY ALL;
        CREATE TABLE p AS SELECT s1, src, idx, {p_col} AS p FROM read_parquet('{p_file.as_posix()}');
    """)
    for country in country_list(split):
        c = country.replace("'", "''")
        con.execute(f"""
            CREATE OR REPLACE TABLE c AS
            SELECT k.s1, k.src, k.idx, k.rank_s, p.p,
                   r.name_core AS r_name, r.house_num AS r_house, r.addr_core AS r_addr,
                   s.name_core AS s_name, s.house_num AS s_house
            FROM read_parquet('{cand}') k
            JOIN p USING (s1, src, idx)
            JOIN (SELECT src, idx, name_core, house_num, addr_core FROM read_parquet('{rec}') WHERE country = '{c}') r
                 USING (src, idx)
            JOIN (SELECT idx AS s1, name_core, house_num FROM read_parquet('{rec}') WHERE country = '{c}' AND src = 1) s
                 USING (s1)
            WHERE k.country = '{c}';
            CREATE OR REPLACE TABLE hp AS
            SELECT s1, r_house, sum(p) AS sp FROM c WHERE r_house IS NOT NULL GROUP BY ALL;
            CREATE OR REPLACE TABLE hmode AS
            SELECT s1, arg_max(r_house, sp) AS mode_house, sum(sp) AS tot FROM hp GROUP BY s1;
        """)
        path = out_dir / f"{country}.parquet"
        con.execute(f"""
            COPY (
                SELECT c.s1, c.src, c.idx, c.rank_s, c.p AS p1,
                       row_number() OVER (PARTITION BY c.s1 ORDER BY c.p DESC, c.src, c.idx) AS s_rank_p,
                       max(c.p) OVER (PARTITION BY c.s1) - c.p AS s_gap_maxp,
                       sum(c.p) OVER (PARTITION BY c.s1) AS s_sum_p,
                       sum(CASE WHEN c.p > 0.5 THEN 1 ELSE 0 END) OVER (PARTITION BY c.s1) AS s_cnt_p50,
                       {_others('c.r_house', 'house')},
                       {_others('c.r_name', 'name')},
                       {_others('c.r_addr', 'addr')},
                       CASE WHEN c.r_house IS NULL THEN NULL ELSE (c.r_house = hm.mode_house)::INTEGER END AS r_house_is_mode,
                       CASE WHEN c.r_house IS NULL OR hm.tot = 0 THEN NULL ELSE hp.sp / hm.tot END AS s_pshare_r_house,
                       CASE WHEN count(*) OVER (PARTITION BY c.src, c.idx) = 1 THEN NULL
                            WHEN row_number() OVER (PARTITION BY c.src, c.idx ORDER BY c.p DESC, c.s1) = 1
                                 THEN nth_value(c.p, 2) OVER (PARTITION BY c.src, c.idx ORDER BY c.p DESC, c.s1
                                                              ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)
                            ELSE max(c.p) OVER (PARTITION BY c.src, c.idx) END AS r_oth_maxp,
                       sum(CASE WHEN c.p > 0.1 THEN 1 ELSE 0 END) OVER (PARTITION BY c.src, c.idx) AS r_cnt_p10,
                       coalesce(g1.n, 0) AS grp_r_name_house,
                       coalesce(g2.n, 0) AS grp_r_name_addr,
                       coalesce(g3.n, 0) AS grp_s_name_house
                FROM c
                LEFT JOIN hmode hm USING (s1)
                LEFT JOIN hp ON hp.s1 = c.s1 AND hp.r_house = c.r_house
                LEFT JOIN g_nh g1 ON g1.name_core = c.r_name AND g1.house_num = c.r_house
                LEFT JOIN g_na g2 ON g2.name_core = c.r_name AND g2.addr_core = c.r_addr
                LEFT JOIN g_nh g3 ON g3.name_core = c.s_name AND g3.house_num = c.s_house
            ) TO '{path.as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
        # exact feature-part alignment: sort like the feature parts and write CHUNK-row groups
        tbl = pq.read_table(path).sort_by([("s1", "ascending"), ("rank_s", "ascending")]).drop_columns(["rank_s"])
        pq.write_table(tbl, path, row_group_size=CHUNK, compression="zstd")
        n = tbl.num_rows
        del tbl
        log(f"stage2 {variant or split}/{country}: {n:,} rows")
    con.close()
    shutil.rmtree(CACHE / "duckdb_tmp", ignore_errors=True)
    log(f"stage2 {variant or split}: {(time.time() - t0) / 60:.1f} min, peak {peak_rss():.1f} GiB")


_PART_RE = re.compile(r"part-(\d+)\.parquet$")


def augment(df: pd.DataFrame, split: str, country: str, path: str, variant: str | None = None) -> pd.DataFrame:
    """Attach the stage-2 columns (incl. p1) to one stage-1 feature part.

    ``variant`` selects a fold-specific test build (``test_m0`` / ``test_m1``).
    """
    i = int(_PART_RE.search(path).group(1))
    pf = pq.ParquetFile(OUT / (variant or split) / f"{country}.parquet")
    g = pf.read_row_group(i).to_pandas()
    assert len(g) == len(df) and np.array_equal(g["s1"].to_numpy(), df["s1"].to_numpy()) \
        and np.array_equal(g["idx"].to_numpy(), df["idx"].to_numpy()), f"stage2 rows misaligned for {path}"
    extra = g.drop(columns=["s1", "src", "idx"]).astype(np.float32)
    return pd.concat([df.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-2 group-consistency features")
    ap.add_argument("--p-tag", default="v2", help="run whose stage-1 predictions are used")
    a = ap.parse_args()
    build("train", a.p_tag)
    for m in (0, 1):  # fold-consistent test inputs, one per stage-1 model
        build("test", a.p_tag, variant=f"test_m{m}", p_col=f"p_m{m}")


if __name__ == "__main__":
    main()
