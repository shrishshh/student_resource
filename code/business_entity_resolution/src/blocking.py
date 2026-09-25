"""Blocking v1: candidate (S1, S2/S3) pairs per country with DuckDB.

Usage (from code/business_entity_resolution/, after src.normalize):
    python -m src.blocking generate --split train --country India   # keys -> all pairs, scored
    python -m src.blocking generate --split train --country US
    python -m src.blocking tune                                       # pick (k, K) on train
    python -m src.blocking generate --split test                      # all test countries
    python -m src.blocking prune --split train|test                   # -> cache/{split}_candidates.parquet

Stages per country
------------------
1. Tokens + document frequencies (S1+S2+S3 of that split and country) per field.
2. Keys per record (hashed with DuckDB ``hash``; one bit per key type):
   name pairs, name singles, phonetic pair, concat prefix, the same name keys from
   name_alt, address pairs, (house number, rarest address token), and
   (rarest name token, rarest address token). S2/S3 records only use tokens present
   in that country's S1 vocabulary. Blocks with n_S1 * n_S2S3 > MAX_BLOCK are skipped.
3. Union -> distinct pairs with key bitmask and number of shared keys.
4. Quick score: sim_name (max token_set_ratio over core/alt/phonetic), sim_addr.
5. Pruning keeps a pair when it is in the record's top-k S1s or the S1's top-K records.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from .data import ARTIFACTS, CACHE, gt_pairs, log, peak_rss
from .metrics import macro_f05
from .normalize import phon

MAX_BLOCK = 20_000
SCORE_CHUNK = 5_000_000
KEY_TYPES = ["name_pair", "name_single", "name_phon", "concat_prefix", "name_alt",
             "addr_pair", "addr_number", "cross"]
GRID_K = [1, 2, 3, 5]      # top-k S1 per record
GRID_BIGK = [10, 20, 30, 50]  # top-K records per S1
BUDGET_PATH = ARTIFACTS / "blocking_budget.json"
RKEY_SHIFT = np.int64(1) << 32  # record key = src << 32 | idx


def connect() -> duckdb.DuckDBPyConnection:
    """DuckDB connection with the memory cap and an on-disk spill directory."""
    tmp = CACHE / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def pairs_path(split: str, country: str) -> str:
    """Scored, unpruned candidate pairs of one split+country."""
    return (CACHE / f"{split}_pairs_all_{country}.parquet").as_posix()


def stats_path(split: str, country: str):
    """Per-country blocking statistics (JSON)."""
    return CACHE / f"{split}_blockstats_{country}.json"


# ----------------------------------------------------------------- keys
def build_keys(con, split: str, country: str) -> dict:
    """Create tables ``rec`` and ``keys`` for one country; return timing/size info."""
    rec_path = (CACHE / f"{split}_records.parquet").as_posix()
    con.execute(f"""
        CREATE OR REPLACE TABLE rec AS
        SELECT src, idx, name_core, name_alt, name_phon, name_concat, addr_core, house_num
        FROM read_parquet('{rec_path}') WHERE country = ?""", [country])
    # name_alt phonetic codes (few rows) computed in Python with the same phon()
    alt = con.execute("SELECT src, idx, name_alt FROM rec WHERE name_alt <> ''").df()
    alt["alt_phon"] = [" ".join(phon(t) for t in s.split()) for s in alt["name_alt"]]
    con.register("alt_df", alt)
    con.execute("CREATE OR REPLACE TABLE altp AS SELECT src, idx, name_alt, alt_phon FROM alt_df")
    con.unregister("alt_df")

    con.execute("""
        CREATE OR REPLACE TABLE ntok AS
        SELECT DISTINCT src, idx, tok, ph FROM (
            SELECT src, idx, unnest(string_split(name_core, ' ')) AS tok,
                             unnest(string_split(name_phon, ' ')) AS ph
            FROM rec WHERE name_core <> '');
        CREATE OR REPLACE TABLE atok_alt AS
        SELECT DISTINCT src, idx, tok, ph FROM (
            SELECT src, idx, unnest(string_split(name_alt, ' ')) AS tok,
                             unnest(string_split(alt_phon, ' ')) AS ph
            FROM altp);
        CREATE OR REPLACE TABLE dtok AS
        SELECT DISTINCT src, idx, tok FROM (
            SELECT src, idx, unnest(string_split(addr_core, ' ')) AS tok FROM rec WHERE addr_core <> '')
        WHERE NOT regexp_matches(tok, '[0-9]');
        -- document frequencies over S1+S2+S3 of this country (IDF ranks = df ascending)
        CREATE OR REPLACE TABLE ndf AS SELECT tok, count(*) AS df, count(*) FILTER (WHERE src = 1) AS s1df FROM ntok GROUP BY tok;
        CREATE OR REPLACE TABLE pdf AS SELECT ph, count(*) AS df, count(*) FILTER (WHERE src = 1) AS s1df
            FROM (SELECT DISTINCT src, idx, ph FROM ntok) GROUP BY ph;
        CREATE OR REPLACE TABLE ddf AS SELECT tok, count(*) AS df, count(*) FILTER (WHERE src = 1) AS s1df FROM dtok GROUP BY tok;
    """)

    def ranked(src_tbl: str, col: str, df_tbl: str, out: str) -> None:
        """Eligible tokens ranked rarest-first (S2/S3: only tokens in S1 vocabulary)."""
        con.execute(f"""
            CREATE OR REPLACE TABLE {out} AS
            SELECT t.src, t.idx, t.{col} AS tok, d.df, d.s1df,
                   row_number() OVER (PARTITION BY t.src, t.idx ORDER BY d.df, t.{col}) AS rn,
                   count(*) OVER (PARTITION BY t.src, t.idx) AS cnt
            FROM (SELECT DISTINCT src, idx, {col} FROM {src_tbl}) t JOIN {df_tbl} d ON d.{'tok' if col == 'tok' else 'ph'} = t.{col}
            WHERE t.src = 1 OR d.s1df > 0""")

    ranked("ntok", "tok", "ndf", "nr")
    ranked("ntok", "ph", "pdf", "pr")
    ranked("atok_alt", "tok", "ndf", "anr")
    ranked("atok_alt", "ph", "pdf", "apr")
    ranked("dtok", "tok", "ddf", "dr")

    def pair_keys(tbl: str, prefix: str, kt: int, top: int) -> str:
        return f"""
            SELECT a.src, a.idx, hash('{prefix}|' || least(a.tok, b.tok) || '|' || greatest(a.tok, b.tok)) AS h, {kt} AS kt
            FROM {tbl} a JOIN {tbl} b ON a.src = b.src AND a.idx = b.idx AND a.rn < b.rn
            WHERE a.rn <= {top} AND b.rn <= {top}
            UNION ALL SELECT src, idx, hash('{prefix}|' || tok), {kt} FROM {tbl} WHERE cnt = 1"""

    kt = {k: i for i, k in enumerate(KEY_TYPES)}
    con.execute(f"""
        CREATE OR REPLACE TABLE keys AS SELECT DISTINCT src::TINYINT AS src, idx::INTEGER AS idx, h, kt::TINYINT AS kt FROM (
            {pair_keys('nr', 'N', kt['name_pair'], 3)}
            UNION ALL SELECT src, idx, hash('S|' || tok), {kt['name_single']} FROM nr WHERE rn <= 2 AND s1df <= 50
            UNION ALL {pair_keys('pr', 'P', kt['name_phon'], 2)}
            UNION ALL SELECT src, idx, hash('C|' || substr(name_concat, 1, 8)), {kt['concat_prefix']}
                      FROM rec WHERE length(name_concat) >= 6
            -- the same name-family keys from name_alt (same hash space, own bit)
            UNION ALL {pair_keys('anr', 'N', kt['name_alt'], 3)}
            UNION ALL SELECT src, idx, hash('S|' || tok), {kt['name_alt']} FROM anr WHERE rn <= 2 AND s1df <= 50
            UNION ALL {pair_keys('apr', 'P', kt['name_alt'], 2)}
            UNION ALL SELECT src, idx, hash('C|' || substr(replace(name_alt, ' ', ''), 1, 8)), {kt['name_alt']}
                      FROM rec WHERE length(replace(name_alt, ' ', '')) >= 6
            UNION ALL {pair_keys('dr', 'A', kt['addr_pair'], 3)}
            UNION ALL SELECT d.src, d.idx, hash('H|' || r.house_num || '|' || d.tok), {kt['addr_number']}
                      FROM dr d JOIN rec r USING (src, idx) WHERE d.rn = 1 AND r.house_num IS NOT NULL
            UNION ALL SELECT n.src, n.idx, hash('X|' || n.tok || '|' || d.tok), {kt['cross']}
                      FROM nr n JOIN dr d USING (src, idx) WHERE n.rn = 1 AND d.rn = 1
        )""")
    for t in ("ntok", "atok_alt", "dtok", "nr", "pr", "anr", "apr", "dr", "altp"):
        con.execute(f"DROP TABLE {t}")
    # block sizes: n1 S1 records, nr S2/S3 records per hash
    con.execute("""
        CREATE OR REPLACE TABLE blocks AS
        SELECT h, count(DISTINCT idx) FILTER (WHERE src = 1) AS n1,
                  count(DISTINCT (src::BIGINT << 32) | idx) FILTER (WHERE src <> 1) AS nr
        FROM keys GROUP BY h;
    """)
    return {"n_keys": con.execute("SELECT count(*) FROM keys").fetchone()[0]}


# ----------------------------------------------------------------- generate
def key_type_stats(con) -> dict:
    """Blocks / skipped blocks / raw pairs per key type.

    A block (hash) belongs to a key type when an S2/S3 record carries that hash with
    that type; its raw pairs for the type are n_S1 x (records of that type in it), so
    types sharing a hash space (name vs name_alt) are not double-charged.
    """
    rows = con.execute(f"""
        WITH kb AS (SELECT h, kt, count(DISTINCT (src::BIGINT << 32) | idx) AS nr_kt
                    FROM keys WHERE src <> 1 GROUP BY h, kt)
        SELECT kb.kt, count(*) FILTER (WHERE b.n1 > 0) AS blocks,
               count(*) FILTER (WHERE b.n1 > 0 AND b.n1 * b.nr > {MAX_BLOCK}) AS skipped,
               sum(b.n1 * kb.nr_kt) FILTER (WHERE b.n1 > 0 AND b.n1 * b.nr <= {MAX_BLOCK}) AS raw_pairs
        FROM kb JOIN blocks b USING (h) GROUP BY kb.kt ORDER BY kb.kt""").fetchall()
    return {KEY_TYPES[k]: {"blocks": int(bl), "skipped": int(sk), "raw_pairs": int(r or 0)} for k, bl, sk, r in rows}


def refresh_key_stats(split: str, country: str) -> None:
    """Rebuild keys only and refresh the key-type section of the stats JSON."""
    con = connect()
    build_keys(con, split, country)
    st = json.loads(stats_path(split, country).read_text(encoding="utf-8"))
    st["key_types"] = key_type_stats(con)
    stats_path(split, country).write_text(json.dumps(st, indent=1), encoding="utf-8")
    con.close()
    log(f"refreshed key stats {split}/{country}")


def generate(split: str, country: str) -> None:
    """Keys -> distinct pairs -> quick scores -> cache/{split}_pairs_all_{country}.parquet."""
    t0 = time.time()
    con = connect()
    stats: dict = {"split": split, "country": country, "timing": {}}
    log(f"[{split}/{country}] building keys")
    stats.update(build_keys(con, split, country))
    stats["timing"]["keys"] = time.time() - t0

    stats["key_types"] = key_type_stats(con)
    raw_total = con.execute(f"SELECT sum(n1 * nr) FROM blocks WHERE n1 > 0 AND nr > 0 AND n1 * nr <= {MAX_BLOCK}").fetchone()[0]
    stats["raw_pairs_total"] = int(raw_total or 0)
    log(f"[{split}/{country}] {stats['n_keys']:,} keys; raw pairs {stats['raw_pairs_total']:,}")

    log(f"[{split}/{country}] joining kept blocks -> distinct pairs")
    out_unscored = CACHE / f"{split}_pairs_unscored_{country}.parquet"
    con.execute(f"""
        CREATE OR REPLACE TABLE kk AS SELECT k.* FROM keys k
        SEMI JOIN (SELECT h FROM blocks WHERE n1 > 0 AND nr > 0 AND n1 * nr <= {MAX_BLOCK}) USING (h)""")
    con.execute(f"""
        COPY (
            SELECT a.idx AS s1, b.src, b.idx, bit_or(1::INTEGER << b.kt::INTEGER)::SMALLINT AS bits,
                   count(DISTINCT a.h)::SMALLINT AS nkeys
            FROM kk a JOIN kk b ON a.h = b.h AND a.src = 1 AND b.src <> 1
            GROUP BY a.idx, b.src, b.idx
        ) TO '{out_unscored.as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
    stats["timing"]["pairs"] = time.time() - t0

    if split == "train":
        # true pairs that share >= 1 key at all vs. only skipped keys
        gt = gt_pairs()
        con.register("gt_df", gt[gt["country"] == country][["s1", "src", "idx"]])
        lost = con.execute(f"""
            WITH m AS (
                SELECT g.s1, g.src, g.idx, max(CASE WHEN b.n1 * b.nr <= {MAX_BLOCK} THEN 1 ELSE 0 END) AS any_kept
                FROM gt_df g JOIN keys a ON a.src = 1 AND a.idx = g.s1
                JOIN keys r ON r.src = g.src AND r.idx = g.idx AND r.h = a.h
                JOIN blocks b ON b.h = a.h
                GROUP BY g.s1, g.src, g.idx)
            SELECT count(*), count(*) FILTER (WHERE any_kept = 0) FROM m""").fetchone()
        stats["true_pairs_sharing_any_key"] = int(lost[0])
        stats["true_pairs_lost_to_skipped_blocks"] = int(lost[1])
        stats["true_pairs_total"] = int((gt["country"] == country).sum())
        con.unregister("gt_df")
    con.close()

    log(f"[{split}/{country}] scoring")
    scored = CACHE / f"{split}_pairs_scored_{country}.parquet"
    score_pairs(out_unscored, scored, split, country)
    stats["timing"]["score"] = time.time() - t0
    log(f"[{split}/{country}] ranking (DuckDB windows)")
    con = connect()
    con.execute(f"""
        COPY (
            SELECT *,
                   row_number() OVER (PARTITION BY s1 ORDER BY score DESC, src, idx)::INTEGER AS rank_s,
                   row_number() OVER (PARTITION BY src, idx ORDER BY score DESC, s1)::INTEGER AS rank_r
            FROM read_parquet('{scored.as_posix()}')
        ) TO '{pairs_path(split, country)}' (FORMAT parquet, COMPRESSION zstd)""")
    con.close()
    scored.unlink()
    stats["timing"]["total"] = time.time() - t0
    stats["peak_rss_gib"] = peak_rss()
    stats_path(split, country).write_text(json.dumps(stats, indent=1), encoding="utf-8")
    out_unscored.unlink()
    shutil.rmtree(CACHE / "duckdb_tmp", ignore_errors=True)
    log(f"[{split}/{country}] done in {stats['timing']['total'] / 60:.1f} min")


# ----------------------------------------------------------------- scoring
def _field_arrays(split: str, country: str) -> tuple[dict, dict]:
    """String columns for S1 (indexed by idx) and S2/S3 (indexed via a position map)."""
    tbl = pq.read_table(CACHE / f"{split}_records.parquet",
                        columns=["src", "idx", "name_core", "name_alt", "name_phon", "addr_core"],
                        filters=[("country", "=", country)])
    src = tbl.column("src").to_numpy()
    idx = tbl.column("idx").to_numpy()
    fields = {}
    pos = {}
    for s in (1, 2, 3):
        m = src == s
        sub = tbl.filter(pa.array(m))
        n_max = int(idx[m].max()) + 1 if m.any() else 0
        p = np.full(n_max, -1, dtype=np.int64)
        p[idx[m]] = np.arange(m.sum())
        pos[s] = p
        fields[s] = {c: sub.column(c).combine_chunks() for c in ("name_core", "name_alt", "name_phon", "addr_core")}
    return fields, pos


def _gather(fields, pos, src: np.ndarray, idx: np.ndarray, col: str) -> list[str]:
    """Strings of ``col`` for records (src, idx) in the given order."""
    out = np.empty(len(idx), dtype=object)
    for s in np.unique(src):
        m = src == s
        out[m] = np.asarray(fields[int(s)][col].take(pa.array(pos[int(s)][idx[m]])).to_pylist(), dtype=object)
    return out.tolist()


def _tsr(a: list[str], b: list[str]) -> np.ndarray:
    return cpdist(a, b, scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32)


def score_pairs(unscored, out_path, split: str, country: str) -> None:
    """Add sim_name / sim_addr / score in chunks and write the scored pair file."""
    fields, pos = _field_arrays(split, country)
    pf = pq.ParquetFile(unscored)
    writer = None
    n_done = 0
    for batch in pf.iter_batches(batch_size=SCORE_CHUNK):
        s1 = batch.column("s1").to_numpy()
        src = batch.column("src").to_numpy()
        idx = batch.column("idx").to_numpy()
        ones = np.ones(len(s1), dtype=np.int8)
        a_core = _gather(fields, pos, ones, s1, "name_core")
        sim = _tsr(a_core, _gather(fields, pos, src, idx, "name_core"))
        r_alt = _gather(fields, pos, src, idx, "name_alt")
        has_alt = np.fromiter((bool(x) for x in r_alt), bool, len(r_alt))
        if has_alt.any():
            w = np.flatnonzero(has_alt)
            sim[w] = np.maximum(sim[w], _tsr([a_core[i] for i in w], [r_alt[i] for i in w]))
        del a_core, r_alt
        sim = np.maximum(sim, _tsr(_gather(fields, pos, ones, s1, "name_phon"),
                                   _gather(fields, pos, src, idx, "name_phon")))
        sim_addr = _tsr(_gather(fields, pos, ones, s1, "addr_core"), _gather(fields, pos, src, idx, "addr_core"))
        tbl = pa.table({"s1": batch.column("s1"), "src": batch.column("src"), "idx": batch.column("idx"),
                        "bits": batch.column("bits"), "nkeys": batch.column("nkeys"),
                        "sim_name": sim, "sim_addr": sim_addr, "score": sim + sim_addr})
        if writer is None:
            writer = pq.ParquetWriter(out_path, tbl.schema, compression="zstd")
        writer.write_table(tbl)
        n_done += len(s1)
        log(f"  scored {n_done:,} / {pf.metadata.num_rows:,}")
    if writer is not None:
        writer.close()


# ----------------------------------------------------------------- ranks / pruning
def pair_codes(s1: np.ndarray, src: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Unique int64 code of an (s1, src, idx) pair."""
    return (s1.astype(np.int64) << 34) | (src.astype(np.int64) << 32) | idx.astype(np.int64)


def iter_pairs(split: str, country: str, columns: list[str], batch_size: int = 10_000_000):
    """Stream the scored, ranked unpruned pairs of a split+country as pandas batches."""
    pf = pq.ParquetFile(pairs_path(split, country))
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def true_mask(gt_codes: np.ndarray, s1, src, idx) -> np.ndarray:
    """True where (s1, src, idx) is a ground-truth pair (``gt_codes`` sorted)."""
    code = pair_codes(np.asarray(s1), np.asarray(src), np.asarray(idx))
    pos = np.searchsorted(gt_codes, code)
    return gt_codes[np.minimum(pos, len(gt_codes) - 1)] == code


def oracle(tp: np.ndarray, t: np.ndarray, s1_ids: np.ndarray, countries: np.ndarray | None = None) -> dict:
    """Oracle macro F0.5 (prediction = GT ∩ candidates) scored with metrics.macro_f05.

    ``tp``/``t`` are per-S1 counts of true matches present / total; because the
    oracle prediction is a subset of the truth, integer placeholder sets with the
    same sizes give the exact entity scores.
    """
    true_map = {int(s): set(range(int(k))) for s, k in zip(s1_ids, t)}
    pred_map = {int(s): set(range(int(k))) for s, k in zip(s1_ids, tp) if k}
    cmap = dict(zip(s1_ids.tolist(), countries.tolist())) if countries is not None else None
    return macro_f05(pred_map, true_map, cmap)


def country_list(split: str) -> list[str]:
    """Countries present in the split's S1 (never hard-coded)."""
    return sorted(pq.read_table(CACHE / f"{split}_records.parquet", columns=["country"],
                                filters=[("src", "=", 1)]).column("country").unique().to_pylist())


def tune() -> dict:
    """Evaluate (k, K) on train; pick the smallest budget within 0.002 of the best oracle F0.5."""
    gt = gt_pairs()
    s1_country = pq.read_table(CACHE / "train_records.parquet", columns=["idx", "country"],
                               filters=[("src", "=", 1)]).to_pandas()
    s1_ids = s1_country["idx"].to_numpy()
    s1_ctry = s1_country["country"].to_numpy()
    n_s1_total = int(s1_ids.max()) + 1
    t = np.bincount(gt["s1"].to_numpy(), minlength=n_s1_total)[s1_ids]
    gt_codes = np.sort(pair_codes(gt["s1"].to_numpy(), gt["src"].to_numpy(), gt["idx"].to_numpy()))
    tp = {(k, K): np.zeros(n_s1_total, np.int64) for k in GRID_K for K in GRID_BIGK}
    npairs = {(k, K): 0 for k in GRID_K for K in GRID_BIGK}
    for country in country_list("train"):
        log(f"tune: {country}")
        for df in iter_pairs("train", country, ["s1", "src", "idx", "rank_r", "rank_s"]):
            is_true = true_mask(gt_codes, df["s1"], df["src"], df["idx"])
            s1 = df["s1"].to_numpy()
            rr, rs = df["rank_r"].to_numpy(), df["rank_s"].to_numpy()
            for k in GRID_K:
                for K in GRID_BIGK:
                    keep = (rr <= k) | (rs <= K)
                    npairs[(k, K)] += int(keep.sum())
                    tp[(k, K)] += np.bincount(s1[keep & is_true], minlength=n_s1_total)
    grid = []
    for (k, K), v in tp.items():
        res = oracle(v[s1_ids], t, s1_ids, s1_ctry)
        grid.append({"k": k, "K": K, "pairs": npairs[(k, K)], "oracle_f05": res["overall"],
                     "by_country": {c: r["score"] for c, r in res["by_country"].items()}})
        log(f"tune: k={k} K={K} pairs={npairs[(k, K)]:,} oracle={res['overall']:.5f}")
    best = max(g["oracle_f05"] for g in grid)
    chosen = min((g for g in grid if g["oracle_f05"] >= best - 0.002), key=lambda g: g["pairs"])
    out = {"k": chosen["k"], "K": chosen["K"], "best_oracle_f05": best, "chosen": chosen, "grid": grid}
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    BUDGET_PATH.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"tune: chosen k={chosen['k']} K={chosen['K']} ({chosen['pairs']:,} pairs, oracle {chosen['oracle_f05']:.5f})")
    return out


def prune(split: str) -> None:
    """Apply the chosen (k, K) and write cache/{split}_candidates.parquet (all countries).

    Ranks and candidate counts are recomputed on the pruned set (``rank_*_all`` keep
    the pre-pruning ranks that defined the selection).
    """
    budget = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    k, K = budget["k"], budget["K"]
    con = connect()
    parts = []
    for country in country_list(split):
        c_esc = country.replace("'", "''")
        parts.append(f"""SELECT s1, src, idx, '{c_esc}' AS country, bits, nkeys, sim_name, sim_addr, score,
                                 rank_r AS rank_r_all, rank_s AS rank_s_all
                          FROM read_parquet('{pairs_path(split, country)}')
                          WHERE rank_r <= {k} OR rank_s <= {K}""")
    out = CACHE / f"{split}_candidates.parquet"
    con.execute(f"""
        COPY (
            SELECT *,
                   row_number() OVER (PARTITION BY s1 ORDER BY score DESC, src, idx)::INTEGER AS rank_s,
                   row_number() OVER (PARTITION BY src, idx ORDER BY score DESC, s1)::INTEGER AS rank_r,
                   count(*) OVER (PARTITION BY s1)::INTEGER AS n_cand_s1,
                   count(*) OVER (PARTITION BY src, idx)::INTEGER AS n_cand_rec
            FROM ({' UNION ALL '.join(parts)})
            ORDER BY country, s1, rank_s
        ) TO '{out.as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    con.close()
    shutil.rmtree(CACHE / "duckdb_tmp", ignore_errors=True)
    log(f"prune {split}: {n:,} pairs kept (k={k}, K={K}) -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Blocking v1 (see module docstring)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--split", choices=["train", "test"], required=True)
    g.add_argument("--country", default=None, help="one country (default: all in the split)")
    ks = sub.add_parser("keystats")
    ks.add_argument("--split", choices=["train", "test"], required=True)
    ks.add_argument("--country", required=True)
    sub.add_parser("tune")
    p = sub.add_parser("prune")
    p.add_argument("--split", choices=["train", "test"], required=True)
    a = ap.parse_args()
    if a.cmd == "generate":
        for c in [a.country] if a.country else country_list(a.split):
            generate(a.split, c)
    elif a.cmd == "keystats":
        refresh_key_stats(a.split, a.country)
    elif a.cmd == "tune":
        tune()
    else:
        prune(a.split)


if __name__ == "__main__":
    main()
