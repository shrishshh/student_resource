"""Blocking v2: learned pruning of the unpruned candidate pairs.

Usage (from code/business_entity_resolution/, after blocking generate for train and test):
    python -m src.pruner all          # train -> score -> rank -> grid -> prune -> report (resumable)

Steps
-----
train  LightGBM pruner (~300 trees, 63 leaves) on ALL unpruned pairs of a random 3% of
       train S1 (label = true pair), using only features cheap enough for every pair.
score  pruner score for every unpruned pair of train and test (chunks of CHUNK).
rank   DuckDB window ranks by pruner score: prank_r (the S1's rank among the record's
       candidates) and prank_s (the record's rank among the S1's candidates).
grid   keep a pair if prank_r <= k OR prank_s <= K; k in {2,3,5} x K in {10,15,20};
       choose the smallest budget within 0.001 of the best train oracle macro F0.5 with
       at most MAX_TRAIN_PAIRS train pairs.
prune  write cache/{split}_candidates.parquet (v1 files moved to cache/v1/ first).
report cache/report_parts/pruner.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

from .blocking import KEY_TYPES, connect, country_list, oracle, pair_codes, pairs_path, true_mask
from .data import ARTIFACTS, CACHE, SEED, gt_pairs, log, peak_rss
from .eda import md_table, p2, pct
from .features import Records

CHUNK = 3_000_000
SAMPLE_FRAC = 0.03
GRID_K, GRID_BIGK = [2, 3, 5], [10, 15, 20]
MAX_TRAIN_PAIRS = 50_000_000
TOL = 0.001
V1_DIR = CACHE / "v1"
MODEL_PATH = ARTIFACTS / "pruner" / "pruner_lgbm.txt"
BUDGET_PATH = ARTIFACTS / "pruner" / "pruner_budget.json"
PARTS = CACHE / "report_parts"
PARAMS = {"objective": "binary", "num_leaves": 63, "learning_rate": 0.1, "min_data_in_leaf": 200,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": SEED,
          "deterministic": True, "force_col_wise": True, "verbose": -1, "num_threads": 16}
N_TREES = 300
FEATS = [f"key_{k}" for k in KEY_TYPES] + [
    "nkeys", "sim_name", "sim_addr", "name_tsort", "name_ratio", "addr_tsort", "house_eq", "house_missing",
    "s_addr_empty", "r_addr_empty", "r_name_is_web", "r_name_has_indic", "r_name_has_marker",
    "chain_s", "chain_r", "r_tok_in_s1_vocab", "rank_r", "rank_s", "n_cand_s", "n_cand_r"]


def scored_path(split: str, country: str):
    return CACHE / f"{split}_pairs_pscored_{country}.parquet"


def ranked_path(split: str, country: str):
    return CACHE / f"{split}_pairs_pranked_{country}.parquet"


class Ctx:
    """Per split+country record data and pre-pruning candidate counts."""

    def __init__(self, split: str, country: str):
        self.R = Records(split, country)
        R = self.R
        self.house = np.array([int(h[:15]) if h else -1 for h in R.house], np.int64)
        # candidate counts before pruning (one cheap pass)
        self.n_s = np.zeros(int(R._pos[1].shape[0]), np.int32)
        self.n_r = np.zeros(R.n, np.int32)
        pf = pq.ParquetFile(pairs_path(split, country))
        for b in pf.iter_batches(batch_size=10_000_000, columns=["s1", "src", "idx"]):
            s1 = b.column("s1").to_numpy()
            self.n_s += np.bincount(s1, minlength=len(self.n_s)).astype(np.int32)
            self.n_r += np.bincount(R.pos(b.column("src").to_numpy(), b.column("idx").to_numpy()),
                                    minlength=R.n).astype(np.int32)

    def features(self, df: pd.DataFrame) -> np.ndarray:
        """Cheap pruner features (float32 matrix, columns = FEATS)."""
        R = self.R
        s1, src, idx = df["s1"].to_numpy(), df["src"].to_numpy(), df["idx"].to_numpy()
        ps = R.pos(np.ones(len(s1), np.int8), s1)
        pr = R.pos(src, idx)
        bits = df["bits"].to_numpy().astype(np.int64)
        na, nb = R.get("name", ps), R.get("name", pr)
        cols = [((bits >> i) & 1) for i in range(len(KEY_TYPES))]
        cols += [df["nkeys"].to_numpy(), df["sim_name"].to_numpy(), df["sim_addr"].to_numpy(),
                 cpdist(na, nb, scorer=fuzz.token_sort_ratio, workers=-1),
                 cpdist(na, nb, scorer=fuzz.ratio, workers=-1)]
        del na, nb
        cols.append(cpdist(R.get("addr", ps), R.get("addr", pr), scorer=fuzz.token_sort_ratio, workers=-1))
        ha, hb = self.house[ps], self.house[pr]
        miss = (ha < 0) | (hb < 0)
        cols += [(ha == hb) & ~miss, miss, R.flags["addr_empty"][ps], R.flags["addr_empty"][pr],
                 R.flags["name_is_web"][pr], R.flags["name_has_indic"][pr], R.flags["name_has_marker"][pr],
                 R.chain[ps], R.chain[pr], R.in_s1_vocab[pr], df["rank_r"].to_numpy(), df["rank_s"].to_numpy(),
                 self.n_s[s1], self.n_r[pr]]
        return np.column_stack([np.asarray(c, dtype=np.float32) for c in cols])


# ----------------------------------------------------------------- steps
def step_train() -> None:
    """Train the pruner on all unpruned pairs of a random 3% of train S1."""
    if MODEL_PATH.exists():
        log("pruner model exists, skipping training")
        return
    rng = np.random.default_rng(SEED)
    gt = gt_pairs()
    gt_codes = np.sort(pair_codes(gt["s1"].to_numpy(), gt["src"].to_numpy(), gt["idx"].to_numpy()))
    n_s1 = pq.read_table(CACHE / "train_records.parquet", columns=["idx"], filters=[("src", "=", 1)]).num_rows
    sample = rng.random(n_s1) < SAMPLE_FRAC
    xs, ys = [], []
    for country in country_list("train"):
        ctx = Ctx("train", country)
        pf = pq.ParquetFile(pairs_path("train", country))
        for b in pf.iter_batches(batch_size=CHUNK):
            df = b.to_pandas()
            df = df[sample[df["s1"].to_numpy()]]
            if len(df):
                xs.append(ctx.features(df))
                ys.append(true_mask(gt_codes, df["s1"], df["src"], df["idx"]).astype(np.int8))
        log(f"pruner sample {country}: {sum(len(y) for y in ys):,} pairs so far")
        del ctx
    x, y = np.concatenate(xs), np.concatenate(ys)
    del xs, ys
    log(f"training pruner on {len(y):,} pairs ({y.mean():.4f} positive), {int(sample.sum()):,} S1")
    bst = lgb.train(PARAMS, lgb.Dataset(x, label=y, feature_name=FEATS), num_boost_round=N_TREES)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(MODEL_PATH))
    gain = sorted(zip(FEATS, bst.feature_importance("gain")), key=lambda t: -t[1])
    (MODEL_PATH.parent / "pruner_train.json").write_text(json.dumps(
        {"pairs": int(len(y)), "positive_rate": float(y.mean()), "s1": int(sample.sum()),
         "gain": [(f, float(g)) for f, g in gain]}, indent=1), encoding="utf-8")


def step_score(split: str) -> None:
    """Pruner score for every unpruned pair of a split."""
    bst = lgb.Booster(model_file=str(MODEL_PATH))
    for country in country_list(split):
        out = scored_path(split, country)
        if out.exists() or ranked_path(split, country).exists():
            log(f"score {split}/{country}: exists, skipping")
            continue
        t0 = time.time()
        ctx = Ctx(split, country)
        pf = pq.ParquetFile(pairs_path(split, country))
        tmp = out.with_suffix(".tmp")
        writer = None
        done = 0
        for b in pf.iter_batches(batch_size=CHUNK):
            df = b.to_pandas()
            ps = bst.predict(ctx.features(df)).astype(np.float32)
            tbl = pa.table({"s1": df["s1"].to_numpy(), "src": df["src"].to_numpy(), "idx": df["idx"].to_numpy(),
                            "bits": df["bits"].to_numpy(), "nkeys": df["nkeys"].to_numpy(),
                            "sim_name": df["sim_name"].to_numpy(), "sim_addr": df["sim_addr"].to_numpy(),
                            "score": df["score"].to_numpy(), "rank_r_all": df["rank_r"].to_numpy(),
                            "rank_s_all": df["rank_s"].to_numpy(), "pscore": ps})
            if writer is None:
                writer = pq.ParquetWriter(tmp, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            done += len(df)
            if done % (10 * CHUNK) < CHUNK:
                log(f"score {split}/{country}: {done:,} / {pf.metadata.num_rows:,}")
        writer.close()
        del pf, ctx
        tmp.rename(out)
        log(f"score {split}/{country}: done in {(time.time() - t0) / 60:.1f} min")


def step_rank(split: str) -> None:
    """DuckDB window ranks by pruner score."""
    for country in country_list(split):
        out = ranked_path(split, country)
        if out.exists():
            continue
        con = connect()
        tmp = out.with_suffix(".tmp")
        con.execute(f"""
            COPY (
                SELECT *,
                       row_number() OVER (PARTITION BY s1 ORDER BY pscore DESC, src, idx)::INTEGER AS prank_s_all,
                       row_number() OVER (PARTITION BY src, idx ORDER BY pscore DESC, s1)::INTEGER AS prank_r_all
                FROM read_parquet('{scored_path(split, country).as_posix()}')
            ) TO '{tmp.as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
        con.close()
        tmp.rename(out)
        scored_path(split, country).unlink()
        shutil.rmtree(CACHE / "duckdb_tmp", ignore_errors=True)
        log(f"rank {split}/{country}: done")


def step_grid() -> dict:
    """Evaluate (k, K) on train and choose the budget."""
    gt = gt_pairs()
    gt_codes = np.sort(pair_codes(gt["s1"].to_numpy(), gt["src"].to_numpy(), gt["idx"].to_numpy()))
    rec1 = pq.read_table(CACHE / "train_records.parquet", columns=["idx", "country"], filters=[("src", "=", 1)]).to_pandas()
    n = int(rec1["idx"].max()) + 1
    t_all = np.bincount(gt["s1"].to_numpy(), minlength=n)
    cfgs = [(k, K) for k in GRID_K for K in GRID_BIGK]
    acc = {c: {"pairs": {}, "tp": np.zeros(n, np.int64), "cps": np.zeros(n, np.int64), "t2": 0, "t3": 0} for c in cfgs}
    n_all = {}
    for country in country_list("train"):
        n_all[country] = 0
        for b in pq.ParquetFile(ranked_path("train", country)).iter_batches(
                batch_size=10_000_000, columns=["s1", "src", "idx", "prank_r_all", "prank_s_all"]):
            df = b.to_pandas()
            s1, src = df["s1"].to_numpy(), df["src"].to_numpy()
            it = true_mask(gt_codes, s1, src, df["idx"].to_numpy())
            rr, rs = df["prank_r_all"].to_numpy(), df["prank_s_all"].to_numpy()
            n_all[country] += len(df)
            for c in cfgs:
                keep = (rr <= c[0]) | (rs <= c[1])
                a = acc[c]
                a["pairs"][country] = a["pairs"].get(country, 0) + int(keep.sum())
                kt = keep & it
                a["tp"] += np.bincount(s1[kt], minlength=n)
                a["cps"] += np.bincount(s1[keep], minlength=n)
                a["t2"] += int(kt[src == 2].sum())
                a["t3"] += int(kt[src == 3].sum())
        log(f"grid: {country} streamed")
    grid = []
    ids, ctry = rec1["idx"].to_numpy(), rec1["country"].to_numpy()
    for c in cfgs:
        a = acc[c]
        tp, t = a["tp"][ids], t_all[ids]
        res = oracle(tp, t, ids, ctry)
        cps = a["cps"][ids]
        row = {"k": c[0], "K": c[1], "pairs": sum(a["pairs"].values()), "pairs_by_country": a["pairs"],
               "recall": float(tp.sum() / len(gt)), "recall_s2": a["t2"] / int((gt["src"] == 2).sum()),
               "recall_s3": a["t3"] / int((gt["src"] == 3).sum()),
               "s1_all": float(((tp == t) & (t > 0)).sum() / (t > 0).sum()), "oracle": res["overall"],
               "oracle_by_country": {k: v["score"] for k, v in res["by_country"].items()},
               "cand_mean": float(cps.mean()), "cand_p95": float(np.percentile(cps, 95))}
        grid.append(row)
        log(f"grid k={c[0]} K={c[1]}: pairs {row['pairs']:,} recall {row['recall']:.4f} oracle {row['oracle']:.5f}")
    ok = [g for g in grid if g["pairs"] <= MAX_TRAIN_PAIRS]
    best = max(g["oracle"] for g in ok)
    chosen = min((g for g in ok if g["oracle"] >= best - TOL), key=lambda g: g["pairs"])
    out = {"k": chosen["k"], "K": chosen["K"], "best_oracle": best, "chosen": chosen, "grid": grid,
           "unpruned_pairs": n_all}
    BUDGET_PATH.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"grid: chosen k={chosen['k']} K={chosen['K']} pairs {chosen['pairs']:,} oracle {chosen['oracle']:.5f}")
    return out


def step_prune(split: str) -> None:
    """Write the v2 candidates (v1 candidates moved to cache/v1/ first)."""
    budget = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    k, K = budget["k"], budget["K"]
    V1_DIR.mkdir(parents=True, exist_ok=True)
    cur = CACHE / f"{split}_candidates.parquet"
    if cur.exists() and not (V1_DIR / cur.name).exists():
        shutil.move(str(cur), str(V1_DIR / cur.name))
    con = connect()
    parts = []
    for country in country_list(split):
        c_esc = country.replace("'", "''")
        parts.append(f"""SELECT s1, src, idx, '{c_esc}' AS country, bits, nkeys, sim_name, sim_addr, score,
                                 rank_r_all, rank_s_all, pscore, prank_r_all, prank_s_all
                          FROM read_parquet('{ranked_path(split, country).as_posix()}')
                          WHERE prank_r_all <= {k} OR prank_s_all <= {K}""")
    con.execute(f"""
        COPY (
            SELECT *,
                   row_number() OVER (PARTITION BY s1 ORDER BY score DESC, src, idx)::INTEGER AS rank_s,
                   row_number() OVER (PARTITION BY src, idx ORDER BY score DESC, s1)::INTEGER AS rank_r,
                   row_number() OVER (PARTITION BY s1 ORDER BY pscore DESC, src, idx)::INTEGER AS prank_s,
                   row_number() OVER (PARTITION BY src, idx ORDER BY pscore DESC, s1)::INTEGER AS prank_r,
                   count(*) OVER (PARTITION BY s1)::INTEGER AS n_cand_s1,
                   count(*) OVER (PARTITION BY src, idx)::INTEGER AS n_cand_rec
            FROM ({' UNION ALL '.join(parts)})
            ORDER BY country, s1, prank_s
        ) TO '{cur.as_posix()}' (FORMAT parquet, COMPRESSION zstd)""")
    n = con.execute(f"SELECT count(*) FROM read_parquet('{cur.as_posix()}')").fetchone()[0]
    con.close()
    shutil.rmtree(CACHE / "duckdb_tmp", ignore_errors=True)
    log(f"prune {split}: {n:,} pairs (k={k}, K={K})")


def step_report(t_start: float, timings: dict) -> None:
    """cache/report_parts/pruner.md."""
    budget = json.loads(BUDGET_PATH.read_text(encoding="utf-8"))
    tr = json.loads((MODEL_PATH.parent / "pruner_train.json").read_text(encoding="utf-8"))
    v1 = json.loads((ARTIFACTS / "blocking_budget.json").read_text(encoding="utf-8"))
    gt = gt_pairs()
    # v1 misses with an empty record address, recovered by v2?
    gcodes = pair_codes(gt["s1"].to_numpy(), gt["src"].to_numpy(), gt["idx"].to_numpy())

    def codes_of(path):
        t = pq.read_table(path, columns=["s1", "src", "idx"])
        return np.sort(pair_codes(t.column("s1").to_numpy(), t.column("src").to_numpy(), t.column("idx").to_numpy()))

    def found(sorted_codes):
        p = np.searchsorted(sorted_codes, gcodes)
        return sorted_codes[np.minimum(p, len(sorted_codes) - 1)] == gcodes

    f1 = found(codes_of(V1_DIR / "train_candidates.parquet"))
    f2 = found(codes_of(CACHE / "train_candidates.parquet"))
    rec = pq.read_table(CACHE / "train_records.parquet", columns=["src", "addr_empty"])
    rsrc = rec.column("src").to_numpy()
    offs = {s: int(np.flatnonzero(rsrc == s)[0]) for s in (1, 2, 3)}
    empty = rec.column("addr_empty").to_numpy(zero_copy_only=False)
    gpos = np.where(gt["src"].to_numpy() == 2, offs[2], offs[3]) + gt["idx"].to_numpy()
    e = empty[gpos]
    rows = [[g["k"], g["K"], f"{g['pairs']:,}", *(f"{g['pairs_by_country'].get(c, 0):,}" for c in sorted(g["pairs_by_country"])),
             f"{g['cand_mean']:.1f}", f"{g['cand_p95']:.0f}", p2(g["recall"], 1), p2(g["recall_s2"], 1), p2(g["recall_s3"], 1),
             p2(g["s1_all"], 1), f"{g['oracle']:.5f}", *(f"{v:.5f}" for _, v in sorted(g["oracle_by_country"].items())),
             "**chosen**" if (g["k"], g["K"]) == (budget["k"], budget["K"]) else ""] for g in budget["grid"]]
    countries = sorted(budget["grid"][0]["pairs_by_country"])
    test_n = {c: pq.read_table(CACHE / "test_candidates.parquet", columns=["country"],
                               filters=[("country", "=", c)]).num_rows for c in country_list("test")}
    test_v1 = {c: pq.read_table(V1_DIR / "test_candidates.parquet", columns=["country"],
                                filters=[("country", "=", c)]).num_rows for c in country_list("test")}
    lines = ["## Part C. Blocking v2: learned pruning", "",
             f"Pruner: LightGBM ({N_TREES} trees, {PARAMS['num_leaves']} leaves, lr {PARAMS['learning_rate']}) trained on all "
             f"{tr['pairs']:,} unpruned pairs of a random {SAMPLE_FRAC:.0%} of train S1 ({tr['s1']:,} S1, "
             f"{tr['positive_rate']:.4f} positive). Features ({len(FEATS)}): " + ", ".join(f"`{f}`" for f in FEATS) + ".", "",
             "Top pruner features by gain: " + ", ".join(f"{f} ({g:,.0f})" for f, g in tr["gain"][:10]) + ".", "",
             f"Grid (keep if the record's top-k OR the S1's top-K by pruner score; train pairs <= {MAX_TRAIN_PAIRS:,}; "
             f"smallest budget within {TOL} of the best oracle):", "",
             md_table(["k", "K", "pairs", *[f"pairs {c}" for c in countries], "cand/S1 mean", "p95", "pair recall",
                       "recall S2", "recall S3", "S1 with ALL matches", "oracle F0.5", *[f"oracle {c}" for c in countries], ""],
                      rows), "",
             f"**v1 (quick-score pruning, k=3, K=10): {v1['chosen']['pairs']:,} pairs, oracle {v1['chosen']['oracle_f05']:.5f}. "
             f"v2 chosen k={budget['k']}, K={budget['K']}: {budget['chosen']['pairs']:,} pairs, oracle "
             f"{budget['chosen']['oracle']:.5f}.**", "",
             md_table(["true pairs", "v1 candidates", "v2 candidates"], [
                 ["all", pct(f1.sum(), len(gt)), pct(f2.sum(), len(gt))],
                 ["record address empty", pct((f1 & e).sum(), e.sum()), pct((f2 & e).sum(), e.sum())],
             ]), "",
             f"Of the {int((~f1 & e).sum()):,} true pairs with an empty record address missed by v1, v2 recovers "
             f"**{int((~f1 & e & f2).sum()):,}**; v2 loses {int((f1 & ~f2).sum()):,} true pairs that v1 had "
             f"(and gains {int((~f1 & f2).sum()):,} in total).", "",
             "Test candidates (v1 -> v2): " + ", ".join(f"{c} {test_v1[c]:,} -> {test_n[c]:,}" for c in test_n) + ".", "",
             "Timing (min): " + ", ".join(f"{k} {v:.1f}" for k, v in timings.items())
             + f"; Part C total {(time.time() - t_start) / 60:.1f} min, peak RSS {peak_rss():.1f} GiB.", ""]
    PARTS.mkdir(parents=True, exist_ok=True)
    (PARTS / "pruner.md").write_text("\n".join(lines), encoding="utf-8")
    log("pruner report written")


def main() -> None:
    ap = argparse.ArgumentParser(description="Blocking v2 learned pruning")
    ap.add_argument("step", choices=["all", "train", "score", "rank", "grid", "prune", "report"])
    a = ap.parse_args()
    t0 = time.time()
    timings: dict = {}

    def run(name, fn, *args):
        t = time.time()
        fn(*args)
        timings[name] = timings.get(name, 0) + (time.time() - t) / 60

    steps = ["train", "score", "rank", "grid", "prune", "report"] if a.step == "all" else [a.step]
    for s in steps:
        if s == "train":
            run("train", step_train)
        elif s in ("score", "rank", "prune"):
            for split in ("train", "test"):
                run(s, {"score": step_score, "rank": step_rank, "prune": step_prune}[s], split)
        elif s == "grid":
            run("grid", step_grid)
        elif s == "report":
            step_report(t0, timings)


if __name__ == "__main__":
    main()
