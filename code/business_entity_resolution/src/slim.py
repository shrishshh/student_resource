"""Slim candidates: threshold + caps on the learned-pruner score (blocking v3).

Usage (from code/business_entity_resolution/, after src.pruner):
    python -m src.slim curve            # evaluate TAUS, choose tau -> artifacts/slim.json
    python -m src.slim apply [--tau T]  # write cache/{split}_candidates.parquet (current ones -> cache/v2/)

A pair is kept when its pruner p >= tau AND it is among the S1's top S_CAP pairs AND the
record's top R_CAP S1s by pruner p. Because the threshold keeps a prefix of each group's
score ordering, the pre-pruning pruner ranks (prank_*_all) give those caps directly.

Proxy loss of a tau = OOF macro F0.5 of run ``s2`` (03_stage2) minus the same score after
removing the 03-accepted pairs that this pruning drops. The chosen tau is the one with the
fewest train pairs whose proxy loss <= MAX_PROXY_LOSS.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .blocking import connect, country_list, oracle, pair_codes, true_mask
from .data import ARTIFACTS, CACHE, log, peak_rss, run_paths
from .decide import Scorer, load_truth, one_home, predict
from .eda import md_table, p2
from .pruner import ranked_path

TAUS = [0.002, 0.005, 0.01, 0.02, 0.05]
S_CAP, R_CAP = 15, 3
MAX_PROXY_LOSS = 0.0007
SLIM_PATH = ARTIFACTS / "slim.json"
DEFAULT_TAU = 0.01  # used by run_all when artifacts/slim.json is absent
V2_DIR = CACHE / "v2"
COLS = ["s1", "src", "idx", "pscore", "prank_s_all", "prank_r_all"]


def keep(ps, rs, rr, tau) -> np.ndarray:
    """Slim rule on pruner score and pre-pruning pruner ranks."""
    return (ps >= tau) & (rs <= S_CAP) & (rr <= R_CAP)


def _stats(c: np.ndarray) -> dict:
    return {"mean": float(c.mean()), "p50": float(np.percentile(c, 50)), "p95": float(np.percentile(c, 95)),
            "max": int(c.max())}


def curve() -> dict:
    """Evaluate every tau on train (and candidate sizes on test); choose tau."""
    t0 = time.time()
    gt, t_all, s1_ids, s1_country = load_truth()
    gt_codes = np.sort(pair_codes(gt["s1"].to_numpy(), gt["src"].to_numpy(), gt["idx"].to_numpy()))
    n = len(t_all)
    acc = {t: {"pairs": 0, "tp": np.zeros(n, np.int64), "cps": np.zeros(n, np.int64)} for t in TAUS}
    for country in country_list("train"):
        for b in pq.ParquetFile(ranked_path("train", country)).iter_batches(batch_size=10_000_000, columns=COLS):
            d = b.to_pandas()
            s1 = d["s1"].to_numpy()
            it = true_mask(gt_codes, s1, d["src"].to_numpy(), d["idx"].to_numpy())
            ps, rs, rr = d["pscore"].to_numpy(), d["prank_s_all"].to_numpy(), d["prank_r_all"].to_numpy()
            for t in TAUS:
                k = keep(ps, rs, rr, t)
                acc[t]["pairs"] += int(k.sum())
                acc[t]["tp"] += np.bincount(s1[k & it], minlength=n)
                acc[t]["cps"] += np.bincount(s1[k], minlength=n)
        log(f"curve: train/{country} streamed")
    # test candidate sizes
    t1 = pq.read_table(CACHE / "test_records.parquet", columns=["idx", "country"], filters=[("src", "=", 1)]).to_pandas()
    nt = int(t1["idx"].max()) + 1
    tcps = {t: np.zeros(nt, np.int64) for t in TAUS}
    tpairs = {t: 0 for t in TAUS}
    for country in country_list("test"):
        for b in pq.ParquetFile(ranked_path("test", country)).iter_batches(batch_size=10_000_000, columns=COLS):
            d = b.to_pandas()
            s1 = d["s1"].to_numpy()
            for t in TAUS:
                k = keep(d["pscore"].to_numpy(), d["prank_s_all"].to_numpy(), d["prank_r_all"].to_numpy(), t)
                tcps[t] += np.bincount(s1[k], minlength=nt)
                tpairs[t] += int(k.sum())
        log(f"curve: test/{country} streamed")
    # proxy loss on 03's OOF decision
    rp = run_paths("s2")
    cfg = json.loads(rp["decision"].read_text(encoding="utf-8"))
    oof = pd.read_parquet(rp["preds"] / "train_oof.parquet", columns=["s1", "src", "idx", "label", "p"])
    s1, src, idx = oof["s1"].to_numpy(), oof["src"].to_numpy(), oof["idx"].to_numpy()
    q = one_home(oof["p"].to_numpy(), src, idx, cfg["d1"])
    pred = predict(q, s1, cfg["d2"], cfg["params"])
    sc = Scorer(s1, oof["label"].to_numpy(), t_all, s1_ids, s1_country)
    base = sc.summary(pred)["overall"]
    cand = pq.read_table(V2_DIR / "train_candidates.parquet" if (V2_DIR / "train_candidates.parquet").exists()
                         else CACHE / "train_candidates.parquet", columns=COLS).to_pandas()
    ccode = pair_codes(cand["s1"].to_numpy(), cand["src"].to_numpy(), cand["idx"].to_numpy())
    order = np.argsort(ccode)
    pos = order[np.searchsorted(ccode[order], pair_codes(s1, src, idx))]
    ps, rs, rr = (cand[c].to_numpy()[pos] for c in ("pscore", "prank_s_all", "prank_r_all"))
    del cand
    rows = []
    for t in TAUS:
        a = acc[t]
        tp, tt = a["tp"][s1_ids], t_all[s1_ids]
        res = oracle(tp, tt, s1_ids, s1_country)
        loss = base - sc.summary(pred & keep(ps, rs, rr, t))["overall"]
        rows.append({"tau": t, "train_pairs": a["pairs"], "test_pairs": tpairs[t],
                     "train_cands": _stats(a["cps"][s1_ids]), "test_cands": _stats(tcps[t][t1["idx"].to_numpy()]),
                     "recall": float(tp.sum() / len(gt)), "oracle": res["overall"],
                     "oracle_by_country": {k: v["score"] for k, v in res["by_country"].items()},
                     "proxy_loss": float(loss)})
        log(f"tau {t}: train pairs {a['pairs']:,}, oracle {res['overall']:.5f}, proxy loss {loss:.5f}")
    ok = [r for r in rows if r["proxy_loss"] <= MAX_PROXY_LOSS]
    chosen = min(ok, key=lambda r: r["train_pairs"]) if ok else max(rows, key=lambda r: r["train_pairs"])
    out = {"tau": chosen["tau"], "s_cap": S_CAP, "r_cap": R_CAP, "max_proxy_loss": MAX_PROXY_LOSS,
           "base_oof_03": base, "curve": rows, "chosen_by_rule": bool(ok), "minutes": (time.time() - t0) / 60}
    SLIM_PATH.write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"chosen tau {chosen['tau']}")
    return out


def apply(tau: float, splits=("train", "test")) -> None:
    """Write the slim candidates for the splits (previous train/test candidates -> cache/v2/)."""
    V2_DIR.mkdir(parents=True, exist_ok=True)
    for split in splits:
        cur = CACHE / f"{split}_candidates.parquet"
        if split in ("train", "test") and cur.exists() and not (V2_DIR / cur.name).exists():
            shutil.move(str(cur), str(V2_DIR / cur.name))
        con = connect()
        parts = []
        for country in country_list(split):
            c_esc = country.replace("'", "''")
            parts.append(f"""SELECT s1, src, idx, '{c_esc}' AS country, bits, nkeys, sim_name, sim_addr, score,
                                     rank_r_all, rank_s_all, pscore, prank_r_all, prank_s_all
                              FROM read_parquet('{ranked_path(split, country).as_posix()}')
                              WHERE pscore >= {tau} AND prank_s_all <= {S_CAP} AND prank_r_all <= {R_CAP}""")
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
        log(f"slim {split}: {n:,} pairs (tau={tau}, S1 cap {S_CAP}, record cap {R_CAP})")


def report() -> None:
    """cache/report_parts/slim.md from artifacts/slim.json."""
    s = json.loads(SLIM_PATH.read_text(encoding="utf-8"))
    cs = sorted(s["curve"][0]["oracle_by_country"])
    rows = [[r["tau"], f"{r['train_pairs']:,}", f"{r['test_pairs']:,}",
             *(f"{r['train_cands'][k]:.1f}" if k == "mean" else f"{r['train_cands'][k]:,.0f}" for k in ("mean", "p50", "p95", "max")),
             *(f"{r['test_cands'][k]:.1f}" if k == "mean" else f"{r['test_cands'][k]:,.0f}" for k in ("mean", "p50", "p95", "max")),
             p2(r["recall"], 1), f"{r['oracle']:.5f}", *(f"{r['oracle_by_country'][c]:.5f}" for c in cs),
             f"{r['proxy_loss']:.5f}", "**chosen**" if r["tau"] == s["tau"] else ""] for r in s["curve"]]
    lines = ["## Part A. Slim candidates", "",
             f"Keep a pair if pruner p >= tau, and it is in the S1's top {s['s_cap']} and the record's top {s['r_cap']} "
             f"by pruner p. Proxy loss = 03's OOF macro F0.5 ({s['base_oof_03']:.5f}) minus the score after dropping the "
             f"03-accepted pairs the slim set removes. Rule: fewest train pairs with proxy loss <= {s['max_proxy_loss']}"
             + ("" if s["chosen_by_rule"] else " (no tau met it; the largest set was kept)") + ".", "",
             md_table(["tau", "train pairs", "test pairs", "train cand/S1 mean", "p50", "p95", "max",
                       "test cand/S1 mean", "p50", "p95", "max", "pair recall", "oracle F0.5",
                       *[f"oracle {c}" for c in cs], "proxy loss", ""], rows), "",
             f"v2 (02/03) used 28,023,030 train / 24,300,351 test pairs (~12.7 / 14.0 per S1), oracle 0.99147. "
             f"Curve computed in {s['minutes']:.1f} min.", ""]
    (CACHE / "report_parts" / "slim.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Slim candidates from the learned-pruner score")
    ap.add_argument("step", choices=["curve", "apply", "report", "all"])
    ap.add_argument("--tau", type=float, default=None)
    args = ap.parse_args()
    if args.step in ("curve", "all"):
        curve()
    if args.step in ("apply", "all"):
        tau = args.tau
        if tau is None:
            tau = json.loads(SLIM_PATH.read_text(encoding="utf-8"))["tau"] if SLIM_PATH.exists() else DEFAULT_TAU
        apply(tau)
    if args.step in ("report", "all"):
        report()
    log(f"slim done (peak {peak_rss():.1f} GiB)")


if __name__ == "__main__":
    main()
