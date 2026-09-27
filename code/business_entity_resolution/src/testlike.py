"""Task 6: do record-side competition features explain the train -> test gap?

Usage (from code/business_entity_resolution/):
    python -m src.testlike build       # derived split "trainD": 19% of train S1 per country dropped, rebuilt
    python -m src.testlike evaluate    # score 04_s1 / N / D in the normal and the test-like world
    python -m src.testlike submit      # 06_* test submissions for the best test-like model

Models (all stage 1 on the slim candidates, same params / halves / 2-fold OOF scheme):
* ``04_s1`` (run v3s1): the existing model.
* ``N`` (run N): trained without the record-side competition / density features (N_DROP).
* ``D`` (run D): trained on "trainD", where 19% of each country's train S1 are removed together
  with their candidate pairs (their S2/S3 records stay as orphans) and every candidate- and
  S1-dependent quantity is rebuilt: quick-score ranks, learned-pruner scores and ranks, slim
  selection, candidate counts, competition features and S1 chain sizes (the pruner model itself
  and the blocking keys are reused). This mimics test, which has ~23% more S2/S3 records per S1.

Worlds: (a) the normal OOF world (slim train candidates), (b) the test-like world (trainD).
In both, each fold model scores only the S1 half it did not train on. Decision: odds one-home
+ expected-F0.5 with gamma in GAMMAS; post-filter R drops an accepted record whose house
number differs from the S1's and is shared by >= 1 other candidate record of that S1.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .blocking import connect, country_list, pairs_path
from .data import CACHE, REPORTS, SEED, log, peak_rss, run_paths
from .decide import Scorer, load_truth, one_home, predict
from .eda import md_table
from .features import FEAT_DIR
from .features import build as build_features
from .io_utils import PROJECT_ROOT, write_id_lists
from .pruner import step_rank, step_score
from .slim import SLIM_PATH
from .slim import apply as slim_apply
from .submit import id_arrays
from .train import feature_parts, s1_halves
from .variants import validate

TD = "trainD"
DROP_FRAC = 0.19
N_DROP = ["r_margin_score", "r_margin_name", "r_margin_addr", "r_oth_score", "r_oth_name", "r_oth_addr",
          "r_n_name90", "rank_r", "rank_r_all", "n_cand_rec", "prank_r", "prank_r_all"]
MODELS = {"04_s1": "v3s1", "N": "N", "D": "D", "D2": "D2"}  # D2: master task 2b
NATIVE = {("v3s1", "a"), ("N", "a"), ("D", "b"), ("D2", "b")}  # (run, world) whose train_oof.parquet already is that world
WORLD_SPLIT = {"a": "train", "b": TD}
GAMMAS = [1.0, 1.5, 2.0, 2.5, 3.0]
OUT = CACHE / "testlike"
PARTS = CACHE / "report_parts"


# ----------------------------------------------------------------- build trainD
def dropped_s1() -> np.ndarray:
    """S1 idx removed in the test-like world: 19% per country, seed 42."""
    rng = np.random.default_rng(SEED)
    t = pq.read_table(CACHE / "train_records.parquet", columns=["idx", "country"], filters=[("src", "=", 1)]).to_pandas()
    out = []
    for c in sorted(t["country"].unique()):
        ids = np.sort(t.loc[t["country"] == c, "idx"].to_numpy())
        out.append(ids[rng.random(len(ids)) < DROP_FRAC])
    return np.sort(np.concatenate(out))


def build() -> None:
    """Records, unpruned pairs, pruner scores, slim candidates and features of trainD."""
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    drop = dropped_s1()
    np.save(OUT / "dropped_s1.npy", drop)
    log(f"trainD: dropping {len(drop):,} S1")
    # records: all S2/S3 rows, S1 rows not dropped
    pf = pq.ParquetFile(CACHE / "train_records.parquet")
    dropset = pa.array(drop.astype(np.int32))
    with pq.ParquetWriter(CACHE / f"{TD}_records.parquet", pf.schema_arrow, compression="zstd") as w:
        for i in range(pf.num_row_groups):
            g = pf.read_row_group(i)
            is_s1 = pc.equal(g.column("src"), 1)
            gone = pc.and_(is_s1, pc.is_in(g.column("idx"), value_set=dropset))
            w.write_table(g.filter(pc.invert(gone)))
    # unpruned pairs without the dropped S1; the S1's rank among a record's candidates is rebuilt
    con = connect()
    con.register("dropped", pd.DataFrame({"s1": drop.astype(np.int32)}))
    for country in country_list("train"):
        con.execute(f"""
            COPY (
                SELECT s1, src, idx, bits, nkeys, sim_name, sim_addr, score, rank_s,
                       row_number() OVER (PARTITION BY src, idx ORDER BY score DESC, s1)::INTEGER AS rank_r
                FROM read_parquet('{pairs_path("train", country)}') ANTI JOIN dropped USING (s1)
            ) TO '{pairs_path(TD, country)}' (FORMAT parquet, COMPRESSION zstd)""")
        log(f"trainD pairs {country} written")
    con.close()
    shutil.rmtree(CACHE / "duckdb_tmp", ignore_errors=True)
    step_score(TD)
    step_rank(TD)
    slim_apply(json.loads(SLIM_PATH.read_text(encoding="utf-8"))["tau"], splits=(TD,))
    build_features(TD)
    for country in country_list(TD):  # the unpruned pairs are no longer needed
        Path(pairs_path(TD, country)).unlink(missing_ok=True)
    (OUT / "build_stats.json").write_text(json.dumps({"minutes": (time.time() - t0) / 60, "dropped": int(len(drop)),
                                                       "peak_rss_gib": peak_rss()}), encoding="utf-8")
    log(f"trainD built in {(time.time() - t0) / 60:.1f} min")


# ----------------------------------------------------------------- evaluation
def world_preds(model: str, world: str) -> pd.DataFrame:
    """(s1, src, idx, label, country, p) of a model in a world; each fold model scores the other half."""
    tag = MODELS[model]
    path = OUT / f"preds_{model}_{world}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    if (tag, world) in NATIVE:
        df = pd.read_parquet(run_paths(tag)["preds"] / "train_oof.parquet",
                             columns=["s1", "src", "idx", "label", "country", "p"])
    else:
        s1_idx, half = s1_halves()
        half_of = np.full(int(s1_idx.max()) + 1, -1, np.int8)
        half_of[s1_idx] = half
        boosters = [lgb.Booster(model_file=str(run_paths(tag)["models"] / f"lgbm_fold{h}.txt")) for h in (0, 1)]
        feats = boosters[0].feature_name()
        frames = []
        for country, part in feature_parts(WORLD_SPLIT[world]):
            d = pq.read_table(part, columns=["s1", "src", "idx", "label"] + feats).to_pandas()
            hh = half_of[d["s1"].to_numpy()]
            p = np.empty(len(d), np.float32)
            for h, b in enumerate(boosters):  # model h trained on half h -> predicts half 1-h
                m = hh != h
                p[m] = b.predict(d.loc[m, feats].to_numpy(np.float32), num_iteration=b.best_iteration)
            frames.append(pd.DataFrame({"s1": d["s1"].to_numpy(), "src": d["src"].to_numpy(), "idx": d["idx"].to_numpy(),
                                        "label": d["label"].to_numpy(), "country": country, "p": p}))
        df = pd.concat(frames, ignore_index=True)
        iso_path = run_paths(tag)["models"] / "isotonic.pkl"
        if iso_path.exists():  # the run calibrated its OOF / test p: apply the same map here
            with open(iso_path, "rb") as fh:
                iso = pickle.load(fh)
            df["p"] = iso.predict(df["p"].to_numpy().astype(np.float64)).astype(np.float32)
    df.to_parquet(path, index=False)
    return df


def lookalike_flags(split: str, cand_split: str | None = None) -> pd.DataFrame:
    """R filter flag per candidate pair: record house number differs from the S1's and is shared by
    at least one other candidate record of that S1."""
    cand = pq.read_table(CACHE / f"{cand_split or split}_candidates.parquet", columns=["s1", "src", "idx"]).to_pandas()
    rec = pq.read_table(CACHE / f"{split}_records.parquet", columns=["src", "idx", "house_num"]).to_pandas()
    con = duckdb.connect()
    con.register("cand", cand)
    con.register("rec", rec)
    out = con.execute("""
        WITH c AS (
            SELECT c.s1, c.src, c.idx, r.house_num AS rh, s.house_num AS sh
            FROM cand c
            LEFT JOIN rec r ON r.src = c.src AND r.idx = c.idx
            LEFT JOIN (SELECT idx, house_num FROM rec WHERE src = 1) s ON s.idx = c.s1)
        SELECT s1, src, idx,
               (sh IS NOT NULL AND rh IS NOT NULL AND rh <> sh
                AND count(*) OVER (PARTITION BY s1, rh) >= 2) AS lookalike
        FROM c""").df()
    con.close()
    return out


def evaluate() -> dict:
    """Decision search per model and world, with and without the R post-filter."""
    t0 = time.time()
    gt, t_all, s1_ids, s1_country = load_truth()
    drop = np.load(OUT / "dropped_s1.npy")
    keep_b = ~np.isin(s1_ids, drop)
    worlds = {"a": (s1_ids, s1_country), "b": (s1_ids[keep_b], s1_country[keep_b])}
    flags = {"a": lookalike_flags("train"), "b": lookalike_flags(TD)}
    results = []
    for world, (ids, ctry) in worlds.items():
        for model in MODELS:
            d = world_preds(model, world).merge(flags[world], on=["s1", "src", "idx"], how="left")
            s1, src, idx = d["s1"].to_numpy(), d["src"].to_numpy(), d["idx"].to_numpy()
            look = d["lookalike"].fillna(False).to_numpy(bool)
            sc = Scorer(s1, d["label"].to_numpy(), t_all, ids, ctry)
            q = one_home(d["p"].to_numpy(), src, idx, "odds")
            per_g = []
            for g in GAMMAS:
                pred = predict(q, s1, "ef", {"gamma": g})
                r = sc.summary(pred)
                rr = sc.summary(pred & ~look)
                size = np.bincount(s1[pred], minlength=len(t_all))[ids].mean()
                per_g.append({"gamma": g, **r, "R": rr["overall"], "R_India": rr["India"], "R_US": rr["US"],
                              "pred_size": float(size), "R_dropped": int((pred & look).sum())})
            best = max(per_g, key=lambda x: x["overall"])
            best_r = max(per_g, key=lambda x: x["R"])
            results.append({"model": model, "world": world, "per_gamma": per_g, "best": best, "best_R": best_r,
                            "true_size": float(t_all[ids].mean())})
            log(f"{model}/{world}: best gamma {best['gamma']} -> {best['overall']:.5f}; "
                f"R at that gamma {best['R']:.5f}; best with R {best_r['R']:.5f} (gamma {best_r['gamma']})")
    (OUT / "evaluate.json").write_text(json.dumps({"results": results, "minutes": (time.time() - t0) / 60},
                                                  indent=1), encoding="utf-8")
    return {"results": results}


# ----------------------------------------------------------------- submissions
def choose() -> tuple[str, float, float, bool]:
    """Model with the best test-like score (tie within 1e-4 -> N), its gamma, next gamma, R helps?"""
    res = json.loads((OUT / "evaluate.json").read_text(encoding="utf-8"))["results"]
    b = {r["model"]: r for r in res if r["world"] == "b"}
    top = max(r["best"]["overall"] for r in b.values())
    model = "N" if b["N"]["best"]["overall"] >= top - 1e-4 else max(b, key=lambda m: b[m]["best"]["overall"])
    g = b[model]["best"]["gamma"]
    i = GAMMAS.index(g)
    g2 = GAMMAS[i + 1] if i + 1 < len(GAMMAS) else g + 0.5
    r_helps = b[model]["best"]["R"] > b[model]["best"]["overall"]
    return model, g, g2, r_helps


def submit() -> list[dict]:
    """06_<model>_g<gamma> submissions (+ R variant if it helped in the test-like world)."""
    model, g, g2, r_helps = choose()
    tag = MODELS[model]
    te = pd.read_parquet(run_paths(tag)["preds"] / "test.parquet", columns=["s1", "src", "idx", "country", "p"])
    look = lookalike_flags("test")
    te = te.merge(look, on=["s1", "src", "idx"], how="left")
    s1, src, idx = te["s1"].to_numpy(), te["src"].to_numpy(), te["idx"].to_numpy()
    q = one_home(te["p"].to_numpy(), src, idx, "odds")
    ids = id_arrays("test")
    rec = np.empty(len(te), dtype=object)
    for s in (2, 3):
        m = src == s
        rec[m] = ids[s][idx[m]]
    t1 = pq.read_table(CACHE / "test_records.parquet", columns=["idx", "country"], filters=[("src", "=", 1)]).to_pandas()
    cand_src = PROJECT_ROOT / "submissions" / "04_s1_g15" / "candidate_pairs.tsv"
    runs = [(g, False), (g2, False)] + ([(g, True)] if r_helps else [])
    out = []
    for gamma, use_r in runs:
        name = f"06_{model}_g{str(gamma).replace('.', '')}" + ("_R" if use_r else "")
        pred = predict(q, s1, "ef", {"gamma": gamma})
        if use_r:
            pred &= ~te["lookalike"].fillna(False).to_numpy(bool)
        folder = PROJECT_ROOT / "submissions" / name
        folder.mkdir(parents=True, exist_ok=True)
        matches: dict = {}
        for a_, b_ in zip(s1[pred].tolist(), rec[pred].tolist()):
            matches.setdefault(ids[1][a_], []).append(b_)
        write_id_lists(folder / "matching_results.tsv", ids[1].tolist(), matches, "matched_entity_ids")
        shutil.copyfile(cand_src, folder / "candidate_pairs.tsv")
        code, txt = validate(folder)
        sizes = pd.Series(s1[pred]).value_counts()
        per_c = {c: float(sizes.reindex(t1.loc[t1["country"] == c, "idx"].to_numpy(), fill_value=0).mean())
                 for c in sorted(t1["country"].unique())}
        out.append({"name": name, "model": model, "gamma": gamma, "R": use_r, "matches": int(pred.sum()),
                    "set_size": per_c, "validator_exit": code})
        log(f"{name}: {int(pred.sum()):,} matches, validator exit {code}")
    (OUT / "submit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ----------------------------------------------------------------- report
def report() -> None:
    """reports/task6_report.md."""
    ev = json.loads((OUT / "evaluate.json").read_text(encoding="utf-8"))
    sub = json.loads((OUT / "submit.json").read_text(encoding="utf-8")) if (OUT / "submit.json").exists() else []
    v = json.loads((PARTS / "variants.json").read_text(encoding="utf-8"))
    rows = []
    for r in ev["results"]:
        b, br = r["best"], r["best_R"]
        rows.append([r["model"], {"a": "(a) normal OOF", "b": "(b) test-like"}[r["world"]], b["gamma"],
                     f"{b['overall']:.5f}", f"{b['India']:.5f}", f"{b['US']:.5f}", f"{b['singleton']:.4f}",
                     f"{b['non_singleton']:.4f}", f"{b['pred_size']:.3f}", f"{r['true_size']:.3f}",
                     f"{b['R']:.5f} ({b['R'] - b['overall']:+.5f})", f"{br['R']:.5f} (g {br['gamma']})"])
    grow = []
    for r in ev["results"]:
        grow.append([r["model"], r["world"], *(f"{x['overall']:.5f}" for x in r["per_gamma"])])
    ref = v["04_s1_g15"]
    srows = [["04_s1_g15 (reference)", f"{ref['matches']:,}", *(f"{ref['set_size'][c]:.3f}" for c in sorted(ref["set_size"])),
              "PASS"]]
    for s in sub:
        srows.append([s["name"], f"{s['matches']:,}", *(f"{s['set_size'][c]:.3f}" for c in sorted(s["set_size"])),
                      "PASS" if s["validator_exit"] == 0 else f"FAIL ({s['validator_exit']})"])
    bs = json.loads((OUT / "build_stats.json").read_text(encoding="utf-8")) if (OUT / "build_stats.json").exists() else {}
    timings = json.loads((PARTS / "timings_t6.json").read_text(encoding="utf-8")) if (PARTS / "timings_t6.json").exists() else {}
    cs = sorted(ref["set_size"])
    lines = ["# Task 6: record-side competition features vs the train -> test gap", "",
             f"Models: **04_s1** (existing slim stage 1), **N** (without {len(N_DROP)} record-side competition features: "
             + ", ".join(f"`{f}`" for f in N_DROP) + "; `pscore` is kept although the pruner saw pre-pruning record "
             "ranks/counts), **D** (trained on the test-like world). Test-like world: 19% of each country's train S1 "
             f"dropped (seed 42; {bs.get('dropped', 0):,} S1) with their pairs; quick-score ranks, pruner scores and ranks, "
             "slim selection, candidate counts, competition features and chain sizes rebuilt (pruner model and "
             "blocking keys reused). Each fold model scores only the half it did not train on. Decision: odds + ef, "
             f"gamma in {GAMMAS}; R = drop an accepted record whose house number differs from the S1's and is shared "
             "by another candidate of that S1.", "",
             md_table(["model", "world", "best gamma", "macro F0.5", "India", "US", "singleton", "non-singleton",
                       "pred set size", "true set size", "with R (same gamma)", "best with R"], rows), "",
             "Macro F0.5 per gamma:", "",
             md_table(["model", "world", *[f"gamma {g}" for g in GAMMAS]], grow), "",
             "Test submissions (odds + ef, slim candidates; candidate_pairs.tsv identical to 04_s1_g15):", "",
             md_table(["submission", "test matches", *[f"mean set size {c}" for c in cs], "validator --check-ids"], srows), ""]
    if timings:
        lines += ["Runtime (minutes): " + ", ".join(f"{k} {x:.0f}" for k, x in timings.items()) + ".", ""]
    (REPORTS / "task6_report.md").write_text("\n".join(lines), encoding="utf-8")
    log("wrote reports/task6_report.md")


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 6: test-like world experiments")
    ap.add_argument("step", choices=["build", "evaluate", "submit", "report"])
    a = ap.parse_args()
    {"build": build, "evaluate": evaluate, "submit": submit, "report": report}[a.step]()


if __name__ == "__main__":
    main()
