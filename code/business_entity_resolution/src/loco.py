"""Leave-one-country-out check (stand-in for the unseen test country).

Usage (from code/business_entity_resolution/):
    python -m src.loco [--feat-dir cache/features] [--tag v1]

For each ordered pair (A -> B) of train countries: train one LightGBM (the v1 params
and early stopping) on a random 30% sample of A's S1 (10% of those S1 held out for
early stopping), predict ALL of B's pairs, apply the fixed decision odds + ef(gamma 1.0)
and score B's macro F0.5 over all of B's S1. Writes cache/report_parts/loco.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .data import CACHE, SEED, log, peak_rss
from .decide import Scorer, load_truth, one_home, predict
from .eda import md_table
from .train import EARLY_STOP, ID_COLS, MAX_ROUNDS, PARAMS, TRAIN_FRAC, VALID_FRAC

DECISION = ("odds", "ef", {"gamma": 1.0})


def parts_of(feat_dir: Path, country: str) -> list[str]:
    """Feature part files of one train country."""
    return [p.as_posix() for p in sorted((feat_dir / "train" / country).glob("part-*.parquet"))]


def main() -> None:
    ap = argparse.ArgumentParser(description="Leave-one-country-out check")
    ap.add_argument("--feat-dir", default=str(CACHE / "features"))
    ap.add_argument("--out", default=str(CACHE / "report_parts" / "loco.md"))
    a = ap.parse_args()
    feat_dir = Path(a.feat_dir)
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    gt, t_all, s1_ids, s1_country = load_truth()
    countries = sorted(d.name for d in (feat_dir / "train").iterdir() if d.is_dir())
    feats = [c for c in pq.read_schema(parts_of(feat_dir, countries[0])[0]).names if c not in ID_COLS]
    results, gains = [], {}
    for src_c in countries:
        for dst_c in countries:
            if src_c == dst_c:
                continue
            tc = time.time()
            members = s1_ids[s1_country == src_c]
            chosen = members[rng.random(len(members)) < TRAIN_FRAC]
            valid = chosen[rng.random(len(chosen)) < VALID_FRAC]
            n = len(t_all)
            tr_m, va_m = np.zeros(n, bool), np.zeros(n, bool)
            tr_m[np.setdiff1d(chosen, valid)] = True
            va_m[valid] = True
            frames_tr, frames_va = [], []
            for path in parts_of(feat_dir, src_c):
                df = pq.read_table(path, columns=ID_COLS + feats).to_pandas()
                s = df["s1"].to_numpy()
                frames_tr.append(df[tr_m[s]])
                frames_va.append(df[va_m[s]])
            dtr, dva = pd.concat(frames_tr), pd.concat(frames_va)
            del frames_tr, frames_va
            log(f"LOCO {src_c} -> {dst_c}: {len(dtr):,} train pairs, {len(dva):,} early-stop pairs")
            ds_tr = lgb.Dataset(dtr[feats].to_numpy(np.float32), label=dtr["label"].to_numpy(), feature_name=feats)
            ds_va = lgb.Dataset(dva[feats].to_numpy(np.float32), label=dva["label"].to_numpy(), reference=ds_tr)
            del dtr, dva
            bst = lgb.train(PARAMS, ds_tr, num_boost_round=MAX_ROUNDS, valid_sets=[ds_va],
                            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(200)])
            del ds_tr, ds_va
            gains[src_c] = dict(zip(feats, bst.feature_importance("gain")))
            cols = {"s1": [], "src": [], "idx": [], "label": [], "p": []}
            for path in parts_of(feat_dir, dst_c):
                df = pq.read_table(path, columns=ID_COLS + feats).to_pandas()
                cols["p"].append(bst.predict(df[feats].to_numpy(np.float32), num_iteration=bst.best_iteration))
                for k in ("s1", "src", "idx", "label"):
                    cols[k].append(df[k].to_numpy())
            c = {k: np.concatenate(v) for k, v in cols.items()}
            q = one_home(c["p"], c["src"], c["idx"], DECISION[0])
            pred = predict(q, c["s1"], DECISION[1], DECISION[2])
            m = s1_country == dst_c
            sc = Scorer(c["s1"], c["label"], t_all, s1_ids[m], s1_country[m])
            r = sc.summary(pred)
            results.append({"train": src_c, "test": dst_c, "rounds": bst.best_iteration, **r,
                            "minutes": (time.time() - tc) / 60})
            log(f"LOCO {src_c} -> {dst_c}: macro F0.5 {r['overall']:.5f} ({results[-1]['minutes']:.1f} min)")
    # gain-rank shifts between the two single-country models
    rows_g = []
    if len(gains) == 2:
        (c1, g1), (c2, g2) = gains.items()
        r1 = {f: i + 1 for i, (f, _) in enumerate(sorted(g1.items(), key=lambda t: -t[1]))}
        r2 = {f: i + 1 for i, (f, _) in enumerate(sorted(g2.items(), key=lambda t: -t[1]))}
        shift = sorted(feats, key=lambda f: -abs(r1[f] - r2[f]))[:10]
        rows_g = [[f, r1[f], r2[f], r2[f] - r1[f], f"{g1[f]:,.0f}", f"{g2[f]:,.0f}"] for f in shift]
    ref = {"India": 0.96381, "US": 0.97532}  # v1 2-fold OOF (all countries in training)
    lines = ["## Part B. Leave-one-country-out (v1 features, fixed decision odds + ef gamma 1.0)", "",
             "One LightGBM per source country (v1 params, 30% S1 sample, 10% of it for early stopping), applied to "
             "every pair of the other country. Compared with the normal 2-fold OOF, where both countries are in training.", "",
             md_table(["train on", "score on", "rounds", "LOCO macro F0.5", "normal OOF", "drop", "singleton",
                       "non-singleton", "minutes"],
                      [[r["train"], r["test"], r["rounds"], f"{r['overall']:.5f}", f"{ref.get(r['test'], float('nan')):.5f}",
                        f"{r['overall'] - ref.get(r['test'], float('nan')):+.5f}", f"{r['singleton']:.4f}",
                        f"{r['non_singleton']:.4f}", f"{r['minutes']:.1f}"] for r in results]), ""]
    if rows_g:
        lines += [f"10 features whose gain rank changes most between the {c1}-only and the {c2}-only model:", "",
                  md_table(["feature", f"rank ({c1} model)", f"rank ({c2} model)", "change", f"gain ({c1})", f"gain ({c2})"],
                           rows_g), ""]
    lines += [f"Part B runtime {(time.time() - t0) / 60:.1f} min, peak RSS {peak_rss():.1f} GiB.", ""]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines), encoding="utf-8")
    Path(a.out).with_suffix(".json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    log("LOCO done")


if __name__ == "__main__":
    main()
