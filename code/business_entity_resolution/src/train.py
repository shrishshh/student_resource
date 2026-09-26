"""LightGBM matcher with out-of-fold predictions.

Usage (from code/business_entity_resolution/, after src.features for train and test):
    python -m src.train [--tag v1] [--stage2]

``--tag`` selects the output locations (data.run_paths); ``--stage2`` joins the stage-2
group-consistency features and the stage-1 probability ``p1`` (src.stage2) to every part.

* Train S1 entities are split into two halves by a CRC32 hash of the S1 entity id.
* For each half h: a random 30% of h's S1s (seed 42) provide training pairs, 10% of
  those S1s are held out for early stopping on logloss; the model then predicts
  every pair of the OTHER half -> an out-of-fold p for every train pair.
* Test p = mean of the two fold models.
* If any of 10 calibration bins of the OOF p is off by > 0.03, an isotonic map is
  fitted on OOF and applied to OOF and test.

Outputs: cache/preds/{train_oof,test}.parquet, artifacts/models/ (git-ignored),
cache/report_parts/train.md and train_stats.json.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import zlib

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

from .data import ARTIFACTS, CACHE, SEED, log, peak_rss, run_paths
from .eda import md_table
from .features import FEAT_DIR

PRED_DIR = CACHE / "preds"
PARTS_DIR = CACHE / "report_parts"
MODEL_DIR = ARTIFACTS / "models"
ID_COLS = ["s1", "src", "idx", "label"]
PARAMS = {
    "objective": "binary", "metric": "binary_logloss", "num_leaves": 127, "learning_rate": 0.05,
    "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "seed": SEED, "deterministic": True, "force_col_wise": True, "verbose": -1,
    "num_threads": 16,
}
MAX_ROUNDS, EARLY_STOP = 3000, 100
TRAIN_FRAC, VALID_FRAC = 0.30, 0.10


def feature_parts(split: str) -> list[tuple[str, str]]:
    """(country, part path) for every feature part of a split."""
    out = []
    for cdir in sorted((FEAT_DIR / split).iterdir()):
        if cdir.is_dir():
            out += [(cdir.name, p.as_posix()) for p in sorted(cdir.glob("part-*.parquet"))]
    return out


def s1_halves() -> tuple[np.ndarray, np.ndarray]:
    """(S1 idx, half 0/1) for every train S1; half = crc32(entity_id) % 2."""
    t = pq.read_table(CACHE / "train_records.parquet", columns=["idx", "entity_id"], filters=[("src", "=", 1)])
    ids = t.column("entity_id").to_pylist()
    half = np.fromiter((zlib.crc32(x.encode()) & 1 for x in ids), np.int8, len(ids))
    return t.column("idx").to_numpy(), half


AUGMENT = None  # optional fn(df, split, country, path) -> df with extra columns (stage 2)


def read_part(split: str, country: str, path: str, columns: list[str] | None = None) -> pd.DataFrame:
    """One feature part, optionally augmented with the stage-2 columns."""
    if AUGMENT is None:
        return pq.read_table(path, columns=columns).to_pandas()
    df = AUGMENT(pq.read_table(path).to_pandas(), split, country, path)
    return df if columns is None else df[columns]


def load_rows(parts, s1_mask: np.ndarray, feats: list[str]) -> pd.DataFrame:
    """Rows (ids + label + features) whose S1 is selected by ``s1_mask`` (indexed by S1 idx)."""
    frames = []
    for country, path in parts:
        df = read_part("train", country, path, ID_COLS + feats)
        frames.append(df[s1_mask[df["s1"].to_numpy()]])
    return pd.concat(frames, ignore_index=True)


def calib_table(p: np.ndarray, y: np.ndarray) -> tuple[list[list], float]:
    """10 equal-width bins of p: pairs, mean p, positive rate, gap; plus the worst |gap|."""
    bins = np.minimum((p * 10).astype(int), 9)
    rows, worst = [], 0.0
    for b in range(10):
        m = bins == b
        if not m.any():
            continue
        mp, pr = float(p[m].mean()), float(y[m].mean())
        worst = max(worst, abs(mp - pr))
        rows.append([f"{b / 10:.1f}-{(b + 1) / 10:.1f}", f"{int(m.sum()):,}", f"{mp:.4f}", f"{pr:.4f}", f"{mp - pr:+.4f}"])
    return rows, worst


def main() -> None:
    global AUGMENT, PRED_DIR, PARTS_DIR, MODEL_DIR
    ap = argparse.ArgumentParser(description="2-fold OOF LightGBM matcher")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--stage2", action="store_true")
    args = ap.parse_args()
    rp = run_paths(args.tag)
    PRED_DIR, PARTS_DIR, MODEL_DIR = rp["preds"], rp["parts"], rp["models"]
    if args.stage2:
        from .stage2 import augment
        AUGMENT = augment
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    for d in (PRED_DIR, PARTS_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)
    s1_idx, half = s1_halves()
    n_s1 = int(s1_idx.max()) + 1
    tr_parts, te_parts = feature_parts("train"), feature_parts("test")
    feats = [c for c in read_part("train", tr_parts[0][0], tr_parts[0][1]).columns if c not in ID_COLS]
    log(f"{len(feats)} features, {len(tr_parts)} train parts, {len(te_parts)} test parts")

    fold_stats, models, oof_frames = [], [], []
    for h in (0, 1):
        members = s1_idx[half == h]
        chosen = members[rng.random(len(members)) < TRAIN_FRAC]
        valid = chosen[rng.random(len(chosen)) < VALID_FRAC]
        tr_mask = np.zeros(n_s1, bool)
        tr_mask[np.setdiff1d(chosen, valid)] = True
        va_mask = np.zeros(n_s1, bool)
        va_mask[valid] = True
        dtr = load_rows(tr_parts, tr_mask, feats)
        dva = load_rows(tr_parts, va_mask, feats)
        log(f"fold {h}: {len(dtr):,} train pairs ({dtr['label'].mean():.4f} positive), {len(dva):,} early-stop pairs")
        ds_tr = lgb.Dataset(dtr[feats].to_numpy(np.float32), label=dtr["label"].to_numpy(), feature_name=feats)
        ds_va = lgb.Dataset(dva[feats].to_numpy(np.float32), label=dva["label"].to_numpy(), reference=ds_tr)
        n_tr_pairs = len(dtr)
        del dtr, dva
        booster = lgb.train(PARAMS, ds_tr, num_boost_round=MAX_ROUNDS, valid_sets=[ds_va], valid_names=["valid"],
                            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(100)])
        del ds_tr, ds_va
        booster.save_model(str(MODEL_DIR / f"lgbm_fold{h}.txt"))
        models.append(booster)
        log(f"fold {h}: best iteration {booster.best_iteration}; predicting the other half")
        other = np.zeros(n_s1, bool)
        other[s1_idx[half != h]] = True
        for country, path in tr_parts:
            df = read_part("train", country, path)
            df = df[other[df["s1"].to_numpy()]]
            p = booster.predict(df[feats].to_numpy(np.float32), num_iteration=booster.best_iteration)
            oof_frames.append(pd.DataFrame({"s1": df["s1"].to_numpy(), "src": df["src"].to_numpy(),
                                            "idx": df["idx"].to_numpy(), "label": df["label"].to_numpy(),
                                            "country": country, "model": np.int8(h), "p": p.astype(np.float32)}))
        gain = booster.feature_importance("gain")
        fold_stats.append({"fold": h, "rounds": booster.best_iteration, "train_s1": int(tr_mask.sum()),
                           "valid_s1": int(va_mask.sum()), "train_pairs": n_tr_pairs,
                           "top_gain": [(f, float(g)) for f, g in sorted(zip(feats, gain), key=lambda t: -t[1])[:30]]})

    oof = pd.concat(oof_frames, ignore_index=True)
    del oof_frames
    y, p = oof["label"].to_numpy(), oof["p"].to_numpy().astype(np.float64)
    ctry, mdl = oof["country"].to_numpy(), oof["model"].to_numpy()
    rows_metrics = []
    for h in (0, 1):
        for c in [None, *sorted(set(ctry))]:
            mm = (mdl == h) & ((ctry == c) if c else True)
            rows_metrics.append([h, c or "ALL", f"{int(mm.sum()):,}", f"{roc_auc_score(y[mm], p[mm]):.5f}",
                                 f"{log_loss(y[mm], np.clip(p[mm], 1e-7, 1 - 1e-7)):.5f}"])
    calib_rows, worst = calib_table(p, y)
    iso = None
    calib_after = None
    if worst > 0.03:
        log(f"calibration gap {worst:.3f} > 0.03: fitting isotonic on OOF")
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        oof["p_raw"] = oof["p"]
        oof["p"] = iso.predict(p).astype(np.float32)
        with open(MODEL_DIR / "isotonic.pkl", "wb") as fh:
            pickle.dump(iso, fh)
        calib_after = calib_table(oof["p"].to_numpy().astype(np.float64), y)[0]
    oof.to_parquet(PRED_DIR / "train_oof.parquet", index=False)
    del oof, y, p

    log("predicting test (mean of the two fold models)")
    te_frames = []
    for country, path in te_parts:
        df = read_part("test", country, path)
        x = df[feats].to_numpy(np.float32)
        p = np.mean([b.predict(x, num_iteration=b.best_iteration) for b in models], axis=0)
        te = pd.DataFrame({"s1": df["s1"].to_numpy(), "src": df["src"].to_numpy(), "idx": df["idx"].to_numpy(),
                           "country": country, "p": p.astype(np.float32)})
        if iso is not None:
            te["p_raw"] = te["p"]
            te["p"] = iso.predict(p).astype(np.float32)
        te_frames.append(te)
    pd.concat(te_frames, ignore_index=True).to_parquet(PRED_DIR / "test.parquet", index=False)

    stats = {"seconds": time.time() - t0, "peak_rss_gib": peak_rss(), "folds": fold_stats,
             "calibration_worst_gap": worst, "isotonic": iso is not None, "n_features": len(feats)}
    (PARTS_DIR / "train_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    shown = {k: v for k, v in PARAMS.items() if k != "verbose"}
    title = "## 2. Model (LightGBM, 2-fold out-of-fold)" if args.tag == "v1" else \
        f"### Model (LightGBM, 2-fold out-of-fold{', stage 2' if args.stage2 else ''}; run `{args.tag}`)"
    lines = [title, "",
             f"{len(feats)} features. Params `{shown}`, up to {MAX_ROUNDS} rounds, early stopping {EARLY_STOP} "
             f"(logloss) on a 10% S1 hold-out. Each fold trains on a random {TRAIN_FRAC:.0%} of its half's S1s and "
             "predicts every pair of the other half; test = mean of both models.", "",
             md_table(["model", "trained on half", "rounds", "train S1", "train pairs", "early-stop S1"],
                      [[f["fold"], f["fold"], f["rounds"], f"{f['train_s1']:,}", f"{f['train_pairs']:,}",
                        f"{f['valid_s1']:,}"] for f in fold_stats]), "",
             "OOF metrics (model h predicts the other half; raw p before any isotonic step):", "",
             md_table(["model", "country", "pairs", "AUC", "logloss"], rows_metrics), "",
             f"Calibration of OOF p (10 bins). Worst gap {worst:.4f} -> isotonic "
             f"{'APPLIED to OOF and test' if iso else 'not needed'}.", "",
             md_table(["bin", "pairs", "mean p", "positive rate", "gap"], calib_rows), ""]
    if calib_after:
        lines += ["After isotonic:", "", md_table(["bin", "pairs", "mean p", "positive rate", "gap"], calib_after), ""]
    for f in fold_stats:
        lines += [f"Top-30 features by gain, model {f['fold']}:", "",
                  md_table(["feature", "gain"], [[n, f"{g:,.0f}"] for n, g in f["top_gain"]]), ""]
    lines += [f"Training + prediction: {stats['seconds'] / 60:.1f} min, peak RSS {stats['peak_rss_gib']:.1f} GiB.", ""]
    (PARTS_DIR / "train.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"train done in {stats['seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
