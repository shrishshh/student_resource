"""Pseudo-labelling check with India as a stand-in for an unseen country (no submission change).

Usage (from code/business_entity_resolution/, on the current stage-1 features):
    python -m src.pseudo [--source US --target India]

1. Train on a 30% S1 sample of the source country (10% of it for early stopping) and score
   the target country with the fixed decision (odds + ef, gamma 1.0) -> baseline.
2. Pseudo-label the target pairs with that model (p >= 0.97 -> 1, p <= 0.03 -> 0, others
   dropped), add the pseudo-labelled pairs of a random 30% of the target's S1 to the
   source training sample, retrain, and score the target again with the TRUE labels.
Writes cache/report_parts/pseudo.md.
"""

from __future__ import annotations

import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .data import CACHE, SEED, log, peak_rss
from .decide import Scorer, load_truth, one_home, predict
from .eda import md_table
from .features import FEAT_DIR
from .loco import DECISION, parts_of
from .train import EARLY_STOP, ID_COLS, MAX_ROUNDS, PARAMS, TRAIN_FRAC, VALID_FRAC

HI, LO = 0.97, 0.03


def _fit(xtr, ytr, xva, yva, feats):
    ds = lgb.Dataset(xtr, label=ytr, feature_name=feats)
    dv = lgb.Dataset(xva, label=yva, reference=ds)
    return lgb.train(PARAMS, ds, num_boost_round=MAX_ROUNDS, valid_sets=[dv],
                     callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(200)])


def main() -> None:
    ap = argparse.ArgumentParser(description="Pseudo-label check")
    ap.add_argument("--source", default="US")
    ap.add_argument("--target", default="India")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    gt, t_all, s1_ids, s1_country = load_truth()
    feats = [c for c in pq.read_schema(parts_of(FEAT_DIR, args.source)[0]).names if c not in ID_COLS]
    n = len(t_all)
    members = s1_ids[s1_country == args.source]
    chosen = members[rng.random(len(members)) < TRAIN_FRAC]
    valid = chosen[rng.random(len(chosen)) < VALID_FRAC]
    tr_m, va_m = np.zeros(n, bool), np.zeros(n, bool)
    tr_m[np.setdiff1d(chosen, valid)] = True
    va_m[valid] = True
    xs, ys, xv, yv = [], [], [], []
    for path in parts_of(FEAT_DIR, args.source):
        d = pq.read_table(path, columns=ID_COLS + feats).to_pandas()
        s = d["s1"].to_numpy()
        xs.append(d.loc[tr_m[s], feats].to_numpy(np.float32)); ys.append(d.loc[tr_m[s], "label"].to_numpy())
        xv.append(d.loc[va_m[s], feats].to_numpy(np.float32)); yv.append(d.loc[va_m[s], "label"].to_numpy())
    xtr, ytr, xva, yva = np.concatenate(xs), np.concatenate(ys), np.concatenate(xv), np.concatenate(yv)
    del xs, ys, xv, yv
    tgt_mask = s1_country == args.target
    tgt_ids = s1_ids[tgt_mask]

    def score(bst):
        cols = {k: [] for k in ("s1", "src", "idx", "label", "p")}
        for path in parts_of(FEAT_DIR, args.target):
            d = pq.read_table(path, columns=ID_COLS + feats).to_pandas()
            cols["p"].append(bst.predict(d[feats].to_numpy(np.float32), num_iteration=bst.best_iteration))
            for k in ("s1", "src", "idx", "label"):
                cols[k].append(d[k].to_numpy())
        c = {k: np.concatenate(v) for k, v in cols.items()}
        pred = predict(one_home(c["p"], c["src"], c["idx"], DECISION[0]), c["s1"], DECISION[1], DECISION[2])
        return Scorer(c["s1"], c["label"], t_all, tgt_ids, s1_country[tgt_mask]).summary(pred), c

    log(f"baseline: {len(ytr):,} {args.source} pairs")
    b0 = _fit(xtr, ytr, xva, yva, feats)
    r0, c0 = score(b0)
    log(f"baseline {args.source} -> {args.target}: {r0['overall']:.5f}")
    # pseudo labels on a random 30% of the target's S1
    samp = np.zeros(n, bool)
    samp[tgt_ids[rng.random(len(tgt_ids)) < TRAIN_FRAC]] = True
    px, py = [], []
    off = 0
    for path in parts_of(FEAT_DIR, args.target):
        d = pq.read_table(path, columns=["s1"] + feats).to_pandas()
        p = c0["p"][off:off + len(d)]
        off += len(d)
        m = samp[d["s1"].to_numpy()] & ((p >= HI) | (p <= LO))
        px.append(d.loc[m, feats].to_numpy(np.float32))
        py.append((p[m] >= HI).astype(np.int8))
    px, py = np.concatenate(px), np.concatenate(py)
    # agreement of the pseudo labels with the truth, for the report
    lab_p = np.concatenate([np.asarray(c0["label"])])
    sel = samp[c0["s1"]] & ((c0["p"] >= HI) | (c0["p"] <= LO))
    agree = float(((c0["p"][sel] >= HI).astype(int) == lab_p[sel]).mean())
    log(f"pseudo-labelled {len(py):,} {args.target} pairs ({py.mean():.4f} positive, {agree:.4f} agree with truth)")
    b1 = _fit(np.concatenate([xtr, px]), np.concatenate([ytr, py]), xva, yva, feats)
    r1, _ = score(b1)
    log(f"with pseudo labels {args.source} -> {args.target}: {r1['overall']:.5f}")
    lines = ["## Part E. Pseudo-labelling check (target country as a stand-in for France)", "",
             f"Stage-1 slim features. Source {args.source}: 30% S1 sample ({len(ytr):,} pairs). Pseudo labels on the "
             f"pairs of a random 30% of {args.target}'s S1: p >= {HI} -> 1, p <= {LO} -> 0, others dropped "
             f"({len(py):,} pairs, {py.mean():.4f} positive, {agree:.2%} agree with the true labels). Scored on ALL "
             f"{args.target} pairs with the true labels, decision odds + ef gamma 1.0.", "",
             md_table(["model", "rounds", f"{args.target} macro F0.5", "singleton", "non-singleton"],
                      [[f"{args.source} only", b0.best_iteration, f"{r0['overall']:.5f}", f"{r0['singleton']:.4f}",
                        f"{r0['non_singleton']:.4f}"],
                       [f"{args.source} + pseudo-labelled {args.target}", b1.best_iteration, f"{r1['overall']:.5f}",
                        f"{r1['singleton']:.4f}", f"{r1['non_singleton']:.4f}"]]), "",
             f"Change: {r1['overall'] - r0['overall']:+.5f}. Runtime {(time.time() - t0) / 60:.1f} min, "
             f"peak RSS {peak_rss():.1f} GiB. No submission was changed.", ""]
    (CACHE / "report_parts" / "pseudo.md").write_text("\n".join(lines), encoding="utf-8")
    (CACHE / "report_parts" / "pseudo.json").write_text(json.dumps(
        {"baseline": r0, "pseudo": r1, "pseudo_pairs": int(len(py)), "agree": agree}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
