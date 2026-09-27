"""Decision-layer submission variants (no retraining).

Usage (from code/business_entity_resolution/, after the stage-1 / stage-2 runs):
    python -m src.variants                       # all VARIANTS -> submissions/<name>/ + summary
    python -m src.variants --only 04_s2_g15 --out ../../output   # one variant into a given folder

Every variant uses odds one-home renormalisation + the expected-F0.5 optimiser with a
gamma (optionally overridden per country). All variants share the same
candidate_pairs.tsv (every pair the models scored).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .data import CACHE, log, run_paths
from .decide import Scorer, load_truth, one_home, predict
from .eda import md_table
from .io_utils import PROJECT_ROOT, write_id_lists
from .submit import id_arrays

STAGE_TAGS = {"s1": "v3s1", "s2": "v3s2", "D": "D"}  # run tags: slim stage 1 / stage 2 / stage 1 trained on the test-like world
# name -> (stage, gamma, {country: gamma override})
VARIANTS = {
    "04_s2_g15": ("s2", 1.5, {}),
    "04_s2_g20": ("s2", 2.0, {}),
    "04_s2_g25": ("s2", 2.5, {}),
    "04_s1_g15": ("s1", 1.5, {}),
    "04_s1_g20": ("s1", 2.0, {}),
    "04_s2_g15_fr25": ("s2", 1.5, {"France": 2.5}),
    "07_D_g15": ("D", 1.5, {}),
}


def decide_variant(p: np.ndarray, s1: np.ndarray, src: np.ndarray, idx: np.ndarray, country: np.ndarray,
                   gamma: float, overrides: dict) -> tuple[np.ndarray, np.ndarray]:
    """(q, predicted mask): odds renormalisation + expected-F0.5 with per-country gamma."""
    q = one_home(p, src, idx, "odds")
    pred = np.zeros(len(p), bool)
    special = np.isin(country, list(overrides)) if overrides else np.zeros(len(p), bool)
    groups = [(~special, gamma)] + [((country == c), g) for c, g in overrides.items()]
    for m, g in groups:
        if m.any():  # S1s never span countries, so each group is a set of whole S1s
            pred[m] = predict(q[m], s1[m], "ef", {"gamma": g})
    return q, pred


def validate(folder: Path) -> tuple[int, str]:
    """Official validator with --check-ids (skipped with a warning when utils/ is missing)."""
    if not (PROJECT_ROOT / "utils" / "validate_submission.py").exists():
        msg = "WARNING: utils/validate_submission.py not found - validation skipped (put the challenge's utils/ next to code/)"
        print(msg, flush=True)
        return 0, msg
    r = subprocess.run([sys.executable, "utils/validate_submission.py",
                        "--matching", str(folder / "matching_results.tsv"),
                        "--candidate", str(folder / "candidate_pairs.tsv"),
                        "--test-dir", "dataset/test", "--check-ids"],
                       cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8")
    return r.returncode, (r.stdout + r.stderr).strip()


def write_variant(name: str, out_dir: Path, ids: dict, te: pd.DataFrame, cand_src: Path | None) -> dict:
    """Write one variant's two files; returns test statistics."""
    stage, gamma, over = VARIANTS[name]
    s1, src, idx = te["s1"].to_numpy(), te["src"].to_numpy(), te["idx"].to_numpy()
    country = te["country"].to_numpy()
    q, pred = decide_variant(te["p"].to_numpy(), s1, src, idx, country, gamma, over)
    rec = np.empty(len(te), dtype=object)
    for s in (2, 3):
        m = src == s
        rec[m] = ids[s][idx[m]]
    s1_order = ids[1].tolist()
    matches: dict = {}
    for a_, b_ in zip(s1[pred].tolist(), rec[pred].tolist()):
        matches.setdefault(ids[1][a_], []).append(b_)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_id_lists(out_dir / "matching_results.tsv", s1_order, matches, "matched_entity_ids")
    if cand_src is not None and cand_src.exists():
        shutil.copyfile(cand_src, out_dir / "candidate_pairs.tsv")
    else:
        cands: dict = {}
        for a_, b_ in zip(s1.tolist(), rec.tolist()):
            cands.setdefault(ids[1][a_], []).append(b_)
        write_id_lists(out_dir / "candidate_pairs.tsv", s1_order, cands, "candidate_entity_ids")
    t1 = pq.read_table(CACHE / "test_records.parquet", columns=["idx", "country"], filters=[("src", "=", 1)]).to_pandas()
    sizes = pd.Series(s1[pred]).value_counts()
    per_c = {}
    for c in sorted(t1["country"].unique()):
        cid = t1.loc[t1["country"] == c, "idx"].to_numpy()
        per_c[c] = float(sizes.reindex(cid, fill_value=0).mean())
    return {"matches": int(pred.sum()), "set_size": per_c}


def oof_score(name: str, truth) -> dict:
    """OOF macro F0.5 of a variant (France overrides do not apply to train)."""
    stage, gamma, over = VARIANTS[name]
    gt, t_all, s1_ids, s1_country = truth
    oof = pd.read_parquet(run_paths(STAGE_TAGS[stage])["preds"] / "train_oof.parquet",
                          columns=["s1", "src", "idx", "label", "country", "p"])
    s1, src, idx = oof["s1"].to_numpy(), oof["src"].to_numpy(), oof["idx"].to_numpy()
    _, pred = decide_variant(oof["p"].to_numpy(), s1, src, idx, oof["country"].to_numpy(), gamma, over)
    return Scorer(s1, oof["label"].to_numpy(), t_all, s1_ids, s1_country).summary(pred)


def main() -> None:
    ap = argparse.ArgumentParser(description="Decision-layer submission variants")
    ap.add_argument("--only", default=None, help="one variant name")
    ap.add_argument("--out", default=None, help="output folder for --only (default submissions/<name>)")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()
    names = [args.only] if args.only else list(VARIANTS)
    ids = id_arrays("test")
    truth = load_truth()
    rows, first_cand = [], None
    stats = {}
    for name in names:
        stage = VARIANTS[name][0]
        te = pd.read_parquet(run_paths(STAGE_TAGS[stage])["preds"] / "test.parquet",
                             columns=["s1", "src", "idx", "country", "p"])
        out_dir = Path(args.out) if (args.only and args.out) else PROJECT_ROOT / "submissions" / name
        st = write_variant(name, out_dir, ids, te, first_cand)
        first_cand = first_cand or out_dir / "candidate_pairs.tsv"
        code, txt = (0, "skipped") if args.no_validate else validate(out_dir)
        oof = oof_score(name, truth)
        stats[name] = {**st, "oof": oof, "validator_exit": code, "validator": txt}
        log(f"{name}: OOF {oof['overall']:.5f}, {st['matches']:,} test matches, validator exit {code}")
        stage_, gamma, over = VARIANTS[name]
        rows.append([name, stage_, gamma, json.dumps(over) if over else "-", f"{oof['overall']:.5f}",
                     f"{oof['India']:.5f}", f"{oof['US']:.5f}", f"{st['matches']:,}",
                     *(f"{st['set_size'][c]:.3f}" for c in sorted(st["set_size"])),
                     "PASS" if code == 0 else f"FAIL ({code})"])
    if not args.only:
        cs = sorted(next(iter(stats.values()))["set_size"])
        md = ["## Part C. Submission variants (decision layer only)", "",
              "All: odds one-home + expected-F0.5 on the slim candidates; the same candidate_pairs.tsv. OOF = macro F0.5 "
              "over all train S1 (France overrides have no train effect).", "",
              md_table(["variant", "stage", "gamma", "override", "OOF", "OOF India", "OOF US", "test matches",
                        *[f"mean set size {c}" for c in cs], "validator --check-ids"], rows), ""]
        (CACHE / "report_parts" / "variants.md").write_text("\n".join(md), encoding="utf-8")
        (CACHE / "report_parts" / "variants.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
