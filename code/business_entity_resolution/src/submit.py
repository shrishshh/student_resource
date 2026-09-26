"""Test submission + matcher report.

Usage (from code/business_entity_resolution/, after src.decide):
    python -m src.submit --name 01_lgbm_v1

Applies artifacts/decision.json to the test predictions, writes
submissions/<name>/{matching_results,candidate_pairs}.tsv (every test S1 in file
order; candidates = exactly the pairs the model scored), runs the official validator
with --check-ids, computes sanity numbers vs train OOF, and assembles
reports/matcher_report.md from the stage fragments in cache/report_parts/.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .data import CACHE, REPORTS, SEED, log
from .decide import DECISION_PATH, PARTS_DIR, PRED_DIR, one_home, predict
from .eda import md_table
from .features import FEAT_DIR
from .io_utils import PROJECT_ROOT, write_id_lists


def id_arrays(split: str) -> dict[int, np.ndarray]:
    """entity_id strings per source indexed by idx."""
    t = pq.read_table(CACHE / f"{split}_records.parquet", columns=["src", "idx", "entity_id"])
    out = {}
    for s in (1, 2, 3):
        sub = t.filter(pc.equal(t.column("src"), s))
        ids = np.empty(len(sub), dtype=object)
        ids[sub.column("idx").to_numpy()] = sub.column("entity_id").to_pylist()
        out[s] = ids
    return out


def sanity(df: pd.DataFrame, pred: np.ndarray, q: np.ndarray, s1_country: pd.Series) -> list[list]:
    """Per country: % S1 predicted empty, mean predicted set size, mean q of accepted pairs."""
    rows = []
    sizes = pd.Series(df["s1"].to_numpy()[pred]).value_counts()
    for c in sorted(s1_country.unique()):
        ids = s1_country.index[s1_country == c]
        sz = sizes.reindex(ids, fill_value=0).to_numpy()
        m = pred & (df["country"].to_numpy() == c)
        rows.append([c, f"{len(ids):,}", f"{(sz == 0).mean() * 100:.2f}%", f"{sz.mean():.3f}",
                     f"{q[m].mean():.4f}" if m.any() else "-"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Write the test submission and the matcher report")
    ap.add_argument("--name", default="01_lgbm_v1")
    name = ap.parse_args().name
    t0 = time.time()
    rng = random.Random(SEED)
    cfg = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    out_dir = PROJECT_ROOT / "submissions" / name

    te = pd.read_parquet(PRED_DIR / "test.parquet")
    s1, src, idx = te["s1"].to_numpy(), te["src"].to_numpy(), te["idx"].to_numpy()
    q = one_home(te["p"].to_numpy(), src, idx, cfg["d1"])
    pred = predict(q, s1, cfg["d2"], cfg["params"])
    ids = id_arrays("test")
    rec_ids = np.empty(len(te), dtype=object)
    for s in (2, 3):
        m = src == s
        rec_ids[m] = ids[s][idx[m]]
    s1_ids_order = ids[1].tolist()  # file order == idx order
    cand_map: dict = {}
    for a, b in zip(s1.tolist(), rec_ids.tolist()):
        cand_map.setdefault(a, []).append(b)
    match_map: dict = {}
    for a, b in zip(s1[pred].tolist(), rec_ids[pred].tolist()):
        match_map.setdefault(a, []).append(b)
    write_id_lists(out_dir / "matching_results.tsv", s1_ids_order,
                   {ids[1][k]: v for k, v in match_map.items()}, "matched_entity_ids")
    write_id_lists(out_dir / "candidate_pairs.tsv", s1_ids_order,
                   {ids[1][k]: v for k, v in cand_map.items()}, "candidate_entity_ids")
    del cand_map
    log(f"wrote {out_dir}: {int(pred.sum()):,} matched pairs, {len(te):,} candidate pairs")
    vr = subprocess.run([sys.executable, "utils/validate_submission.py",
                         "--matching", f"submissions/{name}/matching_results.tsv",
                         "--candidate", f"submissions/{name}/candidate_pairs.tsv",
                         "--test-dir", "dataset/test", "--check-ids"],
                        cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8")
    log(f"validator exit {vr.returncode}")

    # sanity: test vs train OOF with the same decision config
    t1 = pq.read_table(CACHE / "test_records.parquet", columns=["idx", "country"], filters=[("src", "=", 1)]).to_pandas()
    test_rows = sanity(te, pred, q, pd.Series(t1["country"].to_numpy(), index=t1["idx"].to_numpy()))
    oof = pd.read_parquet(PRED_DIR / "train_oof.parquet")
    qo = one_home(oof["p"].to_numpy(), oof["src"].to_numpy(), oof["idx"].to_numpy(), cfg["d1"])
    po = predict(qo, oof["s1"].to_numpy(), cfg["d2"], cfg["params"])
    r1 = pq.read_table(CACHE / "train_records.parquet", columns=["idx", "country"], filters=[("src", "=", 1)]).to_pandas()
    train_rows = sanity(oof, po, qo, pd.Series(r1["country"].to_numpy(), index=r1["idx"].to_numpy()))
    del oof, qo, po

    # France (and any other unseen country) examples
    train_countries = set(r1["country"].unique())
    ex_lines = []
    rt = pq.read_table(CACHE / "test_records.parquet", columns=["src", "entity_id", "name_raw", "addr_raw"])
    rsrc = rt.column("src").to_numpy()
    offs = {k: int(np.flatnonzero(rsrc == k)[0]) for k in (1, 2, 3)}
    for c in sorted(set(t1["country"]) - train_countries):
        pool = t1.loc[t1["country"] == c, "idx"].tolist()
        pick = sorted(rng.sample(pool, min(15, len(pool))))
        rows = []
        order = np.lexsort((-q, s1))
        s1_sorted = s1[order]
        for k in pick:
            a = rt.take(pa.array([offs[1] + k])).to_pylist()[0]
            lo, hi = np.searchsorted(s1_sorted, k), np.searchsorted(s1_sorted, k, side="right")
            members = order[lo:hi]
            acc = [i for i in members if pred[i]]
            rej = [i for i in members if not pred[i]][:3]
            rows.append([a["entity_id"], a["name_raw"], a["addr_raw"], "S1", "", ""])
            for tag, lst in (("MATCH", acc), ("rejected", rej)):
                for i in lst:
                    b = rt.take(pa.array([offs[int(src[i])] + int(idx[i])])).to_pylist()[0]
                    rows.append(["", b["name_raw"], b["addr_raw"], tag, b["entity_id"], f"{q[i]:.3f}"])
            if not members.size:
                rows.append(["", "(no candidates)", "", "", "", ""])
        ex_lines += [f"### {c}: 15 random test S1 (accepted matches, then top-3 rejected candidates)", "",
                     md_table(["S1 id", "name", "address", "role", "record id", "q"], rows), ""]

    sub_md = ["## 4. Test submission", "",
              f"`submissions/{name}/` with decision `{cfg['d1']}` + `{cfg['d2']}` `{json.dumps(cfg['params'])}`: "
              f"{int(pred.sum()):,} matched pairs over {len(s1_ids_order):,} test S1; candidate_pairs.tsv lists "
              f"all {len(te):,} scored pairs.", "",
              "Validator (`--check-ids`):", "", "```", (vr.stdout + vr.stderr).strip(), "```", "",
              "Sanity, test vs train OOF (same decision config):", "",
              md_table(["split", "country", "S1", "% S1 predicted empty", "mean predicted set size",
                        "mean q of accepted pairs"],
                       [["train OOF", *r] for r in train_rows] + [["test", *r] for r in test_rows]), "",
              *ex_lines,
              f"Submission step {(time.time() - t0) / 60:.1f} min.", ""]
    (PARTS_DIR / "submit.md").write_text("\n".join(sub_md), encoding="utf-8")

    # ---- assemble the matcher report
    fstats = {s: json.loads((FEAT_DIR / f"{s}_stats.json").read_text(encoding="utf-8")) for s in ("train", "test")}
    tstats = json.loads((PARTS_DIR / "train_stats.json").read_text(encoding="utf-8"))
    feat_names = [c for c in pq.read_schema(next((FEAT_DIR / "train").rglob("part-*.parquet"))).names
                  if c not in ("s1", "src", "idx", "label")]
    head = ["# Matcher report (v1)", "",
            f"Decision: **{cfg['d1']} + {cfg['d2']} {json.dumps(cfg['params'])}**, OOF macro F0.5 "
            f"**{cfg['oof']['overall']:.5f}** ({', '.join(f'{k} {v:.5f}' for k, v in cfg['oof'].items() if k != 'overall')}).", "",
            "## 1. Pair features", "",
            f"{len(feat_names)} features per candidate pair (country is not a feature): "
            + ", ".join(f"`{f}`" for f in feat_names) + ".", "",
            md_table(["split", "country", "pairs", "minutes"],
                     [[s, c, f"{v['pairs']:,}", f"{v['seconds'] / 60:.1f}"] for s in ("train", "test")
                      for c, v in fstats[s]["countries"].items()]
                     + [[s, "total", "", f"{fstats[s]['seconds'] / 60:.1f} (peak RSS {fstats[s]['peak_rss_gib']:.1f} GiB)"]
                        for s in ("train", "test")]), ""]
    parts = [(PARTS_DIR / n).read_text(encoding="utf-8") for n in ("train.md", "decide.md", "submit.md")]
    (REPORTS / "matcher_report.md").write_text("\n".join(head) + "\n" + "\n".join(parts), encoding="utf-8")
    json.dump({"name": name, "validator_exit": vr.returncode, "matched_pairs": int(pred.sum()),
               "test_rows": test_rows, "train_rows": train_rows, "train_minutes": tstats["seconds"] / 60},
              open(PARTS_DIR / "submit_stats.json", "w"), indent=1)
    log("wrote reports/matcher_report.md")


if __name__ == "__main__":
    main()
