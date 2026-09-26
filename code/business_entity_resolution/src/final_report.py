"""Assemble reports/final_report.md (Task 5: slim candidates, retrain, variants, package, pseudo labels).

Usage (from code/business_entity_resolution/):
    python -m src.final_report
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .data import CACHE, REPORTS, run_paths
from .eda import md_table
from .variants import decide_variant

PARTS = CACHE / "report_parts"
REF = {"02_prune_v2 (stage 1, v2 cands)": "v2", "03_stage2 (stage 2, v2 cands)": "s2",
       "slim stage 1 (v3s1)": "v3s1", "slim stage 2 (v3s2)": "v3s2"}


def _read(name: str) -> str:
    p = PARTS / name
    return p.read_text(encoding="utf-8") if p.exists() else f"*{name}: not run / failed (see notes).*"


def set_sizes(tag: str, gamma: float = 1.0) -> dict | None:
    """Mean predicted set size per country (odds + ef gamma) on train OOF and test."""
    rp = run_paths(tag)
    out = {}
    for split, fname in (("train OOF", "train_oof.parquet"), ("test", "test.parquet")):
        f = rp["preds"] / fname
        if not f.exists():
            return None
        d = pd.read_parquet(f, columns=["s1", "src", "idx", "country", "p"])
        _, pred = decide_variant(d["p"].to_numpy(), d["s1"].to_numpy(), d["src"].to_numpy(), d["idx"].to_numpy(),
                                 d["country"].to_numpy(), gamma, {})
        rec = f"{'train' if split == 'train OOF' else 'test'}_records.parquet"
        s1c = pq.read_table(CACHE / rec, columns=["idx", "country"], filters=[("src", "=", 1)]).to_pandas()
        sizes = pd.Series(d["s1"].to_numpy()[pred]).value_counts()
        for c in sorted(s1c["country"].unique()):
            ids = s1c.loc[s1c["country"] == c, "idx"].to_numpy()
            out[(split, c)] = float(sizes.reindex(ids, fill_value=0).mean())
    return out


def main() -> None:
    notes = json.loads((PARTS / "notes_final.json").read_text(encoding="utf-8")) if (PARTS / "notes_final.json").exists() else []
    timings = json.loads((PARTS / "timings_final.json").read_text(encoding="utf-8")) if (PARTS / "timings_final.json").exists() else {}
    slim = json.loads((CACHE.parent / "artifacts" / "slim.json").read_text(encoding="utf-8")) \
        if (CACHE.parent / "artifacts" / "slim.json").exists() else None
    rows = []
    for name, tag in REF.items():
        p = run_paths(tag)["parts"] / "decide_stats.json"
        if p.exists():
            st = json.loads(p.read_text(encoding="utf-8"))
            o = st["oof"]
            rows.append([name, f"{o['overall']:.5f}", f"{o['India']:.5f}", f"{o['US']:.5f}", f"{o['singleton']:.4f}",
                         f"{o['non_singleton']:.4f}", f"{st['missed_by_blocking']:,}", f"{st['fn_in_candidates']:,}",
                         f"{st['fp']:,}"])
    size_rows = []
    for name, tag in REF.items():
        s = set_sizes(tag)
        if s:
            cs = sorted({c for (_, c) in s})
            size_rows.append([name, *(f"{s[('train OOF', c)]:.3f}" if ("train OOF", c) in s else "-" for c in cs),
                              *(f"{s[('test', c)]:.3f}" for c in cs)])
    head_cs = cs if size_rows else []
    lines = ["# Final report (Task 5): slim candidates, fold-consistent stage 2, variants, package", ""]
    if slim:
        ch = next(r for r in slim["curve"] if r["tau"] == slim["tau"])
        lines += [f"**Chosen slim candidates: tau = {slim['tau']}** (S1 top-{slim['s_cap']}, record top-{slim['r_cap']}): "
                  f"{ch['train_pairs']:,} train / {ch['test_pairs']:,} test pairs, candidates per S1 "
                  f"{ch['train_cands']['mean']:.1f} train / {ch['test_cands']['mean']:.1f} test (p95 "
                  f"{ch['train_cands']['p95']:.0f} / {ch['test_cands']['p95']:.0f}), pair recall {ch['recall']:.2%}, "
                  f"oracle {ch['oracle']:.5f}, proxy loss {ch['proxy_loss']:.5f}.", ""]
    if timings:
        lines += ["Runtime per part (minutes): " + ", ".join(f"{k} {v:.0f}" for k, v in timings.items()) + ".", ""]
    if notes:
        lines += ["Notes:", "", *[f"- {n}" for n in notes], ""]
    lines += [_read("slim.md"), "",
              "## Part B. Retrain on the slim candidates (fold-consistent stage-2 test inputs)", "",
              "OOF macro F0.5 over all train S1 with the decision chosen by the limited search (all runs picked odds + ef "
              "gamma 1.0), and the loss split in train pairs:", "",
              md_table(["run", "OOF", "India", "US", "singleton", "non-singleton", "missed by blocking",
                        "false negatives", "false positives"], rows), "",
              "Mean predicted set size per S1 (odds + ef, gamma 1.0): train OOF vs test. 03 used the mean of both "
              "stage-1 models' test p for every stage-2 model; the slim stage 2 applies each stage-2 model to test "
              "features built from the stage-1 model whose p it was trained on.", "",
              md_table(["run", *[f"train OOF {c}" for c in head_cs], *[f"test {c}" for c in head_cs]],
                       [r[:1] + [x for x in r[1:1 + len(head_cs)]] + r[1 + len(head_cs):] for r in size_rows]), "",
              "### Stage 1 (slim)", "", _read("v3s1/train.md"), "", _read("v3s1/decide.md"), "",
              "### Stage 2 (slim)", "", _read("v3s2/train.md"), "", _read("v3s2/decide.md"), "",
              _read("variants.md"), "",
              "## Part D. Final package", "", _read("package.md"), "",
              _read("pseudo.md"), ""]
    (REPORTS / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote reports/final_report.md")


if __name__ == "__main__":
    main()
