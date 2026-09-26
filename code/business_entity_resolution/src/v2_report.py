"""Assemble reports/v2_report.md from the stage fragments of Parts B-E.

Usage (from code/business_entity_resolution/):
    python -m src.v2_report

Missing fragments are reported as "not run / failed" so the report can be rebuilt after
every checkpoint.
"""

from __future__ import annotations

import json

from .data import CACHE, REPORTS, run_paths
from .eda import md_table

PARTS = CACHE / "report_parts"
V1_OOF = {"overall": 0.97071, "India": 0.96381, "US": 0.97532, "singleton": 0.95219, "non_singleton": 0.97181}


def _read(path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def _stats(tag: str) -> dict | None:
    p = run_paths(tag)["parts"] / "decide_stats.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> None:
    timings = json.loads((PARTS / "timings.json").read_text(encoding="utf-8")) if (PARTS / "timings.json").exists() else {}
    notes = json.loads((PARTS / "notes.json").read_text(encoding="utf-8")) if (PARTS / "notes.json").exists() else []
    v1_loss = json.loads((PARTS / "decide_stats.json").read_text(encoding="utf-8")) if (PARTS / "decide_stats.json").exists() else None
    runs = {"v1 (01_lgbm_v1)": V1_OOF, "v2 (02_prune_v2)": (_stats("v2") or {}).get("oof"),
            "stage 2 (03_stage2)": (_stats("s2") or {}).get("oof")}
    rows = [[k, *(f"{v[c]:.5f}" if v else "-" for c in ("overall", "India", "US", "singleton", "non_singleton"))]
            for k, v in runs.items()]
    loss_rows = []
    v1_split = {"missed_by_blocking": 290922, "fn_in_candidates": 208272, "fp": 50186}  # from reports/matcher_report.md
    for name, st in (("v1", v1_loss or v1_split), ("v2", _stats("v2")), ("stage 2", _stats("s2"))):
        if st:
            loss_rows.append([name, f"{st['missed_by_blocking']:,}", f"{st['fn_in_candidates']:,}", f"{st['fp']:,}"])
    lines = ["# v2 report: LOCO check, learned pruning, matcher v2, stage 2", "",
             "Out-of-fold macro F0.5 over all train S1 (scored with the `metrics.py` rule):", "",
             md_table(["run", "overall", "India", "US", "singleton", "non-singleton"], rows), "",
             "Loss split (train pairs): true pairs never in the candidates / true pairs rejected by the decision / "
             "false-positive pairs:", "",
             md_table(["run", "missed by blocking", "false negatives", "false positives"], loss_rows), ""]
    if timings:
        lines += ["Runtime per part (minutes): " + ", ".join(f"{k} {v:.0f}" for k, v in timings.items()) + ".", ""]
    if notes:
        lines += ["Notes:", "", *[f"- {n}" for n in notes], ""]
    sections = [("Part B", PARTS / "loco.md"), ("Part C", PARTS / "pruner.md")]
    for title, tag in (("Part D. Matcher on the v2 candidates -> submissions/02_prune_v2/", "v2"),
                       ("Part E. Stage-2 model with group-consistency features -> submissions/03_stage2/", "s2")):
        sections.append((title, None))
        for frag in ("train.md", "decide.md", "submit.md"):
            sections.append((None, run_paths(tag)["parts"] / frag))
    for title, path in sections:
        if path is None:
            lines += [f"## {title}", ""]
            continue
        txt = _read(path)
        lines += [txt if txt else f"*{title or path.name}: not run / failed (see notes).*", ""]
    (REPORTS / "v2_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote reports/v2_report.md")


if __name__ == "__main__":
    main()
