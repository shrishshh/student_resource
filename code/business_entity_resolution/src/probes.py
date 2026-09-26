"""Leaderboard probe submissions derived from the v1 test predictions (no retraining).

Usage (from code/business_entity_resolution/, after src.submit):
    python -m src.probes

* submissions/01b_gamma15/     : 01_lgbm_v1 decision, but the expected-F0.5 optimiser uses gamma = 1.5.
* submissions/01c_france_empty/: 01_lgbm_v1 with every S1 of the countries unseen in train set to an
                                 empty match list (candidate_pairs.tsv unchanged).
Both are validated with utils/validate_submission.py --check-ids.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .data import CACHE, log
from .decide import DECISION_PATH, PRED_DIR, one_home, predict
from .io_utils import PROJECT_ROOT, read_tsv, write_id_lists
from .submit import id_arrays

SUB = PROJECT_ROOT / "submissions"


def validate(name: str) -> tuple[int, str]:
    """Run the official validator (--check-ids) on submissions/<name>/."""
    r = subprocess.run([sys.executable, "utils/validate_submission.py",
                        "--matching", f"submissions/{name}/matching_results.tsv",
                        "--candidate", f"submissions/{name}/candidate_pairs.tsv",
                        "--test-dir", "dataset/test", "--check-ids"],
                       cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8")
    return r.returncode, (r.stdout + r.stderr).strip()


def n_matches(path) -> tuple[int, int]:
    """(total matched ids, rows with an empty list) of a matching_results.tsv."""
    df = read_tsv(path)
    ids = df["matched_entity_ids"]
    return int(ids.str.count(",").sum() + (ids != "").sum()), int((ids == "").sum())


def main() -> None:
    cfg = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    base = SUB / "01_lgbm_v1"
    out = {}

    # --- 01b: gamma 1.5
    name = "01b_gamma15"
    te = pd.read_parquet(PRED_DIR / "test.parquet", columns=["s1", "src", "idx", "p"])
    s1, src, idx = te["s1"].to_numpy(), te["src"].to_numpy(), te["idx"].to_numpy()
    q = one_home(te["p"].to_numpy(), src, idx, cfg["d1"])
    pred = predict(q, s1, "ef", {"gamma": 1.5})
    ids = id_arrays("test")
    rec = np.where(src == 2, ids[2][np.minimum(idx, len(ids[2]) - 1)], ids[3][np.minimum(idx, len(ids[3]) - 1)])
    match_map: dict = {}
    for a, b in zip(s1[pred].tolist(), rec[pred].tolist()):
        match_map.setdefault(ids[1][a], []).append(b)
    (SUB / name).mkdir(parents=True, exist_ok=True)
    write_id_lists(SUB / name / "matching_results.tsv", ids[1].tolist(), match_map, "matched_entity_ids")
    shutil.copyfile(base / "candidate_pairs.tsv", SUB / name / "candidate_pairs.tsv")
    out[name] = {"matches": int(pred.sum()), "validator": validate(name)}
    log(f"{name}: {int(pred.sum()):,} matches")
    del te, q, pred, match_map

    # --- 01c: unseen-in-train countries empty
    name = "01c_france_empty"
    train_c = set(pq.read_table(CACHE / "train_records.parquet", columns=["country"],
                                filters=[("src", "=", 1)]).column("country").unique().to_pylist())
    t1 = pq.read_table(CACHE / "test_records.parquet", columns=["entity_id", "country"], filters=[("src", "=", 1)]).to_pandas()
    unseen = set(t1.loc[~t1["country"].isin(train_c), "entity_id"])
    m = read_tsv(base / "matching_results.tsv")
    is_unseen = m["source1_entity_id"].isin(unseen)
    already_empty = int((is_unseen & (m["matched_entity_ids"] == "")).sum())
    mapping = {s: [x for x in v.split(",") if x] for s, v in zip(m["source1_entity_id"], m["matched_entity_ids"])
               if s not in unseen}
    (SUB / name).mkdir(parents=True, exist_ok=True)
    write_id_lists(SUB / name / "matching_results.tsv", m["source1_entity_id"].tolist(), mapping, "matched_entity_ids")
    shutil.copyfile(base / "candidate_pairs.tsv", SUB / name / "candidate_pairs.tsv")
    out[name] = {"unseen_countries": sorted(set(t1["country"]) - train_c), "unseen_s1_rows": int(is_unseen.sum()),
                 "already_empty_in_01": already_empty, "validator": validate(name)}
    out["01_lgbm_v1"] = {"matches_and_empty_rows": n_matches(base / "matching_results.tsv")}
    out["01c_france_empty"]["matches_and_empty_rows"] = n_matches(SUB / name / "matching_results.tsv")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
