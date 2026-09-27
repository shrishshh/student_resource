"""France region fill (no retraining) -> submissions/08_D_g15_frstate/.

Usage (from code/business_entity_resolution/):
    python -m src.france_state

~35% of France S2/S3 addresses carry only street + city, so their region is missing and the
pair feature state_match is -1 far more often than in train. Steps:
1. city -> region table learned (unsupervised) from the test S1 France addresses, which always
   carry city and region: every digit-free address component that is not the region itself,
   counted per region; keep components seen >= MIN_COUNT times with exactly one region.
2. For France S2/S3 records without a state, find a known city among their address components
   (exact component match, else the longest known city appearing as a token sequence) and fill
   the region.
3. Recompute state_match for the France test pairs, re-predict them with the D fold models (mean
   of both), rerun odds + expected-F0.5 (gamma 1.5); US / India rows are unchanged.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import unicodedata
from collections import Counter, defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import resources as R
from .data import CACHE, REPORTS, log, run_paths
from .decide import one_home, predict
from .eda import md_table, pct
from .features import FEAT_DIR
from .io_utils import PROJECT_ROOT, write_id_lists
from .submit import id_arrays
from .variants import validate

COUNTRY = "France"
MIN_COUNT = 20
GAMMA = 1.5
BASE = "07_D_g15"
NAME = "08_D_g15_frstate"
OUT = CACHE / "france_state"
_NONALNUM = re.compile(r"[^a-z0-9]+")


def norm(s: str) -> str:
    """Lower-case, strip accents, punctuation -> space (like the pipeline's cleaning)."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    return _NONALNUM.sub(" ", s).strip()


def components(addr: str) -> list[str]:
    return [c for c in (norm(x) for x in addr.split(",")) if c]


def city_table(s1: pd.DataFrame) -> dict[str, str]:
    """city component -> region code, from S1 addresses (unambiguous, frequent components only)."""
    regions = R.STATE_LOOKUP[COUNTRY]
    cnt: dict[str, Counter] = defaultdict(Counter)
    for addr, st in zip(s1["addr_raw"], s1["state"]):
        if not st:
            continue
        for c in components(addr):
            if c in regions or any(ch.isdigit() for ch in c):
                continue
            cnt[c][st] += 1
    return {c: next(iter(v)) for c, v in cnt.items() if len(v) == 1 and sum(v.values()) >= MIN_COUNT}


def find_region(addr: str, table: dict[str, str], by_len: list[str]) -> str:
    """Region of the city found in a record address ('' if none or ambiguous)."""
    comps = components(addr)
    hits = {table[c] for c in comps if c in table}
    if not hits:
        full = " " + " ".join(comps) + " "
        for city in by_len:  # longest first: "la teste de buch" before "la"
            if " " + city + " " in full:
                hits = {table[city]}
                break
    return next(iter(hits)) if len(hits) == 1 else ""


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    rec = pq.read_table(CACHE / "test_records.parquet", columns=["src", "idx", "addr_raw", "state"],
                        filters=[("country", "=", COUNTRY)]).to_pandas()
    table = city_table(rec[rec["src"] == 1])
    by_len = sorted(table, key=len, reverse=True)
    log(f"city table: {len(table)} cities -> {dict(Counter(table.values()))}")
    miss = (rec["src"] != 1) & (rec["state"] == "")
    filled = [find_region(a, table, by_len) for a in rec.loc[miss, "addr_raw"]]
    rec["state_new"] = rec["state"]
    rec.loc[miss, "state_new"] = filled
    n_miss, n_fill = int(miss.sum()), int(sum(1 for f in filled if f))
    log(f"France S2/S3 without state: {n_miss:,}; filled {n_fill:,}")
    # state lookup arrays
    st = {s: rec[rec["src"] == s].set_index("idx")["state_new"] for s in (1, 2, 3)}

    # recompute state_match on the France feature parts and re-predict with the D fold models
    boosters = [lgb.Booster(model_file=str(run_paths("D")["models"] / f"lgbm_fold{h}.txt")) for h in (0, 1)]
    feats = boosters[0].feature_name()
    frames, before, after = [], Counter(), Counter()
    for part in sorted((FEAT_DIR / "test" / COUNTRY).glob("part-*.parquet")):
        d = pq.read_table(part, columns=["s1", "src", "idx"] + feats).to_pandas()
        sa = st[1].reindex(d["s1"].to_numpy()).fillna("").to_numpy()
        sb = np.empty(len(d), dtype=object)
        for s in (2, 3):
            m = d["src"].to_numpy() == s
            sb[m] = st[s].reindex(d["idx"].to_numpy()[m]).fillna("").to_numpy()
        new = np.where((sa == "") | (sb == ""), -1.0, (sa == sb).astype(np.float32)).astype(np.float32)
        before.update(d["state_match"].to_numpy().tolist())
        after.update(new.tolist())
        d["state_match"] = new
        x = d[feats].to_numpy(np.float32)
        p = np.mean([b.predict(x, num_iteration=b.best_iteration) for b in boosters], axis=0)
        frames.append(pd.DataFrame({"s1": d["s1"].to_numpy(), "src": d["src"].to_numpy(),
                                    "idx": d["idx"].to_numpy(), "p_new": p.astype(np.float32)}))
    newp = pd.concat(frames, ignore_index=True)
    log(f"re-predicted {len(newp):,} France pairs")

    # decision on the full test set with the France p replaced
    te = pd.read_parquet(run_paths("D")["preds"] / "test.parquet", columns=["s1", "src", "idx", "country", "p"])
    te = te.merge(newp, on=["s1", "src", "idx"], how="left")
    fr = (te["country"] == COUNTRY).to_numpy()
    assert te.loc[fr, "p_new"].notna().all(), "France pairs without a new prediction"
    s1, src, idx = te["s1"].to_numpy(), te["src"].to_numpy(), te["idx"].to_numpy()
    p_old = te["p"].to_numpy()
    p_new = np.where(fr, te["p_new"].to_numpy(), p_old)
    pred_old = predict(one_home(p_old, src, idx, "odds"), s1, "ef", {"gamma": GAMMA})
    pred_new = predict(one_home(p_new, src, idx, "odds"), s1, "ef", {"gamma": GAMMA})
    assert (pred_old[~fr] == pred_new[~fr]).all(), "US / India predictions changed"
    t1 = pq.read_table(CACHE / "test_records.parquet", columns=["idx", "country"], filters=[("src", "=", 1)]).to_pandas()
    fr_ids = t1.loc[t1["country"] == COUNTRY, "idx"].to_numpy()

    def fr_stats(pred):
        sz = pd.Series(s1[pred]).value_counts().reindex(fr_ids, fill_value=0).to_numpy()
        return {"empty": float((sz == 0).mean()), "mean_size": float(sz.mean()), "matches": int(pred[fr].sum())}

    b_stats, a_stats = fr_stats(pred_old), fr_stats(pred_new)
    ids = id_arrays("test")
    rec_ids = np.empty(len(te), dtype=object)
    for s in (2, 3):
        m = src == s
        rec_ids[m] = ids[s][idx[m]]
    matches: dict = {}
    for a_, b_ in zip(s1[pred_new].tolist(), rec_ids[pred_new].tolist()):
        matches.setdefault(ids[1][a_], []).append(b_)
    folder = PROJECT_ROOT / "submissions" / NAME
    folder.mkdir(parents=True, exist_ok=True)
    write_id_lists(folder / "matching_results.tsv", ids[1].tolist(), matches, "matched_entity_ids")
    shutil.copyfile(PROJECT_ROOT / "submissions" / BASE / "candidate_pairs.tsv", folder / "candidate_pairs.tsv")
    # US / India rows identical to the base submission?
    base = pd.read_csv(PROJECT_ROOT / "submissions" / BASE / "matching_results.tsv", sep="\t", dtype=str,
                       keep_default_na=False)
    new = pd.read_csv(folder / "matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    fr_s1 = set(ids[1][fr_ids].tolist())
    non_fr = ~base["source1_entity_id"].isin(fr_s1)
    same_rows = bool((base.loc[non_fr].values == new.loc[non_fr.values].values).all())
    code, txt = validate(folder)
    log(f"{NAME}: {int(pred_new.sum()):,} test matches; US/India rows identical: {same_rows}; validator exit {code}")

    def dist(c: Counter):
        n = sum(c.values())
        return [pct(c.get(k, 0), n) for k in (-1.0, 0.0, 1.0)]

    lines = ["# France region fill -> 08_D_g15_frstate", "",
             f"City -> region table from test S1 France addresses (digit-free components, >= {MIN_COUNT} occurrences, one "
             f"region only): **{len(table)} cities** ({', '.join(f'{k} {v}' for k, v in sorted(Counter(table.values()).items()))}).", "",
             f"France S2/S3 records without a region: {n_miss:,}; **filled: {n_fill:,}** ({n_fill / n_miss:.1%}).", "",
             "state_match on the France test pairs:", "",
             md_table(["", "-1 (missing)", "0 (different)", "1 (same)"], [["before", *dist(before)], ["after", *dist(after)]]), "",
             f"Decision odds + ef gamma {GAMMA} with model D (mean of both fold models); US and India rows identical to {BASE}: "
             f"**{same_rows}**.", "",
             md_table(["France", "% S1 predicted empty", "mean set size", "France matches"],
                      [[BASE, f"{b_stats['empty']:.2%}", f"{b_stats['mean_size']:.3f}", f"{b_stats['matches']:,}"],
                       [NAME, f"{a_stats['empty']:.2%}", f"{a_stats['mean_size']:.3f}", f"{a_stats['matches']:,}"]]), "",
             f"Total test matches: {int(pred_old.sum()):,} ({BASE}) -> **{int(pred_new.sum()):,}** ({NAME}). "
             f"Validator (--check-ids): {'PASS' if code == 0 else f'FAIL ({code})'}. Runtime {(time.time() - t0) / 60:.1f} min.", "",
             "Sample of the city table: " + ", ".join(f"{c} -> {r}" for c, r in sorted(table.items())[:25]) + ".", ""]
    (REPORTS / "france_state.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "stats.json").write_text(json.dumps({"cities": len(table), "missing": n_miss, "filled": n_fill,
                                                "before": {str(k): v for k, v in before.items()},
                                                "after": {str(k): v for k, v in after.items()},
                                                "france_before": b_stats, "france_after": a_stats,
                                                "total_before": int(pred_old.sum()), "total_after": int(pred_new.sum()),
                                                "same_non_france_rows": same_rows, "validator": code}, indent=1),
                                    encoding="utf-8")
    log("wrote reports/france_state.md")


if __name__ == "__main__":
    main()
