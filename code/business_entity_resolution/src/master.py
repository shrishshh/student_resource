"""Master task: diagnostics (1a data audit, 1b adversarial validation), collective smoothing and
shift-robustness experiments in the normal and test-like worlds, and the final 07 submissions.

Usage (from code/business_entity_resolution/):
    python -m src.master audit        # 1a -> cache/master/audit.json
    python -m src.master adversarial  # 1b -> cache/master/adversarial.json
    python -m src.master evaluate     # 2a/2c on existing 04_s1 / D predictions -> cache/master/evaluate.json
    python -m src.master submit       # 07_* submissions for the best test-like variant
    python -m src.master report       # reports/master.md

Fair play: no signal uses file row order or numeric id values; links use only name / address text.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

import duckdb
import lightgbm as lgb
import numba
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from .data import CACHE, REPORTS, SEED, gt_pairs, log, run_paths
from .decide import Scorer, load_truth, one_home, predict
from .eda import md_table
from .features import FEAT_DIR
from .io_utils import PROJECT_ROOT, write_id_lists
from .submit import id_arrays
from .testlike import OUT as TL_OUT
from .testlike import world_preds
from .variants import validate

OUT = CACHE / "master"
GAMMAS = [1.0, 1.5, 2.0]
PREC_MIN = 0.98
REF_GAMMA = 1.5  # 04_s1_g15 decision used for the false-negative link count
# record-to-record link rules (same source and country)
RULES = {
    "name_core+addr_core": "CASE WHEN name_core <> '' AND addr_core <> '' THEN name_core || '|' || addr_core END",
    "raw name": "CASE WHEN name_raw <> '' THEN name_raw END",
    "raw address": "CASE WHEN addr_raw <> '' THEN addr_raw END",
    "name_core, both addr empty": "CASE WHEN name_core <> '' AND addr_core = '' THEN name_core END",
}
# link tokens that the cleaning removes (extracted from the raw name; phones also from the address)
TOKENS = {
    "phone (>=7 digits)": r"""list_filter(list_transform(regexp_extract_all(name_raw || ' ' || addr_raw,
                             '\+?(?:\d[ ().-]{0,2}){6,}\d'), x -> regexp_replace(x, '[^0-9]', '', 'g')),
                             x -> length(x) >= 7)""",
    "(ID: n) tag": r"""regexp_extract_all(name_raw, '(?i)\bid\s*:?\s*(\d+)', 1)""",
    "#tag": r"""regexp_extract_all(name_raw, '#([A-Za-z][A-Za-z0-9_]+)', 1)""",
    "@handle": r"""regexp_extract_all(name_raw, '@([A-Za-z0-9_.]+)', 1)""",
    "web domain": r"""list_transform(regexp_extract_all(name_raw,
                      '(?i)\b([a-z0-9-]+\.(?:com|net|org|in|co|fr|io|biz|info|us))\b', 1), x -> lower(x))""",
}


def con_with_entities(con, split: str = "train") -> None:
    """Table ``rec`` of a split's records with an entity id (S1 idx; orphans get a unique negative id)."""
    rec = (CACHE / f"{split}_records.parquet").as_posix()
    gt = gt_pairs()
    con.register("gt_df", gt[["s1", "src", "idx"]])
    con.execute(f"""
        CREATE OR REPLACE TABLE rec AS
        SELECT r.src, r.idx, r.country, r.name_raw, r.addr_raw, r.name_core, r.addr_core,
               coalesce(CASE WHEN r.src = 1 THEN r.idx END, g.s1, -(r.src::BIGINT * 1000000000 + r.idx) - 1) AS ent
        FROM read_parquet('{rec}') r LEFT JOIN gt_df g ON g.src = r.src AND g.idx = r.idx""")
    con.unregister("gt_df")


# ----------------------------------------------------------------- 1a audit
def audit() -> dict:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    tr, te = (CACHE / "train_records.parquet").as_posix(), (CACHE / "test_records.parquet").as_posix()
    res: dict = {"overlap": [], "tokens": [], "rules": [], "fn_links": {}}
    # (i) train-test overlap per source
    for src in (1, 2, 3):
        n, raw, norm = con.execute(f"""
            WITH t AS (SELECT * FROM read_parquet('{te}') WHERE src = {src}),
                 a AS (SELECT DISTINCT name_raw, addr_raw FROM read_parquet('{tr}') WHERE src = {src}),
                 b AS (SELECT DISTINCT name_core, addr_core FROM read_parquet('{tr}') WHERE src = {src})
            SELECT count(*),
                   count(*) FILTER (WHERE (name_raw, addr_raw) IN (SELECT (name_raw, addr_raw) FROM a)),
                   count(*) FILTER (WHERE (name_core, addr_core) IN (SELECT (name_core, addr_core) FROM b))
            FROM t""").fetchone()
        res["overlap"].append({"src": src, "test_rows": n, "raw": raw, "normalized": norm})
        log(f"overlap S{src}: raw {raw / n:.4%}, normalized {norm / n:.4%}")
    con_with_entities(con)
    n_multi = con.execute("SELECT count(*) FROM (SELECT ent FROM rec WHERE ent >= 0 GROUP BY ent HAVING count(*) >= 2)").fetchone()[0]
    # (ii) link tokens
    for name, expr in TOKENS.items():
        con.execute(f"""CREATE OR REPLACE TABLE tok AS
            SELECT DISTINCT src, idx, country, ent, unnest({expr}) AS v FROM rec""")
        carriers = con.execute("SELECT count(DISTINCT (src, idx)) FROM tok").fetchone()[0]
        tot, same = con.execute("""
            SELECT (SELECT coalesce(sum(n * (n - 1) / 2), 0) FROM (SELECT count(*) n FROM tok GROUP BY country, v)),
                   (SELECT coalesce(sum(m * (m - 1) / 2), 0) FROM (SELECT count(*) m FROM tok GROUP BY country, v, ent))""").fetchone()
        cov = con.execute("""SELECT count(DISTINCT ent) FROM (SELECT ent FROM tok WHERE ent >= 0
                             GROUP BY ent, country, v HAVING count(*) >= 2)""").fetchone()[0]
        res["tokens"].append({"type": name, "records": int(carriers), "pairs": int(tot), "same_pairs": int(same),
                              "precision": same / tot if tot else None, "coverage": cov / n_multi})
        log(f"token {name}: {carriers:,} records, precision {same / tot if tot else float('nan'):.4f}, coverage {cov / n_multi:.4%}")
    # (iii) strict same-source record links (S2/S3)
    den = con.execute("""SELECT sum(m * (m - 1) / 2) FROM (SELECT count(*) m FROM rec WHERE src IN (2, 3) AND ent >= 0
                         GROUP BY src, ent)""").fetchone()[0]
    for name, expr in RULES.items():
        con.execute(f"CREATE OR REPLACE TABLE k AS SELECT src, idx, country, ent, {expr} AS key FROM rec WHERE src IN (2, 3)")
        tot, same = con.execute("""
            SELECT (SELECT coalesce(sum(n * (n - 1) / 2), 0) FROM (SELECT count(*) n FROM k WHERE key IS NOT NULL GROUP BY src, country, key)),
                   (SELECT coalesce(sum(m * (m - 1) / 2), 0) FROM (SELECT count(*) m FROM k WHERE key IS NOT NULL AND ent >= 0 GROUP BY src, country, key, ent))""").fetchone()
        res["rules"].append({"rule": name, "pairs": int(tot), "same_pairs": int(same),
                             "precision": same / tot if tot else None, "coverage": same / den})
        log(f"rule {name}: precision {same / tot if tot else float('nan'):.4f}, coverage {same / den:.4%}")
    # FN of 04_s1 (gamma 1.5) linked to an accepted true record of the same S1
    oof = pd.read_parquet(run_paths("v3s1")["preds"] / "train_oof.parquet", columns=["s1", "src", "idx", "label", "p"])
    q = one_home(oof["p"].to_numpy(), oof["src"].to_numpy(), oof["idx"].to_numpy(), "odds")
    pred = predict(q, oof["s1"].to_numpy(), "ef", {"gamma": REF_GAMMA})
    lab = oof["label"].to_numpy().astype(bool)
    con.register("fn", oof.loc[lab & ~pred, ["s1", "src", "idx"]])
    con.register("acc", oof.loc[lab & pred, ["s1", "src", "idx"]])
    gt = gt_pairs()
    missed = len(gt) - int(lab.sum())
    res["fn_total"] = int((lab & ~pred).sum())
    res["fn_missed_by_blocking"] = missed
    for name, expr in RULES.items():
        n = con.execute(f"""
            WITH k AS (SELECT src, idx, {expr} AS key FROM rec WHERE src IN (2, 3))
            SELECT count(DISTINCT (f.s1, f.src, f.idx)) FROM fn f
            JOIN k kf ON kf.src = f.src AND kf.idx = f.idx
            JOIN acc a ON a.s1 = f.s1 AND a.src = f.src
            JOIN k ka ON ka.src = a.src AND ka.idx = a.idx AND ka.key = kf.key
            WHERE kf.key IS NOT NULL""").fetchone()[0]
        res["fn_links"][name] = int(n)
    con.close()
    res["minutes"] = (time.time() - t0) / 60
    (OUT / "audit.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    log(f"audit done in {res['minutes']:.1f} min")
    return res


# ----------------------------------------------------------------- 1b adversarial validation
def _sample(split: str, countries: list[str], n: int, rng) -> pd.DataFrame:
    frames = []
    for c in countries:
        for p in sorted((FEAT_DIR / split / c).glob("part-*.parquet")):
            frames.append(pq.read_table(p).to_pandas().drop(columns=["label"], errors="ignore"))
    d = pd.concat(frames, ignore_index=True)
    return d.iloc[np.sort(rng.choice(len(d), size=min(n, len(d)), replace=False))]


def adversarial() -> dict:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    tr = _sample("train", ["India", "US"], 1_000_000, rng)
    out = {}
    for name, cs in (("US+India", ["India", "US"]), ("France", ["France"])):
        te = _sample("test", cs, 1_000_000, rng)
        feats = [c for c in tr.columns if c not in ("s1", "src", "idx")]
        x = np.vstack([tr[feats].to_numpy(np.float32), te[feats].to_numpy(np.float32)])
        y = np.r_[np.zeros(len(tr)), np.ones(len(te))]
        hold = rng.random(len(y)) < 0.25
        bst = lgb.train({"objective": "binary", "num_leaves": 63, "learning_rate": 0.1, "min_data_in_leaf": 200,
                         "feature_fraction": 0.8, "seed": SEED, "deterministic": True, "force_col_wise": True,
                         "verbose": -1, "num_threads": 16},
                        lgb.Dataset(x[~hold], label=y[~hold], feature_name=feats), num_boost_round=200)
        auc = roc_auc_score(y[hold], bst.predict(x[hold]))
        gain = sorted(zip(feats, bst.feature_importance("gain")), key=lambda t: -t[1])[:15]
        out[name] = {"auc": float(auc), "n_test": int(len(te)),
                     "top": [{"feature": f, "gain": float(g), "train_mean": float(tr[f].mean()),
                              "test_mean": float(te[f].mean())} for f, g in gain]}
        log(f"adversarial {name}: AUC {auc:.4f}; top {[f for f, _ in gain[:5]]}")
    out["minutes"] = (time.time() - t0) / 60
    (OUT / "adversarial.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ----------------------------------------------------------------- 2a collective smoothing
@numba.njit(cache=True)
def _find(par, i):
    while par[i] != i:
        par[i] = par[par[i]]
        i = par[i]
    return i


@numba.njit(cache=True)
def _union_sorted(par, order, grp):
    """Union consecutive rows (in ``order``) that share the same group code (code < 0 = no key)."""
    for j in range(1, len(order)):
        a, b = order[j - 1], order[j]
        if grp[a] >= 0 and grp[a] == grp[b]:
            ra, rb = _find(par, a), _find(par, b)
            if ra != rb:
                par[rb] = ra


@numba.njit(cache=True)
def _roots(par):
    out = np.empty(len(par), np.int64)
    for i in range(len(par)):
        out[i] = _find(par, i)
    return out


def link_groups(d: pd.DataFrame, keys: pd.DataFrame, rules: list[str]) -> np.ndarray:
    """Component id per candidate row: rows of the same S1 and same source linked by any rule."""
    m = d[["s1", "src", "idx"]].merge(keys, on=["src", "idx"], how="left")
    par = np.arange(len(d), dtype=np.int64)
    s1a, srca = d["s1"].to_numpy(), d["src"].to_numpy()
    for r in rules:
        codes, _ = pd.factorize(m[r])  # -1 where the record has no key for this rule
        codes = codes.astype(np.int64)
        order = np.lexsort((codes, srca, s1a)).astype(np.int64)  # rows of equal (s1, src, key) become adjacent
        c, s_, t_ = codes[order], s1a[order], srca[order]
        same = np.r_[False, (s_[1:] == s_[:-1]) & (t_[1:] == t_[:-1]) & (c[1:] == c[:-1]) & (c[1:] >= 0)]
        gg = np.empty(len(d), np.int64)
        gg[order] = np.cumsum(~same)
        gg[codes < 0] = -1
        _union_sorted(par, order, gg)
    return _roots(par)


def record_keys(split: str) -> pd.DataFrame:
    """Rule keys per S2/S3 record of a split."""
    con = duckdb.connect()
    rec = (CACHE / f"{split}_records.parquet").as_posix()
    cols = ", ".join(f'{e} AS "{n}"' for n, e in RULES.items())
    k = con.execute(f"SELECT src, idx, {cols} FROM read_parquet('{rec}') WHERE src IN (2, 3)").df()
    con.close()
    return k


def smooth(p: np.ndarray, comp: np.ndarray, how: str) -> np.ndarray:
    """Shared probability per component."""
    if how == "none":
        return p
    s = pd.Series(p)
    g = s.groupby(comp)
    if how == "mean":
        return g.transform("mean").to_numpy()
    if how == "median":
        return g.transform("median").to_numpy()
    if how == "maxif":
        mx, mean = g.transform("max").to_numpy(), g.transform("mean").to_numpy()
        return np.where(mx >= 0.9, mx, mean)
    raise ValueError(how)


SHIFT_DROP_TOP = 5
MAX_NORMAL_DROP = 0.002


def eval_models() -> list[str]:
    """04_s1 and D always; D2 (2b shift-robust model) when it has been trained."""
    return ["04_s1", "D"] + (["D2"] if (run_paths("D2")["preds"] / "train_oof.parquet").exists() else [])


def evaluate() -> dict:
    t0 = time.time()
    audit_res = json.loads((OUT / "audit.json").read_text(encoding="utf-8"))
    good = [r["rule"] for r in audit_res["rules"] if r["precision"] is not None and r["precision"] >= PREC_MIN]
    log(f"rules with precision >= {PREC_MIN}: {good}")
    gt, t_all, s1_ids, s1_country = load_truth()
    drop = np.load(TL_OUT / "dropped_s1.npy")
    keep_b = ~np.isin(s1_ids, drop)
    worlds = {"a": ("train", s1_ids, s1_country), "b": ("trainD", s1_ids[keep_b], s1_country[keep_b])}
    results = []
    for world, (split, ids, ctry) in worlds.items():
        keys = record_keys(split)
        for model in eval_models():
            d = world_preds(model, world)
            comp = link_groups(d, keys, good) if good else np.arange(len(d))
            n_linked = int(len(d) - len(np.unique(comp)))
            s1, src, idx = d["s1"].to_numpy(), d["src"].to_numpy(), d["idx"].to_numpy()
            sc = Scorer(s1, d["label"].to_numpy(), t_all, ids, ctry)
            for how in ("none", "mean", "median", "maxif"):
                ps = smooth(d["p"].to_numpy().astype(np.float64), comp, how)
                q = one_home(ps, src, idx, "odds")
                for g in GAMMAS:
                    r = sc.summary(predict(q, s1, "ef", {"gamma": g}))
                    results.append({"world": world, "model": model, "smooth": how, "gamma": g, **r,
                                    "linked_rows": n_linked})
                best = max((x for x in results if x["world"] == world and x["model"] == model and x["smooth"] == how),
                           key=lambda x: x["overall"])
                log(f"{world}/{model}/{how}: best gamma {best['gamma']} -> {best['overall']:.5f}")
    out = {"rules_used": good, "results": results, "minutes": (time.time() - t0) / 60}
    (OUT / "evaluate.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ----------------------------------------------------------------- phase 3
def choose(results: list[dict]) -> tuple[str, str, float, float]:
    """Best test-like-world (model, smoothing, gamma); ties (<1e-4) -> simpler (no smoothing, 04_s1).

    D2 is excluded when its best normal-world score is more than MAX_NORMAL_DROP below D's (rule 2b).
    """
    def best_a(m):
        return max(r["overall"] for r in results if r["world"] == "a" and r["model"] == m)
    banned = {"D2"} if any(r["model"] == "D2" for r in results) and best_a("D2") < best_a("D") - MAX_NORMAL_DROP else set()
    b = [r for r in results if r["world"] == "b" and r["model"] not in banned]
    top = max(r["overall"] for r in b)
    simple = {"none": 0, "mean": 1, "median": 2, "maxif": 3}
    cands = [r for r in b if r["overall"] >= top - 1e-4]
    best = min(cands, key=lambda r: (simple[r["smooth"]], r["model"] != "04_s1", r["gamma"]))
    g = best["gamma"]
    g2 = g + 0.5
    return best["model"], best["smooth"], g, g2


def submit() -> list[dict]:
    ev = json.loads((OUT / "evaluate.json").read_text(encoding="utf-8"))
    model, how, g, g2 = choose(ev["results"])
    tag = {"04_s1": "v3s1", "D": "D", "D2": "D2"}[model]
    te = pd.read_parquet(run_paths(tag)["preds"] / "test.parquet", columns=["s1", "src", "idx", "country", "p"])
    comp = link_groups(te, record_keys("test"), ev["rules_used"]) if (how != "none" and ev["rules_used"]) else np.arange(len(te))
    ps = smooth(te["p"].to_numpy().astype(np.float64), comp, how)
    s1, src, idx = te["s1"].to_numpy(), te["src"].to_numpy(), te["idx"].to_numpy()
    q = one_home(ps, src, idx, "odds")
    ids = id_arrays("test")
    rec = np.empty(len(te), dtype=object)
    for s in (2, 3):
        m = src == s
        rec[m] = ids[s][idx[m]]
    t1 = pq.read_table(CACHE / "test_records.parquet", columns=["idx", "country"], filters=[("src", "=", 1)]).to_pandas()
    out = []
    for gamma in (g, g2):
        name = f"07_{model}{'' if how == 'none' else '_' + how}_g{str(gamma).replace('.', '')}"
        pred = predict(q, s1, "ef", {"gamma": gamma})
        folder = PROJECT_ROOT / "submissions" / name
        folder.mkdir(parents=True, exist_ok=True)
        matches: dict = {}
        for a_, b_ in zip(s1[pred].tolist(), rec[pred].tolist()):
            matches.setdefault(ids[1][a_], []).append(b_)
        write_id_lists(folder / "matching_results.tsv", ids[1].tolist(), matches, "matched_entity_ids")
        shutil.copyfile(PROJECT_ROOT / "submissions" / "04_s1_g15" / "candidate_pairs.tsv", folder / "candidate_pairs.tsv")
        code, _ = validate(folder)
        sizes = pd.Series(s1[pred]).value_counts()
        per_c = {c: float(sizes.reindex(t1.loc[t1["country"] == c, "idx"].to_numpy(), fill_value=0).mean())
                 for c in sorted(t1["country"].unique())}
        out.append({"name": name, "model": model, "smooth": how, "gamma": gamma, "matches": int(pred.sum()),
                    "set_size": per_c, "validator_exit": code})
        log(f"{name}: {int(pred.sum()):,} matches, validator exit {code}")
    (OUT / "submit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ----------------------------------------------------------------- report
def report() -> None:
    def load(n):
        p = OUT / n
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    au, ad, ev, sub = load("audit.json"), load("adversarial.json"), load("evaluate.json"), load("submit.json")
    notes = load("notes.json") or []
    L = ["# Master task: diagnostics, collective smoothing, final 07 submissions", ""]
    if notes:
        L += ["Notes:", "", *[f"- {n}" for n in notes], ""]
    if au:
        L += ["## 1a. Data audit (train)", "", "(i) Train-test overlap (test rows whose values exactly equal a train row of the same source):", "",
              md_table(["source", "test rows", "raw (name, address)", "normalised (name_core, addr_core)"],
                       [[f"S{o['src']}", f"{o['test_rows']:,}", f"{o['raw'] / o['test_rows']:.3%}",
                         f"{o['normalized'] / o['test_rows']:.3%}"] for o in au["overlap"]]), "",
              "(ii) Link tokens removed by our cleaning (all sources; same value + same country):", "",
              md_table(["token", "records", "same-value pairs", "precision (same S1 entity)", "coverage (multi-record entities)"],
                       [[t["type"], f"{t['records']:,}", f"{t['pairs']:,}",
                         f"{t['precision']:.4f}" if t["precision"] is not None else "-", f"{t['coverage']:.3%}"]
                        for t in au["tokens"]]), "",
              "(iii) Strict record-to-record links within the same source (S2 or S3) and country; coverage = share of "
              "same-entity same-source record pairs captured; last column = 04_s1 (gamma 1.5) OOF false negatives "
              f"linked to a record accepted for the correct S1 (of {au['fn_total']:,} false negatives; "
              f"{au['fn_missed_by_blocking']:,} further true pairs are not in the candidates):", "",
              md_table(["rule", "linked pairs", "precision", "coverage", "linked false negatives"],
                       [[r["rule"], f"{r['pairs']:,}", f"{r['precision']:.4f}" if r["precision"] is not None else "-",
                         f"{r['coverage']:.3%}", f"{au['fn_links'][r['rule']]:,}"] for r in au["rules"]]), ""]
    if ad:
        L += ["## 1b. Adversarial validation (train vs test pair features)", ""]
        for name in ("US+India", "France"):
            a = ad[name]
            L += [f"**{name}** test pairs vs train pairs: AUC **{a['auc']:.4f}** ({a['n_test']:,} test pairs). Top 15 features:", "",
                  md_table(["feature", "gain", "train mean", "test mean"],
                           [[t["feature"], f"{t['gain']:,.0f}", f"{t['train_mean']:.4f}", f"{t['test_mean']:.4f}"]
                            for t in a["top"]]), ""]
    if ev:
        rows = []
        for model in sorted({r["model"] for r in ev["results"]}, key=lambda m: ["04_s1", "D", "D2"].index(m)):
            for how in ("none", "mean", "median", "maxif"):
                ra = [r for r in ev["results"] if r["world"] == "a" and r["model"] == model and r["smooth"] == how]
                rb = [r for r in ev["results"] if r["world"] == "b" and r["model"] == model and r["smooth"] == how]
                ba, bb = max(ra, key=lambda r: r["overall"]), max(rb, key=lambda r: r["overall"])
                rows.append([model, how, *(f"{r['overall']:.5f}" for r in sorted(ra, key=lambda r: r["gamma"])),
                             f"{ba['overall']:.5f} (g {ba['gamma']})",
                             *(f"{r['overall']:.5f}" for r in sorted(rb, key=lambda r: r["gamma"])),
                             f"{bb['overall']:.5f} (g {bb['gamma']})"])
        L += ["## 2. Variants in the normal (a) and test-like (b) worlds (odds + ef)", "",
              f"Collective smoothing groups records of the same S1 and source linked by rules with train precision >= "
              f"{PREC_MIN}: {ev['rules_used'] or 'none'}.", "",
              md_table(["model", "smoothing", *[f"(a) g {g}" for g in GAMMAS], "(a) best",
                        *[f"(b) g {g}" for g in GAMMAS], "(b) best"], rows), ""]
    if sub:
        L += ["## 3. Final submissions", "",
              md_table(["submission", "test matches", *[f"mean set size {c}" for c in sorted(sub[0]["set_size"])], "validator"],
                       [[s["name"], f"{s['matches']:,}", *(f"{s['set_size'][c]:.3f}" for c in sorted(s["set_size"])),
                         "PASS" if s["validator_exit"] == 0 else f"FAIL ({s['validator_exit']})"] for s in sub]), ""]
    (REPORTS / "master.md").write_text("\n".join(L), encoding="utf-8")
    log("wrote reports/master.md")


def main() -> None:
    ap = argparse.ArgumentParser(description="Master task")
    ap.add_argument("step", choices=["audit", "adversarial", "evaluate", "submit", "report"])
    {"audit": audit, "adversarial": adversarial, "evaluate": evaluate, "submit": submit, "report": report}[
        ap.parse_args().step]()


if __name__ == "__main__":
    main()
