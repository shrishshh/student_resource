"""Decision layer: pair probabilities -> predicted match set per S1.

Usage (from code/business_entity_resolution/, after src.train):
    python -m src.decide [--tag v1] [--limited]

``--limited`` restricts the search to odds + ef (gamma 0.85/1.0/1.2/1.5) and odds + two
(t1 0.50-0.70, t2 0.60-0.80).

D1 (one-home, per record r):
    odds   q = odds(p) / (1 + sum of odds over r's candidates), p clipped to [1e-6, 1-1e-6]
    none   q = p
    argmax q = p for r's best S1, 0 for its other candidates
D2 (per S1):
    thr    global threshold t
    two    first match needs q >= t1, further matches q >= t2
    ef     expected-F0.5 optimiser (numba) over the top 15 candidates with q >= 0.01,
           optionally on q ** gamma
Every combination is scored on the out-of-fold predictions over ALL train S1
(S1 without candidates count as empty predictions) with a vectorised evaluator that
implements exactly the rule of ``metrics.entity_f05``; the winner is re-scored with
``metrics.macro_f05`` itself as a check. The winner is saved to artifacts/decision.json.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import numba
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .data import ARTIFACTS, CACHE, SEED, gt_pairs, log, peak_rss, run_paths
from .eda import md_table, p2
from .metrics import macro_f05

PRED_DIR = CACHE / "preds"
PARTS_DIR = CACHE / "report_parts"
DECISION_PATH = ARTIFACTS / "decision.json"
EF_TOP, EF_MIN_Q = 15, 0.01
THRESHOLDS = np.round(np.arange(0.30, 0.901, 0.05), 2)
T1S = np.round(np.arange(0.30, 0.701, 0.05), 2)
T2S = np.round(np.arange(0.50, 0.901, 0.05), 2)
GAMMAS = [0.7, 0.85, 1.0, 1.2, 1.5]


# ----------------------------------------------------------------- D1
def one_home(p: np.ndarray, src: np.ndarray, idx: np.ndarray, mode: str) -> np.ndarray:
    """Exclusive probabilities over each record's S1 candidates."""
    if mode == "none":
        return p.astype(np.float32)
    rkey = (src.astype(np.int64) << 32) | idx.astype(np.int64)
    _, inv = np.unique(rkey, return_inverse=True)
    if mode == "odds":
        pc_ = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
        odds = pc_ / (1 - pc_)
        return (odds / (1.0 + np.bincount(inv, weights=odds)[inv])).astype(np.float32)
    if mode == "argmax":
        order = np.lexsort((-p, inv))            # best p first inside each record
        first = np.r_[True, inv[order][1:] != inv[order][:-1]]
        keep = np.zeros(len(p), bool)
        keep[order[first]] = True
        return np.where(keep, p, 0.0).astype(np.float32)
    raise ValueError(mode)


# ----------------------------------------------------------------- D2
@numba.njit(cache=True)
def _pb(q: np.ndarray) -> np.ndarray:
    """Poisson-binomial distribution of the number of successes."""
    d = np.zeros(len(q) + 1)
    d[0] = 1.0
    for i in range(len(q)):
        for k in range(i + 1, 0, -1):
            d[k] = d[k] * (1 - q[i]) + d[k - 1] * q[i]
        d[0] *= 1 - q[i]
    return d


@numba.njit(cache=True)
def _ef_choose(q: np.ndarray, starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """For each group (q sorted descending) the j in 0..n maximising E[F0.5]."""
    out = np.zeros(len(starts), np.int32)
    for g in range(len(starts)):
        s, n = starts[g], lens[g]
        qq = q[s:s + n]
        best_j, best = 0, 1.0
        for i in range(n):
            best *= 1 - qq[i]                      # E[F | predict nothing] = P(no true match)
        for j in range(1, n + 1):
            pa_ = _pb(qq[:j])
            pb_ = _pb(qq[j:])
            e = 0.0
            for a in range(1, j + 1):
                if pa_[a] == 0.0:
                    continue
                for b in range(n - j + 1):
                    e += pa_[a] * pb_[b] * 1.25 * a / (j + 0.25 * (a + b))
            if e > best:
                best, best_j = e, j
        out[g] = best_j
    return out


class Scorer:
    """Vectorised macro F0.5 over all train S1 (identical rule to metrics.entity_f05)."""

    def __init__(self, s1: np.ndarray, label: np.ndarray, t_all: np.ndarray, s1_ids: np.ndarray,
                 s1_country: np.ndarray):
        self.s1, self.label, self.t_all = s1, label.astype(bool), t_all
        self.s1_ids, self.s1_country = s1_ids, s1_country
        self.n = len(t_all)
        self.T = t_all[s1_ids]

    def per_s1(self, pred: np.ndarray) -> np.ndarray:
        """Entity F0.5 of every train S1 for a boolean pair prediction."""
        tp = np.bincount(self.s1[pred & self.label], minlength=self.n)[self.s1_ids]
        pp = np.bincount(self.s1[pred], minlength=self.n)[self.s1_ids]
        T = self.T
        f = np.where((pp == 0) & (T == 0), 1.0,
                     np.where((pp == 0) | (T == 0), 0.0, 1.25 * tp / np.maximum(pp + 0.25 * T, 1e-12)))
        return f

    def summary(self, pred: np.ndarray) -> dict:
        """Overall, per-country and singleton / non-singleton macro F0.5."""
        f = self.per_s1(pred)
        out = {"overall": float(f.mean())}
        for c in np.unique(self.s1_country):
            out[c] = float(f[self.s1_country == c].mean())
        out["singleton"] = float(f[self.T == 0].mean())
        out["non_singleton"] = float(f[self.T > 0].mean())
        return out


def sorted_view(s1: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(order by s1 then q desc, 1-based rank of each pair inside its S1 in that order)."""
    order = np.lexsort((-q, s1))
    g = s1[order]
    start = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    rank = np.empty(len(q), np.int32)
    rank[order] = (np.arange(len(q)) - np.repeat(start, np.diff(np.r_[start, len(q)])) + 1).astype(np.int32)
    return order, rank


def predict(q: np.ndarray, s1: np.ndarray, method: str, params: dict, view=None) -> np.ndarray:
    """Boolean pair prediction for a D2 method."""
    if method == "thr":
        return q >= params["t"]
    order, rank = view if view is not None else sorted_view(s1, q)
    if method == "two":
        first_ok = np.zeros(int(s1.max()) + 1, bool)
        r1 = rank == 1
        first_ok[s1[r1]] = q[r1] >= params["t1"]
        return np.where(rank == 1, q >= params["t1"], (q >= params["t2"]) & first_ok[s1])
    if method == "ef":
        qg = q.astype(np.float64) ** params["gamma"]
        elig = (rank <= EF_TOP) & (q >= EF_MIN_Q)
        o = order[elig[order]]                    # eligible pairs grouped by S1, q descending
        g = s1[o]
        starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
        lens = np.diff(np.r_[starts, len(o)])
        j = _ef_choose(qg[o], starts.astype(np.int64), lens.astype(np.int64))
        pos_in_group = np.arange(len(o)) - np.repeat(starts, lens)
        pred = np.zeros(len(q), bool)
        pred[o] = pos_in_group < np.repeat(j, lens)
        return pred
    raise ValueError(method)


def grid(limited: bool = False) -> list[tuple[str, dict]]:
    """All D2 configurations (or the limited v2 search)."""
    if limited:
        out = [("ef", {"gamma": g}) for g in (0.85, 1.0, 1.2, 1.5)]
        out += [("two", {"t1": float(a), "t2": float(b)}) for a in np.round(np.arange(0.50, 0.701, 0.05), 2)
                for b in np.round(np.arange(0.60, 0.801, 0.05), 2)]
        return out
    out = [("thr", {"t": float(t)}) for t in THRESHOLDS]
    out += [("two", {"t1": float(a), "t2": float(b)}) for a in T1S for b in T2S]
    out += [("ef", {"gamma": g}) for g in GAMMAS]
    return out


def load_truth():
    """Ground-truth counts per S1 and the train S1 list with countries."""
    gt = gt_pairs()
    t = pq.read_table(CACHE / "train_records.parquet", columns=["idx", "country"], filters=[("src", "=", 1)])
    s1_ids = t.column("idx").to_numpy()
    s1_country = np.array(t.column("country").to_pylist(), dtype=object)
    t_all = np.bincount(gt["s1"].to_numpy(), minlength=int(s1_ids.max()) + 1)
    return gt, t_all, s1_ids, s1_country


def main() -> None:
    ap = argparse.ArgumentParser(description="Decision layer search on OOF predictions")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--limited", action="store_true")
    args = ap.parse_args()
    rp = run_paths(args.tag)
    parts_dir, decision_path = rp["parts"], rp["decision"]
    d1_modes = ("odds",) if args.limited else ("odds", "none", "argmax")
    t0 = time.time()
    rng = random.Random(SEED)
    parts_dir.mkdir(parents=True, exist_ok=True)
    oof = pd.read_parquet(rp["preds"] / "train_oof.parquet")
    gt, t_all, s1_ids, s1_country = load_truth()
    s1, src, idx = oof["s1"].to_numpy(), oof["src"].to_numpy(), oof["idx"].to_numpy()
    p, label = oof["p"].to_numpy(), oof["label"].to_numpy().astype(bool)
    sc = Scorer(s1, label, t_all, s1_ids, s1_country)
    countries = sorted(set(s1_country))
    results = []
    for mode in d1_modes:
        q = one_home(p, src, idx, mode)
        view = sorted_view(s1, q)
        for method, params in grid(args.limited):
            r = sc.summary(predict(q, s1, method, params, view))
            results.append({"d1": mode, "d2": method, "params": params, **r})
        log(f"D1={mode}: best so far {max(x['overall'] for x in results):.5f}")
    res = pd.DataFrame(results)
    win = res.loc[res["overall"].idxmax()]
    cfg = {"d1": win["d1"], "d2": win["d2"], "params": win["params"]}
    log(f"winner {cfg} -> {win['overall']:.5f}")

    # verify with metrics.py (ids = integer record keys)
    q = one_home(p, src, idx, cfg["d1"])
    pred = predict(q, s1, cfg["d2"], cfg["params"])
    rkey = (src.astype(np.int64) << 32) | idx
    gkey = (gt["src"].to_numpy().astype(np.int64) << 32) | gt["idx"].to_numpy()
    true_map = {int(s): set() for s in s1_ids}
    for a, b in zip(gt["s1"].to_numpy().tolist(), gkey.tolist()):
        true_map[a].add(b)
    pred_map: dict = {}
    for a, b in zip(s1[pred].tolist(), rkey[pred].tolist()):
        pred_map.setdefault(a, set()).add(b)
    check = macro_f05(pred_map, true_map, dict(zip(s1_ids.tolist(), s1_country.tolist())))
    log(f"metrics.py check: {check['overall']:.6f} (fast evaluator {win['overall']:.6f})")
    del true_map

    # set sizes
    pp = np.bincount(s1[pred], minlength=len(t_all))[s1_ids]
    T = t_all[s1_ids]

    def bucket(x):
        return np.select([x <= 5, x <= 10], [x.astype(str).astype(object), "6-10"], ">10")

    sizes = pd.crosstab(pd.Series(bucket(T), name="true"), pd.Series(bucket(pp), name="pred"))
    order = [c for c in ["0", "1", "2", "3", "4", "5", "6-10", ">10"]]
    size_rows = [[k, f"{int((bucket(T) == k).sum()):,}", f"{int((bucket(pp) == k).sum()):,}"] for k in order]
    # best per country
    best_c = []
    for c in countries:
        b = res.loc[res[c].idxmax()]
        best_c.append([c, b["d1"], b["d2"], json.dumps(b["params"]), f"{b[c]:.5f}", f"{win[c]:.5f}"])
    # error analysis
    fp = np.flatnonzero(pred & ~label)
    fn = np.flatnonzero(~pred & label)
    pick_fp = sorted(rng.sample(fp.tolist(), min(15, len(fp))))
    pick_fn = sorted(rng.sample(fn.tolist(), min(15, len(fn))))
    rec_cols = ["entity_id", "name_raw", "addr_raw", "name_core", "name_alt", "addr_core"]
    rt = pq.read_table(CACHE / "train_records.parquet", columns=["src"] + rec_cols)
    rsrc = rt.column("src").to_numpy()
    offs = {k: int(np.flatnonzero(rsrc == k)[0]) for k in (1, 2, 3)}

    def rows_for(ids):
        a = rt.take(pa.array([offs[1] + int(s1[i]) for i in ids])).to_pylist()
        b = rt.take(pa.array([offs[int(src[i])] + int(idx[i]) for i in ids])).to_pylist()
        return [[oof["country"].iat[i], x["entity_id"], x["name_raw"], x["addr_raw"], x["name_core"],
                 y["entity_id"], y["name_raw"], y["addr_raw"], y["name_core"], y["name_alt"], f"{p[i]:.3f}", f"{q[i]:.3f}"]
                for i, x, y in zip(ids, a, b)]

    err_hdr = ["country", "S1 id", "S1 name", "S1 address", "S1 name_core", "rec id", "rec name", "rec address",
               "rec name_core", "rec name_alt", "p", "q"]
    fp_rows, fn_rows = rows_for(pick_fp), rows_for(pick_fn)
    del rt
    missing_true = len(gt) - int(label.sum())

    decision_path.write_text(json.dumps({**cfg, "oof": {k: float(win[k]) for k in ["overall", *countries,
                                                                                   "singleton", "non_singleton"]},
                                         "metrics_py_check": check["overall"], "ef_top": EF_TOP,
                                         "ef_min_q": EF_MIN_Q}, indent=1), encoding="utf-8")
    fmt = lambda r: [r["d1"], r["d2"], json.dumps(r["params"]), f"{r['overall']:.5f}",  # noqa: E731
                     *[f"{r[c]:.5f}" for c in countries], f"{r['singleton']:.4f}", f"{r['non_singleton']:.4f}"]
    hdr = ["D1", "D2", "params", "OOF F0.5", *countries, "singleton", "non-singleton"]
    best_per_family = res.loc[res.groupby(["d1", "d2"])["overall"].idxmax()].sort_values("overall", ascending=False)
    thr_rows = res[res["d2"].isin(["thr", "ef"])].sort_values(["d1", "d2", "overall"])
    search = ("limited: odds + ef (gamma 0.85/1.0/1.2/1.5) and odds + two (t1 0.50-0.70, t2 0.60-0.80)" if args.limited
              else f"3 D1 x ({len(THRESHOLDS)} thresholds + {len(T1S) * len(T2S)} two-threshold pairs + "
                   f"{len(GAMMAS)} expected-F gammas)")
    title = "## 3. Decision layer (tuned on OOF over all train S1)" if args.tag == "v1" else \
        f"### Decision layer (run `{args.tag}`, tuned on OOF over all train S1)"
    lines = [title, "",
             f"{len(res)} configurations evaluated ({search}). S1s without candidates and true pairs "
             f"missing from the candidates ({missing_true:,}) count against recall.", "",
             f"**Winner: D1 = `{cfg['d1']}`, D2 = `{cfg['d2']}`, params `{json.dumps(cfg['params'])}` -> OOF macro "
             f"F0.5 {win['overall']:.5f}** (re-scored with `metrics.macro_f05`: {check['overall']:.5f}).", "",
             "Best configuration of each (D1, D2) family:", "", md_table(hdr, [fmt(r) for _, r in best_per_family.iterrows()]), "",
             "All global-threshold and expected-F configurations:", "", md_table(hdr, [fmt(r) for _, r in thr_rows.iterrows()]), "",
             "Winner breakdown:", "",
             md_table(["slice", "OOF macro F0.5"], [[k, f"{win[k]:.5f}"] for k in ["overall", *countries, "singleton", "non_singleton"]]), "",
             "Predicted vs true set size (number of train S1):", "", md_table(["size", "true", "predicted"], size_rows), "",
             "Best configuration per country (vs the global winner on that country):", "",
             md_table(["country", "D1", "D2", "params", "best F0.5", "winner F0.5"], best_c), "",
             f"Error analysis: {len(fp):,} false-positive pairs, {len(fn):,} false-negative pairs among candidates "
             f"(+ {missing_true:,} true pairs never in the candidates).", "",
             "15 random false positives:", "", md_table(err_hdr, fp_rows), "",
             "15 random false negatives (true pairs in the candidates that were rejected):", "", md_table(err_hdr, fn_rows), "",
             f"Decision search time {(time.time() - t0) / 60:.1f} min, peak RSS {peak_rss():.1f} GiB.", ""]
    (parts_dir / "decide.md").write_text("\n".join(lines), encoding="utf-8")
    (parts_dir / "decide_sizes.json").write_text(sizes.to_json(), encoding="utf-8")
    (parts_dir / "decide_stats.json").write_text(json.dumps(
        {"oof": {k: float(win[k]) for k in ["overall", *countries, "singleton", "non_singleton"]},
         "cfg": cfg, "fp": int(len(fp)), "fn_in_candidates": int(len(fn)), "missed_by_blocking": missing_true,
         "tp": int((pred & label).sum()), "true_pairs": int(len(gt))}, indent=1), encoding="utf-8")
    log("decide done")


if __name__ == "__main__":
    main()
