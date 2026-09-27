"""Pair features for every candidate pair -> cache/features/{split}/{country}/part-*.parquet.

Usage (from code/business_entity_resolution/, after blocking prune):
    python -m src.features --split train
    python -m src.features --split test

Per country (one at a time):
1. Competition features with DuckDB window functions over the candidate list
   (how this pair compares with the record's other S1 candidates and with the S1's
   other records), written to a temporary ordered parquet.
2. Record-level values computed once (IDF totals, chain sizes, orphan signal, ...).
3. Pairs streamed in chunks of CHUNK: rapidfuzz ``cpdist`` string scores plus a
   Python pass for token-set features. float32 everywhere.

Country is never a feature (it only selects the IDF / vocabulary tables).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import time
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from .blocking import KEY_TYPES, connect, country_list, pair_codes, true_mask
from .data import CACHE, gt_pairs, log, peak_rss
from .normalize import phon

CHUNK = 3_000_000
FEAT_DIR = CACHE / "features"
# learned-pruner columns (blocking v2): used as features when the candidates carry them
PRUNER_COLS = ["pscore", "prank_r", "prank_s", "prank_r_all", "prank_s_all"]
_LEAD_ZERO = re.compile(r"\b0+(?=\d)")


# ----------------------------------------------------------------- record prep
def _prep_name(s: str) -> str:
    """Feature-time name tweak: strip leading zeros of numbers, drop 'cie'."""
    return " ".join(t for t in _LEAD_ZERO.sub("", s).split() if t != "cie")


def _prep_addr(s: str) -> str:
    """Feature-time address tweak: strip leading zeros of numbers (0046 -> 46)."""
    return _LEAD_ZERO.sub("", s)


class Records:
    """String fields and record-level features of one split+country.

    Records are stored in position order: S1 rows first, then S2, then S3;
    ``pos(src, idx)`` maps record keys to positions.
    """

    FIELDS = ["src", "idx", "name_core", "name_alt", "name_concat", "addr_core", "addr_nums", "house_num", "state",
              "addr_empty", "name_has_indic", "name_is_web", "name_has_marker", "addr_has_landmark", "addr_has_pobox"]

    def __init__(self, split: str, country: str):
        tbl = pq.read_table(CACHE / f"{split}_records.parquet", columns=self.FIELDS,
                            filters=[("country", "=", country)])
        src = tbl.column("src").to_numpy()
        idx = tbl.column("idx").to_numpy()
        order = np.lexsort((idx, src))
        tbl = tbl.take(pa.array(order))
        src, idx = src[order], idx[order]
        self.n = len(src)
        self.src = src
        self._pos = {}
        for s in (1, 2, 3):
            m = np.flatnonzero(src == s)
            p = np.full(int(idx[m].max()) + 1 if len(m) else 0, -1, np.int64)
            p[idx[m]] = m
            self._pos[s] = p
        self.name = [_prep_name(x) for x in tbl.column("name_core").to_pylist()]
        self.alt = [_prep_name(x) for x in tbl.column("name_alt").to_pylist()]
        self.concat = [x.replace(" ", "") for x in self.name]
        self.phon = [" ".join(phon(t) for t in x.split()) for x in self.name]
        self.addr = [_prep_addr(x) for x in tbl.column("addr_core").to_pylist()]
        self.nums = tbl.column("addr_nums").to_pylist()
        self.house = [h or "" for h in tbl.column("house_num").to_pylist()]
        states = tbl.column("state").to_pylist()
        st_codes = {s: i for i, s in enumerate(sorted(set(states)))}
        self.state = np.array([st_codes[s] if s else -1 for s in states], np.int32)
        self.flags = {c: tbl.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
                      for c in ("addr_empty", "name_has_indic", "name_is_web", "name_has_marker",
                                "addr_has_landmark", "addr_has_pobox")}
        del tbl
        # IDF over all records of the country (S1+S2+S3), per field
        n = self.n
        ndf, adf = Counter(), Counter()
        for x in self.name:
            ndf.update(set(x.split()))
        for x in self.addr:
            adf.update(set(x.split()))
        self.nidf = {t: math.log(n / c) for t, c in ndf.items()}
        self.aidf = {t: math.log(n / c) for t, c in adf.items()}
        is1 = src == 1
        s1_names = [self.name[i] for i in np.flatnonzero(is1)]
        chain = Counter(s1_names)
        s1_vocab = set()
        for x in s1_names:
            s1_vocab.update(x.split())
        # record-level features
        rarest, n_ntok, n_atok, chain_n, orphan, nlen = [], [], [], [], [], []
        for x, a in zip(self.name, self.addr):
            toks = x.split()
            rarest.append(max(toks, key=lambda t: (self.nidf.get(t, 0.0), t)) if toks else "")
            n_ntok.append(len(toks))
            n_atok.append(len(a.split()))
            chain_n.append(chain.get(x, 0))
            orphan.append(sum(t in s1_vocab for t in toks) / len(toks) if toks else 0.0)
            nlen.append(len(x))
        self.rarest = rarest
        self.n_ntok = np.array(n_ntok, np.float32)
        self.n_atok = np.array(n_atok, np.float32)
        self.chain = np.array(chain_n, np.float32)
        self.in_s1_vocab = np.array(orphan, np.float32)
        self.nlen = np.array(nlen, np.float32)

    def pos(self, src: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Positions of records (src, idx)."""
        out = np.empty(len(idx), np.int64)
        for s in (1, 2, 3):
            m = src == s
            if m.any():
                out[m] = self._pos[s][idx[m]]
        return out

    def get(self, field: str, pos: np.ndarray) -> list:
        """Python list of a string field at positions."""
        lst = getattr(self, field)
        return [lst[i] for i in pos]


# ----------------------------------------------------------------- competition (DuckDB)
def competition_table(split: str, country: str, out_path) -> int:
    """Candidate rows of one country plus competition features, ordered by (s1, rank_s)."""
    cand = (CACHE / f"{split}_candidates.parquet").as_posix()
    extra = [c for c in PRUNER_COLS if c in pq.read_schema(cand).names]
    extra_sql = "".join(f"{c}, " for c in extra)
    con = connect()

    def other_best(col: str) -> str:
        """Best value of ``col`` among the record's OTHER S1 candidates (NULL if none)."""
        w = f"(PARTITION BY src, idx ORDER BY {col} DESC, s1)"
        return (f"CASE WHEN row_number() OVER {w} = 1 "
                f"THEN nth_value({col}, 2) OVER ({w[1:-1]} ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) "
                f"ELSE first_value({col}) OVER {w} END")

    con.execute(f"""
        COPY (
            SELECT s1, src, idx, bits, nkeys, sim_name, sim_addr, score, rank_r, rank_s, rank_r_all, rank_s_all,
                   n_cand_s1, n_cand_rec, {extra_sql}
                   {other_best('sim_name')} AS r_oth_name,
                   {other_best('sim_addr')} AS r_oth_addr,
                   {other_best('score')} AS r_oth_score,
                   sum(CASE WHEN sim_name >= 90 THEN 1 ELSE 0 END) OVER (PARTITION BY src, idx) AS r_n_name90,
                   rank() OVER (PARTITION BY s1 ORDER BY sim_name DESC) AS s_rank_name,
                   rank() OVER (PARTITION BY s1 ORDER BY sim_addr DESC) AS s_rank_addr,
                   max(score) OVER (PARTITION BY s1) - score AS s_gap_best,
                   sum(CASE WHEN sim_name >= 90 AND sim_addr >= 85 THEN 1 ELSE 0 END) OVER (PARTITION BY s1) AS s_n_near
            FROM read_parquet('{cand}') WHERE country = ?
            ORDER BY s1, rank_s
        ) TO '{out_path.as_posix()}' (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)""", [country])
    n = con.execute(f"SELECT count(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
    con.close()
    return n


# ----------------------------------------------------------------- per-pair features
def _cp(scorer, a: list, b: list) -> np.ndarray:
    return cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)


def _set_features(R: Records, ps: np.ndarray, pr: np.ndarray) -> dict[str, np.ndarray]:
    """Token-set features computed in one Python pass."""
    n = len(ps)
    out = {k: np.zeros(n, np.float32) for k in (
        "name_jaccard", "name_cov_s", "name_cov_r", "name_first_tok", "name_rarest_in_r", "name_exact",
        "concat_contain", "addr_cov_s", "addr_cov_r", "nums_jaccard", "nums_frac_s_in_r",
        "house_eq", "house_logdiff", "house_prefix", "house_missing")}
    nidf, aidf = R.nidf, R.aidf
    for j, (i, k) in enumerate(zip(ps.tolist(), pr.tolist())):
        a, b = R.name[i], R.name[k]
        ta, tb = a.split(), b.split()
        sa, sb = set(ta), set(tb)
        inter = sa & sb
        if sa or sb:
            out["name_jaccard"][j] = len(inter) / len(sa | sb)
        wi = sum(nidf.get(t, 0.0) for t in inter)
        wa = sum(nidf.get(t, 0.0) for t in sa)
        wb = sum(nidf.get(t, 0.0) for t in sb)
        out["name_cov_s"][j] = wi / wa if wa else 0.0
        out["name_cov_r"][j] = wi / wb if wb else 0.0
        out["name_first_tok"][j] = bool(ta and tb and ta[0] == tb[0])
        out["name_rarest_in_r"][j] = R.rarest[i] in sb
        out["name_exact"][j] = bool(a) and a == b
        ca, cb = R.concat[i], R.concat[k]
        out["concat_contain"][j] = bool(ca and cb and (ca in cb or cb in ca))
        # address tokens
        xa, xb = set(R.addr[i].split()), set(R.addr[k].split())
        if xa and xb:
            wi = sum(aidf.get(t, 0.0) for t in xa & xb)
            wa = sum(aidf.get(t, 0.0) for t in xa)
            wb = sum(aidf.get(t, 0.0) for t in xb)
            out["addr_cov_s"][j] = wi / wa if wa else 0.0
            out["addr_cov_r"][j] = wi / wb if wb else 0.0
        na, nb = R.nums[i].split(), R.nums[k].split()
        if na or nb:
            sna, snb = set(na), set(nb)
            out["nums_jaccard"][j] = len(sna & snb) / len(sna | snb)
            out["nums_frac_s_in_r"][j] = (sum(x in snb for x in na) / len(na)) if na else 0.0
        ha, hb = R.house[i], R.house[k]
        if ha and hb:
            out["house_eq"][j] = ha == hb
            out["house_logdiff"][j] = math.log1p(abs(int(ha[:15]) - int(hb[:15])))
            out["house_prefix"][j] = ha != hb and (ha.startswith(hb) or hb.startswith(ha))
        else:
            out["house_missing"][j] = 1.0
            out["house_logdiff"][j] = np.nan
    return out


def chunk_features(R: Records, df) -> dict[str, np.ndarray]:
    """All features for one chunk of candidate rows (a pandas DataFrame)."""
    s1 = df["s1"].to_numpy()
    src = df["src"].to_numpy()
    idx = df["idx"].to_numpy()
    ps = R.pos(np.ones(len(s1), np.int8), s1)
    pr = R.pos(src, idx)
    f: dict[str, np.ndarray] = {}
    na, nb = R.get("name", ps), R.get("name", pr)
    f["name_tsr"] = _cp(fuzz.token_set_ratio, na, nb)
    f["name_tsort"] = _cp(fuzz.token_sort_ratio, na, nb)
    f["name_ratio"] = _cp(fuzz.ratio, na, nb)
    f["name_partial"] = _cp(fuzz.partial_ratio, na, nb)
    f["name_jw"] = _cp(JaroWinkler.normalized_similarity, na, nb)
    alt = R.get("alt", pr)
    has_alt = np.fromiter((bool(x) for x in alt), bool, len(alt))
    f["name_alt_tsr"] = np.zeros(len(s1), np.float32)
    if has_alt.any():
        w = np.flatnonzero(has_alt)
        f["name_alt_tsr"][w] = _cp(fuzz.token_set_ratio, [na[i] for i in w], [alt[i] for i in w])
    del na, nb, alt
    f["concat_ratio"] = _cp(fuzz.ratio, R.get("concat", ps), R.get("concat", pr))
    f["phon_tsr"] = _cp(fuzz.token_set_ratio, R.get("phon", ps), R.get("phon", pr))
    aa, ab = R.get("addr", ps), R.get("addr", pr)
    f["addr_tsr"] = _cp(fuzz.token_set_ratio, aa, ab)
    f["addr_tsort"] = _cp(fuzz.token_sort_ratio, aa, ab)
    f["addr_ratio"] = _cp(fuzz.ratio, aa, ab)
    f["addr_partial"] = _cp(fuzz.partial_ratio, aa, ab)
    del aa, ab
    f.update(_set_features(R, ps, pr))
    # record-level values gathered to the pair
    f["name_ntok_s"], f["name_ntok_r"] = R.n_ntok[ps], R.n_ntok[pr]
    f["addr_ntok_s"], f["addr_ntok_r"] = R.n_atok[ps], R.n_atok[pr]
    f["name_len_diff"] = np.abs(R.nlen[ps] - R.nlen[pr])
    sa, sb = R.state[ps], R.state[pr]
    f["state_match"] = np.where((sa < 0) | (sb < 0), -1, (sa == sb).astype(np.float32)).astype(np.float32)
    f["rec_src_s3"] = (src == 3).astype(np.float32)
    for k, v in R.flags.items():
        f[f"rec_{k}"] = v[pr]
    f["chain_s"] = R.chain[ps]
    f["chain_r"] = R.chain[pr]          # number of S1 whose name_core equals r's
    f["r_tok_in_s1_vocab"] = R.in_s1_vocab[pr]
    # blocking / competition columns
    bits = df["bits"].to_numpy().astype(np.int64)
    for i, kt in enumerate(KEY_TYPES):
        f[f"key_{kt}"] = ((bits >> i) & 1).astype(np.float32)
    for c in ("nkeys", "score", "sim_name", "sim_addr", "rank_r", "rank_s", "rank_r_all", "rank_s_all",
              "n_cand_s1", "n_cand_rec", "r_n_name90", "s_rank_name", "s_rank_addr", "s_gap_best", "s_n_near"):
        f[c] = df[c].to_numpy().astype(np.float32)
    for c in PRUNER_COLS:
        if c in df.columns:
            f[c] = df[c].to_numpy().astype(np.float32)
    for c in ("name", "addr", "score"):
        other = df[f"r_oth_{c}"].to_numpy().astype(np.float32)  # NaN when r has no other candidate
        f[f"r_oth_{c}"] = other
        this = df["score" if c == "score" else f"sim_{c}"].to_numpy().astype(np.float32)
        f[f"r_margin_{c}"] = this - other
    return f


def _complete(split: str, country: str) -> int | None:
    """Pair count if this country's parts already cover all its candidates, else None."""
    out_dir = FEAT_DIR / split / country
    parts = sorted(out_dir.glob("part-*.parquet"))
    if not parts:
        return None
    have = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
    n = pq.read_table(CACHE / f"{split}_candidates.parquet", columns=["country"],
                      filters=[("country", "=", country)]).num_rows
    return n if have == n else None


def build(split: str, resume: bool = False) -> None:
    """Features for every candidate pair of a split, one country at a time.

    ``resume`` skips countries whose part files already hold all their pairs.
    """
    t0 = time.time()
    gt_codes = None
    if split.startswith("train"):  # train and derived train splits (e.g. trainD)
        gt = gt_pairs()
        gt_codes = np.sort(pair_codes(gt["s1"].to_numpy(), gt["src"].to_numpy(), gt["idx"].to_numpy()))
    stats = {"split": split, "countries": {}}
    for country in country_list(split):
        tc = time.time()
        if resume and (n_done := _complete(split, country)) is not None:
            log(f"[{split}/{country}] already complete ({n_done:,} pairs), skipped")
            stats["countries"][country] = {"pairs": n_done, "seconds": float("nan"), "resumed": True}
            continue
        out_dir = FEAT_DIR / split / country
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        comp = CACHE / f"_comp_{split}_{country}.parquet"
        n = competition_table(split, country, comp)
        log(f"[{split}/{country}] {n:,} pairs; competition features done; loading records")
        R = Records(split, country)
        log(f"[{split}/{country}] records ready ({R.n:,})")
        pf = pq.ParquetFile(comp)
        done = 0
        for part, batch in enumerate(pf.iter_batches(batch_size=CHUNK)):
            df = batch.to_pandas()
            f = chunk_features(R, df)
            cols = {"s1": df["s1"].to_numpy().astype(np.int32), "src": df["src"].to_numpy().astype(np.int8),
                    "idx": df["idx"].to_numpy().astype(np.int32)}
            if gt_codes is not None:
                cols["label"] = true_mask(gt_codes, cols["s1"], cols["src"], cols["idx"]).astype(np.int8)
            cols.update({k: v.astype(np.float32) for k, v in f.items()})
            pq.write_table(pa.table(cols), out_dir / f"part-{part:04d}.parquet", compression="zstd")
            done += len(df)
            log(f"[{split}/{country}] {done:,} / {n:,}")
        del R, pf  # close the parquet handle before deleting (Windows)
        comp.unlink()
        stats["countries"][country] = {"pairs": n, "seconds": time.time() - tc}
    shutil.rmtree(CACHE / "duckdb_tmp", ignore_errors=True)
    stats["seconds"] = time.time() - t0
    stats["peak_rss_gib"] = peak_rss()
    (FEAT_DIR / f"{split}_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    log(f"[{split}] features done in {stats['seconds'] / 60:.1f} min, peak {stats['peak_rss_gib']:.1f} GiB")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pair features -> cache/features/{split}/{country}/")
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--resume", action="store_true", help="skip countries already fully built")
    a = ap.parse_args()
    build(a.split, a.resume)


if __name__ == "__main__":
    main()
