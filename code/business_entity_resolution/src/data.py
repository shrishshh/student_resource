"""Shared paths and loaders for the pipeline stages.

Records are addressed by ``(src, idx)``: ``src`` in {1, 2, 3} is the source number
and ``idx`` (int32) is the 0-based row inside that source file. This keeps big
joins on small integers instead of string ids.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .io_utils import DATASET_DIR, PROJECT_ROOT, read_tsv

PKG_DIR = Path(__file__).resolve().parents[1]
CACHE = PKG_DIR / "cache"
ARTIFACTS = PKG_DIR / "artifacts"
REPORTS = PROJECT_ROOT / "reports"
SEED = 42
LARGE = pa.large_string()
COLS = {"entity_id": "id", "business_name": "name", "business_address": "addr", "country": "country"}

_T0 = time.time()
_PEAK = [0.0]


def log(msg: str) -> None:
    """Progress line with elapsed time, current and peak RSS (GiB)."""
    rss = psutil.Process().memory_info().rss / 2**30
    _PEAK[0] = max(_PEAK[0], rss)
    print(f"[{time.time() - _T0:7.1f}s | {rss:4.1f} GiB, peak {_PEAK[0]:4.1f}] {msg}", flush=True)


def peak_rss() -> float:
    """Peak RSS observed by :func:`log` (GiB); psutil's peak_wset on Windows when available."""
    mi = psutil.Process().memory_info()
    return max(_PEAK[0], getattr(mi, "peak_wset", 0) / 2**30)


def arrow_col(series: pd.Series) -> pa.Array:
    """pandas string Series -> contiguous pyarrow large_string Array."""
    a = pa.array(series)
    if isinstance(a, pa.ChunkedArray):
        a = a.combine_chunks()
    return a.cast(LARGE)


def load_sources(split: str) -> dict[int, dict[str, pa.Array]]:
    """{1|2|3: {"id","name","addr","country"}} raw columns of a split as Arrow arrays."""
    out = {}
    for i in (1, 2, 3):
        df = read_tsv(DATASET_DIR / split / f"{split}_source{i}.tsv")
        out[i] = {short: arrow_col(df[col]) for col, short in COLS.items()}
        del df
    return out


def gt_pairs(sources: dict | None = None) -> pd.DataFrame:
    """Train ground truth as integer pairs (cached in cache/train_gt_pairs.parquet).

    Returns:
        DataFrame with ``s1`` (int32 S1 row), ``src`` (int8, 2 or 3), ``idx`` (int32
        row in that source) and ``country`` (S1 country).
    """
    path = CACHE / "train_gt_pairs.parquet"
    if path.exists():
        return pd.read_parquet(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    sources = sources or load_sources("train")
    gt = read_tsv(DATASET_DIR / "train" / "train_ground_truth.tsv")
    s1_pos = pc.index_in(arrow_col(gt["source1_entity_id"]), value_set=sources[1]["id"]).to_numpy(zero_copy_only=False)
    lists = pc.split_pattern(arrow_col(gt["matched_entity_ids"]), ",")
    flat = pc.list_flatten(lists)
    parent = pc.list_parent_indices(lists).to_numpy()
    keep = pc.not_equal(flat, "").to_numpy(zero_copy_only=False)
    flat, parent = pc.filter(flat, pa.array(keep)), parent[keep]
    src = np.where(pc.starts_with(flat, "S2-").to_numpy(zero_copy_only=False), 2, 3).astype(np.int8)
    idx = np.full(len(flat), -1, dtype=np.int64)
    for s in (2, 3):
        m = src == s
        idx[m] = pc.fill_null(pc.index_in(pc.filter(flat, pa.array(m)), value_set=sources[s]["id"]), -1).to_numpy()
    s1 = s1_pos[parent]
    assert (idx >= 0).all() and (s1 >= 0).all(), "ground truth ids missing from sources"
    country = pc.take(sources[1]["country"], pa.array(s1)).to_numpy(zero_copy_only=False)
    df = pd.DataFrame({"s1": s1.astype(np.int32), "src": src, "idx": idx.astype(np.int32), "country": country})
    df.to_parquet(path, index=False)
    return df


def read_records(split: str, columns: list[str] | None = None, country: str | None = None,
                 src: int | None = None) -> pd.DataFrame:
    """Read cache/{split}_records.parquet (optionally filtered by country / source)."""
    filters = []
    if country is not None:
        filters.append(("country", "=", country))
    if src is not None:
        filters.append(("src", "=", src))
    return pq.read_table(CACHE / f"{split}_records.parquet", columns=columns,
                         filters=filters or None).to_pandas()
