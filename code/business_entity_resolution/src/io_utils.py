"""I/O helpers for the Business Entity Resolution challenge.

Every TSV in the project is read through :func:`read_tsv` so that parsing is
identical everywhere: all columns as strings, no NA inference (so an empty
field stays ``""`` and a name like ``"NA"`` or ``"null"`` is not turned into
NaN), and no quote handling (a ``"`` inside a business name is literal text,
not a CSV quote).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

# student_resource/ root: src -> business_entity_resolution -> code -> root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_ROOT / "dataset"

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def read_tsv(path: str | Path) -> pd.DataFrame:
    """Read a challenge TSV with the project's single canonical parser.

    Args:
        path: Path to a tab-separated file with a header row.

    Returns:
        DataFrame with every column as string dtype and no missing values
        (empty fields are ``""``).
    """
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
    ).fillna("")


def load_split(split: str, dataset_dir: str | Path = DATASET_DIR) -> dict[str, pd.DataFrame]:
    """Load the three source files of a split.

    Args:
        split: ``"train"`` or ``"test"``.
        dataset_dir: Folder containing ``train/`` and ``test/``.

    Returns:
        ``{"S1": df, "S2": df, "S3": df}`` with columns
        ``entity_id, business_name, business_address, country``.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    base = Path(dataset_dir) / split
    return {f"S{i}": read_tsv(base / f"{split}_source{i}.tsv") for i in (1, 2, 3)}


def parse_id_list(value: str) -> set[str]:
    """Split a comma-joined ID list into a set, ignoring blanks and whitespace."""
    return {x.strip() for x in value.split(",") if x.strip()}


def load_ground_truth(dataset_dir: str | Path = DATASET_DIR) -> dict[str, set[str]]:
    """Load the training ground truth.

    Returns:
        Mapping ``source1_entity_id -> set of matched S2/S3 ids`` (empty set for
        singletons). If an S1 id appears on several rows its sets are unioned.
    """
    gt = read_tsv(Path(dataset_dir) / "train" / "train_ground_truth.tsv")
    mapping: dict[str, set[str]] = {}
    for s1, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        mapping.setdefault(s1, set()).update(parse_id_list(ids))
    return mapping


def write_id_lists(
    path: str | Path,
    s1_ids_in_file_order: Iterable[str],
    mapping: Mapping[str, Iterable[str]],
    second_col_name: str,
) -> None:
    """Write a submission-style file (matching_results / candidate_pairs).

    One row per S1 id, in the given order; the second column is the comma-joined
    (sorted, de-duplicated) id list, or an empty string when there are none.
    Nothing is quoted. Written as UTF-8 with ``\\n`` line endings.

    Args:
        path: Output TSV path (parent folders are created).
        s1_ids_in_file_order: S1 ids in the order they appear in the source file.
        mapping: ``S1 id -> iterable of S2/S3 ids``; missing keys are written empty.
        second_col_name: ``"matched_entity_ids"`` or ``"candidate_entity_ids"``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{second_col_name}\n")
        for s1 in s1_ids_in_file_order:
            ids = sorted(set(mapping.get(s1, ())))
            f.write(f"{s1}\t{','.join(ids)}\n")
