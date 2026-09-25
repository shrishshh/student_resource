"""Write an all-empty submission (format sanity check / trivial baseline).

Usage:
    python -m src.make_empty_submission [out_dir]   (from code/business_entity_resolution/)
"""

from __future__ import annotations

import sys
from pathlib import Path

from .io_utils import DATASET_DIR, PROJECT_ROOT, read_tsv, write_id_lists


def write_all_empty(out_dir: str | Path, dataset_dir: str | Path = DATASET_DIR) -> int:
    """Write matching_results.tsv and candidate_pairs.tsv with every list empty.

    Returns:
        Number of Source-1 rows written.
    """
    out_dir = Path(out_dir)
    s1_ids = read_tsv(Path(dataset_dir) / "test" / "test_source1.tsv")["entity_id"].tolist()
    write_id_lists(out_dir / "matching_results.tsv", s1_ids, {}, "matched_entity_ids")
    write_id_lists(out_dir / "candidate_pairs.tsv", s1_ids, {}, "candidate_entity_ids")
    return len(s1_ids)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else PROJECT_ROOT / "submissions" / "00_all_empty"
    print(f"wrote {write_all_empty(target)} rows to {target}")
