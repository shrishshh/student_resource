"""Build the final submission zip.

Usage (from code/business_entity_resolution/):
    python -m src.make_package --variant 04_s2_g15 --team TEAM

Creates student_resource/<team>_submission.zip with:
    output/matching_results.tsv, output/candidate_pairs.tsv   (from submissions/<variant>/)
    code/business_entity_resolution/src/*.py, README.md, requirements.txt
    Documentation_template.md                                 (the filled-in write-up)
No data, cache, artifacts or model files are included (run_all regenerates them).
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from .io_utils import PROJECT_ROOT

PKG = PROJECT_ROOT / "code" / "business_entity_resolution"


def build(variant: str, team: str) -> Path:
    """Write the zip and return its path."""
    sub = PROJECT_ROOT / "submissions" / variant
    files: list[tuple[Path, str]] = [(sub / "matching_results.tsv", "output/matching_results.tsv"),
                                     (sub / "candidate_pairs.tsv", "output/candidate_pairs.tsv")]
    files += [(p, f"code/business_entity_resolution/src/{p.name}") for p in sorted((PKG / "src").glob("*.py"))]
    files += [(PKG / "README.md", "code/business_entity_resolution/README.md"),
              (PKG / "requirements.txt", "code/business_entity_resolution/requirements.txt"),
              (PROJECT_ROOT / "Documentation_template.md", "Documentation_template.md")]
    missing = [str(p) for p, _ in files if not p.exists()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    out = PROJECT_ROOT / f"{team}_submission.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arc in files:
            z.write(src, arc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build <team>_submission.zip")
    ap.add_argument("--variant", required=True, help="folder under submissions/")
    ap.add_argument("--team", required=True)
    args = ap.parse_args()
    out = build(args.variant, args.team)
    with zipfile.ZipFile(out) as z:
        for info in z.infolist():
            print(f"{info.file_size:>13,}  {info.compress_size:>13,}  {info.filename}")
    print(f"{out.name}: {out.stat().st_size / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
