"""End-to-end pipeline: raw dataset -> output/matching_results.tsv + output/candidate_pairs.tsv.

Usage (from code/business_entity_resolution/):
    python -m src.run_all --variant 04_s2_g15            # full run (~8-9 h on a 16 GB / 16-thread laptop)
    python -m src.run_all --variant 04_s2_g15 --dry-run  # list the steps only
    python -m src.run_all --variant 04_s2_g15 --from-step features   # resume

Steps (each is a module that can also be run on its own):
    translit   learn the Indic token dictionary + native-state map from train pairs
    normalize  cache/{train,test}_records.parquet
    blocking   blocking keys -> unpruned scored pairs per country (DuckDB)
    pruner     learned pruner: train, score every pair, rank, choose (k, K), prune
    slim       slim candidates (pruner p >= tau, S1 top-15, record top-3); tau from artifacts/slim.json
    features   pair features for every candidate
    stage1     2-fold out-of-fold LightGBM (run tag v3s1)
    stage2f    stage-2 group-consistency features (train OOF p; fold-consistent test p)
    stage2     2-fold stage-2 LightGBM (run tag v3s2)
    output     decision layer of --variant -> output/ (+ official validator)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from .io_utils import PROJECT_ROOT
from .variants import VARIANTS

OUTPUT = PROJECT_ROOT / "output"


def steps(variant: str) -> list[tuple[str, list[list[str]]]]:
    """(name, module command lines) in execution order."""
    return [
        ("translit", [["src.translit"]]),
        ("normalize", [["src.normalize", "--split", "train"], ["src.normalize", "--split", "test"]]),
        ("blocking", [["src.blocking", "generate", "--split", "train"], ["src.blocking", "generate", "--split", "test"]]),
        ("pruner", [["src.pruner", s] for s in ("train", "score", "rank", "grid", "prune")]),
        ("slim", [["src.slim", "apply"]]),
        ("features", [["src.features", "--split", "train"], ["src.features", "--split", "test"]]),
        ("stage1", [["src.train", "--tag", "v3s1"]]),
        ("stage2f", [["src.stage2", "--p-tag", "v3s1"]]),
        ("stage2", [["src.train", "--tag", "v3s2", "--stage2"]]),
        ("output", [["src.variants", "--only", variant, "--out", str(OUTPUT)]]),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the whole pipeline")
    ap.add_argument("--variant", default="04_s2_g15", choices=sorted(VARIANTS))
    ap.add_argument("--from-step", default=None, help="resume from this step name")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    plan = steps(args.variant)
    names = [n for n, _ in plan]
    start = names.index(args.from_step) if args.from_step else 0
    t0 = time.time()
    for name, cmds in plan[start:]:
        for cmd in cmds:
            line = [sys.executable, "-m", *cmd]
            print(f"[run_all] {name}: {' '.join(cmd)}", flush=True)
            if args.dry_run:
                continue
            t = time.time()
            r = subprocess.run(line)
            if r.returncode != 0:
                sys.exit(f"[run_all] step '{name}' failed ({' '.join(cmd)}); resume with --from-step {name}")
            print(f"[run_all] {name} done in {(time.time() - t) / 60:.1f} min", flush=True)
    if not args.dry_run:
        print(f"[run_all] finished in {(time.time() - t0) / 3600:.1f} h -> {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
