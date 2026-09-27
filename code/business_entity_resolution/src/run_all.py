"""End-to-end pipeline: raw dataset -> output/matching_results.tsv + output/candidate_pairs.tsv.

Usage (from code/business_entity_resolution/):
    python -m src.run_all                                # full run, default --variant 04_s1_g15 (~4 h of measured steps)
    python -m src.run_all --variant 04_s1_g15 --dry-run  # list the steps only
    python -m src.run_all --variant 04_s1_g15 --from-step features   # resume

Steps (each is a module that can also be run on its own):
    translit   learn the Indic token dictionary + native-state map from train pairs
    normalize  cache/{train,test}_records.parquet
    blocking   blocking keys -> unpruned scored pairs per country (DuckDB)
    pruner     learned pruner: train, score every pair, rank, choose (k, K), prune
    slim       slim candidates (pruner p >= tau, S1 top-15, record top-3); tau from artifacts/slim.json
    features   pair features for every candidate
    stage1     2-fold out-of-fold LightGBM (run v3s1, or run D on the test-like world)
    testlike   (D variants) test-like world: 19% of train S1 dropped per country, features rebuilt
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
    """(name, module command lines) in execution order for a stage-1 variant.

    ``s1`` variants train the slim stage-1 model (run v3s1); ``D`` variants build the test-like
    world (19% of train S1 dropped per country, everything S1/candidate-dependent rebuilt) and
    train stage 1 on it (run D), exactly as in Task 6. Stage 2 is not part of the final pipeline.
    """
    stage = VARIANTS[variant][0]
    if stage not in ("s1", "D"):
        raise SystemExit(f"{variant}: only stage-1 variants (s1 / D) are supported by run_all")
    plan = [
        ("translit", [["src.translit"]]),
        ("normalize", [["src.normalize", "--split", "train"], ["src.normalize", "--split", "test"]]),
        ("blocking", [["src.blocking", "generate", "--split", "train"], ["src.blocking", "generate", "--split", "test"]]),
        ("pruner", [["src.pruner", s] for s in ("train", "score", "rank", "grid", "prune")]),
        ("slim", [["src.slim", "apply"]]),
        ("features", [["src.features", "--split", "train"], ["src.features", "--split", "test"]]),
    ]
    if stage == "s1":
        plan.append(("stage1", [["src.train", "--tag", "v3s1"]]))
    else:
        plan += [("testlike", [["src.testlike", "build"]]),
                 ("stage1", [["src.train", "--tag", "D", "--train-split", "trainD"]])]
    plan.append(("output", [["src.variants", "--only", variant, "--out", str(OUTPUT)]]))
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the whole pipeline")
    ap.add_argument("--variant", default="04_s1_g15", choices=sorted(VARIANTS))
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
