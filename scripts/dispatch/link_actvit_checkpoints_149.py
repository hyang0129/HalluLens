"""link_actvit_checkpoints_149.py — expose existing act_vit checkpoints to the
issue #149 eval-only prefix curve, without retraining and without mutating the
canonical baseline runs.

``run_experiment`` derives its output directory from ``experiment_name``, and the
act_vit routine now skips training when ``artifacts/final_weights.pt`` already
exists. So to score an already-trained act_vit at k=16/32/48/64 we only need that
file to be present under the new experiment's path.

We **symlink** rather than copy, and we write into a NEW experiment directory
(``prefix149_actvit_<ds>``) rather than re-running inside
``baseline_comparison_*``. Re-running in place would rewrite the canonical
metrics.json that ``scripts/results_table.py`` reads — adding fields to the
single source of truth for published numbers as a side effect of an experiment.
A separate directory keeps the eval-only arm additive and reversible: delete the
directory and nothing else changed.

Usage:
    python scripts/dispatch/link_actvit_checkpoints_149.py           # link
    python scripts/dispatch/link_actvit_checkpoints_149.py --check   # report only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# (source experiment, dataset key, target experiment)
_TARGETS = [
    ("baseline_comparison_sciq_memmap", "sciq_memmap", "prefix149_actvit_sciq"),
    ("baseline_comparison_hotpotqa_memmap", "hotpotqa_memmap", "prefix149_actvit_hotpotqa"),
]
_SEEDS = (0, 1, 2, 3, 4)
_SRC_METHOD = "act_vit"
_DST_METHOD = "act_vit_prefix_eval"


def run(check_only: bool) -> int:
    runs = _PROJECT_ROOT / "runs"
    linked = missing = existing = 0
    for src_exp, ds, dst_exp in _TARGETS:
        for seed in _SEEDS:
            src = runs / src_exp / ds / _SRC_METHOD / f"seed_{seed}" / "artifacts" / "final_weights.pt"
            dst_dir = runs / dst_exp / ds / _DST_METHOD / f"seed_{seed}" / "artifacts"
            dst = dst_dir / "final_weights.pt"

            if not src.exists():
                print(f"  MISSING source: {src.relative_to(_PROJECT_ROOT)}", file=sys.stderr)
                missing += 1
                continue
            if dst.exists() or dst.is_symlink():
                existing += 1
                continue
            if check_only:
                print(f"  would link: {dst.relative_to(_PROJECT_ROOT)}")
                linked += 1
                continue

            dst_dir.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src.resolve())
            print(f"  linked: {dst.relative_to(_PROJECT_ROOT)} -> {src.relative_to(_PROJECT_ROOT)}")
            linked += 1

    verb = "would link" if check_only else "linked"
    print(f"\n{verb} {linked}, already present {existing}, missing sources {missing}")
    if missing:
        print(
            "Missing sources mean those seeds were never trained; the eval-only "
            "arm cannot cover them without a real act_vit training run.",
            file=sys.stderr,
        )
    return 1 if missing else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="report without creating links")
    return run(ap.parse_args().check)


if __name__ == "__main__":
    raise SystemExit(main())
