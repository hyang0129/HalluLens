"""build_b5norm_cells.py — queue b5_norm (input-normalized contrastive) cells.

Tests the `normalize_input=true` lever (method `contrastive_logprob_recon_b5_norm`)
against plain b5, reusing the SAME flipped experiment configs (so same dataset,
seeds, and train/test splits → apples-to-apples vs the b5 grid).

One cell per matching flipped experiment config; the cell's seeds = that config's
own training_seeds (preserves the split_seeds pairing). run_experiment.py's
inner-loop resume skips seeds whose predictions.csv already exists.

Usage:
    # full grid:
    python scripts/dispatch/build_b5norm_cells.py --dispatch-root shared/issue_135_norm_dispatch
    # diagnostic subset (config-stem substrings, comma-sep):
    python scripts/dispatch/build_b5norm_cells.py --dispatch-root shared/issue_135_norm_dispatch \
        --filter popqa_flipped_memmap_fill234,sciq_flipped_memmap_fill234
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.dispatch.claim import init_dispatch_dirs  # noqa: E402

_METHOD = "contrastive_logprob_recon_b5_norm"
_CFG_DIR = _ROOT / "configs" / "experiments"


def _has(root: Path, cid: str) -> bool:
    fn = cid + ".json"
    if any((root / sub / fn).exists() for sub in ("pending", "done", "failed")):
        return True
    claimed = root / "claimed"
    return claimed.exists() and any(
        (wd / fn).exists() for wd in claimed.iterdir() if wd.is_dir())


def build(root: Path, filt: list[str]) -> int:
    init_dispatch_dirs(root)
    n = 0
    for cfg in sorted(_CFG_DIR.glob("baseline_comparison_*flipped*.json")):
        exp = cfg.stem
        if filt and not any(f in exp for f in filt):
            continue
        d = json.loads(cfg.read_text())
        ds = d.get("dataset")
        seeds = d.get("training_seeds") or []
        if not ds or not seeds:
            print(f"  skip (no dataset/seeds): {exp}", file=sys.stderr)
            continue
        seeds_csv = ",".join(str(s) for s in seeds)
        last = seeds[-1]
        cid = f"b5norm__{exp}"
        if _has(root, cid):
            print(f"  skip (already queued): {cid}")
            continue
        cell = {
            "cell_id": cid,
            "experiment_config": f"configs/experiments/{exp}.json",
            "dataset": ds,
            "method": _METHOD,
            "seed": seeds_csv,
            "output_check": f"runs/{exp}/{ds}/{_METHOD}/seed_{last}/predictions.csv",
        }
        (root / "pending" / f"{cid}.json").write_text(json.dumps(cell, indent=2),
                                                      encoding="utf-8")
        n += 1
        print(f"  queued: {cid}  (seeds {seeds_csv})")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Queue b5_norm cells (issue #135 input-norm lever).")
    ap.add_argument("--dispatch-root", default="shared/issue_135_norm_dispatch")
    ap.add_argument("--filter", default="", help="comma-sep config-stem substrings; empty = full grid")
    a = ap.parse_args()
    root = Path(a.dispatch_root)
    if not root.is_absolute():
        root = _ROOT / root
    filt = [x for x in a.filter.split(",") if x]
    total = build(root, filt)
    print(f"Done — {total} b5_norm cells queued in {root}/pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
