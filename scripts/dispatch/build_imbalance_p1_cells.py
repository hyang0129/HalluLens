"""Queue P1 imbalance-sweep cells (#140): one cell per (imbalance config, method).

Each cell runs all 3 seeds of one arm at one prevalence via run_experiment.py's
inner-loop resume. Filter by config-stem substring (e.g. 'hotpotqa') to stage one
dataset at a time.

    python scripts/dispatch/build_imbalance_p1_cells.py \
        --dispatch-root shared/issue_140_dispatch --filter imbalance_hotpotqa
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

_CFG_DIR = _ROOT / "configs" / "experiments"
_SEEDS = "0,1,2"
_LAST = 2


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
    for cfg_path in sorted(_CFG_DIR.glob("imbalance_*.json")):
        exp = cfg_path.stem
        if filt and not any(f in exp for f in filt):
            continue
        cfg = json.loads(cfg_path.read_text())
        ds = cfg["dataset"]
        for method in cfg["methods"]:
            cid = f"p1__{exp}__{method}"
            if _has(root, cid):
                print(f"  skip (queued): {cid}")
                continue
            cell = {
                "cell_id": cid,
                "experiment_config": f"configs/experiments/{exp}.json",
                "dataset": ds,
                "method": method,
                "seed": _SEEDS,
                "output_check": f"runs/{exp}/{ds}/{method}/seed_{_LAST}/predictions.csv",
            }
            (root / "pending" / f"{cid}.json").write_text(json.dumps(cell, indent=2),
                                                          encoding="utf-8")
            n += 1
            print(f"  queued: {cid}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dispatch-root", default="shared/issue_140_dispatch")
    ap.add_argument("--filter", default="", help="comma-sep config-stem substrings")
    a = ap.parse_args()
    root = Path(a.dispatch_root)
    if not root.is_absolute():
        root = _ROOT / root
    filt = [x for x in a.filter.split(",") if x]
    total = build(root, filt)
    print(f"Done — {total} P1 cells queued in {root}/pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
