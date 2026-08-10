"""build_issue_149_cells.py — queue the issue #149 early-detection pilot.

Question: the label says the whole generation is a hallucination — can we catch
it at token 16, before the model finishes saying it?

Grid: 2 datasets x 3 view geometries, 5 seeds bundled per cell.

    dataset  x  {prefix_mixed, prefix_only, layer_only}

The three methods are the ablation over the (layer, prefix) view space:

    prefix_mixed  positives differ in BOTH layer and prefix  (headline)
    prefix_only   positives differ in prefix at a fixed layer (time axis alone)
    layer_only    positives differ in layer at full length    (control; this is
                  the unmodified pre-#149 method, so its arm doubles as a
                  regression check that nothing changed for existing configs)

Every arm is evaluated at k = 16/32/48/64 by ``eval_prefix_curve``, so the
layer_only arm also supplies the "train on full length, test on a prefix"
regime — the deployability question — while the prefix arms supply the
"matched-k" regime.

Seeds are bundled inside a cell (``"seed": "0,1,2,3,4"``) so a re-claimed cell
resumes at the first seed lacking predictions.csv rather than redoing the whole
bundle, matching build_issue_135_cells.py.

Deliberately NOT queued here:
  - act_vit. Its activation tensor is adaptive-pooled to a fixed (L_p, N_p)
    grid, so evaluating a 64-token-tuned patch geometry at k=16 would handicap
    the strongest competitor. It needs a per-k retune first.
  - the k=0 prompt-only arm. It is not a response-token slice; it needs
    prompt_activations.npy and is a separate arm.

Usage:
    python scripts/dispatch/build_issue_149_cells.py \
        --dispatch-root shared/issue_149_dispatch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.dispatch.claim import init_dispatch_dirs  # noqa: E402

_METHODS = [
    "contrastive_logprob_recon_prefix_mixed",
    "contrastive_logprob_recon_prefix_only",
    "contrastive_logprob_recon",
]

# (experiment_name, dataset_key)
_TARGETS = [
    ("prefix149_sciq", "sciq_memmap"),
    ("prefix149_hotpotqa", "hotpotqa_memmap"),
]

_SEEDS_CSV = "0,1,2,3,4"
_LAST_SEED = 4


def _dispatch_has_cell(dispatch_root: Path, cell_id: str) -> bool:
    fname = cell_id + ".json"
    for sub in ("pending", "done", "failed"):
        if (dispatch_root / sub / fname).exists():
            return True
    claimed = dispatch_root / "claimed"
    if claimed.exists():
        for wd in claimed.iterdir():
            if wd.is_dir() and (wd / fname).exists():
                return True
    return False


def build(dispatch_root: Path) -> int:
    init_dispatch_dirs(dispatch_root)
    written = 0
    for experiment_name, dataset_key in _TARGETS:
        cfg_rel = f"configs/experiments/{experiment_name}.json"
        if not (_PROJECT_ROOT / cfg_rel).exists():
            print(f"  skip (config missing): {cfg_rel}", file=sys.stderr)
            continue
        for method in _METHODS:
            method_cfg = _PROJECT_ROOT / "configs" / "methods" / f"{method}.json"
            if not method_cfg.exists():
                print(f"  skip (method missing): {method}", file=sys.stderr)
                continue
            cell_id = f"1_issue149__{dataset_key}__{method}"
            if _dispatch_has_cell(dispatch_root, cell_id):
                print(f"  skip (already queued): {cell_id}")
                continue
            output_check = (
                f"runs/{experiment_name}/{dataset_key}/{method}/"
                f"seed_{_LAST_SEED}/predictions.csv"
            )
            cell = {
                "cell_id": cell_id,
                "experiment_config": cfg_rel,
                "dataset": dataset_key,
                "method": method,
                "seed": _SEEDS_CSV,
                "output_check": output_check,
            }
            (dispatch_root / "pending" / f"{cell_id}.json").write_text(
                json.dumps(cell, indent=2) + "\n"
            )
            written += 1
            print(f"  queued: {cell_id}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dispatch-root",
        default="shared/issue_149_dispatch",
        help="Dispatch queue root (default: shared/issue_149_dispatch)",
    )
    args = ap.parse_args()
    root = Path(args.dispatch_root)
    if not root.is_absolute():
        root = _PROJECT_ROOT / root
    n = build(root)
    print(f"\nqueued {n} cells under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
