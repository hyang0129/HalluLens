"""
recover_act_vit_eval.py — finish a killed act_vit run by re-running val+test
evaluation from the saved best_checkpoint.pt.

Used when run_act_vit was interrupted between best-val-checkpoint save and the
final test-eval step. Reads <run-dir>/config.json (the per-run snapshot
written by run_experiment.py) so dataset, method, split-seed, and training
seed are exactly what the original run used.

Writes <run-dir>/eval_metrics.json and <run-dir>/predictions.csv in the same
schema run_experiment.py would have written them.

Usage:
    python scripts/recover_act_vit_eval.py \\
        --run-dir runs/baseline_comparison_hotpotqa_memmap/hotpotqa_memmap/act_vit/seed_0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="Path to the seed_X directory containing config.json "
                             "and artifacts/best_checkpoint.pt")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override eval batch size (default: training batch_size from config)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Override DataLoader num_workers")
    args = parser.parse_args()

    import torch
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    from activation_research.act_vit import ACTViT, ACTViTConfig
    from activation_research.act_vit_dataset import ACTViTDataset
    from activation_research.memmap_activation_parser import MemmapActivationParser

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"run-dir does not exist: {run_dir}", file=sys.stderr)
        return 2

    config_path = run_dir / "config.json"
    ckpt_path = run_dir / "artifacts" / "best_checkpoint.pt"
    if not config_path.exists():
        print(f"missing {config_path}", file=sys.stderr)
        return 2
    if not ckpt_path.exists():
        print(f"missing {ckpt_path}", file=sys.stderr)
        return 2

    cfg_blob = json.loads(config_path.read_text())
    dataset_cfg = cfg_blob["dataset"]
    method_cfg = cfg_blob["method"]
    experiment_cfg = cfg_blob["experiment"]
    training_seed = cfg_blob["training_seed"]
    split_seed = cfg_blob.get("split_seed", experiment_cfg.get("split_seed", 42))

    icr_cfg = dataset_cfg["icr_capture"]
    model_params = method_cfg.get("model_params", {})
    train_cfg = method_cfg["training"]
    batch_size = args.batch_size or train_cfg["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None \
        else experiment_cfg.get("num_workers", 4)
    persistent_workers = experiment_cfg.get("persistent_workers", True) and num_workers > 0
    prefetch_factor = 4 if num_workers > 0 else None

    # Rebuild splits identically to run_act_vit.
    train_parser = MemmapActivationParser(
        icr_cfg["train_dir"],
        random_seed=split_seed,
        split_strategy="three_way",
    )
    val_df = train_parser.df[train_parser.df["split"] == "val"]
    val_idx = val_df["sample_index"].values

    test_parser = MemmapActivationParser(
        icr_cfg["test_dir"],
        random_seed=split_seed,
        split_strategy="none",
    )
    test_idx = test_parser.df["sample_index"].values

    # n_train is not strictly needed for eval but we record it to match
    # the original eval_metrics.json schema.
    train_df = train_parser.df[train_parser.df["split"] == "train"]
    n_train = int(len(train_df))

    val_ds = ACTViTDataset(icr_cfg["train_dir"], val_idx)
    test_ds = ACTViTDataset(icr_cfg["test_dir"], test_idx)

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        persistent_workers=persistent_workers, pin_memory=True,
        prefetch_factor=prefetch_factor,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        persistent_workers=persistent_workers, pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    cfg = ACTViTConfig(
        input_dim=dataset_cfg["input_dim"],
        L_p=model_params.get("L_p", 8),
        N_p=model_params.get("N_p", 100),
        patch_h=model_params.get("patch_h", 2),
        patch_w=model_params.get("patch_w", 10),
        d_adapter=model_params.get("d_adapter", 256),
        d_model=model_params.get("d_model", 256),
        num_heads=model_params.get("num_heads", 8),
        depth=model_params.get("depth", 4),
        mlp_ratio=model_params.get("mlp_ratio", 4.0),
        dropout=model_params.get("dropout", 0.1),
    )
    model = ACTViT(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    best_epoch = int(ckpt.get("epoch", -1))
    model.eval()

    def _eval(loader):
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                x = batch["activations"].to(device, non_blocking=True)
                logits = model(x).squeeze(1)
                preds.append(torch.sigmoid(logits).cpu())
                labels.append(batch["label"].cpu())
        scores = torch.cat(preds).numpy()
        lbls = torch.cat(labels).numpy()
        auroc = float(roc_auc_score(lbls, scores)) if len(set(lbls.tolist())) >= 2 else float("nan")
        return auroc, scores, lbls

    val_auroc, _, _ = _eval(val_loader)
    test_auroc, test_scores, test_labels = _eval(test_loader)

    print(f"[recover] best_epoch={best_epoch}  val_auroc={val_auroc:.4f}  test_auroc={test_auroc:.4f}")

    eval_metrics = {
        "method": method_cfg["name"],
        "dataset": dataset_cfg["name"],
        "seed": training_seed,
        "split_seed": split_seed,
        "n_train": n_train,
        "n_val": int(len(val_ds)),
        "n_test": int(len(test_ds)),
        "auroc": test_auroc,
        "best_val_auroc": val_auroc,
        "best_epoch": best_epoch,
        "recovered_from_checkpoint": True,
    }

    eval_metrics_path = run_dir / "eval_metrics.json"
    with open(eval_metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)

    pred_path = run_dir / "predictions.csv"
    with open(pred_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["example_id", "score_halu", "label_halu", "split"])
        writer.writeheader()
        for i, (s, l) in enumerate(zip(test_scores, test_labels)):
            writer.writerow({
                "example_id": i,
                "score_halu": float(s),
                "label_halu": int(l),
                "split": "test",
            })

    print(f"[recover] wrote {eval_metrics_path}")
    print(f"[recover] wrote {pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
