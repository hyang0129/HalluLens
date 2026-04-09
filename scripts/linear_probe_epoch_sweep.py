"""Train linear probe for extended epochs and evaluate each snapshot.

Usage:
    python scripts/linear_probe_epoch_sweep.py \
        --dataset configs/datasets/nq_test_hallu_cor.json \
        --method configs/methods/linear_probe.json \
        --seed 42 \
        --max-epochs 150 \
        --snapshot-every 10 \
        --output-dir runs/epoch_sweep
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from loguru import logger
from sklearn.metrics import roc_auc_score


def parse_layer_range(spec: str) -> list[int]:
    if "-" not in spec:
        return [int(spec)]
    start, end = spec.split("-")
    return list(range(int(start), int(end) + 1))


def evaluate_checkpoint(model, checkpoint_path, probe_test, batch_size, device):
    """Load checkpoint and compute AUROC."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    from torch.utils.data import DataLoader

    loader = DataLoader(probe_test, batch_size=batch_size, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["views_activations"].to(device)
            if x.dim() == 4:
                x = x.squeeze(1)
            preds = model(x)
            all_preds.append(preds.cpu())
            all_labels.append(batch["halu"].cpu())

    all_preds_np = torch.cat(all_preds).squeeze().numpy()
    all_labels_np = torch.cat(all_labels).numpy()
    if len(set(all_labels_np)) < 2:
        return float("nan")
    return float(roc_auc_score(all_labels_np, all_preds_np))


def main():
    parser = argparse.ArgumentParser(description="Linear probe epoch sweep")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="runs/epoch_sweep")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    with open(args.dataset) as f:
        dataset_cfg = json.load(f)
    with open(args.method) as f:
        method_cfg = json.load(f)

    for path_key in ("inference_json", "activations_path", "eval_json", "raw_eval_jsonl"):
        if path_key in dataset_cfg and not os.path.isabs(dataset_cfg[path_key]):
            dataset_cfg[path_key] = os.path.join(str(project_root), dataset_cfg[path_key])

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    run_name = f"linear_probe_{dataset_cfg['name']}_seed{args.seed}_ep{args.max_epochs}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(os.path.join(output_dir, "artifacts"), exist_ok=True)

    from utils.seeding import seed_everything
    seed_everything(args.seed)

    from activation_logging.activation_parser import ActivationParser

    ap = ActivationParser(
        inference_json=dataset_cfg["inference_json"],
        eval_json=dataset_cfg["eval_json"],
        activations_path=dataset_cfg["activations_path"],
        logger_type=dataset_cfg.get("backend", "zarr"),
        random_seed=42,
        split_strategy="two_way",
        verbose=True,
    )

    data_cfg = method_cfg["data"]
    train_cfg = method_cfg["training"]
    relevant_layers = parse_layer_range(data_cfg["relevant_layers"]) if "relevant_layers" in data_cfg else list(range(14, 30))
    probe_layer = data_cfg["probe_layer"]

    ds_kwargs = dict(
        relevant_layers=relevant_layers,
        num_views=2,
        preload=data_cfg.get("preload", True),
        include_response_logprobs=data_cfg.get("include_response_logprobs", False),
        response_logprobs_top_k=data_cfg.get("response_logprobs_top_k", 20),
        check_ram=False,
    )

    train_ds = ap.get_dataset("train", **ds_kwargs)
    test_ds = ap.get_dataset("test", **ds_kwargs)

    probe_train = train_ds.get_single_layer_dataset(probe_layer)
    probe_test = test_ds.get_single_layer_dataset(probe_layer)
    logger.info(f"Probe train: {len(probe_train)}, test: {len(probe_test)} (layer {probe_layer})")

    steps_per_epoch = math.ceil(len(probe_train) / train_cfg["batch_size"])
    logger.info(f"Natural steps/epoch: {steps_per_epoch}, total steps: {steps_per_epoch * args.max_epochs}")

    from activation_research.model import LinearProbe
    from activation_research.trainer import LinearProbeTrainer, LinearProbeTrainerConfig

    model = LinearProbe(
        input_dim=dataset_cfg["input_dim"],
        pooling=method_cfg["model_params"].get("pooling", "mean"),
    )

    config = LinearProbeTrainerConfig(
        max_epochs=args.max_epochs,
        batch_size=train_cfg["batch_size"],
        lr=train_cfg["lr"],
        steps_per_epoch_override=train_cfg.get("steps_per_epoch_override"),
        grad_clip_norm=train_cfg.get("grad_clip_norm"),
        balanced_sampling=train_cfg.get("balanced_sampling", True),
        pooling=method_cfg["model_params"].get("pooling", "mean"),
        device=device,
        num_workers=4,
        persistent_workers=True,
        checkpoint_dir=os.path.join(output_dir, "artifacts"),
        save_every=1,
        snapshot_every=args.snapshot_every,
        snapshot_keep_last=999,
    )

    trainer = LinearProbeTrainer(model, config=config)
    trainer.fit(train_dataset=probe_train, val_dataset=probe_test)

    torch.save(
        {"model_state_dict": model.state_dict()},
        os.path.join(output_dir, "artifacts", "final_weights.pt"),
    )

    # --- Evaluate each snapshot ---
    import glob

    snapshot_dir = os.path.join(
        output_dir, "artifacts",
        "linearprobetrainer__linearprobe",
    )
    snapshots = sorted(glob.glob(os.path.join(snapshot_dir, "*.pt")))
    final_path = os.path.join(output_dir, "artifacts", "final_weights.pt")
    if os.path.exists(final_path):
        snapshots.append(final_path)

    logger.info(f"Evaluating {len(snapshots)} checkpoints...")

    results = []
    for snap_path in snapshots:
        snap_name = os.path.basename(snap_path)

        if "epoch_" in snap_name:
            epoch = int(snap_name.split("epoch_")[1].split(".")[0]) + 1
        elif snap_name == "final_weights.pt":
            epoch = args.max_epochs
        else:
            epoch = -1

        logger.info(f"Evaluating {snap_name} (epoch {epoch})...")
        try:
            auroc = evaluate_checkpoint(
                model, snap_path, probe_test, train_cfg["batch_size"], device,
            )
            results.append({"epoch": epoch, "snapshot": snap_name, "auroc": auroc})
            logger.info(f"  Epoch {epoch}: auroc={auroc:.4f}")
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results.append({"epoch": epoch, "snapshot": snap_name, "error": str(e)})

    results_path = os.path.join(output_dir, "epoch_sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Linear Probe Epoch Sweep ===")
    print(f"{'Epoch':>6}  {'AUROC':>8}")
    print("-" * 18)
    for r in results:
        if "error" in r:
            print(f"{r['epoch']:>6}  {'ERROR':>8}")
        else:
            print(f"{r['epoch']:>6}  {r['auroc']:>8.4f}")

    valid = [r for r in results if "error" not in r]
    if valid:
        best = max(valid, key=lambda r: r["auroc"])
        print(f"\nBest: epoch {best['epoch']} ({best['auroc']:.4f})")


if __name__ == "__main__":
    main()
