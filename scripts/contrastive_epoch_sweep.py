"""Train contrastive model for extended epochs and evaluate each snapshot.

Trains for 150 epochs (3x default), saving snapshots every 10 epochs,
then evaluates each snapshot to find the optimal training length.

Usage:
    python scripts/contrastive_epoch_sweep.py \
        --dataset configs/datasets/nq_test_hallu_cor.json \
        --method configs/methods/contrastive.json \
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


def parse_layer_range(spec: str) -> list[int]:
    if "-" not in spec:
        return [int(spec)]
    start, end = spec.split("-")
    return list(range(int(start), int(end) + 1))


def evaluate_checkpoint(
    model, checkpoint_path, ap, dataset_cfg, method_cfg, device, seed,
    train_ds, test_ds,
):
    """Load a checkpoint into model and run OOD evaluation. Returns metrics dict."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    data_cfg = method_cfg["data"]
    eval_cfg = method_cfg["evaluation"]
    target_layers = data_cfg["target_layers"]

    from torch.utils.data import DataLoader
    from activation_research.metric_evaluator import MultiMetricHallucinationEvaluator

    train_ds_target = train_ds.slice_layers(target_layers)
    test_ds_target = test_ds.slice_layers(target_layers)

    train_loader = DataLoader(train_ds_target, batch_size=64, shuffle=False)
    eval_loader = DataLoader(test_ds_target, batch_size=64, shuffle=False)

    metrics_list = []
    for m in eval_cfg["metrics"]:
        if m == "knn":
            knn_params = dict(eval_cfg.get("knn_params", {}))
            knn_params["sample_seed"] = seed
            metrics_list.append(
                {"metric": "knn", "kwargs": knn_params, "train_selection": "all"}
            )
        else:
            metrics_list.append(m)

    evaluator = MultiMetricHallucinationEvaluator(
        activation_parser_df=ap.df,
        train_data_loader=train_loader,
        metrics=metrics_list,
        batch_size=eval_cfg.get("eval_batch_size", 256),
        sub_batch_size=eval_cfg.get("sub_batch_size", 64),
        device=device,
        num_workers=4,
        persistent_workers=False,
        outlier_class=dataset_cfg.get("outlier_class", 1),
    )

    return evaluator.compute(eval_loader, model)


def main():
    parser = argparse.ArgumentParser(description="Contrastive epoch sweep")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset config JSON")
    parser.add_argument("--method", type=str, required=True, help="Method config JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="runs/epoch_sweep")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Load configs
    with open(args.dataset) as f:
        dataset_cfg = json.load(f)
    with open(args.method) as f:
        method_cfg = json.load(f)

    # Resolve paths
    for path_key in ("inference_json", "activations_path", "eval_json", "raw_eval_jsonl"):
        if path_key in dataset_cfg and not os.path.isabs(dataset_cfg[path_key]):
            dataset_cfg[path_key] = os.path.join(str(project_root), dataset_cfg[path_key])

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Build output dir
    run_name = f"{dataset_cfg['name']}_seed{args.seed}_ep{args.max_epochs}"
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(os.path.join(output_dir, "artifacts"), exist_ok=True)
    logger.info(f"Output: {output_dir}")

    # Seed
    from utils.seeding import seed_everything
    seed_everything(args.seed)

    # Build ActivationParser
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

    # Build datasets
    data_cfg = method_cfg["data"]
    train_cfg = method_cfg["training"]
    relevant_layers = parse_layer_range(data_cfg["relevant_layers"])

    ds_kwargs = dict(
        relevant_layers=relevant_layers,
        num_views=data_cfg.get("num_views", 2),
        pad_length=data_cfg.get("pad_length", 63),
        preload=data_cfg.get("preload", True),
        include_response_logprobs=data_cfg.get("include_response_logprobs", False),
        response_logprobs_top_k=data_cfg.get("response_logprobs_top_k", 20),
        check_ram=False,
    )

    train_ds = ap.get_dataset("train", **ds_kwargs)
    test_ds = ap.get_dataset("test", **ds_kwargs)
    logger.info(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    steps_per_epoch = math.ceil(len(train_ds) / train_cfg["batch_size"])
    logger.info(f"Natural steps/epoch: {steps_per_epoch}, total steps: {steps_per_epoch * args.max_epochs}")

    # Build model
    from activation_research.model import ProgressiveCompressor
    model = ProgressiveCompressor(
        input_dim=dataset_cfg["input_dim"],
        final_dim=method_cfg["model_params"].get("final_dim", 512),
        input_dropout=method_cfg["model_params"].get("input_dropout", 0.3),
    )

    # Build trainer
    from activation_research.trainer import ContrastiveTrainer, ContrastiveTrainerConfig

    config = ContrastiveTrainerConfig(
        max_epochs=args.max_epochs,
        batch_size=train_cfg["batch_size"],
        lr=train_cfg["lr"],
        temperature=train_cfg["temperature"],
        grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
        use_labels=train_cfg.get("use_labels", False),
        ignore_label=train_cfg.get("ignore_label", -1),
        use_infinite_index_stream=train_cfg.get("use_infinite_index_stream", True),
        use_infinite_index_stream_eval=train_cfg.get("use_infinite_index_stream_eval", True),
        infinite_stream_seed=args.seed,
        infinite_eval_seed=args.seed,
        balanced_sampling=train_cfg.get("balanced_sampling", False),
        num_views=data_cfg.get("num_views", 2),
        device=device,
        num_workers=4,
        persistent_workers=True,
        checkpoint_dir=os.path.join(output_dir, "artifacts"),
        save_every=1,
        snapshot_every=args.snapshot_every,
        snapshot_keep_last=999,  # keep all snapshots
    )

    trainer = ContrastiveTrainer(model, config=config)
    trainer.fit(train_dataset=train_ds, val_dataset=test_ds)

    # Save final weights
    torch.save(
        {"model_state_dict": model.state_dict()},
        os.path.join(output_dir, "artifacts", "final_weights.pt"),
    )

    # --- Evaluate each snapshot ---
    import glob

    snapshot_dir = os.path.join(
        output_dir, "artifacts",
        "contrastivetrainer__progressivecompressor",
    )
    snapshots = sorted(glob.glob(os.path.join(snapshot_dir, "*.pt")))
    # Also evaluate final weights
    final_path = os.path.join(output_dir, "artifacts", "final_weights.pt")
    if os.path.exists(final_path):
        snapshots.append(final_path)

    logger.info(f"Evaluating {len(snapshots)} checkpoints...")

    results = []
    for snap_path in snapshots:
        snap_name = os.path.basename(snap_path)

        # Extract epoch from snapshot name
        if "epoch_" in snap_name:
            epoch = int(snap_name.split("epoch_")[1].split(".")[0]) + 1  # 0-indexed -> 1-indexed
        elif snap_name == "final_weights.pt":
            epoch = args.max_epochs
        else:
            epoch = -1

        logger.info(f"Evaluating {snap_name} (epoch {epoch})...")
        try:
            ood_stats = evaluate_checkpoint(
                model, snap_path, ap, dataset_cfg, method_cfg, device, args.seed,
                train_ds, test_ds,
            )

            result = {
                "epoch": epoch,
                "snapshot": snap_name,
                "cosine_auroc": ood_stats.get("cosine_auroc"),
                "mahalanobis_auroc": ood_stats.get("mahalanobis_auroc"),
                "knn_auroc": ood_stats.get("knn_auroc"),
            }
            results.append(result)
            logger.info(
                f"  Epoch {epoch}: "
                f"cosine={result['cosine_auroc']:.4f}, "
                f"mahal={result['mahalanobis_auroc']:.4f}, "
                f"knn={result['knn_auroc']:.4f}"
            )
        except Exception as e:
            logger.error(f"  Failed to evaluate {snap_name}: {e}")
            results.append({"epoch": epoch, "snapshot": snap_name, "error": str(e)})

    # Save results
    results_path = os.path.join(output_dir, "epoch_sweep_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Print summary table
    print("\n=== Epoch Sweep Results ===")
    print(f"{'Epoch':>6}  {'Cosine':>8}  {'Mahal':>8}  {'KNN':>8}")
    print("-" * 36)
    for r in results:
        if "error" in r:
            print(f"{r['epoch']:>6}  {'ERROR':>8}")
        else:
            print(
                f"{r['epoch']:>6}  "
                f"{r['cosine_auroc']:>8.4f}  "
                f"{r['mahalanobis_auroc']:>8.4f}  "
                f"{r['knn_auroc']:>8.4f}"
            )

    # Find best epoch per metric
    valid = [r for r in results if "error" not in r]
    if valid:
        best_knn = max(valid, key=lambda r: r["knn_auroc"])
        best_mahal = max(valid, key=lambda r: r["mahalanobis_auroc"])
        print(f"\nBest KNN:   epoch {best_knn['epoch']} ({best_knn['knn_auroc']:.4f})")
        print(f"Best Mahal: epoch {best_mahal['epoch']} ({best_mahal['mahalanobis_auroc']:.4f})")


if __name__ == "__main__":
    main()
