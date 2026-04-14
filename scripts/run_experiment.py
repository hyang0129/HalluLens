"""Config-driven experiment runner for reproducible multi-seed benchmarks.

Loads experiment/dataset/method configs and orchestrates training + evaluation
across multiple seeds, writing structured output for aggregation.

Usage:
    # Full experiment
    python scripts/run_experiment.py --experiment configs/experiments/baseline_comparison_hotpotqa.json

    # Override seeds
    python scripts/run_experiment.py --experiment configs/experiments/baseline_comparison_hotpotqa.json --seeds 0,42

    # Single run (no experiment config needed)
    python scripts/run_experiment.py --dataset configs/datasets/hotpotqa_train.json --method configs/methods/contrastive.json --seed 42

    # Force re-run
    python scripts/run_experiment.py --experiment configs/experiments/baseline_comparison_hotpotqa.json --force

    # Override max epochs (for quick testing)
    python scripts/run_experiment.py --experiment configs/experiments/baseline_comparison_hotpotqa.json --max-epochs 1
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from loguru import logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_layer_range(spec: str) -> list[int]:
    """Parse a layer range string like '14-29' into a list of ints.

    Also accepts a single integer string like '22'.
    """
    if "-" not in spec:
        return [int(spec)]
    start, end = spec.split("-")
    return list(range(int(start), int(end) + 1))


def write_run_manifest(output_dir: str) -> None:
    """Write run_manifest.json with environment metadata."""
    manifest = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _safe_git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_safe_git("status", "--porcelain")),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        ),
        "hostname": platform.node(),
        "command": " ".join(sys.argv),
    }
    with open(os.path.join(output_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def _safe_git(*args: str) -> str:
    """Run a git command, returning '' on failure."""
    try:
        return (
            subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------


def run_contrastive(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    training_seed: int,
    test_ap=None,
) -> tuple[dict, list[dict]]:
    """Train and evaluate a contrastive (ProgressiveCompressor) model."""
    data_cfg = method_cfg["data"]
    train_cfg = method_cfg["training"]
    eval_cfg = method_cfg["evaluation"]

    relevant_layers = parse_layer_range(data_cfg["relevant_layers"])
    target_layers = data_cfg["target_layers"]

    # Common dataset kwargs
    ds_kwargs = dict(
        relevant_layers=relevant_layers,
        num_views=data_cfg.get("num_views", 2),
        pad_length=data_cfg.get("pad_length", 63),
        preload=data_cfg.get("preload", True),
        include_response_logprobs=data_cfg.get("include_response_logprobs", False),
        response_logprobs_top_k=data_cfg.get("response_logprobs_top_k", 20),
        check_ram=False,
    )

    # Build datasets
    train_ds = ap.get_dataset("train", **ds_kwargs)
    eval_ap = test_ap if test_ap is not None else ap
    test_ds = eval_ap.get_dataset("test", **ds_kwargs)

    # Use val split for training validation when available (avoids data leakage)
    has_val = ap.split_strategy == "three_way"
    val_ds = ap.get_dataset("val", **ds_kwargs) if has_val else test_ds

    # Build model
    from activation_research.model import ProgressiveCompressor

    model = ProgressiveCompressor(
        input_dim=dataset_cfg["input_dim"],
        final_dim=method_cfg["model_params"].get("final_dim", 512),
        input_dropout=method_cfg["model_params"].get("input_dropout", 0.3),
    )

    # Build trainer
    from activation_research.trainer import ContrastiveTrainer, ContrastiveTrainerConfig

    checkpoint_dir = os.path.join(output_dir, "artifacts")
    config = ContrastiveTrainerConfig(
        max_epochs=train_cfg["max_epochs"],
        batch_size=train_cfg["batch_size"],
        lr=train_cfg["lr"],
        temperature=train_cfg["temperature"],
        steps_per_epoch_override=train_cfg.get("steps_per_epoch_override"),
        min_total_steps=train_cfg.get("min_total_steps"),
        grad_clip_norm=train_cfg.get("grad_clip_norm"),
        use_labels=train_cfg.get("use_labels", False),
        ignore_label=train_cfg.get("ignore_label", -1),
        use_infinite_index_stream=train_cfg.get("use_infinite_index_stream", True),
        use_infinite_index_stream_eval=train_cfg.get(
            "use_infinite_index_stream_eval", True
        ),
        infinite_stream_seed=training_seed,
        infinite_eval_seed=training_seed,
        balanced_sampling=train_cfg.get("balanced_sampling", False),
        num_views=data_cfg.get("num_views", 2),
        device=device,
        num_workers=experiment_cfg.get("num_workers", 4),
        persistent_workers=experiment_cfg.get("persistent_workers", True),
        checkpoint_dir=checkpoint_dir,
        save_every=1,
        snapshot_every=10,
        snapshot_keep_last=3,
    )

    trainer = ContrastiveTrainer(model, config=config)
    trainer.fit(train_dataset=train_ds, val_dataset=val_ds)

    # Save final weights
    torch.save(
        {"model_state_dict": model.state_dict()},
        os.path.join(output_dir, "artifacts", "final_weights.pt"),
    )

    # OOD evaluation on target layers
    from torch.utils.data import DataLoader

    from activation_research.metric_evaluator import MultiMetricHallucinationEvaluator

    train_ds_target = train_ds.slice_layers(target_layers)
    test_ds_target = test_ds.slice_layers(target_layers)

    train_loader = DataLoader(train_ds_target, batch_size=64, shuffle=False)
    eval_loader = DataLoader(test_ds_target, batch_size=64, shuffle=False)

    model.eval()

    # Build metrics list from config
    metrics_list: list = []
    for m in eval_cfg["metrics"]:
        if m == "knn":
            knn_params = dict(eval_cfg.get("knn_params", {}))
            knn_params["sample_seed"] = training_seed
            metrics_list.append(
                {
                    "metric": "knn",
                    "kwargs": knn_params,
                    "train_selection": "all",
                }
            )
        else:
            metrics_list.append(m)

    evaluator = MultiMetricHallucinationEvaluator(
        activation_parser_df=eval_ap.df,
        train_data_loader=train_loader,
        metrics=metrics_list,
        batch_size=eval_cfg.get("eval_batch_size", 256),
        sub_batch_size=eval_cfg.get("sub_batch_size", 64),
        device=device,
        num_workers=experiment_cfg.get("num_workers", 4),
        persistent_workers=False,
        outlier_class=dataset_cfg.get("outlier_class", 1),
    )

    ood_stats = evaluator.compute(eval_loader, model)

    # Build eval_metrics
    eval_metrics: dict = {
        "method": method_cfg["name"],
        "dataset": dataset_cfg["name"],
        "seed": training_seed,
        "split_seed": experiment_cfg.get("split_seed", 42),
        "n_train": len(train_ds),
        "n_test": len(test_ds),
    }
    eval_metrics.update(ood_stats)

    # Contrastive OOD doesn't produce per-example predictions easily
    predictions: list[dict] = []
    return eval_metrics, predictions


def run_contrastive_logprob_recon(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    training_seed: int,
    test_ap=None,
) -> tuple[dict, list[dict]]:
    """Train and evaluate a LogprobReconProgressiveCompressor model."""
    data_cfg = method_cfg["data"]
    train_cfg = method_cfg["training"]
    eval_cfg = method_cfg["evaluation"]

    relevant_layers = parse_layer_range(data_cfg["relevant_layers"])
    target_layers = data_cfg["target_layers"]

    ds_kwargs = dict(
        relevant_layers=relevant_layers,
        num_views=data_cfg.get("num_views", 2),
        pad_length=data_cfg.get("pad_length", 63),
        preload=data_cfg.get("preload", True),
        include_response_logprobs=True,
        response_logprobs_top_k=data_cfg.get("response_logprobs_top_k", 20),
        check_ram=False,
    )

    train_ds = ap.get_dataset("train", **ds_kwargs)
    eval_ap = test_ap if test_ap is not None else ap
    test_ds = eval_ap.get_dataset("test", **ds_kwargs)

    has_val = ap.split_strategy == "three_way"
    val_ds = ap.get_dataset("val", **ds_kwargs) if has_val else test_ds

    from activation_research.model import LogprobReconProgressiveCompressor

    model_params = method_cfg.get("model_params", {})
    model = LogprobReconProgressiveCompressor(
        input_dim=dataset_cfg["input_dim"],
        final_dim=model_params.get("final_dim", 512),
        input_dropout=model_params.get("input_dropout", 0.3),
        recon_seq_len=model_params.get("recon_seq_len", 64),
        recon_hidden_dim=model_params.get("recon_hidden_dim", 256),
        recon_lambda=model_params.get("recon_lambda", 1.0),
        logprob_var_threshold=model_params.get("logprob_var_threshold", 1e-4),
    )

    train_device = device if device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    from activation_research.training import train_contrastive_logprob_recon

    checkpoint_dir = os.path.join(output_dir, "artifacts")
    train_contrastive_logprob_recon(
        model=model,
        train_dataset=train_ds,
        test_dataset=val_ds,
        epochs=train_cfg["max_epochs"],
        batch_size=train_cfg["batch_size"],
        lr=train_cfg["lr"],
        temperature=train_cfg["temperature"],
        device=train_device,
        num_workers=experiment_cfg.get("num_workers", 4),
        sub_batch_size=train_cfg.get("sub_batch_size", 64),
        checkpoint_dir=checkpoint_dir,
        save_every=1,
        snapshot_every=10,
        snapshot_keep_last=3,
        use_labels=train_cfg.get("use_labels", False),
        ignore_label=train_cfg.get("ignore_label", -1),
        persistent_workers=experiment_cfg.get("persistent_workers", True),
        recon_lambda=model_params.get("recon_lambda", 1.0),
        use_infinite_index_stream=train_cfg.get("use_infinite_index_stream", True),
        infinite_stream_shuffle=True,
        infinite_stream_seed=training_seed,
        steps_per_epoch_override=train_cfg.get("steps_per_epoch_override"),
        balanced_sampling=train_cfg.get("balanced_sampling", False),
        grad_clip_norm=train_cfg.get("grad_clip_norm"),
    )

    torch.save(
        {"model_state_dict": model.state_dict()},
        os.path.join(output_dir, "artifacts", "final_weights.pt"),
    )

    # OOD evaluation on target layers
    from torch.utils.data import DataLoader

    from activation_research.metric_evaluator import MultiMetricHallucinationEvaluator

    train_ds_target = train_ds.slice_layers(target_layers)
    test_ds_target = test_ds.slice_layers(target_layers)

    train_loader = DataLoader(train_ds_target, batch_size=64, shuffle=False)
    eval_loader = DataLoader(test_ds_target, batch_size=64, shuffle=False)

    model.eval()

    metrics_list: list = []
    for m in eval_cfg["metrics"]:
        if m == "knn":
            knn_params = dict(eval_cfg.get("knn_params", {}))
            knn_params["sample_seed"] = training_seed
            metrics_list.append(
                {
                    "metric": "knn",
                    "kwargs": knn_params,
                    "train_selection": "all",
                }
            )
        else:
            metrics_list.append(m)

    evaluator = MultiMetricHallucinationEvaluator(
        activation_parser_df=eval_ap.df,
        train_data_loader=train_loader,
        metrics=metrics_list,
        batch_size=eval_cfg.get("eval_batch_size", 256),
        sub_batch_size=eval_cfg.get("sub_batch_size", 64),
        device=train_device,
        num_workers=experiment_cfg.get("num_workers", 4),
        persistent_workers=False,
        outlier_class=dataset_cfg.get("outlier_class", 1),
    )

    ood_stats = evaluator.compute(eval_loader, model)

    eval_metrics: dict = {
        "method": method_cfg["name"],
        "dataset": dataset_cfg["name"],
        "seed": training_seed,
        "split_seed": experiment_cfg.get("split_seed", 42),
        "n_train": len(train_ds),
        "n_test": len(test_ds),
    }
    eval_metrics.update(ood_stats)

    predictions: list[dict] = []
    return eval_metrics, predictions


def run_linear_probe(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    training_seed: int,
    test_ap=None,
) -> tuple[dict, list[dict]]:
    """Train and evaluate a linear probe on a single layer."""
    data_cfg = method_cfg["data"]
    train_cfg = method_cfg["training"]

    relevant_layers = (
        parse_layer_range(data_cfg["relevant_layers"])
        if "relevant_layers" in data_cfg
        else list(range(14, 30))
    )
    probe_layer = data_cfg["probe_layer"]

    # Common dataset kwargs
    ds_kwargs = dict(
        relevant_layers=relevant_layers,
        num_views=2,
        preload=data_cfg.get("preload", True),
        include_response_logprobs=data_cfg.get("include_response_logprobs", False),
        response_logprobs_top_k=data_cfg.get("response_logprobs_top_k", 20),
        check_ram=False,
    )

    # Build full datasets
    train_ds = ap.get_dataset("train", **ds_kwargs)
    eval_ap = test_ap if test_ap is not None else ap
    test_ds = eval_ap.get_dataset("test", **ds_kwargs)

    # Use val split for training validation when available (avoids data leakage)
    has_val = ap.split_strategy == "three_way"
    val_ds = ap.get_dataset("val", **ds_kwargs) if has_val else test_ds

    # Get single-layer datasets
    probe_train = train_ds.get_single_layer_dataset(probe_layer)
    probe_test = test_ds.get_single_layer_dataset(probe_layer)
    probe_val = val_ds.get_single_layer_dataset(probe_layer) if has_val else probe_test

    from activation_research.model import LinearProbe
    from activation_research.trainer import LinearProbeTrainer, LinearProbeTrainerConfig

    model = LinearProbe(
        input_dim=dataset_cfg["input_dim"],
        pooling=method_cfg["model_params"].get("pooling", "mean"),
    )

    checkpoint_dir = os.path.join(output_dir, "artifacts")
    config = LinearProbeTrainerConfig(
        max_epochs=train_cfg["max_epochs"],
        batch_size=train_cfg["batch_size"],
        lr=train_cfg["lr"],
        steps_per_epoch_override=train_cfg.get("steps_per_epoch_override"),
        min_total_steps=train_cfg.get("min_total_steps"),
        grad_clip_norm=train_cfg.get("grad_clip_norm"),
        balanced_sampling=train_cfg.get("balanced_sampling", True),
        use_infinite_index_stream=train_cfg.get("use_infinite_index_stream", False),
        infinite_stream_seed=training_seed,
        pooling=method_cfg["model_params"].get("pooling", "mean"),
        device=device,
        num_workers=experiment_cfg.get("num_workers", 4),
        persistent_workers=experiment_cfg.get("persistent_workers", True),
        checkpoint_dir=checkpoint_dir,
        save_every=1,
    )

    trainer = LinearProbeTrainer(model, config=config)
    trainer.fit(train_dataset=probe_train, val_dataset=probe_val)

    # Save final weights
    torch.save(
        {"model_state_dict": model.state_dict()},
        os.path.join(output_dir, "artifacts", "final_weights.pt"),
    )

    # Evaluate
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    model.eval()
    eval_device = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(eval_device)

    probe_eval_loader = DataLoader(
        probe_test, batch_size=train_cfg["batch_size"], shuffle=False
    )
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in probe_eval_loader:
            x = batch["views_activations"].to(eval_device)
            if x.dim() == 4:
                x = x.squeeze(1)
            preds = model(x)
            all_preds.append(preds.cpu())
            all_labels.append(batch["halu"].cpu())

    all_preds_np = torch.cat(all_preds).squeeze().numpy()
    all_labels_np = torch.cat(all_labels).numpy()
    if len(set(all_labels_np)) < 2:
        auroc = float("nan")
    else:
        auroc = roc_auc_score(all_labels_np, all_preds_np)

    eval_metrics = {
        "method": method_cfg["name"],
        "dataset": dataset_cfg["name"],
        "seed": training_seed,
        "split_seed": experiment_cfg.get("split_seed", 42),
        "n_train": len(probe_train),
        "n_test": len(probe_test),
        "auroc": float(auroc),
        "probe_layer": probe_layer,
    }

    predictions = [
        {"example_id": i, "score_halu": float(s), "label_halu": int(l)}
        for i, (s, l) in enumerate(zip(all_preds_np, all_labels_np))
    ]

    return eval_metrics, predictions


def run_multi_layer_linear_probe(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    training_seed: int,
    test_ap=None,
) -> tuple[dict, list[dict]]:
    """Train and evaluate a multi-layer linear probe on all relevant layers."""
    data_cfg = method_cfg["data"]
    train_cfg = method_cfg["training"]

    relevant_layers = (
        parse_layer_range(data_cfg["relevant_layers"])
        if "relevant_layers" in data_cfg
        else list(range(14, 30))
    )

    # Common dataset kwargs
    ds_kwargs = dict(
        relevant_layers=relevant_layers,
        num_views=2,
        preload=data_cfg.get("preload", True),
        include_response_logprobs=data_cfg.get("include_response_logprobs", False),
        response_logprobs_top_k=data_cfg.get("response_logprobs_top_k", 20),
        check_ram=False,
    )

    # Build full datasets
    train_ds = ap.get_dataset("train", **ds_kwargs)
    eval_ap = test_ap if test_ap is not None else ap
    test_ds = eval_ap.get_dataset("test", **ds_kwargs)

    # Use val split for training validation when available (avoids data leakage)
    has_val = ap.split_strategy == "three_way"
    val_ds = ap.get_dataset("val", **ds_kwargs) if has_val else test_ds

    # Get deterministic multi-layer datasets
    ml_train = train_ds.get_multi_layer_dataset(relevant_layers)
    ml_test = test_ds.get_multi_layer_dataset(relevant_layers)
    ml_val = val_ds.get_multi_layer_dataset(relevant_layers) if has_val else ml_test

    from activation_research.model import MultiLayerLinearProbe
    from activation_research.trainer import LinearProbeTrainer, LinearProbeTrainerConfig

    model = MultiLayerLinearProbe(
        input_dim=dataset_cfg["input_dim"],
        num_layers=len(relevant_layers),
        pooling=method_cfg["model_params"].get("pooling", "mean"),
    )

    checkpoint_dir = os.path.join(output_dir, "artifacts")
    config = LinearProbeTrainerConfig(
        max_epochs=train_cfg["max_epochs"],
        batch_size=train_cfg["batch_size"],
        lr=train_cfg["lr"],
        steps_per_epoch_override=train_cfg.get("steps_per_epoch_override"),
        min_total_steps=train_cfg.get("min_total_steps"),
        grad_clip_norm=train_cfg.get("grad_clip_norm"),
        balanced_sampling=train_cfg.get("balanced_sampling", True),
        use_infinite_index_stream=train_cfg.get("use_infinite_index_stream", False),
        infinite_stream_seed=training_seed,
        pooling=method_cfg["model_params"].get("pooling", "mean"),
        device=device,
        num_workers=experiment_cfg.get("num_workers", 4),
        persistent_workers=experiment_cfg.get("persistent_workers", True),
        checkpoint_dir=checkpoint_dir,
        save_every=1,
    )

    trainer = LinearProbeTrainer(model, config=config)
    trainer.fit(train_dataset=ml_train, val_dataset=ml_val)

    # Save final weights
    torch.save(
        {"model_state_dict": model.state_dict()},
        os.path.join(output_dir, "artifacts", "final_weights.pt"),
    )

    # Evaluate
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    model.eval()
    eval_device = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(eval_device)

    eval_loader = DataLoader(
        ml_test, batch_size=train_cfg["batch_size"], shuffle=False
    )
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in eval_loader:
            x = batch["views_activations"].to(eval_device)
            preds = model(x)
            all_preds.append(preds.cpu())
            all_labels.append(batch["halu"].cpu())

    all_preds_np = torch.cat(all_preds).squeeze().numpy()
    all_labels_np = torch.cat(all_labels).numpy()
    if len(set(all_labels_np)) < 2:
        auroc = float("nan")
    else:
        auroc = roc_auc_score(all_labels_np, all_preds_np)

    eval_metrics = {
        "method": method_cfg["name"],
        "dataset": dataset_cfg["name"],
        "seed": training_seed,
        "split_seed": experiment_cfg.get("split_seed", 42),
        "n_train": len(ml_train),
        "n_test": len(ml_test),
        "auroc": float(auroc),
        "relevant_layers": relevant_layers,
    }

    predictions = [
        {"example_id": i, "score_halu": float(s), "label_halu": int(l)}
        for i, (s, l) in enumerate(zip(all_preds_np, all_labels_np))
    ]

    return eval_metrics, predictions


def run_token_entropy(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    test_ap=None,
) -> tuple[dict, list[dict]]:
    """Run token-entropy baseline (no training)."""
    data_cfg = method_cfg["data"]
    relevant_layers = parse_layer_range(data_cfg["relevant_layers"])

    eval_ap = test_ap if test_ap is not None else ap
    test_ds = eval_ap.get_dataset(
        "test",
        relevant_layers=relevant_layers,
        num_views=data_cfg.get("num_views", 2),
        preload=data_cfg.get("preload", True),
        include_response_logprobs=True,
        response_logprobs_top_k=data_cfg.get("response_logprobs_top_k", 20),
        check_ram=False,
    )

    from activation_research.token_entropy import TokenEntropyDetector

    detector = TokenEntropyDetector(
        outlier_class=dataset_cfg.get("outlier_class", 1)
    )
    stats = detector.score(
        test_ds,
        batch_size=256,
        num_workers=experiment_cfg.get("num_workers", 4),
    )

    eval_metrics: dict = {
        "method": method_cfg["name"],
        "dataset": dataset_cfg["name"],
        "seed": None,
        "split_seed": experiment_cfg.get("split_seed", 42),
        "n_test": len(test_ds),
    }
    eval_metrics.update(stats)

    return eval_metrics, []


def run_logprob_baseline(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    test_ap=None,
) -> tuple[dict, list[dict]]:
    """Run logprob baseline (no training)."""
    data_cfg = method_cfg["data"]
    relevant_layers = parse_layer_range(data_cfg["relevant_layers"])

    eval_ap = test_ap if test_ap is not None else ap
    test_ds = eval_ap.get_dataset(
        "test",
        relevant_layers=relevant_layers,
        num_views=data_cfg.get("num_views", 2),
        preload=data_cfg.get("preload", True),
        include_response_logprobs=True,
        response_logprobs_top_k=data_cfg.get("response_logprobs_top_k", 20),
        check_ram=False,
    )

    # Collect records from dataset
    from torch.utils.data import DataLoader

    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    records = []
    for batch in loader:
        bsz = batch["halu"].shape[0]
        for i in range(bsz):
            records.append(
                {
                    "halu": int(batch["halu"][i].item()),
                    "response_token_logprobs": batch["response_token_logprobs"][i],
                    "response_logprob_mask": batch["response_logprob_mask"][i],
                }
            )

    from activation_research.baselines import logprob_baseline_auroc

    stats = logprob_baseline_auroc(
        records, outlier_class=dataset_cfg.get("outlier_class", 1)
    )

    eval_metrics: dict = {
        "method": method_cfg["name"],
        "dataset": dataset_cfg["name"],
        "seed": None,
        "split_seed": experiment_cfg.get("split_seed", 42),
        "n_test": len(test_ds),
    }
    eval_metrics.update(stats)

    return eval_metrics, []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Config-driven experiment runner for reproducible multi-seed benchmarks.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Path to experiment config JSON (e.g. configs/experiments/baseline_comparison_hotpotqa.json)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset config JSON (single-run mode)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Path to method config JSON (single-run mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single training seed (single-run mode)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated training seeds, overrides experiment config",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if eval_metrics.json already exists",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max_epochs for all methods (useful for quick testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda, cpu, auto)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run plan (expected/complete/failed/pending) and exit without executing",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Pre-loaded config lookups (populated in single-run mode)
    _preloaded_dataset_cfgs: dict = {}
    _preloaded_method_cfgs: dict = {}

    # ---- Load configs ----
    if args.experiment:
        with open(args.experiment) as f:
            experiment_cfg = json.load(f)

        dataset_value = experiment_cfg.get("dataset") or experiment_cfg.get("datasets")
        if dataset_value is None:
            logger.error("Experiment config must have 'dataset' or 'datasets' key")
            sys.exit(1)
        if isinstance(dataset_value, str):
            datasets = [dataset_value]
        else:
            datasets = list(dataset_value)

        methods = experiment_cfg["methods"]

        if args.seeds is not None:
            training_seeds = [int(s) for s in args.seeds.split(",")]
        else:
            training_seeds = experiment_cfg.get("training_seeds", [42])
    elif args.dataset and args.method:
        with open(args.dataset) as f:
            single_dataset_cfg = json.load(f)
        with open(args.method) as f:
            single_method_cfg = json.load(f)

        datasets = [single_dataset_cfg["name"]]
        methods = [single_method_cfg["name"]]
        training_seeds = [args.seed] if args.seed is not None else [42]

        # Store pre-loaded configs so the main loop doesn't re-load from hardcoded paths
        _preloaded_dataset_cfgs[single_dataset_cfg["name"]] = single_dataset_cfg
        _preloaded_method_cfgs[single_method_cfg["name"]] = single_method_cfg

        experiment_cfg = {
            "experiment_name": f"{single_dataset_cfg['name']}_{single_method_cfg['name']}",
            "split_seed": 42,
            "training_seeds": training_seeds,
            "device": "auto",
            "num_workers": 4,
            "persistent_workers": True,
            "output_dir": "runs",
        }
    else:
        logger.error(
            "Must provide either --experiment or both --dataset and --method"
        )
        sys.exit(1)

    max_epochs_override = args.max_epochs

    # ---- Resolve device ----
    device = args.device or experiment_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")


    output_base = experiment_cfg.get("output_dir", "runs")
    exp_name = experiment_cfg.get("experiment_name", "default")
    split_strategy = experiment_cfg.get("split_strategy", "two_way")

    # ---- Dry-run mode: show plan and exit ----
    if args.dry_run:
        from scripts.experiment_utils import (
            RunStatus,
            classify_run,
            enumerate_runs,
            load_experiment_config,
        )

        if args.experiment:
            exp_cfg_loaded = load_experiment_config(
                args.experiment, project_root=str(project_root)
            )
        else:
            # Single-run mode: build a minimal config for enumerate_runs
            exp_cfg_loaded = dict(experiment_cfg)
            exp_cfg_loaded["datasets"] = datasets
            exp_cfg_loaded["methods"] = methods
            exp_cfg_loaded["training_seeds"] = training_seeds
            # Load method configs for is_learned detection
            method_configs = {}
            for m in methods:
                if m in _preloaded_method_cfgs:
                    method_configs[m] = _preloaded_method_cfgs[m]
                else:
                    mcfg_path = os.path.join(
                        str(project_root), "configs", "methods", f"{m}.json"
                    )
                    if os.path.exists(mcfg_path):
                        with open(mcfg_path) as f:
                            method_configs[m] = json.load(f)
            exp_cfg_loaded["method_configs"] = method_configs

        run_specs = enumerate_runs(
            exp_cfg_loaded,
            output_base=output_base,
            project_root=str(project_root),
        )

        complete = failed = running = pending = 0
        for spec in run_specs:
            status = classify_run(spec.run_dir)
            seed_str = f"seed={spec.seed}" if spec.seed is not None else "no-seed"
            symbol = {
                RunStatus.COMPLETE: "✓",
                RunStatus.FAILED: "✗",
                RunStatus.RUNNING: "~",
                RunStatus.PENDING: "·",
            }[status]
            logger.info(f"  {symbol} {spec.dataset_name}/{spec.method_name}/{seed_str} -> {status.value}")
            if status == RunStatus.COMPLETE:
                complete += 1
            elif status == RunStatus.FAILED:
                failed += 1
            elif status == RunStatus.RUNNING:
                running += 1
            else:
                pending += 1

        total = len(run_specs)
        logger.info(
            f"\nOverall: {complete}/{total} complete, {failed} failed, "
            f"{running} running, {pending} pending"
        )
        sys.exit(0)

    # ---- Lazy import heavy deps ----
    from activation_logging.activation_parser import ActivationParser

    # ---- Main loop ----
    for dataset_name in datasets:
        # Load dataset config (use pre-loaded if available, else load from configs/)
        if dataset_name in _preloaded_dataset_cfgs:
            dataset_cfg = _preloaded_dataset_cfgs[dataset_name]
        else:
            dataset_cfg_path = os.path.join(
                str(project_root), "configs", "datasets", f"{dataset_name}.json"
            )
            with open(dataset_cfg_path) as f:
                dataset_cfg = json.load(f)

        def _resolve_paths(cfg, root):
            """Resolve relative paths in a dataset config dict."""
            for path_key in ("inference_json", "activations_path", "eval_json", "raw_eval_jsonl"):
                if path_key in cfg and not os.path.isabs(cfg[path_key]):
                    cfg[path_key] = os.path.join(str(root), cfg[path_key])

        def _build_eval_json(cfg):
            """Build eval JSON for ActivationParser if it doesn't exist."""
            eval_json_path = cfg["eval_json"]
            if not os.path.exists(eval_json_path):
                from utils.eval_builder import build_eval_for_activation_parser
                build_eval_for_activation_parser(
                    cfg["inference_json"],
                    cfg.get("eval_json", ""),
                    cfg.get("raw_eval_jsonl", ""),
                    eval_json_path,
                )
            return eval_json_path

        # Detect unified config format (has "train" and/or "test" sub-keys)
        has_train_test = "train" in dataset_cfg and isinstance(dataset_cfg["train"], dict)

        test_ap = None
        if has_train_test:
            # Unified config: separate train and test ActivationParsers
            train_cfg = {**dataset_cfg, **dataset_cfg["train"]}
            _resolve_paths(train_cfg, project_root)
            train_eval_json = _build_eval_json(train_cfg)

            logger.info(f"Loading train ActivationParser from: {train_cfg['activations_path']}")
            ap = ActivationParser(
                inference_json=train_cfg["inference_json"],
                eval_json=train_eval_json,
                activations_path=train_cfg["activations_path"],
                logger_type=dataset_cfg.get("backend", "zarr"),
                random_seed=experiment_cfg.get("split_seed", 42),
                split_strategy=split_strategy,
                verbose=True,
            )

            if "test" in dataset_cfg and isinstance(dataset_cfg["test"], dict):
                test_cfg = {**dataset_cfg, **dataset_cfg["test"]}
                _resolve_paths(test_cfg, project_root)
                test_eval_json = _build_eval_json(test_cfg)

                logger.info(f"Loading test ActivationParser from: {test_cfg['activations_path']}")
                test_ap = ActivationParser(
                    inference_json=test_cfg["inference_json"],
                    eval_json=test_eval_json,
                    activations_path=test_cfg["activations_path"],
                    logger_type=dataset_cfg.get("backend", "zarr"),
                    random_seed=experiment_cfg.get("split_seed", 42),
                    split_strategy="none",
                    verbose=True,
                )
        else:
            # Legacy flat config: single ActivationParser with internal splitting
            _resolve_paths(dataset_cfg, project_root)
            eval_json_path = _build_eval_json(dataset_cfg)

            ap = ActivationParser(
                inference_json=dataset_cfg["inference_json"],
                eval_json=eval_json_path,
                activations_path=dataset_cfg["activations_path"],
                logger_type=dataset_cfg.get("backend", "zarr"),
                random_seed=experiment_cfg.get("split_seed", 42),
                split_strategy=split_strategy,
                verbose=True,
            )

        for method_name in methods:
            # Load method config (use pre-loaded if available)
            if method_name in _preloaded_method_cfgs:
                method_cfg = _preloaded_method_cfgs[method_name]
            else:
                method_cfg_path = os.path.join(
                    str(project_root), "configs", "methods", f"{method_name}.json"
                )
                with open(method_cfg_path) as f:
                    method_cfg = json.load(f)

            # Apply max_epochs override
            if max_epochs_override is not None and method_cfg.get("training"):
                method_cfg = dict(method_cfg)  # shallow copy to avoid mutating cached
                method_cfg["training"] = dict(method_cfg["training"])
                method_cfg["training"]["max_epochs"] = max_epochs_override

            is_learned = method_cfg.get("training") is not None
            seeds = training_seeds if is_learned else [None]

            for seed in seeds:
                # Build output directory
                if seed is not None:
                    run_dir = os.path.join(
                        output_base, exp_name, dataset_name, method_name, f"seed_{seed}"
                    )
                else:
                    run_dir = os.path.join(
                        output_base, exp_name, dataset_name, method_name
                    )

                # Resume check
                eval_metrics_path = os.path.join(run_dir, "eval_metrics.json")
                if os.path.exists(eval_metrics_path) and not args.force:
                    logger.info(
                        f"Skipping {method_name} seed={seed} (already complete)"
                    )
                    continue

                os.makedirs(run_dir, exist_ok=True)
                os.makedirs(os.path.join(run_dir, "artifacts"), exist_ok=True)

                logger.info(f"Running {method_name} seed={seed} -> {run_dir}")

                try:
                    # Seed
                    if seed is not None:
                        from utils.seeding import seed_everything

                        seed_everything(seed)

                    # Write merged config
                    merged_config = {
                        "dataset": dataset_cfg,
                        "method": method_cfg,
                        "experiment": {k: v for k, v in experiment_cfg.items()},
                        "training_seed": seed,
                        "split_seed": experiment_cfg.get("split_seed", 42),
                    }
                    with open(os.path.join(run_dir, "config.json"), "w") as f:
                        json.dump(merged_config, f, indent=2)

                    # Write manifest
                    write_run_manifest(run_dir)

                    # Dispatch to method runner
                    routine = method_cfg.get("routine", method_cfg["name"])
                    if routine == "contrastive":
                        eval_metrics, predictions = run_contrastive(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, seed,
                            test_ap=test_ap,
                        )
                    elif routine == "contrastive_logprob_recon":
                        eval_metrics, predictions = run_contrastive_logprob_recon(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, seed,
                            test_ap=test_ap,
                        )
                    elif routine == "linear_probe":
                        eval_metrics, predictions = run_linear_probe(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, seed,
                            test_ap=test_ap,
                        )
                    elif routine == "multi_layer_linear_probe":
                        eval_metrics, predictions = run_multi_layer_linear_probe(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, seed,
                            test_ap=test_ap,
                        )
                    elif routine == "token_entropy":
                        eval_metrics, predictions = run_token_entropy(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device,
                            test_ap=test_ap,
                        )
                    elif routine == "logprob_baseline":
                        eval_metrics, predictions = run_logprob_baseline(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device,
                            test_ap=test_ap,
                        )
                    else:
                        logger.warning(f"Unknown routine: {routine}, skipping")
                        continue

                    # Write eval_metrics.json
                    with open(eval_metrics_path, "w") as f:
                        json.dump(eval_metrics, f, indent=2)

                    # Write predictions.csv
                    if predictions:
                        pred_path = os.path.join(run_dir, "predictions.csv")
                        with open(pred_path, "w", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
                            writer.writeheader()
                            writer.writerows(predictions)

                    logger.info(f"Completed {method_name} seed={seed}: {eval_metrics}")

                except Exception:
                    import traceback

                    tb = traceback.format_exc()
                    logger.error(
                        f"Failed {method_name} seed={seed}: {tb}"
                    )
                    # Write error record so we know this run failed
                    error_path = os.path.join(run_dir, "run_error.json")
                    with open(error_path, "w") as f:
                        json.dump(
                            {"method": method_name, "seed": seed, "error": tb},
                            f,
                            indent=2,
                        )

    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()
