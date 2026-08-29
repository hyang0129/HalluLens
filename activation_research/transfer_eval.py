"""Transfer evaluation: apply source-trained checkpoints to target datasets."""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from activation_logging.activation_parser import ActivationParser
from activation_research.metrics import knn_ood_stats, mahalanobis_ood_stats
from activation_research.model import (
    LinearProbe,
    LogprobReconProgressiveCompressor,
    SimpleHaluClassifier,
)


_CHECKPOINT_FILE = {
    "contrastive_logprob_recon": "contrastive_last.pt",
    "linear_probe": "linear_probe_last.pt",
    "saplma": "linear_probe_last.pt",
}

_FALLBACK_CHECKPOINT = "trainer_last.pt"


def _build_activation_parser(zarr_path: str, eval_json_path: str) -> ActivationParser:
    """Construct ActivationParser for a pre-split zarr store (split_strategy='none')."""
    inference_json_path = os.path.join(os.path.dirname(eval_json_path), "generation.jsonl")
    return ActivationParser(
        inference_json=inference_json_path,
        eval_json=eval_json_path,
        activations_path=zarr_path,
        logger_type="zarr",
        split_strategy="none",
        verbose=False,
    )


def load_checkpoint_model(
    method: str,
    checkpoint_path: str,
    dataset_cfg: dict,
) -> torch.nn.Module:
    """Load model from checkpoint. Returns model in eval mode on CPU."""
    input_dim = dataset_cfg.get("input_dim", 4096)

    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
    model_params = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            run_cfg = json.load(f)
        method_cfg = run_cfg.get("method_cfg", {})
        model_params = method_cfg.get("model_params", {})

    if method == "contrastive_logprob_recon":
        params = dict(
            input_dim=input_dim,
            final_dim=512,
            input_dropout=0.3,
            recon_seq_len=64,
            recon_hidden_dim=256,
            recon_lambda=1.0,
            logprob_var_threshold=1e-4,
        )
        params.update(model_params)
        model = LogprobReconProgressiveCompressor(**params)
    elif method == "linear_probe":
        params = dict(input_dim=input_dim, pooling="mean")
        params.update(model_params)
        model = LinearProbe(**params)
    elif method == "saplma":
        params = dict(input_dim=input_dim, hidden_dims=[2048, 1024, 512], dropout=0.1)
        params.update(model_params)
        model = SimpleHaluClassifier(**params)
    else:
        raise ValueError(f"Unknown method: {method}")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def get_embeddings_contrastive(
    model: torch.nn.Module,
    zarr_path: str,
    eval_json_path: str,
    relevant_layers: list,
    device: str = "cpu",
    batch_size: int = 128,
    num_workers: int = 4,
) -> list:
    """Run forward pass through contrastive model; return embedding records.

    Returns list of dicts: [{"hashkey": str, "z_views": Tensor(1, D), "halu": int}, ...]
    """
    ap = _build_activation_parser(zarr_path, eval_json_path)
    ds = ap.get_dataset(
        split="test",
        relevant_layers=relevant_layers,
        num_views=1,
        include_response_logprobs=False,
        preload=False,
        check_ram=False,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
    )

    records = []
    with torch.no_grad():
        for batch in dl:
            x = batch["views_activations"][:, 0, :, :]  # (B, T, H) — first (only) view
            z = model(x.to(device)).cpu()               # (B, D)
            for i in range(len(z)):
                records.append({
                    "hashkey": batch["hashkey"][i],
                    "z_views": z[i].unsqueeze(0),  # (1, D) — metrics.py expects (K, D)
                    "halu": int(batch["halu"][i]),
                })
    return records


def get_scores_probe(
    model: torch.nn.Module,
    zarr_path: str,
    eval_json_path: str,
    probe_layer: int,
    device: str = "cpu",
    batch_size: int = 256,
    num_workers: int = 4,
) -> tuple:
    """Forward-pass a LinearProbe or SimpleHaluClassifier on a dataset split.

    Returns (scores, labels): float32 arrays of shape (N,).
    scores are sigmoid probabilities in [0, 1].
    """
    ap = _build_activation_parser(zarr_path, eval_json_path)
    ds = ap.get_dataset(
        split="test",
        relevant_layers=[probe_layer],
        num_views=1,
        include_response_logprobs=False,
        preload=False,
        check_ram=False,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
    )

    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in dl:
            x = batch["views_activations"][:, 0, :, :]  # (B, T, H)
            out = model(x.to(device)).squeeze(-1).cpu()  # (B,)
            all_scores.append(out.numpy())
            all_labels.append(batch["halu"].numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


def evaluate_transfer_cell(
    source_run_dir: str,
    source_dataset_cfg: dict,
    target_test_cfg: dict,
    method: str,
    relevant_layers: list,
    probe_layer: int,
    device: str = "cpu",
) -> dict:
    """Evaluate one transfer matrix cell. Returns metrics dict."""
    ckpt_file = _CHECKPOINT_FILE[method]
    checkpoint_path = os.path.join(source_run_dir, ckpt_file)

    if not os.path.exists(checkpoint_path):
        fallback = os.path.join(source_run_dir, _FALLBACK_CHECKPOINT)
        if os.path.exists(fallback):
            checkpoint_path = fallback
        else:
            return {"status": "missing_checkpoint"}

    model = load_checkpoint_model(method, checkpoint_path, source_dataset_cfg)
    model = model.to(device)

    if method == "contrastive_logprob_recon":
        src_train_records = get_embeddings_contrastive(
            model,
            zarr_path=source_dataset_cfg["train"]["activations_path"],
            eval_json_path=source_dataset_cfg["train"]["eval_json"],
            relevant_layers=relevant_layers,
            device=device,
        )
        tgt_test_records = get_embeddings_contrastive(
            model,
            zarr_path=target_test_cfg["test"]["activations_path"],
            eval_json_path=target_test_cfg["test"]["eval_json"],
            relevant_layers=relevant_layers,
            device=device,
        )
        maha_stats = mahalanobis_ood_stats(src_train_records, tgt_test_records, outlier_class=1)
        knn_stats = knn_ood_stats(
            src_train_records,
            tgt_test_records,
            outlier_class=1,
            k=50,
            metric="euclidean",
            calibrate_k=False,
        )
        return {
            "status": "ok",
            "auroc": maha_stats["mahalanobis_auroc"],
            "mahalanobis_auroc": maha_stats["mahalanobis_auroc"],
            "mahalanobis_mean_id": maha_stats["mahalanobis_mean_id"],
            "mahalanobis_std_id": maha_stats["mahalanobis_std_id"],
            "mahalanobis_mean_ood": maha_stats["mahalanobis_mean_ood"],
            "mahalanobis_std_ood": maha_stats["mahalanobis_std_ood"],
            "knn_auroc": knn_stats["knn_auroc"],
            "n_src_train": len(src_train_records),
            "n_tgt_test": len(tgt_test_records),
        }
    else:
        scores, labels = get_scores_probe(
            model,
            zarr_path=target_test_cfg["test"]["activations_path"],
            eval_json_path=target_test_cfg["test"]["eval_json"],
            probe_layer=probe_layer,
            device=device,
        )
        if len(np.unique(labels)) < 2:
            auroc = float("nan")
        else:
            auroc = float(roc_auc_score(labels, scores))
        return {
            "status": "ok",
            "auroc": auroc,
            "n_tgt_test": len(labels),
        }


def discover_runs(runs_root: str, method: str) -> list:
    """Scan runs_root for completed runs of the given method.

    Returns list of dicts: experiment_name, dataset, method, seed, run_dir, config.
    Only includes runs where both eval_metrics.json and the checkpoint file exist.
    """
    ckpt_file = _CHECKPOINT_FILE[method]
    results = []

    runs_root_path = Path(runs_root)
    if not runs_root_path.exists():
        return results

    # Walk: runs_root/{experiment_name}/{dataset}/{method}/seed_{seed}/
    for exp_dir in sorted(runs_root_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        experiment_name = exp_dir.name
        for dataset_dir in sorted(exp_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            method_dir = dataset_dir / method
            if not method_dir.is_dir():
                continue
            for seed_dir in sorted(method_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                try:
                    seed = int(seed_dir.name.split("_", 1)[1])
                except (ValueError, IndexError):
                    continue

                eval_metrics = seed_dir / "eval_metrics.json"
                checkpoint = seed_dir / ckpt_file
                if not eval_metrics.exists():
                    continue
                if not checkpoint.exists():
                    fallback = seed_dir / _FALLBACK_CHECKPOINT
                    if not fallback.exists():
                        continue

                config_path = seed_dir / "config.json"
                config = {}
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)

                results.append({
                    "experiment_name": experiment_name,
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "run_dir": str(seed_dir),
                    "config": config,
                })

    return results
