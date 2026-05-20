"""eval_d2_heads_knn.py — Per-head + ensemble KNN re-eval for D2 shared-trunk checkpoints.

Issue #111.

Loads existing SharedTrunkProjectionHeadCompressor checkpoints produced by
run_experiment.py (PR #103/issue #102), extracts trunk + Head A + Head B
embeddings via model.forward_with_heads(), runs KNN AUROC on each, and
computes a train-distance-normalized ensemble of Head A + Head B.

Writes a sidecar eval_metrics_heads.json next to the existing eval_metrics.json.

Usage
-----
# Single run:
python scripts/eval_d2_heads_knn.py \\
    --run-dir runs/sharedtrunk_grid_sciq_llama_memmap/sciq_memmap/contrastive_logprob_recon_d2a/seed_0

# Batch over all matching method×seed dirs in a grid:
python scripts/eval_d2_heads_knn.py \\
    --grid sharedtrunk_grid_sciq_llama_memmap \\
    --methods contrastive_logprob_recon_d2a
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Reuse the existing KNN AUROC helper from activation_research (DO NOT reimplement).
# ---------------------------------------------------------------------------
from activation_research.metrics import knn_ood_stats, _safe_auroc

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_layer_range(spec) -> list[int]:
    """Parse '14-29' → [14, 15, …, 29]; or a list of ints directly."""
    if isinstance(spec, list):
        return [int(x) for x in spec]
    spec = str(spec)
    if "-" not in spec:
        return [int(spec)]
    start, end = spec.split("-")
    return list(range(int(start), int(end) + 1))


def _load_run_config(run_dir: Path) -> dict:
    """Load and return the merged config.json from a run directory."""
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    with cfg_path.open() as fh:
        return json.load(fh)


def _load_existing_metrics(run_dir: Path) -> dict:
    """Load eval_metrics.json from run directory. Raises if missing."""
    metrics_path = run_dir / "eval_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"eval_metrics.json not found in {run_dir}")
    with metrics_path.open() as fh:
        return json.load(fh)


def _build_model(cfg: dict) -> "SharedTrunkProjectionHeadCompressor":
    """Rebuild SharedTrunkProjectionHeadCompressor from merged run config."""
    from activation_research.model import SharedTrunkProjectionHeadCompressor

    # config.json stores {"dataset": ..., "method": {..., "model_params": ...}, ...}
    # The spec says config["method_params"]["model_params"] but the actual stored
    # structure is config["method"]["model_params"].
    method_cfg = cfg.get("method", cfg.get("method_params", {}))
    mp = method_cfg.get("model_params", {})

    dataset_cfg = cfg.get("dataset", {})
    input_dim = int(dataset_cfg.get("input_dim", mp.get("input_dim", 4096)))

    model = SharedTrunkProjectionHeadCompressor(
        input_dim=input_dim,
        trunk_dim=int(mp.get("trunk_dim", 512)),
        head_dim=int(mp.get("head_dim", 256)),
        head_hidden_dim=int(mp.get("head_hidden_dim", 256)),
        recon_seq_len=int(mp.get("recon_seq_len", 64)),
        recon_hidden_dim=int(mp.get("recon_hidden_dim", 256)),
        recon_lambda=float(mp.get("recon_lambda", 1.0)),
        input_dropout=float(mp.get("input_dropout", 0.3)),
        logprob_var_threshold=float(mp.get("logprob_var_threshold", 1e-4)),
    )
    return model


def _load_weights(model: torch.nn.Module, run_dir: Path, device: str) -> None:
    """Load final_weights.pt into model (in-place)."""
    weights_path = run_dir / "artifacts" / "final_weights.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"final_weights.pt not found at {weights_path}")
    ckpt = torch.load(str(weights_path), map_location=device)
    # Checkpoints are saved as {"model_state_dict": ...}
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)


def _build_datasets(cfg: dict):
    """Rebuild train/test MemmapContrastiveDataset from run config.

    Returns (train_ds, test_ds) with num_views=1 (eval mode).
    """
    from activation_research.memmap_activation_parser import MemmapActivationParser

    dataset_cfg = cfg.get("dataset", {})
    method_cfg = cfg.get("method", cfg.get("method_params", {}))
    data_cfg = method_cfg.get("data", {})
    experiment_cfg = cfg.get("experiment", {})

    split_seed = int(cfg.get("split_seed", experiment_cfg.get("split_seed", 42)))

    # Build the project root relative path resolver.
    # Paths in dataset_cfg["icr_capture"] may be relative to the project root.
    # We resolve relative to the script's grandparent (project root).
    project_root = Path(__file__).resolve().parent.parent

    icr = dataset_cfg.get("icr_capture", {})
    train_dir = icr.get("train_dir", "")
    test_dir = icr.get("test_dir", "")

    if not os.path.isabs(train_dir):
        train_dir = str(project_root / train_dir)
    if not os.path.isabs(test_dir):
        test_dir = str(project_root / test_dir)

    relevant_layers = _parse_layer_range(
        data_cfg.get("relevant_layers", "14-29")
    )
    target_layers = data_cfg.get("target_layers", [22, 26])
    if isinstance(target_layers, str):
        target_layers = _parse_layer_range(target_layers)
    else:
        target_layers = [int(x) for x in target_layers]

    pad_length = data_cfg.get("pad_length", 63)
    include_response_logprobs = data_cfg.get("include_response_logprobs", False)
    random_seed = data_cfg.get("random_seed", split_seed)

    # Match production eval (run_experiment.py:477-489): num_views=2 → both views
    # are forwarded through the model and averaged via knn_ood_stats's z_views.mean(0).
    # Using num_views=1 here drops half the signal and the sanity check fails.
    common_kwargs = dict(
        relevant_layers=relevant_layers,
        num_views=data_cfg.get("num_views", 2),
        pad_length=pad_length,
        include_response_logprobs=include_response_logprobs,
    )

    # Train parser (three_way split — 90% train, 10% val)
    train_ap = MemmapActivationParser(
        capture_dir=train_dir,
        random_seed=split_seed,
        split_strategy="three_way",
        verbose=False,
    )
    train_ds_full = train_ap.get_dataset("train", **common_kwargs)
    train_ds = train_ds_full.slice_layers(target_layers)

    # Test parser (none split — all rows as test)
    test_ap = MemmapActivationParser(
        capture_dir=test_dir,
        random_seed=split_seed,
        split_strategy="none",
        verbose=False,
    )
    test_ds_full = test_ap.get_dataset("test", **common_kwargs)
    test_ds = test_ds_full.slice_layers(target_layers)

    return train_ds, test_ds


@torch.no_grad()
def _extract_embeddings(
    model: torch.nn.Module,
    dataset,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract trunk, head_A, head_B embeddings and labels for all samples.

    Returns (trunk_z, z_A, z_B, labels) as numpy arrays of shapes:
        trunk_z : (N, trunk_dim)
        z_A     : (N, head_dim)
        z_B     : (N, head_dim)
        labels  : (N,) int32
    """
    model.eval()
    model.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    trunk_list, zA_list, zB_list, label_list = [], [], [], []

    for batch in loader:
        # batch["views_activations"] has shape (B, num_views, seq_len, hidden_dim).
        # Match production eval: forward each view separately, then mean over views
        # (knn_ood_stats does z_views.mean(0); we replicate it explicitly here so
        # downstream LOO-NN distances also use the reduced embeddings).
        x = batch["views_activations"].to(device).float()
        if x.dim() == 3:
            x = x.unsqueeze(1)  # promote (B, L, D) → (B, 1, L, D)
        bsz, num_views, seq_len, hidden = x.shape
        x_flat = x.reshape(bsz * num_views, seq_len, hidden)

        trunk_flat, zA_flat, zB_flat, _ = model.forward_with_heads(x_flat)

        # Reshape back to (B, num_views, dim) and mean over views.
        trunk_z = trunk_flat.reshape(bsz, num_views, -1).mean(dim=1)
        z_A = zA_flat.reshape(bsz, num_views, -1).mean(dim=1)
        z_B = zB_flat.reshape(bsz, num_views, -1).mean(dim=1)

        trunk_list.append(trunk_z.cpu().float().numpy())
        zA_list.append(z_A.cpu().float().numpy())
        zB_list.append(z_B.cpu().float().numpy())

        halu = batch["halu"]
        if isinstance(halu, torch.Tensor):
            halu = halu.cpu().numpy()
        label_list.append(np.asarray(halu, dtype=np.int32).reshape(-1))

    trunk_z = np.concatenate(trunk_list, axis=0)
    z_A = np.concatenate(zA_list, axis=0)
    z_B = np.concatenate(zB_list, axis=0)
    labels = np.concatenate(label_list, axis=0)

    return trunk_z, z_A, z_B, labels


def _make_knn_records(arr: np.ndarray, labels: np.ndarray) -> list[dict]:
    """Build a list of {"z1": Tensor, "halu": int} records for knn_ood_stats."""
    return [
        {"z1": torch.from_numpy(arr[i]).float(), "halu": int(labels[i])}
        for i in range(len(labels))
    ]


def _knn_stats_for_rep(
    train_arr: np.ndarray,
    test_arr: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    k: int,
    metric: str,
    outlier_class: int = 1,
) -> dict:
    """Run knn_ood_stats for one representation (trunk, head_A, or head_B).

    Additionally computes train_nn_mean and per-sample test distances needed
    for ensemble normalization. knn_ood_stats is called for AUROC and all
    standard stats; the NearestNeighbors index is rebuilt once for train-NN
    distances (leave-one-out k+1 query) — this avoids duplicating AUROC math.

    Returns dict with all knn_ood_stats fields plus:
        "train_nn_mean"  : mean of LOO k-NN train distances (scalar, for normalization)
        "test_distances" : (N_test,) array of mean k-NN distances (for ensemble)
    """
    train_records = _make_knn_records(train_arr, train_labels)
    test_records = _make_knn_records(test_arr, test_labels)

    # Use knn_ood_stats for AUROC and standard metrics (reuse, don't reimplement).
    stats = knn_ood_stats(
        train_records=train_records,
        test_records=test_records,
        outlier_class=outlier_class,
        k=k,
        metric=metric,
        train_label_filter="all",
        calibrate_k=False,
        include_per_sample=True,
    )

    # Extract per-sample test distances (added by include_per_sample=True).
    test_distances = stats.pop("knn_scores", None)
    stats.pop("knn_labels", None)  # not needed downstream
    if test_distances is None:
        raise RuntimeError(
            "knn_ood_stats did not return knn_scores despite include_per_sample=True"
        )
    if isinstance(test_distances, torch.Tensor):
        test_distances = test_distances.numpy()
    test_distances = np.asarray(test_distances, dtype=np.float64)

    # Compute train leave-one-out k-NN distances for ensemble normalization.
    # Query k+1 neighbors: the closest is the point itself (distance 0); drop it.
    n_train = train_arr.shape[0]
    n_neighbors_loo = min(k + 1, n_train)
    nn = NearestNeighbors(n_neighbors=n_neighbors_loo, metric=metric)
    nn.fit(train_arr)
    train_dists, _ = nn.kneighbors(train_arr)
    # Drop the self-match (first neighbor, distance ~0) and take remaining k.
    k_actual = min(k, n_neighbors_loo - 1)
    train_knn_dists = train_dists[:, 1: k_actual + 1]  # (N_train, k)
    train_nn_mean = float(train_knn_dists.mean())

    stats["train_nn_mean"] = train_nn_mean
    stats["test_distances"] = test_distances
    return stats


def _summarize_rep(stats: dict, labels: np.ndarray, outlier_class: int = 1) -> dict:
    """Extract the standard summary fields for one representation."""
    return {
        "auroc": float(stats.get("knn_auroc", float("nan"))),
        "mean_id": float(stats.get("knn_mean_id", float("nan"))),
        "std_id": float(stats.get("knn_std_id", float("nan"))),
        "mean_ood": float(stats.get("knn_mean_ood", float("nan"))),
        "std_ood": float(stats.get("knn_std_ood", float("nan"))),
        "train_nn_mean": float(stats.get("train_nn_mean", float("nan"))),
    }


def _ensemble_stats(
    dists_A: np.ndarray,
    dists_B: np.ndarray,
    scale_A: float,
    scale_B: float,
    test_labels: np.ndarray,
    outlier_class: int = 1,
) -> dict:
    """Compute ensemble AUROC and summary stats.

    Ensemble score = (d_A / scale_A + d_B / scale_B) / 2
    where scale_X = mean(train_nn_distances_X) so each head contributes equally.
    """
    if scale_A <= 0:
        scale_A = 1.0
    if scale_B <= 0:
        scale_B = 1.0

    d_ensemble = (dists_A / scale_A + dists_B / scale_B) / 2.0

    binary_labels = (test_labels == int(outlier_class)).astype(np.int32)
    auroc = float(_safe_auroc(binary_labels, d_ensemble))

    labels_t = torch.from_numpy(test_labels.astype(np.int32))
    d_t = torch.from_numpy(d_ensemble.astype(np.float32))

    id_mask = labels_t == 0
    ood_mask = labels_t == 1
    if outlier_class == 0:
        id_mask, ood_mask = ood_mask, id_mask

    id_scores = d_t[id_mask]
    ood_scores = d_t[ood_mask]

    return {
        "auroc": auroc,
        "mean_id": float(id_scores.mean().item()) if id_scores.numel() else float("nan"),
        "std_id": float(id_scores.std().item()) if id_scores.numel() else float("nan"),
        "mean_ood": float(ood_scores.mean().item()) if ood_scores.numel() else float("nan"),
        "std_ood": float(ood_scores.std().item()) if ood_scores.numel() else float("nan"),
    }


def eval_run(
    run_dir: Path,
    knn_k: Optional[int],
    knn_metric: Optional[str],
    output_name: str,
    device: str,
    batch_size: int,
) -> bool:
    """Evaluate one run directory. Returns True on success, False on failure."""
    run_dir = Path(run_dir)
    logger.info(f"--- Evaluating {run_dir} ---")

    # ---- Sanity: required files ----
    for fname in ("config.json", "eval_metrics.json", "artifacts/final_weights.pt"):
        if not (run_dir / fname).exists():
            logger.error(f"  Skipping: {fname} not found in {run_dir}")
            return False

    cfg = _load_run_config(run_dir)
    existing_metrics = _load_existing_metrics(run_dir)

    # ---- Resolve KNN params ----
    # Default: read from existing eval_metrics.json to match production eval.
    effective_k = knn_k if knn_k is not None else int(existing_metrics.get("knn_k", 50))
    effective_metric = knn_metric if knn_metric is not None else str(
        existing_metrics.get("knn_metric", "euclidean")
    )

    # ---- Rebuild model ----
    logger.info("  Building model from config...")
    model = _build_model(cfg)
    _load_weights(model, run_dir, device)
    model.eval()

    # ---- Rebuild datasets ----
    logger.info("  Building train/test datasets...")
    train_ds, test_ds = _build_datasets(cfg)

    logger.info(f"  Train samples: {len(train_ds)}  Test samples: {len(test_ds)}")

    # ---- Extract embeddings ----
    logger.info(f"  Extracting embeddings on {device} (batch_size={batch_size})...")
    train_trunk, train_A, train_B, train_labels = _extract_embeddings(
        model, train_ds, device, batch_size
    )
    test_trunk, test_A, test_B, test_labels = _extract_embeddings(
        model, test_ds, device, batch_size
    )
    logger.info(
        f"  Extracted: trunk={train_trunk.shape[1]}d, "
        f"head_A={train_A.shape[1]}d, head_B={train_B.shape[1]}d"
    )

    outlier_class = int(cfg.get("dataset", {}).get("outlier_class", 1))
    experiment_cfg = cfg.get("experiment", {})
    method_cfg = cfg.get("method", cfg.get("method_params", {}))
    dataset_cfg = cfg.get("dataset", {})

    # ---- KNN for each representation ----
    logger.info(f"  Running KNN (k={effective_k}, metric={effective_metric})...")

    trunk_stats = _knn_stats_for_rep(
        train_trunk, test_trunk, train_labels, test_labels,
        k=effective_k, metric=effective_metric, outlier_class=outlier_class,
    )
    head_a_stats = _knn_stats_for_rep(
        train_A, test_A, train_labels, test_labels,
        k=effective_k, metric=effective_metric, outlier_class=outlier_class,
    )
    head_b_stats = _knn_stats_for_rep(
        train_B, test_B, train_labels, test_labels,
        k=effective_k, metric=effective_metric, outlier_class=outlier_class,
    )

    # ---- Trunk sanity check (REQUIRED — abort on mismatch) ----
    existing_knn_auroc = float(existing_metrics.get("knn_auroc", float("nan")))
    recomputed_knn_auroc = float(trunk_stats.get("knn_auroc", float("nan")))

    if abs(recomputed_knn_auroc - existing_knn_auroc) > 1e-6:
        logger.error(
            f"  SANITY CHECK FAILED for {run_dir}:\n"
            f"    Recomputed trunk KNN AUROC : {recomputed_knn_auroc:.10f}\n"
            f"    Existing eval_metrics.json  : {existing_knn_auroc:.10f}\n"
            f"    Diff                        : {abs(recomputed_knn_auroc - existing_knn_auroc):.2e}\n"
            f"  Not writing {output_name}. Fix the mismatch before proceeding."
        )
        return False

    logger.info(f"  Trunk sanity check PASSED (AUROC={recomputed_knn_auroc:.6f})")

    # ---- Ensemble ----
    dists_A = head_a_stats.pop("test_distances")
    dists_B = head_b_stats.pop("test_distances")
    trunk_stats.pop("test_distances", None)

    scale_A = head_a_stats.get("train_nn_mean", 1.0)
    scale_B = head_b_stats.get("train_nn_mean", 1.0)

    ens_stats = _ensemble_stats(
        dists_A, dists_B, scale_A, scale_B, test_labels, outlier_class=outlier_class,
    )

    # ---- Build output ----
    split_seed = int(cfg.get("split_seed", experiment_cfg.get("split_seed", 42)))
    training_seed = cfg.get("training_seed", 0)

    output: dict = {
        "method": str(existing_metrics.get("method", method_cfg.get("name", ""))),
        "dataset": str(existing_metrics.get("dataset", dataset_cfg.get("name", ""))),
        "seed": int(training_seed) if training_seed is not None else 0,
        "split_seed": split_seed,
        "n_train": int(len(train_ds)),
        "n_test": int(len(test_ds)),
        "knn_k": int(effective_k),
        "knn_metric": str(effective_metric),
        "knn_train_label_filter": "all",
        "trunk": _summarize_rep(trunk_stats, test_labels, outlier_class),
        "head_a": _summarize_rep(head_a_stats, test_labels, outlier_class),
        "head_b": _summarize_rep(head_b_stats, test_labels, outlier_class),
        "ensemble_a_b_normalized": ens_stats,
    }

    # ---- Log AUROC values ----
    logger.info(
        f"  AUROC — trunk: {output['trunk']['auroc']:.4f}  "
        f"head_A: {output['head_a']['auroc']:.4f}  "
        f"head_B: {output['head_b']['auroc']:.4f}  "
        f"ensemble: {output['ensemble_a_b_normalized']['auroc']:.4f}"
    )

    # ---- Write sidecar ----
    out_path = run_dir / output_name
    with out_path.open("w") as fh:
        json.dump(output, fh, indent=2)
    logger.info(f"  Written: {out_path}")

    return True


def _collect_run_dirs(grid: str, methods_filter: Optional[list[str]]) -> list[Path]:
    """Find all seed dirs under runs/<grid>/<dataset>/<method>/seed_*."""
    project_root = Path(__file__).resolve().parent.parent
    grid_dir = project_root / "runs" / grid
    if not grid_dir.exists():
        raise FileNotFoundError(f"Grid directory not found: {grid_dir}")

    run_dirs = []
    for dataset_dir in sorted(grid_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for method_dir in sorted(dataset_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            if methods_filter and method_dir.name not in methods_filter:
                continue
            for seed_dir in sorted(method_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                # Only include if it looks like a run dir (has config.json)
                if (seed_dir / "config.json").exists():
                    run_dirs.append(seed_dir)

    return run_dirs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-head + ensemble KNN re-eval for D2 shared-trunk checkpoints."
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--run-dir", type=str,
        help="Path to a single run directory containing config.json + artifacts/final_weights.pt",
    )
    mode.add_argument(
        "--grid", type=str,
        help="Grid name (subdirectory of runs/) for batch evaluation.",
    )

    p.add_argument(
        "--methods", type=str, default=None,
        help=(
            "Comma-separated list of method names to include when using --grid. "
            "Default: process every method dir found."
        ),
    )
    p.add_argument(
        "--knn-k", type=int, default=None,
        help="Number of neighbors for KNN. Default: read from existing eval_metrics.json.",
    )
    p.add_argument(
        "--knn-metric", type=str, default=None,
        help="Distance metric for KNN. Default: read from existing eval_metrics.json.",
    )
    p.add_argument(
        "--output-name", type=str, default="eval_metrics_heads.json",
        help="Filename for the sidecar output file (default: eval_metrics_heads.json).",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Torch device (e.g. 'cuda', 'cpu'). Default: cuda if available, else cpu.",
    )
    p.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size for embedding extraction (default: 256).",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    methods_filter = None
    if args.methods:
        methods_filter = [m.strip() for m in args.methods.split(",") if m.strip()]

    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    else:
        # --grid mode
        run_dirs = _collect_run_dirs(args.grid, methods_filter)
        if not run_dirs:
            logger.error(
                f"No matching run directories found under runs/{args.grid} "
                f"(methods filter: {methods_filter})"
            )
            sys.exit(1)
        logger.info(f"Found {len(run_dirs)} run dir(s) to evaluate.")

    success_count = 0
    fail_count = 0
    for run_dir in run_dirs:
        ok = eval_run(
            run_dir=run_dir,
            knn_k=args.knn_k,
            knn_metric=args.knn_metric,
            output_name=args.output_name,
            device=device,
            batch_size=args.batch_size,
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1

    logger.info(
        f"Done. {success_count} succeeded, {fail_count} failed."
    )
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
