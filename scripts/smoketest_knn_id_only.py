"""
smoketest_knn_id_only.py

Compare KNN AUROC with train_label_filter="all" vs "id_only" for a saved
contrastive_logprob_recon run.  Loads final_weights.pt and reuses the exact
data pipeline from run_experiment.py, but swaps in an id_only KNN metric.

Usage (from repo root, on a GPU node):
    python scripts/smoketest_knn_id_only.py \
        --run-dir runs/baseline_comparison_hotpotqa_memmap/hotpotqa_memmap/contrastive_logprob_recon/seed_0
"""

import argparse
import json
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from activation_research.memmap_activation_parser import MemmapActivationParser
from activation_research.model import LogprobReconProgressiveCompressor
from activation_research.metrics import knn_ood_stats
from activation_research.metric_evaluator import MultiMetricHallucinationEvaluator


def parse_layer_range(spec: str):
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x.strip()) for x in spec.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to seed_N run directory")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_dir = args.run_dir
    config_path = os.path.join(run_dir, "config.json")
    weights_path = os.path.join(run_dir, "artifacts", "final_weights.pt")

    assert os.path.exists(config_path), f"config.json not found: {config_path}"
    assert os.path.exists(weights_path), f"final_weights.pt not found: {weights_path}"

    with open(config_path) as f:
        cfg = json.load(f)

    dataset_cfg = cfg["dataset"]
    method_cfg = cfg["method"]
    experiment_cfg = cfg.get("experiment", {})
    split_seed = cfg.get("split_seed", 42)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Build data parsers ----
    icr = dataset_cfg["icr_capture"]
    train_dir = os.path.join(ROOT, icr["train_dir"])
    test_dir = os.path.join(ROOT, icr["test_dir"])

    print(f"Loading train MemmapActivationParser from: {train_dir}")
    ap = MemmapActivationParser(
        capture_dir=train_dir,
        random_seed=split_seed,
        split_strategy="three_way",
        verbose=True,
    )
    print(f"Loading test MemmapActivationParser from: {test_dir}")
    test_ap = MemmapActivationParser(
        capture_dir=test_dir,
        random_seed=split_seed,
        split_strategy="none",
        verbose=True,
    )

    # ---- Build datasets ----
    data_cfg = method_cfg["data"]
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

    train_ds = ap.get_dataset("train", **ds_kwargs).slice_layers(target_layers)
    test_ds = test_ap.get_dataset("test", **ds_kwargs).slice_layers(target_layers)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
    eval_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # ---- Build and load model ----
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
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ---- Compute embeddings once, run KNN with both filters ----
    knn_params = dict(eval_cfg.get("knn_params", {}))

    metrics_list = [
        # (1) all train, calibrated k  — current production behaviour (with bug fixed)
        {"metric": "knn", "kwargs": {**knn_params, "train_label_filter": "all",     "include_per_sample": False}, "train_selection": "all"},
        # (2) id_only, calibrated k   — the proposed change
        {"metric": "knn", "kwargs": {**knn_params, "train_label_filter": "id_only", "include_per_sample": False}, "train_selection": "all",
         "prefix": "knn_id"},
        # (3) all train, k=50 fixed   — isolates calibrate_k effect from filter effect
        {"metric": "knn", "kwargs": {**knn_params, "train_label_filter": "all", "calibrate_k": False, "k": 50, "include_per_sample": False}, "train_selection": "all",
         "prefix": "knn_all_k50"},
    ]

    outlier_class = dataset_cfg.get("outlier_class", 1)

    # Combine train + test dfs so _assign_hallucination_labels can label baseline
    # (train) records. Without this, train hashes are absent from test_ap.df and
    # all baseline records fall back to unlabeled → train_label_filter is a no-op.
    combined_df = pd.concat([ap.df, test_ap.df], ignore_index=True).drop_duplicates("prompt_hash")
    print(f"Combined df: {len(combined_df)} rows  halu counts: {combined_df['halu'].value_counts().to_dict()}")

    evaluator = MultiMetricHallucinationEvaluator(
        activation_parser_df=combined_df,
        train_data_loader=train_loader,
        metrics=metrics_list,
        batch_size=eval_cfg.get("eval_batch_size", 256),
        sub_batch_size=eval_cfg.get("sub_batch_size", 64),
        device=device,
        num_workers=0,
        persistent_workers=False,
        outlier_class=outlier_class,
    )

    print("Running evaluation (single forward pass, two KNN variants)...")
    stats = evaluator.compute(eval_loader, model)

    knn_all     = stats.get("knn_auroc")
    knn_id      = stats.get("knn_id_knn_auroc")
    knn_all_k50 = stats.get("knn_all_k50_knn_auroc")
    k_all       = stats.get("knn_k")
    k_id        = stats.get("knn_id_knn_k")

    print()
    print("=" * 60)
    print(f"Dataset : {dataset_cfg['name']}  ({dataset_cfg['model_name']})")
    print(f"Run dir : {run_dir}")
    print(f"KNN (all,    calibrated k={k_all:4d})  AUROC: {knn_all:.4f}")
    print(f"KNN (all,    fixed    k= 50)  AUROC: {knn_all_k50:.4f}   <- isolates calibrate_k effect")
    print(f"KNN (id_only,calibrated k={k_id:4d})  AUROC: {knn_id:.4f}   <- isolates filter effect at same k")
    print()
    if knn_all is not None and knn_id is not None and knn_all_k50 is not None:
        print(f"  calibrate_k effect (all k=1000 vs all k=50):   {knn_all - knn_all_k50:+.4f}")
        print(f"  filter effect      (id_only k=50 vs all k=50): {knn_id - knn_all_k50:+.4f}")
        print(f"  combined delta     (id_only k=50 vs all k=1000): {knn_id - knn_all:+.4f}")
    print("=" * 60)
    print()
    print("All stats keys:", sorted(stats.keys()))
    print(json.dumps({k: v for k, v in stats.items() if "auroc" in k.lower()}, indent=2))


if __name__ == "__main__":
    main()
