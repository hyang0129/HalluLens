"""Per-slice driver for the Issue #75 lambda sweep.

Loads one method config from configs/methods/, then sequentially trains and
evaluates `LogprobAttnReconProgressiveCompressor` over a comma-separated list
of dataset config names (default: 6 Llama datasets). Writes per-dataset
results + a slice-level summary.csv. Failures in one dataset don't abort
the chain.

Spec: specs/issue_75_lambda_sweep.md
Smoketest reference: scripts/smoketest_issue_75.py (single-cell version)

Usage::

    python scripts/sweep_issue_75_lambda.py \\
        --method-config configs/methods/contrastive_logprob_attn_recon_l10_a10.json \\
        --datasets hotpotqa,popqa,nq,sciq,searchqa,mmlu \\
        --seed 0 \\
        --output-dir runs/issue_75_lambda_sweep/contrastive_logprob_attn_recon_l10_a10/seed_0
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from activation_research.memmap_contrastive_dataset import MemmapContrastiveDataset
from activation_research.model import LogprobAttnReconProgressiveCompressor
from activation_research.training import train_contrastive_logprob_attn_recon


DEFAULT_DATASETS = "hotpotqa,popqa,nq,sciq,searchqa,mmlu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method-config", required=True, help="Path to a configs/methods/*.json file")
    p.add_argument(
        "--datasets",
        default=DEFAULT_DATASETS,
        help=f"Comma-separated dataset config names (default: {DEFAULT_DATASETS})",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None, help="Default: runs/issue_75_lambda_sweep/{method_name}/seed_{seed}")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    # Smoketest overrides — bypass the method config's epoch / step count for
    # quick driver validation runs. Use sparingly; sweep production runs
    # should leave these unset and rely on the method config.
    p.add_argument("--max-epochs", type=int, default=None, help="Override training.max_epochs from the method config.")
    p.add_argument("--steps-per-epoch", type=int, default=None, help="Override training.steps_per_epoch_override.")
    p.add_argument("--sub-batch-size", type=int, default=None, help="Override training.sub_batch_size from the method config.")
    return p.parse_args()


def parse_layers(spec):
    if isinstance(spec, list):
        return [int(x) for x in spec]
    s = str(spec).strip()
    if "-" in s and "," not in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",") if x.strip()]


def resolve_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def load_dataset_paths(dataset_name: str) -> tuple[Path, Path, dict]:
    """Load configs/datasets/{name}.json, return (train_capture_dir, test_capture_dir, raw_cfg)."""
    cfg_path = PROJECT_ROOT / "configs" / "datasets" / f"{dataset_name}.json"
    with cfg_path.open() as fh:
        cfg = json.load(fh)
    icr = cfg.get("icr_capture")
    if not icr or "train_dir" not in icr or "test_dir" not in icr:
        raise KeyError(
            f"{cfg_path} is missing icr_capture.{{train_dir,test_dir}} — "
            "this driver requires #72-format captures."
        )
    train_dir = (PROJECT_ROOT / icr["train_dir"]).resolve()
    test_dir = (PROJECT_ROOT / icr["test_dir"]).resolve()
    if not train_dir.exists():
        raise FileNotFoundError(f"capture train_dir not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"capture test_dir not found: {test_dir}")
    return train_dir, test_dir, cfg


def build_label_dataframe(*capture_dirs: Path) -> pd.DataFrame:
    """Build a DataFrame with prompt_hash + halu from one or more meta.jsonl files."""
    rows = []
    for d in capture_dirs:
        with (d / "meta.jsonl").open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                rows.append({
                    "prompt_hash": r["prompt_hash"],
                    "halu": int(bool(r["hallucinated"])),
                })
    return pd.DataFrame(rows)


def build_dataset(capture_dir: Path, split: str, data_cfg: dict, seed: int, *, relevant_layers):
    """Build MemmapContrastiveDataset for given split using the method's data config block."""
    return MemmapContrastiveDataset(
        capture_dir,
        split=split,
        num_views=int(data_cfg.get("num_views", 2)),
        relevant_layers=relevant_layers,
        include_response_logprobs=bool(data_cfg.get("include_response_logprobs", True)),
        response_logprobs_top_k=int(data_cfg.get("response_logprobs_top_k", 20)),
        pad_length=int(data_cfg.get("pad_length", 63)),
        include_response_attention=bool(data_cfg.get("include_response_attention", True)),
        attention_summary=str(data_cfg.get("attention_summary", "stats")),
        attention_target_layer_offset_backward=data_cfg.get("attention_target_layer_offset_backward"),
        attention_target_layer_offset_forward=data_cfg.get("attention_target_layer_offset_forward"),
        random_seed=int(seed),
    )


def build_model(input_dim: int, model_params: dict) -> LogprobAttnReconProgressiveCompressor:
    return LogprobAttnReconProgressiveCompressor(
        input_dim=int(input_dim),
        final_dim=int(model_params.get("final_dim", 512)),
        input_dropout=float(model_params.get("input_dropout", 0.3)),
        recon_seq_len=int(model_params.get("recon_seq_len", 64)),
        recon_hidden_dim=int(model_params.get("recon_hidden_dim", 256)),
        recon_lambda=float(model_params.get("recon_lambda", 1.0)),
        logprob_var_threshold=float(model_params.get("logprob_var_threshold", 1e-4)),
        attn_direction=str(model_params.get("attn_direction", "both")),
        attn_offset_k=int(model_params.get("attn_offset_k", 1)),
        attn_target=str(model_params.get("attn_target", "stats")),
        attn_num_stat_features=int(model_params.get("attn_num_stat_features", 3)),
        attn_recon_hidden_dim=int(model_params.get("attn_recon_hidden_dim", 256)),
        attn_recon_lambda=float(model_params.get("attn_recon_lambda", 1.0)),
        attn_var_threshold=float(model_params.get("attn_var_threshold", 1e-5)),
    )


def evaluate(model, train_ds_eval, test_ds_eval, eval_cfg, device, seed, output_dir, num_workers: int = 0):
    """Run MultiMetricHallucinationEvaluator over test_ds_eval using train_ds_eval as baseline.

    Builds a combined label DataFrame from both splits' meta.jsonl, then runs all metrics
    configured in eval_cfg (cosine, mds, knn).
    """
    from torch.utils.data import DataLoader

    from activation_research.metric_evaluator import MultiMetricHallucinationEvaluator

    # Build a label DataFrame from both splits' meta.jsonl — keeps the
    # evaluator decoupled from the dataset's internal state.
    capture_train = Path(train_ds_eval._capture_dir)  # type: ignore[attr-defined]
    capture_test = Path(test_ds_eval._capture_dir)    # type: ignore[attr-defined]
    label_df = build_label_dataframe(capture_train, capture_test)

    eval_sub_batch = int(eval_cfg.get("sub_batch_size", 64))
    persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds_eval, batch_size=eval_sub_batch, shuffle=False,
        num_workers=num_workers, persistent_workers=persistent,
    )
    eval_loader = DataLoader(
        test_ds_eval, batch_size=eval_sub_batch, shuffle=False,
        num_workers=num_workers, persistent_workers=persistent,
    )

    model.eval()

    metrics_list = []
    for m in eval_cfg.get("metrics", ["cosine", "mds", "knn"]):
        if m == "knn":
            knn_params = dict(eval_cfg.get("knn_params", {}))
            knn_params["sample_seed"] = seed
            metrics_list.append({"metric": "knn", "kwargs": knn_params, "train_selection": "all"})
        else:
            metrics_list.append(m)

    evaluator = MultiMetricHallucinationEvaluator(
        activation_parser_df=label_df,
        train_data_loader=train_loader,
        metrics=metrics_list,
        batch_size=int(eval_cfg.get("eval_batch_size", 256)),
        sub_batch_size=eval_sub_batch,
        device=device,
        num_workers=num_workers,
        persistent_workers=persistent,
        outlier_class=1,
    )

    return evaluator.compute(eval_loader, model)


def append_summary_row(summary_csv: Path, row: dict) -> None:
    new_file = not summary_csv.exists()
    with summary_csv.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


def run_one_dataset(
    *, dataset_name: str, method_cfg: dict, seed: int, device: str, output_root: Path,
    num_workers: int, summary_csv: Path,
    max_epochs_override: int | None = None,
    steps_per_epoch_override: int | None = None,
    sub_batch_size_override: int | None = None,
) -> dict:
    """Train + evaluate on one dataset. Returns a result dict; on failure raises."""
    method_name = method_cfg["name"]
    data_cfg = method_cfg["data"]
    train_cfg = method_cfg["training"]
    eval_cfg = method_cfg.get("evaluation", {})
    model_params = method_cfg.get("model_params", {})

    train_dir, test_dir, dataset_cfg = load_dataset_paths(dataset_name)

    cell_dir = output_root / dataset_name
    artifacts_dir = cell_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{method_name}/{dataset_name}] capture_train={train_dir}")
    logger.info(f"[{method_name}/{dataset_name}] capture_test={test_dir}")

    # Load capture config to get input_dim
    with (train_dir / "config.json").open() as fh:
        capture_cfg = json.load(fh)
    input_dim = int(capture_cfg["hidden_dim"])

    # Layer setup
    relevant_layers = parse_layers(data_cfg.get("relevant_layers", "14-29"))
    target_layers = data_cfg.get("target_layers", [22, 26])
    if isinstance(target_layers, str):
        target_layers = parse_layers(target_layers)

    # --- Build datasets ---
    # Training datasets: full relevant_layers range, 2 views.
    train_ds = build_dataset(train_dir, "train", data_cfg, seed, relevant_layers=relevant_layers)
    train_ds._capture_dir = str(train_dir)  # type: ignore[attr-defined]

    # Use train_dir's "val" split as the test-set proxy during training (3-way).
    val_ds = build_dataset(train_dir, "val", data_cfg, seed, relevant_layers=relevant_layers)
    val_ds._capture_dir = str(train_dir)  # type: ignore[attr-defined]

    # Eval datasets: restrict views to target_layers (one view per target layer pair).
    eval_train_ds = build_dataset(
        train_dir, "all", data_cfg, seed, relevant_layers=list(target_layers),
    )
    eval_train_ds._capture_dir = str(train_dir)  # type: ignore[attr-defined]
    eval_test_ds = build_dataset(
        test_dir, "all", data_cfg, seed, relevant_layers=list(target_layers),
    )
    eval_test_ds._capture_dir = str(test_dir)  # type: ignore[attr-defined]

    logger.info(
        f"[{method_name}/{dataset_name}] sizes: train={len(train_ds)} val={len(val_ds)} "
        f"eval_train={len(eval_train_ds)} eval_test={len(eval_test_ds)}"
    )

    # --- Build model ---
    model = build_model(input_dim, model_params)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"[{method_name}/{dataset_name}] model params={n_params:,} "
        f"lambdas=(lp={model.recon_lambda}, attn={model.attn_recon_lambda}, dir={model.attn_direction})"
    )

    # --- Train ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    epochs = int(max_epochs_override) if max_epochs_override is not None else int(train_cfg.get("max_epochs", 100))
    steps_per_epoch = (
        int(steps_per_epoch_override)
        if steps_per_epoch_override is not None
        else train_cfg.get("steps_per_epoch_override")
    )
    t0 = time.time()
    sub_batch_size = (
        int(sub_batch_size_override)
        if sub_batch_size_override is not None
        else int(train_cfg.get("sub_batch_size", 64))
    )
    train_contrastive_logprob_attn_recon(
        model=model,
        train_dataset=train_ds,
        test_dataset=val_ds,
        epochs=epochs,
        batch_size=int(train_cfg.get("batch_size", 512)),
        sub_batch_size=sub_batch_size,
        lr=float(train_cfg.get("lr", 1e-5)),
        temperature=float(train_cfg.get("temperature", 0.25)),
        device=device,
        num_workers=int(num_workers),
        checkpoint_dir=str(artifacts_dir),
        save_every=1,
        snapshot_every=0,
        use_labels=bool(train_cfg.get("use_labels", True)),
        ignore_label=int(train_cfg.get("ignore_label", 1)),
        use_infinite_index_stream=bool(train_cfg.get("use_infinite_index_stream", True)),
        infinite_stream_shuffle=True,
        infinite_stream_seed=int(seed),
        steps_per_epoch_override=steps_per_epoch,
        balanced_sampling=bool(train_cfg.get("balanced_sampling", False)),
        grad_clip_norm=train_cfg.get("grad_clip_norm"),
    )
    train_secs = time.time() - t0
    logger.info(f"[{method_name}/{dataset_name}] train: {train_secs:.0f}s")

    torch.save(
        {"model_state_dict": model.state_dict()},
        artifacts_dir / "final_weights.pt",
    )

    # --- Eval ---
    t1 = time.time()
    ood_stats = evaluate(
        model, eval_train_ds, eval_test_ds, eval_cfg, device=device, seed=seed,
        output_dir=cell_dir, num_workers=num_workers,
    )
    eval_secs = time.time() - t1
    logger.info(f"[{method_name}/{dataset_name}] eval: {eval_secs:.0f}s")

    result = {
        "method": method_name,
        "dataset": dataset_name,
        "seed": seed,
        "recon_lambda": float(model.recon_lambda),
        "attn_recon_lambda": float(model.attn_recon_lambda),
        "attn_direction": model.attn_direction,
        "attn_offset_k": int(model.attn_offset_k),
        "n_train": len(train_ds),
        "n_eval_train": len(eval_train_ds),
        "n_eval_test": len(eval_test_ds),
        "train_secs": round(train_secs, 1),
        "eval_secs": round(eval_secs, 1),
        **{k: v for k, v in ood_stats.items() if isinstance(v, (int, float, str))},
    }

    # Per-cell results JSON (full ood_stats, no scalar coercion)
    with (cell_dir / "results.json").open("w") as fh:
        json.dump({**result, "ood_stats_full": ood_stats}, fh, indent=2, default=str)

    append_summary_row(summary_csv, result)
    return result


def write_run_manifest(output_dir: Path, method_cfg: dict, datasets: list[str], seed: int) -> None:
    manifest = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "method": method_cfg["name"],
        "method_config_file": method_cfg.get("__source_path__"),
        "datasets": datasets,
        "seed": seed,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "hostname": platform.node(),
        "command": " ".join(sys.argv),
    }
    with (output_dir / "run_manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)


def main() -> int:
    args = parse_args()
    method_path = Path(args.method_config).resolve()
    with method_path.open() as fh:
        method_cfg = json.load(fh)
    method_cfg["__source_path__"] = str(method_path)

    method_name = method_cfg["name"]
    seed = int(args.seed)
    output_dir = Path(args.output_dir or f"runs/issue_75_lambda_sweep/{method_name}/seed_{seed}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    summary_csv = output_dir / "summary.csv"

    device = resolve_device(args.device)
    logger.info(f"sweep driver: method={method_name} seed={seed} device={device}")
    logger.info(f"datasets: {datasets}")
    logger.info(f"output_dir: {output_dir}")

    write_run_manifest(output_dir, method_cfg, datasets, seed)

    n_ok, n_fail = 0, 0
    fail_details: list[dict] = []
    for ds_name in datasets:
        try:
            res = run_one_dataset(
                dataset_name=ds_name,
                method_cfg=method_cfg,
                seed=seed,
                device=device,
                output_root=output_dir,
                num_workers=int(args.num_workers),
                summary_csv=summary_csv,
                max_epochs_override=args.max_epochs,
                steps_per_epoch_override=args.steps_per_epoch,
                sub_batch_size_override=args.sub_batch_size,
            )
            n_ok += 1
            logger.success(f"[{method_name}/{ds_name}] OK ({res.get('train_secs', '?')}s train, {res.get('eval_secs','?')}s eval)")
        except Exception as e:
            n_fail += 1
            tb = traceback.format_exc()
            logger.error(f"[{method_name}/{ds_name}] FAIL: {type(e).__name__}: {e}\n{tb}")
            fail_details.append({"dataset": ds_name, "exception": type(e).__name__, "message": str(e)})
            # Persist the failure to the per-cell dir so it's findable later
            (output_dir / ds_name).mkdir(parents=True, exist_ok=True)
            with (output_dir / ds_name / "FAILED.json").open("w") as fh:
                json.dump({"exception": type(e).__name__, "message": str(e), "traceback": tb}, fh, indent=2)
            # Don't free GPU here — torch will reclaim on next allocation. Continue.

    logger.info(f"sweep driver done: {n_ok} ok / {n_fail} fail / {len(datasets)} total")
    if fail_details:
        logger.warning(f"failures: {fail_details}")
        with (output_dir / "failures.json").open("w") as fh:
            json.dump(fail_details, fh, indent=2)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
