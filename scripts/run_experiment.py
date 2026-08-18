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
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# When HALLULENS_SHARED_DIR is set, paths starting with "shared/" in dataset
# configs are resolved against that directory instead of project_root. Lets a
# worktree checkout point at the canonical 20TB shared data dir without
# symlinking or modifying configs. Unset → existing behavior unchanged.
_shared_root_override = os.environ.get("HALLULENS_SHARED_DIR")


def _resolve_shared(rel_path: str) -> str:
    if _shared_root_override and rel_path.startswith("shared/"):
        return str(Path(_shared_root_override) / rel_path[len("shared/"):])
    return str(project_root / rel_path)


import torch
from loguru import logger

from scripts.experiment_utils import load_method_config


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


def _make_validation_knn_scorer(
    *,
    activation_parser_df,
    train_dataset,
    val_dataset,
    device: str,
    num_workers: int,
    eval_batch_size: int,
    sub_batch_size: int,
    outlier_class: int,
    k: int,
    max_train_size: int,
    sample_seed: int,
    prefix_length: int | None = None,
):
    """Build a fixed held-out KNN-AUROC checkpoint scorer.

    The reference bank always contains all training examples. ``k`` and the
    raw-Euclidean metric are fixed before training; validation labels are used
    only to compute AUROC, never to tune the scorer. A prefix wrapper is used
    for the layer-wise early-token model so selection matches its k=1 report.
    """
    from torch.utils.data import DataLoader

    from activation_research.metric_evaluator import (
        MultiMetricHallucinationEvaluator,
    )

    k = int(k)
    if k <= 0:
        raise ValueError("validation KNN k must be positive")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    metric_spec = {
        "metric": "knn",
        "kwargs": {
            "k": k,
            "metric": "euclidean",
            "calibrate_k": False,
            "max_train_size": int(max_train_size),
            "sample_seed": int(sample_seed),
        },
        "train_selection": "all",
    }

    def _score(model) -> float:
        score_model = model
        if prefix_length is not None:
            from activation_research.prefix_views import PrefixEvalWrapper

            score_model = PrefixEvalWrapper(model, int(prefix_length)).to(device)
        evaluator = MultiMetricHallucinationEvaluator(
            activation_parser_df=activation_parser_df,
            train_activation_parser_df=activation_parser_df,
            train_data_loader=train_loader,
            metrics=[metric_spec],
            batch_size=int(eval_batch_size),
            sub_batch_size=int(sub_batch_size),
            device=device,
            num_workers=int(num_workers),
            persistent_workers=False,
            outlier_class=int(outlier_class),
        )
        stats = evaluator.compute(val_loader, score_model)
        return float(stats["knn_auroc"])

    return _score


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


def _resolve_run_split_seed(experiment_cfg: dict, training_seed: int | None) -> int:
    """Return the fold seed paired with this training seed.

    Experiment files may specify ``split_seeds`` in parallel with
    ``training_seeds``. Routines that reconstruct their own memmap splits
    (notably ACT-ViT and ICR) must use the paired value rather than the global
    fallback, or every reported seed silently trains on the same fold.
    """
    fallback = int(experiment_cfg.get("split_seed", 42))
    split_seeds = experiment_cfg.get("split_seeds")
    training_seeds = experiment_cfg.get("training_seeds")
    if training_seed is None or split_seeds is None or training_seeds is None:
        return fallback
    try:
        index = list(training_seeds).index(training_seed)
    except ValueError:
        return fallback
    return int(split_seeds[index]) if index < len(split_seeds) else fallback


def _score_single_layer_classifier(
    model,
    dataset,
    *,
    batch_size: int,
    eval_device: torch.device,
    prefix_len: int | None = None,
    length_aware: bool = False,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Score a LinearProbe/SAPLMA-style sequence classifier.

    ``prefix_len`` selects the first k response tokens.  When either a prefix
    is requested or ``length_aware`` is true, the model receives an explicit
    mask capped by each sample's recorded response length.
    """
    import numpy as np
    from torch.utils.data import DataLoader

    from activation_research.prefix_views import build_prefix_mask

    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["views_activations"].to(eval_device)
            if x.dim() == 4:
                x = x.squeeze(1)
            if prefix_len is not None or length_aware:
                if "response_len" not in batch:
                    raise KeyError(
                        "prefix evaluation requires response_len in the "
                        "single-layer dataset"
                    )
                k = x.shape[1] if prefix_len is None else int(prefix_len)
                mask = build_prefix_mask(
                    batch["response_len"].to(eval_device),
                    seq_len=x.shape[1],
                    prefix_len=k,
                    device=eval_device,
                )
                preds = model(x, token_mask=mask)
            else:
                preds = model(x)
            all_preds.append(preds.detach().cpu())
            all_labels.append(batch["halu"].cpu())

    return (
        torch.cat(all_preds).reshape(-1).numpy().astype(np.float32),
        torch.cat(all_labels).reshape(-1).numpy().astype(np.int32),
    )


def _single_layer_prefix_curve(
    model,
    dataset,
    *,
    batch_size: int,
    eval_device: torch.device,
    eval_cfg: dict,
) -> tuple[dict, dict[int, "np.ndarray"]]:
    """Return namespaced AUROCs and per-sample scores for every requested k."""
    from sklearn.metrics import roc_auc_score

    from activation_research.prefix_views import resolve_eval_prefixes

    if not eval_cfg.get("eval_prefix_curve", False):
        return {}, {}
    max_prefix = int(getattr(dataset, "max_response_len", 64))
    if hasattr(dataset, "cache") and getattr(dataset.cache, "ndim", 0) >= 3:
        max_prefix = int(dataset.cache.shape[-2])
    ks = resolve_eval_prefixes(eval_cfg.get("eval_prefix_lengths"), max_prefix)
    curve: dict = {}
    scores_by_k: dict[int, "np.ndarray"] = {}
    for k in ks:
        scores, labels = _score_single_layer_classifier(
            model,
            dataset,
            batch_size=batch_size,
            eval_device=eval_device,
            prefix_len=k,
        )
        auroc = (
            float(roc_auc_score(labels, scores))
            if len(set(labels.tolist())) >= 2
            else float("nan")
        )
        curve[f"k{k}_auroc"] = auroc
        scores_by_k[int(k)] = scores
        logger.info(f"[single-layer prefix] k={k}: auroc={auroc}")
    return curve, scores_by_k


def _act_vit_logits_at_prefix(
    model,
    activations: torch.Tensor,
    response_lens: torch.Tensor,
    prefix_len: int,
) -> torch.Tensor:
    """Run ACT-ViT without padding beyond each sample's effective prefix.

    ACT-ViT accepts variable token width through adaptive pooling, but a batch
    tensor has one shared width.  Grouping by ``min(response_len, k)`` keeps
    early-EOS rows information-correct while preserving gradients.
    """
    if activations.dim() != 4:
        raise ValueError(
            "ACT-ViT activations must have shape (B, L, N, D), got "
            f"{tuple(activations.shape)}"
        )
    if int(prefix_len) < 1:
        raise ValueError(f"prefix_len must be >= 1, got {prefix_len}")
    lens = torch.as_tensor(
        response_lens, dtype=torch.long, device=activations.device
    ).clamp(min=1, max=min(int(prefix_len), int(activations.shape[2])))
    logits_chunks = []
    index_chunks = []
    for effective_len in lens.unique(sorted=True):
        idx = torch.nonzero(lens == effective_len, as_tuple=False).reshape(-1)
        width = int(effective_len.item())
        logits_chunks.append(
            model(activations.index_select(0, idx)[:, :, :width, :]).reshape(-1)
        )
        index_chunks.append(idx)
    joined_logits = torch.cat(logits_chunks, dim=0)
    joined_indices = torch.cat(index_chunks, dim=0)
    return joined_logits[torch.argsort(joined_indices)]


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

    # Build augmentation function if configured
    aug_cfg = data_cfg.get("augmentations", None)
    augment_fn = None
    if aug_cfg:
        from activation_research.augmentations import AugmentationComposer
        augment_fn = AugmentationComposer(
            augmentations=aug_cfg.get("ops", []),
            asymmetric=aug_cfg.get("asymmetric", False),
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

    trainer = ContrastiveTrainer(model, config=config, augment_fn=augment_fn)
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

    # flip_auroc=True (ignore_label=0 configs): hallu is the compact cluster,
    # so correct examples are the anomalies — pass outlier_class=0 to the evaluator.
    flip_auroc: bool = bool(eval_cfg.get("flip_auroc", False))
    effective_outlier_class = 0 if flip_auroc else dataset_cfg.get("outlier_class", 1)

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
        outlier_class=effective_outlier_class,
    )

    ood_stats = evaluator.compute(eval_loader, model)

    # Build eval_metrics
    eval_metrics: dict = {
        "method": method_cfg["name"],
        "dataset": dataset_cfg["name"],
        "seed": training_seed,
        "flip_auroc": flip_auroc,
        "split_seed": _resolve_run_split_seed(experiment_cfg, training_seed),
        "n_train": len(train_ds),
        "n_test": len(test_ds),
    }
    eval_metrics.update(ood_stats)

    # Contrastive OOD doesn't produce per-example predictions easily
    predictions: list[dict] = []
    return eval_metrics, predictions


def _apply_train_prevalence(parser, experiment_cfg, run_seed=None):
    """In-place subsample of a parser's 'train'-split rows to a target positive
    (halu) prevalence at a fixed total N — for the P1 imbalance sweep (#140).

    Non-selected train rows are re-labeled split='__unused__' so get_dataset()
    never returns them for any split. Val/test are left untouched (held fixed
    across all prevalence conditions). No-op when experiment_cfg has no
    'train_prevalence', so all other experiments are unaffected.

    Reads from experiment_cfg: train_prevalence (float in (0,1)), train_total_n
    (int; default = all available), resample_seed (default split_seed).

    Returns the realized kept train-row count, or None when no subsample is
    applied — so callers can assert their built train set matches it. (#140)
    """
    prev = experiment_cfg.get("train_prevalence", None)
    if prev is None:
        return None
    import numpy as np

    total_n = experiment_cfg.get("train_total_n", None)
    # Per-run resample seed so each training seed draws a distinct subsample
    # (gives honest resample-variance error bars across seeds).
    seed = run_seed if run_seed is not None else \
        experiment_cfg.get("resample_seed", experiment_cfg.get("split_seed", 42))
    df = parser.df
    tr = df[df["split"] == "train"]
    pos_idx = tr.index[tr["halu"] == 1].to_numpy()
    neg_idx = tr.index[tr["halu"] == 0].to_numpy()
    avail_pos, avail_neg = len(pos_idx), len(neg_idx)
    if total_n is None:
        total_n = avail_pos + avail_neg
    n_pos = int(round(prev * total_n))
    n_neg = total_n - n_pos
    # Cap to availability while preserving the target ratio as closely as possible.
    if n_pos > avail_pos or n_neg > avail_neg:
        scale = min(avail_pos / max(n_pos, 1), avail_neg / max(n_neg, 1), 1.0)
        n_pos, n_neg = int(n_pos * scale), int(n_neg * scale)
    rng = np.random.default_rng(seed)
    keep_pos = rng.choice(pos_idx, size=n_pos, replace=False) if n_pos > 0 else pos_idx[:0]
    keep_neg = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg > 0 else neg_idx[:0]
    keep = set(keep_pos.tolist()) | set(keep_neg.tolist())
    drop = [i for i in tr.index if i not in keep]
    df.loc[drop, "split"] = "__unused__"
    logger.info(
        f"[P1 prevalence] target_prev={prev} N={total_n} seed={seed} -> "
        f"kept pos={n_pos} neg={n_neg} (achieved_prev={n_pos/max(n_pos+n_neg,1):.3f}; "
        f"avail pos={avail_pos} neg={avail_neg}); dropped {len(drop)} train rows"
    )
    # Realized kept train-row count — runners assert their built train set
    # matches this so a future subsample-propagation regression fails loudly. (#140)
    return int((df["split"] == "train").sum())


def run_contrastive_logprob_recon(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    training_seed: int,
    test_ap=None,
    *,
    _tokenwise: bool = False,
) -> tuple[dict, list[dict]]:
    """Train and evaluate a LogprobReconProgressiveCompressor model.

    ``_tokenwise`` is used by the dedicated issue #151 entry point below.  It
    changes only the memmap view geometry and evaluation surface; model size,
    SupCon semantics, and full-response logprob reconstruction stay identical
    to the established layer-wise routine.
    """
    _p1_expected_train_n = _apply_train_prevalence(ap, experiment_cfg, run_seed=training_seed)  # P1 sweep (#140); no-op otherwise
    data_cfg = method_cfg["data"]
    train_cfg = method_cfg["training"]
    eval_cfg = method_cfg["evaluation"]

    relevant_layers = parse_layer_range(data_cfg["relevant_layers"])
    target_layers = data_cfg.get("target_layers", [])

    ds_kwargs = dict(
        relevant_layers=relevant_layers,
        num_views=data_cfg.get("num_views", 2),
        pad_length=data_cfg.get("pad_length", 63),
        preload=data_cfg.get("preload", True),
        include_response_logprobs=True,
        response_logprobs_top_k=data_cfg.get("response_logprobs_top_k", 20),
        check_ram=False,
    )

    # Build the ordinary contrastive dataset exactly once per split.  In the
    # token-wise routine, lightweight adapters below only transpose the final
    # cache slice; they reuse the base cache, labels, hashes, split rows, and
    # full-response logprob arrays.
    train_base_ds = ap.get_dataset("train", **ds_kwargs)
    train_ds = train_base_ds
    if (
        _p1_expected_train_n is not None
        and not _tokenwise
        and len(train_ds) != _p1_expected_train_n
    ):
        raise RuntimeError(
            f"[P1 guard] train_prevalence set (expected {_p1_expected_train_n} train rows) "
            f"but get_dataset('train') returned {len(train_ds)} — subsample did not "
            f"propagate to the contrastive train set."
        )
    has_val = ap.split_strategy == "three_way"
    eval_ap = test_ap if test_ap is not None else ap
    if _tokenwise:
        from activation_research.tokenwise_contrastive_dataset import (
            TokenwiseContrastiveDataset,
        )

        view_adapter = data_cfg.get(
            "view_adapter", "tokenwise_shared_contrastive_cache"
        )
        if view_adapter != "tokenwise_shared_contrastive_cache":
            raise ValueError(
                "token-wise routine requires "
                "data.view_adapter='tokenwise_shared_contrastive_cache'; "
                f"got {view_adapter!r}"
            )

        # MemmapContrastiveDataset.cache retains the complete model-layer axis,
        # so model layer IDs are also cache positions. A RAM-preloaded base
        # cache is already restricted to relevant_layers and uses 0..L-1.
        layer_positions = (
            list(relevant_layers)
            if hasattr(train_base_ds, "_relevant_layers")
            else list(range(len(relevant_layers)))
        )
        pair_mode = data_cfg.get("token_pair_mode", "first_anchored")
        shuffle_length_bucket_size = int(
            data_cfg.get("shuffle_length_bucket_size", 8)
        )
        later_token_sampling = str(
            data_cfg.get("later_token_sampling", "uniform")
        )
        later_view_probability = float(
            data_cfg.get("later_view_probability", 0.5)
        )
        pair_min_response_tokens = int(
            data_cfg.get("min_response_tokens", 2)
        )
        emit_view_logprob_targets = bool(
            data_cfg.get("emit_view_logprob_targets", False)
        )

        train_ds = TokenwiseContrastiveDataset(
            train_base_ds,
            layer_positions=layer_positions,
            num_views=data_cfg.get("num_views", 2),
            token_pair_mode=pair_mode,
            min_response_tokens=pair_min_response_tokens,
            shuffle_length_bucket_size=shuffle_length_bucket_size,
            later_token_sampling=later_token_sampling,
            later_view_probability=later_view_probability,
            emit_view_logprob_targets=emit_view_logprob_targets,
        )
        val_base_ds = (
            ap.get_dataset("val", **ds_kwargs)
            if has_val
            else eval_ap.get_dataset("test", **ds_kwargs)
        )
        val_ds = TokenwiseContrastiveDataset(
            val_base_ds,
            layer_positions=layer_positions,
            num_views=data_cfg.get("num_views", 2),
            token_pair_mode=pair_mode,
            min_response_tokens=pair_min_response_tokens,
            shuffle_length_bucket_size=shuffle_length_bucket_size,
            later_token_sampling=later_token_sampling,
            later_view_probability=later_view_probability,
            emit_view_logprob_targets=emit_view_logprob_targets,
        )
        test_base_ds = eval_ap.get_dataset("test", **ds_kwargs)
        train_eval_ds = TokenwiseContrastiveDataset(
            train_base_ds,
            layer_positions=layer_positions,
            num_views=1,
            token_pair_mode=pair_mode,
            fixed_token=0,
            min_response_tokens=1,
            shuffle_length_bucket_size=shuffle_length_bucket_size,
            later_token_sampling=later_token_sampling,
            later_view_probability=later_view_probability,
            emit_view_logprob_targets=False,
        )
        val_eval_ds = TokenwiseContrastiveDataset(
            val_base_ds,
            layer_positions=layer_positions,
            num_views=1,
            token_pair_mode=pair_mode,
            fixed_token=0,
            min_response_tokens=1,
            shuffle_length_bucket_size=shuffle_length_bucket_size,
            later_token_sampling=later_token_sampling,
            later_view_probability=later_view_probability,
            emit_view_logprob_targets=False,
        )
        test_ds = TokenwiseContrastiveDataset(
            test_base_ds,
            layer_positions=layer_positions,
            num_views=1,
            token_pair_mode=pair_mode,
            fixed_token=0,
            min_response_tokens=1,
            shuffle_length_bucket_size=shuffle_length_bucket_size,
            later_token_sampling=later_token_sampling,
            later_view_probability=later_view_probability,
            emit_view_logprob_targets=False,
        )
        logger.info(
            "token-wise cache adapter={} pair_mode={} later_sampling={} "
            "pair_min_tokens={} causal_objective={} "
            "source_aligned_recon={} base_cache_id={} "
            "train_base={} train_pairs={} train_t0={} test_t0={} "
            "depth_sequence={}",
            view_adapter,
            pair_mode,
            later_token_sampling,
            pair_min_response_tokens,
            train_cfg.get("contrastive_objective", "legacy_supcon"),
            emit_view_logprob_targets,
            id(train_base_ds.cache),
            len(train_base_ds),
            len(train_ds),
            len(train_eval_ds),
            len(test_ds),
            relevant_layers,
        )
    else:
        test_ds = eval_ap.get_dataset("test", **ds_kwargs)
        val_ds = ap.get_dataset("val", **ds_kwargs) if has_val else test_ds
        train_eval_ds = None
        val_eval_ds = None

    contrastive_objective = str(
        train_cfg.get("contrastive_objective", "legacy_supcon")
    ).strip().lower()
    if contrastive_objective == "tokenwise_causal_control":
        if not _tokenwise:
            raise ValueError(
                "tokenwise_causal_control is valid only for the token-wise routine"
            )
        if int(data_cfg.get("num_views", 2)) < 2:
            raise ValueError("tokenwise causal controls require at least two views")
        if not bool(data_cfg.get("emit_view_logprob_targets", False)):
            raise ValueError(
                "tokenwise causal controls require source-aligned per-view "
                "logprob targets"
            )
        if pair_mode not in {
            "first_anchored",
            "first_same",
            "first_mixed",
            "shuffled_later",
        }:
            raise ValueError(
                "tokenwise causal control pair mode must be one of "
                "{'first_anchored', 'first_same', 'first_mixed', "
                "'shuffled_later'}"
            )

    model_params = method_cfg.get("model_params", {})
    model_class = str(method_cfg.get("model_class", "logprob_recon_progressive_compressor")).strip().lower()
    if model_class == "logprob_recon_projected_progressive_compressor":
        from activation_research.model import (
            LogprobReconProjectedProgressiveCompressor,
        )

        model = LogprobReconProjectedProgressiveCompressor(
            input_dim=dataset_cfg["input_dim"],
            final_dim=model_params.get("final_dim", 512),
            projection_hidden_dim=model_params.get(
                "projection_hidden_dim", 512
            ),
            projection_dim=model_params.get("projection_dim", 128),
            projection_l2_normalize=model_params.get(
                "projection_l2_normalize", True
            ),
            depth_pooling=model_params.get("depth_pooling", "attention"),
            pool_num_queries=model_params.get("pool_num_queries", 1),
            dropout=model_params.get("dropout", 0.1),
            input_dropout=model_params.get("input_dropout", 0.3),
            normalize_input=model_params.get("normalize_input", False),
            block_dims=model_params.get("block_dims"),
            pre_norm=model_params.get("pre_norm", False),
            recon_seq_len=model_params.get("recon_seq_len", 64),
            recon_hidden_dim=model_params.get("recon_hidden_dim", 256),
            recon_lambda=model_params.get("recon_lambda", 1.0),
            logprob_var_threshold=model_params.get(
                "logprob_var_threshold", 1e-4
            ),
        )
    elif model_class == "logprob_recon_adapter_vit_compressor":
        from activation_research.model import LogprobReconAdapterViTCompressor

        model = LogprobReconAdapterViTCompressor(
            input_dim=dataset_cfg["input_dim"],
            d_model=model_params.get("d_model", 256),
            depth=model_params.get("depth", 4),
            num_heads=model_params.get("num_heads", 8),
            mlp_ratio=model_params.get("mlp_ratio", 4),
            dropout=model_params.get("dropout", 0.1),
            input_dropout=model_params.get("input_dropout", 0.2),
            pool=model_params.get("pool", "mean"),
            adapter_norm=model_params.get("adapter_norm", False),
            recon_seq_len=model_params.get("recon_seq_len", 64),
            recon_hidden_dim=model_params.get("recon_hidden_dim", 256),
            recon_lambda=model_params.get("recon_lambda", 1.0),
            logprob_var_threshold=model_params.get("logprob_var_threshold", 1e-4),
        )
    elif model_class == "logprob_recon_attn_pool_progressive_compressor":
        from activation_research.model import LogprobReconAttentionPoolProgressiveCompressor

        model = LogprobReconAttentionPoolProgressiveCompressor(
            input_dim=dataset_cfg["input_dim"],
            final_dim=model_params.get("final_dim", 512),
            dropout=model_params.get("dropout", 0.1),
            input_dropout=model_params.get("input_dropout", 0.3),
            normalize_input=model_params.get("normalize_input", False),
            recon_seq_len=model_params.get("recon_seq_len", 64),
            recon_hidden_dim=model_params.get("recon_hidden_dim", 256),
            recon_lambda=model_params.get("recon_lambda", 1.0),
            logprob_var_threshold=model_params.get("logprob_var_threshold", 1e-4),
            block_dims=model_params.get("block_dims"),
            pre_norm=model_params.get("pre_norm", False),
            pool_num_queries=model_params.get("pool_num_queries", 1),
        )
    elif model_class == "logprob_recon_mlp_intake":
        from activation_research.model import LogprobReconMLPIntake

        model = LogprobReconMLPIntake(
            input_dim=dataset_cfg["input_dim"],
            final_dim=model_params.get("final_dim", 256),
            intake_type=model_params.get("intake_type", "mlp"),
            hidden_dim=model_params.get("hidden_dim", 512),
            depth=model_params.get("depth", 2),
            normalize_input=model_params.get("normalize_input", False),
            input_dropout=model_params.get("input_dropout", 0.3),
            recon_seq_len=model_params.get("recon_seq_len", 64),
            recon_hidden_dim=model_params.get("recon_hidden_dim", 256),
            recon_lambda=model_params.get("recon_lambda", 1.0),
            logprob_var_threshold=model_params.get("logprob_var_threshold", 1e-4),
        )
    else:
        from activation_research.model import LogprobReconProgressiveCompressor

        model = LogprobReconProgressiveCompressor(
            input_dim=dataset_cfg["input_dim"],
            final_dim=model_params.get("final_dim", 512),
            dropout=model_params.get("dropout", 0.1),
            input_dropout=model_params.get("input_dropout", 0.3),
            normalize_input=model_params.get("normalize_input", False),
            recon_seq_len=model_params.get("recon_seq_len", 64),
            recon_hidden_dim=model_params.get("recon_hidden_dim", 256),
            recon_lambda=model_params.get("recon_lambda", 1.0),
            logprob_var_threshold=model_params.get("logprob_var_threshold", 1e-4),
            block_dims=model_params.get("block_dims"),
            pre_norm=model_params.get("pre_norm", False),
        )

    score_projection_surface = bool(
        eval_cfg.get("score_projection_surface", False)
    )
    if score_projection_surface:
        if not hasattr(model, "project"):
            raise ValueError(
                "evaluation.score_projection_surface=true requires a model "
                "with a projection head"
            )
        if not bool(eval_cfg.get("dump_embeddings", False)):
            raise ValueError(
                "projection-surface scoring requires evaluation.dump_embeddings=true"
            )

    model_total_params = sum(p.numel() for p in model.parameters())
    expected_total_params = model_params.get("expected_total_params")
    if (
        expected_total_params is not None
        and model_total_params != int(expected_total_params)
    ):
        raise RuntimeError(
            f"model parameter guard failed: expected {expected_total_params}, "
            f"constructed {model_total_params}"
        )
    reference_total_params = model_params.get("reference_total_params")
    model_parameter_delta_pct = None
    if reference_total_params is not None:
        model_parameter_delta_pct = 100.0 * (
            model_total_params - int(reference_total_params)
        ) / int(reference_total_params)
    logger.info(
        "model_class={} total_params={} reference_params={} delta_pct={}",
        model_class,
        model_total_params,
        reference_total_params,
        model_parameter_delta_pct,
    )

    train_device = device if device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Training-skip resume: if final_weights.pt already exists, load and skip training.
    checkpoint_dir = os.path.join(output_dir, "artifacts")
    final_weights_path = os.path.join(checkpoint_dir, "final_weights.pt")
    training_skipped = os.path.exists(final_weights_path)
    training_summary: dict = {}
    if training_skipped:
        logger.info(f"final_weights.pt found — skipping training, loading weights for eval")
        ckpt = torch.load(final_weights_path, map_location=train_device)
        model.load_state_dict(ckpt["model_state_dict"])
        training_summary = dict(ckpt.get("training_summary", {}))

    aug_cfg = data_cfg.get("augmentations", None)
    augment_fn = None
    if aug_cfg:
        from activation_research.augmentations import AugmentationComposer
        augment_fn = AugmentationComposer(
            augmentations=aug_cfg.get("ops", []),
            asymmetric=aug_cfg.get("asymmetric", False),
        )

    from activation_research.training import train_contrastive_logprob_recon

    select_on_val = bool(train_cfg.get("select_on_val", False))
    checkpoint_selection_metric = str(
        train_cfg.get("checkpoint_selection_metric", "validation_loss")
    ).strip().lower()
    validation_score_fn = None
    validation_score_name = "validation_score"
    if select_on_val and checkpoint_selection_metric == "knn_auroc":
        if not has_val:
            raise ValueError(
                "checkpoint_selection_metric='knn_auroc' requires a held-out "
                "validation split; refusing to select on the test capture"
            )
        selection_cfg = dict(train_cfg.get("validation_knn", {}))
        if str(selection_cfg.get("metric", "euclidean")).lower() != "euclidean":
            raise ValueError("validation checkpoint KNN must use raw Euclidean distance")
        if bool(selection_cfg.get("calibrate_k", False)):
            raise ValueError("validation checkpoint KNN must use a fixed predeclared k")
        if str(selection_cfg.get("train_selection", "all")).lower() != "all":
            raise ValueError("validation checkpoint KNN must use the all-example bank")

        if _tokenwise:
            selection_train_ds = train_eval_ds
            selection_val_ds = val_eval_ds
            selection_prefix_length = None
        else:
            selection_train_ds = train_ds.slice_layers(target_layers)
            selection_val_ds = val_ds.slice_layers(target_layers)
            selection_prefix_length = selection_cfg.get("prefix_length")

        selection_outlier_class = (
            0
            if bool(eval_cfg.get("flip_auroc", False))
            else int(dataset_cfg.get("outlier_class", 1))
        )
        validation_score_fn = _make_validation_knn_scorer(
            activation_parser_df=ap.df,
            train_dataset=selection_train_ds,
            val_dataset=selection_val_ds,
            device=train_device,
            num_workers=experiment_cfg.get("num_workers", 4),
            eval_batch_size=eval_cfg.get("eval_batch_size", 256),
            sub_batch_size=eval_cfg.get("sub_batch_size", 64),
            outlier_class=selection_outlier_class,
            k=selection_cfg.get("k", 50),
            max_train_size=selection_cfg.get("max_train_size", 200000),
            sample_seed=training_seed,
            prefix_length=selection_prefix_length,
        )
        validation_score_name = "validation_knn_auroc"
    elif select_on_val and checkpoint_selection_metric != "validation_loss":
        raise ValueError(
            "checkpoint_selection_metric must be one of "
            "{'validation_loss', 'knn_auroc'}"
        )

    if not training_skipped:
        training_summary = train_contrastive_logprob_recon(
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
            contrastive_objective=contrastive_objective,
            temporal_loss_weight=train_cfg.get("temporal_loss_weight", 1.0),
            class_loss_weight=train_cfg.get("class_loss_weight", 1.0),
            causal_temporal_mode=train_cfg.get(
                "causal_temporal_mode", "symmetric"
            ),
            persistent_workers=experiment_cfg.get("persistent_workers", True),
            recon_lambda=model_params.get("recon_lambda", 1.0),
            use_infinite_index_stream=train_cfg.get("use_infinite_index_stream", True),
            infinite_stream_shuffle=True,
            infinite_stream_seed=training_seed,
            steps_per_epoch_override=train_cfg.get("steps_per_epoch_override"),
            min_total_steps=train_cfg.get("min_total_steps"),
            balanced_sampling=train_cfg.get("balanced_sampling", False),
            grad_clip_norm=train_cfg.get("grad_clip_norm"),
            augment_fn=augment_fn,
            # Issue #149 prefix views. Default "layer_only" keeps every existing
            # config byte-identical to its pre-#149 behaviour.
            prefix_view_mode=train_cfg.get("prefix_view_mode", "layer_only"),
            prefix_min_tokens=train_cfg.get("prefix_min_tokens", 8),
            prefix_min_gap=train_cfg.get("prefix_min_gap", 8),
            prefix_sampling_lengths=train_cfg.get("prefix_sampling_lengths"),
            prefix_seed=training_seed,
            select_on_val=select_on_val,
            validation_score_fn=validation_score_fn,
            validation_score_name=validation_score_name,
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "training_summary": training_summary,
            },
            os.path.join(output_dir, "artifacts", "final_weights.pt"),
        )

    # OOD evaluation.  The established routine averages target-layer views.
    # Token-wise evaluation deliberately exposes exactly one view: token 0
    # encoded across every selected post-block layer.
    from torch.utils.data import DataLoader

    from activation_research.metric_evaluator import MultiMetricHallucinationEvaluator

    if _tokenwise:
        train_ds_target = train_eval_ds
        test_ds_target = test_ds
    else:
        train_ds_target = train_ds.slice_layers(target_layers)
        test_ds_target = test_ds.slice_layers(target_layers)

    train_loader = DataLoader(train_ds_target, batch_size=64, shuffle=False)
    eval_loader = DataLoader(test_ds_target, batch_size=64, shuffle=False)

    model.eval()

    metrics_list: list = []
    for m in eval_cfg["metrics"]:
        if _tokenwise and m == "cosine":
            # A token-wise eval item deliberately has one view (t=0), so the
            # legacy within-sample view-cosine scorer is identically zero.
            logger.warning(
                "ignoring invalid intra-view cosine metric for one-view "
                "token-wise evaluation; use cosine KNN instead"
            )
            continue
        if m == "knn":
            knn_params = dict(eval_cfg.get("knn_params", {}))
            knn_params["sample_seed"] = training_seed
            knn_params["include_per_sample"] = True
            metrics_list.append(
                {
                    "metric": "knn",
                    "kwargs": knn_params,
                    "train_selection": "all",
                }
            )
            if _tokenwise:
                cosine_knn_params = dict(
                    eval_cfg.get("cosine_knn_params", knn_params)
                )
                cosine_knn_params.update(
                    {
                        "metric": "cosine",
                        "l2_normalize": True,
                        "sample_seed": training_seed,
                        "include_per_sample": True,
                    }
                )
                metrics_list.append(
                    {
                        "name": "cosine_knn",
                        "metric": "knn",
                        "prefix": "cosine",
                        "kwargs": cosine_knn_params,
                        "train_selection": "all",
                    }
                )
        elif _tokenwise and m == "linear_probe":
            linear_probe_params = dict(eval_cfg.get("linear_probe_params", {}))
            linear_probe_params.update(
                {
                    "sample_seed": training_seed,
                    "include_per_sample": True,
                }
            )
            metrics_list.append(
                {
                    "metric": "linear_probe",
                    "kwargs": linear_probe_params,
                    "train_selection": "all",
                }
            )
        else:
            metrics_list.append(m)

    flip_auroc: bool = bool(eval_cfg.get("flip_auroc", False))
    effective_outlier_class = 0 if flip_auroc else dataset_cfg.get("outlier_class", 1)

    evaluator = MultiMetricHallucinationEvaluator(
        activation_parser_df=eval_ap.df,
        train_activation_parser_df=ap.df,
        train_data_loader=train_loader,
        metrics=metrics_list,
        batch_size=eval_cfg.get("eval_batch_size", 256),
        sub_batch_size=eval_cfg.get("sub_batch_size", 64),
        device=train_device,
        num_workers=experiment_cfg.get("num_workers", 4),
        persistent_workers=False,
        outlier_class=effective_outlier_class,
    )

    ood_stats = evaluator.compute(eval_loader, model)

    knn_scores_arr = ood_stats.pop("knn_scores", None)
    knn_labels_arr = ood_stats.pop("knn_labels", None)
    cosine_knn_scores_arr = ood_stats.pop("cosine_knn_scores", None)
    cosine_knn_labels_arr = ood_stats.pop("cosine_knn_labels", None)
    linear_probe_scores_arr = ood_stats.pop("linear_probe_scores", None)
    linear_probe_labels_arr = ood_stats.pop("linear_probe_labels", None)

    # ---- Issue #149: AUROC as a function of the response prefix ------------
    # The headline question is whether a hallucination is detectable at token 16,
    # before the generation finishes. The full-length numbers above are the k=64
    # point of that curve; the loop below fills in the rest.
    #
    # Both the train (KNN reference) and test loaders are re-encoded at each k.
    # Scoring a k-truncated test set against a full-length KNN base would
    # measure distribution shift between the two, not early detectability.
    prefix_curve: dict = {}
    eval_prefixes_cfg = eval_cfg.get("eval_prefix_lengths", None)
    if eval_cfg.get("eval_prefix_curve", False):
        from activation_research.prefix_views import (
            PrefixEvalWrapper,
            resolve_eval_prefixes,
        )

        _capture_r_max = getattr(test_ds, "_max_resp", None) or 64
        _ks = resolve_eval_prefixes(eval_prefixes_cfg, int(_capture_r_max))
        logger.info(f"prefix curve: evaluating at k={_ks} (capture width {_capture_r_max})")

        for _k in _ks:
            _wrapped = PrefixEvalWrapper(model, _k).to(train_device)
            _wrapped.eval()
            _evaluator_k = MultiMetricHallucinationEvaluator(
                activation_parser_df=eval_ap.df,
                train_activation_parser_df=ap.df,
                train_data_loader=DataLoader(
                    train_ds_target, batch_size=64, shuffle=False
                ),
                metrics=metrics_list,
                batch_size=eval_cfg.get("eval_batch_size", 256),
                sub_batch_size=eval_cfg.get("sub_batch_size", 64),
                device=train_device,
                num_workers=experiment_cfg.get("num_workers", 4),
                persistent_workers=False,
                outlier_class=effective_outlier_class,
            )
            _stats_k = _evaluator_k.compute(
                DataLoader(test_ds_target, batch_size=64, shuffle=False), _wrapped
            )
            _stats_k.pop("knn_scores", None)
            _stats_k.pop("knn_labels", None)
            _stats_k.pop("cosine_knn_scores", None)
            _stats_k.pop("cosine_knn_labels", None)
            _stats_k.pop("linear_probe_scores", None)
            _stats_k.pop("linear_probe_labels", None)
            for _metric_name, _v in _stats_k.items():
                prefix_curve[f"k{_k}_{_metric_name}"] = _v
            logger.info(
                f"  k={_k}: knn_auroc={_stats_k.get('knn_auroc')} "
                f"cosine_auroc={_stats_k.get('cosine_auroc')} "
                f"mahalanobis_auroc={_stats_k.get('mahalanobis_auroc')}"
            )

    # ---- Issue #151: token-position diagnostic curve --------------------
    # The primary score above is t=0.  Later-token diagnostics always rebuild
    # BOTH the KNN reference bank and the test surface at the same token index;
    # mixing token indices would confound detection with representation shift.
    token_curve: dict = {}
    if _tokenwise and eval_cfg.get("eval_token_curve", False):
        # The requested frozen linear probe is a token-zero scorer. Avoid
        # fitting six additional probes per run for the later-token diagnostic
        # curve; distance metrics remain comparable at every position.
        token_curve_metrics = [
            spec
            for spec in metrics_list
            if not (
                isinstance(spec, dict)
                and str(spec.get("metric", "")).lower() == "linear_probe"
            )
        ]
        token_positions = [
            int(t) for t in eval_cfg.get(
                "eval_token_positions", [0, 1, 3, 7, 15, 31, 63]
            )
        ]
        for token_index in token_positions:
            if token_index == 0:
                token_curve["t0_n_train"] = len(train_ds_target)
                token_curve["t0_n_test"] = len(test_ds_target)
                for metric_name, value in ood_stats.items():
                    token_curve[f"t0_{metric_name}"] = value
                continue
            try:
                train_at_t = train_eval_ds.fixed_token_view(token_index)
                test_at_t = test_ds.fixed_token_view(token_index)
            except ValueError as exc:
                logger.warning(
                    "token curve t={}: no eligible rows ({})", token_index, exc
                )
                token_curve[f"t{token_index}_n_train"] = 0
                token_curve[f"t{token_index}_n_test"] = 0
                continue

            evaluator_at_t = MultiMetricHallucinationEvaluator(
                activation_parser_df=eval_ap.df,
                train_activation_parser_df=ap.df,
                train_data_loader=DataLoader(
                    train_at_t, batch_size=64, shuffle=False
                ),
                metrics=token_curve_metrics,
                batch_size=eval_cfg.get("eval_batch_size", 256),
                sub_batch_size=eval_cfg.get("sub_batch_size", 64),
                device=train_device,
                num_workers=experiment_cfg.get("num_workers", 4),
                persistent_workers=False,
                outlier_class=effective_outlier_class,
            )
            stats_at_t = evaluator_at_t.compute(
                DataLoader(test_at_t, batch_size=64, shuffle=False), model
            )
            stats_at_t.pop("knn_scores", None)
            stats_at_t.pop("knn_labels", None)
            stats_at_t.pop("cosine_knn_scores", None)
            stats_at_t.pop("cosine_knn_labels", None)
            token_curve[f"t{token_index}_n_train"] = len(train_at_t)
            token_curve[f"t{token_index}_n_test"] = len(test_at_t)
            for metric_name, value in stats_at_t.items():
                token_curve[f"t{token_index}_{metric_name}"] = value

    eval_metrics: dict = {
        "method": method_cfg["name"],
        "dataset": dataset_cfg["name"],
        "seed": training_seed,
        "flip_auroc": flip_auroc,
        "split_seed": _resolve_run_split_seed(experiment_cfg, training_seed),
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "training_min_total_steps": train_cfg.get("min_total_steps"),
        "checkpoint_selection": training_summary.get(
            "checkpoint_selection",
            "maximum_validation_knn_auroc"
            if select_on_val and checkpoint_selection_metric == "knn_auroc"
            else "minimum_validation_loss"
            if select_on_val
            else "final_epoch",
        ),
        "best_validation_score": training_summary.get(
            "best_validation_score"
        ),
        "best_validation_epoch": training_summary.get(
            "best_validation_epoch"
        ),
    }
    if _tokenwise:
        eval_metrics.update(
            {
                "view_axis": "token",
                "view_adapter": data_cfg.get(
                    "view_adapter", "tokenwise_shared_contrastive_cache"
                ),
                "token_pair_mode": data_cfg.get(
                    "token_pair_mode", "first_anchored"
                ),
                "contrastive_objective": contrastive_objective,
                "temporal_loss_weight": float(
                    train_cfg.get("temporal_loss_weight", 1.0)
                ),
                "class_loss_weight": float(
                    train_cfg.get("class_loss_weight", 1.0)
                ),
                "causal_temporal_mode": str(
                    train_cfg.get("causal_temporal_mode", "symmetric")
                ),
                "source_aligned_view_reconstruction": bool(
                    data_cfg.get("emit_view_logprob_targets", False)
                ),
                "shuffle_length_bucket_size": int(
                    data_cfg.get("shuffle_length_bucket_size", 8)
                ),
                "later_token_sampling": str(
                    data_cfg.get("later_token_sampling", "uniform")
                ),
                "later_view_probability": (
                    float(data_cfg.get("later_view_probability", 0.5))
                    if pair_mode == "first_mixed"
                    else None
                ),
                "pair_min_response_tokens": int(
                    data_cfg.get("min_response_tokens", 2)
                ),
                "primary_eval_token": 0,
                "activation_cache_reused": bool(
                    train_ds.cache is train_eval_ds.cache
                ),
                "n_train_base": len(train_base_ds),
                "n_train_pair_eligible": len(train_ds),
                "n_train_t0": len(train_ds_target),
                "model_class": model_class,
                "model_total_params": model_total_params,
                "model_encoder_params": sum(
                    p.numel() for p in model.encoder.parameters()
                ),
                "model_reference_total_params": reference_total_params,
                "model_parameter_delta_pct": model_parameter_delta_pct,
                "study_issue": method_cfg.get("study_issue"),
                "architecture_arm": method_cfg.get("architecture_arm"),
                "training_recipe": method_cfg.get("training_recipe"),
                "base_method_config": method_cfg.get("extends"),
            }
        )
        architecture_metadata = getattr(model, "architecture_metadata", None)
        if architecture_metadata is not None:
            eval_metrics.update(architecture_metadata())
    eval_metrics.update(ood_stats)
    # Prefix-curve cells are namespaced k{K}_* so they never collide with the
    # full-length metrics above.
    eval_metrics.update(prefix_curve)
    eval_metrics.update(token_curve)

    predictions: list[dict] = []
    if knn_scores_arr is not None and knn_labels_arr is not None:
        # Under flip_auroc=True (B5: ignore_label=0), the contrastive loss compacts hallu
        # and the KNN base contains all train samples — so low knn_score = close to hallu cluster = halu.
        # binary_outlier_labels is (test_label == 0), i.e. 1 = correct. Negate both so predictions.csv
        # always has score_halu="higher = more halu" and label_halu="1 = halu" regardless of convention;
        # AUROC computed downstream from these columns then matches eval_metrics.knn_auroc.
        if flip_auroc:
            predictions = [
                {"example_id": i, "score_halu": -float(s), "label_halu": 1 - int(l)}
                for i, (s, l) in enumerate(zip(knn_scores_arr, knn_labels_arr))
            ]
        else:
            predictions = [
                {"example_id": i, "score_halu": float(s), "label_halu": int(l)}
                for i, (s, l) in enumerate(zip(knn_scores_arr, knn_labels_arr))
            ]

        def _secondary_halu_scores(scores, labels, *, probability: bool):
            if scores is None or labels is None:
                return None
            if len(scores) != len(predictions) or len(labels) != len(predictions):
                raise RuntimeError("secondary scorer output is not aligned with KNN predictions")
            if flip_auroc:
                # The positive outlier is label 0 under flip_auroc. Probability
                # scores invert via 1-p; unbounded distance scores via negation.
                return [
                    1.0 - float(score) if probability else -float(score)
                    for score in scores
                ]
            return [float(score) for score in scores]

        cosine_halu_scores = _secondary_halu_scores(
            cosine_knn_scores_arr, cosine_knn_labels_arr, probability=False
        )
        linear_halu_scores = _secondary_halu_scores(
            linear_probe_scores_arr, linear_probe_labels_arr, probability=True
        )
        for i, prediction in enumerate(predictions):
            if cosine_halu_scores is not None:
                prediction["score_halu_cosine_knn"] = cosine_halu_scores[i]
            if linear_halu_scores is not None:
                prediction["score_halu_linear_probe"] = linear_halu_scores[i]

    # Optional: dump predicted train+test embeddings as memmap-friendly .npy files
    # for downstream reuse (e.g. KNN-k sweeps, cosine-collapse analysis). Opt-in via
    # eval_cfg["dump_embeddings"] — runs only at the final test eval, and downstream
    # callers must explicitly load via np.load(..., mmap_mode='r'); no path in this
    # codebase reads these files automatically.
    if eval_cfg.get("dump_embeddings", False):
        # Dump failures must not invalidate the eval — the metrics + predictions
        # are already in hand. Log the traceback and move on.
        try:
            from activation_research.evaluation import dump_embeddings_to_memmap

            emb_dir = os.path.join(output_dir, "embeddings")

            # Test records: the evaluator's labeling used eval_ap.df which is the
            # test parser's df, so test hashkeys resolve and labels are correct.
            test_records = getattr(evaluator, "_labeled_test_embeddings", None)

            # Baseline records were resolved against the train parser (ap.df),
            # independently of the test parser used above.
            train_records = getattr(evaluator, "_labeled_baseline_embeddings", None)

            if train_records:
                train_meta = dump_embeddings_to_memmap(train_records, emb_dir, "train")
                logger.info(f"dumped train embeddings: {train_meta['n']} × {train_meta['z_shape'][1:]} -> {emb_dir}")
            else:
                logger.warning("dump_embeddings=true but no labeled train records (check ap.df); skipping train dump")
            if test_records:
                test_meta = dump_embeddings_to_memmap(test_records, emb_dir, "test")
                logger.info(f"dumped test embeddings: {test_meta['n']} × {test_meta['z_shape'][1:]} -> {emb_dir}")
            else:
                logger.warning("dump_embeddings=true but evaluator has no _labeled_test_embeddings; skipping test dump")
        except Exception:
            if score_projection_surface:
                raise
            logger.exception("dump_embeddings failed; continuing without embeddings dump")

    # Issue #156: the projection-only arm reports both deployment surfaces in
    # one cell. The encoder has already produced and dumped its token-zero
    # trunk, so applying the small saved projection MLP avoids a second encoder
    # pass and prevents a scoring dependency from racing the training cell.
    if score_projection_surface:
        from activation_research.projection_scoring import (
            merge_projection_surface_results,
            score_saved_projection,
        )

        projection_metrics, projection_predictions = score_saved_projection(
            output_dir,
            evaluation_cfg=eval_cfg,
            outlier_class=int(effective_outlier_class),
            sample_seed=int(training_seed),
            device=train_device,
        )
        merge_projection_surface_results(
            eval_metrics,
            predictions,
            projection_metrics,
            projection_predictions,
        )

    return eval_metrics, predictions


def run_tokenwise_contrastive_logprob_recon(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    training_seed: int,
    test_ap=None,
) -> tuple[dict, list[dict]]:
    """Issue #151 token-pair contrastive training with token-0 KNN scoring."""
    return run_contrastive_logprob_recon(
        ap,
        dataset_cfg,
        method_cfg,
        experiment_cfg,
        output_dir,
        device,
        training_seed,
        test_ap=test_ap,
        _tokenwise=True,
    )


def run_tokenwise_projection_rescore(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    training_seed: int,
    test_ap=None,
) -> tuple[dict, list[dict]]:
    """Score a trained token-wise projection head without encoder retraining.

    The source run is the sibling method directory named by ``source_method``.
    Its dumped 512-d token-zero trunk embeddings are passed through the saved
    projection MLP, then evaluated with the same KNN/probe implementations used
    by the ordinary token-wise evaluation path.
    """
    del ap, test_ap
    from pathlib import Path

    from activation_research.projection_scoring import score_saved_projection

    source_method = str(method_cfg["source_method"])
    source_run_dir = (
        Path(output_dir).parent.parent / source_method / f"seed_{training_seed}"
    )
    metrics, predictions = score_saved_projection(
        source_run_dir,
        evaluation_cfg=method_cfg["evaluation"],
        outlier_class=int(dataset_cfg.get("outlier_class", 1)),
        sample_seed=int(training_seed),
        device=device,
    )
    metrics.update(
        {
            "method": method_cfg["name"],
            "dataset": dataset_cfg["name"],
            "seed": int(training_seed),
            "split_seed": _resolve_run_split_seed(experiment_cfg, training_seed),
            "source_method": source_method,
            "training_performed": False,
        }
    )
    return metrics, predictions


def run_contrastive_logprob_recon_shared_trunk(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    training_seed: int,
    test_ap=None,
) -> tuple[dict, list[dict]]:
    """Train a shared-trunk twin-head model (D1 or D2) and evaluate.

    Reads ``method_cfg["model_params"]["variant"]`` to select the model class:
    - ``"split_output"`` → ``SharedTrunkSplitOutputCompressor`` (D1):
        single trunk with ``final_dim = 2 * half_dim``.  Eval surface is
        the full 2D embedding — KNN/Maha only (no cosine).
    - ``"projection_head"`` → ``SharedTrunkProjectionHeadCompressor`` (D2):
        trunk + two projection heads.  Heads discarded at eval.  Eval
        surface is the trunk — KNN/Maha + cosine apply.

    A single checkpoint ``artifacts/final_weights.pt`` is saved.  The model
    is passed directly to ``MultiMetricHallucinationEvaluator`` (no
    ``TwinConcatModel`` wrapper needed).
    """
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

    from activation_research.model import (
        SharedTrunkSplitOutputCompressor,
        SharedTrunkProjectionHeadCompressor,
    )
    from activation_research.training import train_contrastive_logprob_recon_dualloss

    model_params = method_cfg.get("model_params", {})
    variant = model_params.get("variant", "split_output")

    train_device = device if device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if variant == "split_output":
        model = SharedTrunkSplitOutputCompressor(
            input_dim=dataset_cfg["input_dim"],
            half_dim=model_params.get("half_dim", 256),
            input_dropout=model_params.get("input_dropout", 0.3),
            recon_seq_len=model_params.get("recon_seq_len", 64),
            recon_hidden_dim=model_params.get("recon_hidden_dim", 256),
            recon_lambda=model_params.get("recon_lambda", 1.0),
            logprob_var_threshold=model_params.get("logprob_var_threshold", 1e-4),
        )
    elif variant == "projection_head":
        model = SharedTrunkProjectionHeadCompressor(
            input_dim=dataset_cfg["input_dim"],
            trunk_dim=model_params.get("trunk_dim", 512),
            head_dim=model_params.get("head_dim", 256),
            head_hidden_dim=model_params.get("head_hidden_dim", 256),
            input_dropout=model_params.get("input_dropout", 0.3),
            recon_seq_len=model_params.get("recon_seq_len", 64),
            recon_hidden_dim=model_params.get("recon_hidden_dim", 256),
            recon_lambda=model_params.get("recon_lambda", 1.0),
            logprob_var_threshold=model_params.get("logprob_var_threshold", 1e-4),
        )
    else:
        raise ValueError(
            f"Unknown shared-trunk variant '{variant}'. "
            "Expected 'split_output' or 'projection_head'."
        )

    artifacts_dir = os.path.join(output_dir, "artifacts")
    final_weights = os.path.join(artifacts_dir, "final_weights.pt")

    if os.path.exists(final_weights):
        logger.info("final_weights.pt found — skipping training, loading weights for eval")
        ckpt = torch.load(final_weights, map_location=train_device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        os.makedirs(artifacts_dir, exist_ok=True)
        train_contrastive_logprob_recon_dualloss(
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
            checkpoint_dir=artifacts_dir,
            save_every=1,
            snapshot_every=10,
            snapshot_keep_last=3,
            persistent_workers=experiment_cfg.get("persistent_workers", True),
            recon_lambda=model_params.get("recon_lambda", 1.0),
            use_infinite_index_stream=train_cfg.get("use_infinite_index_stream", True),
            infinite_stream_shuffle=True,
            infinite_stream_seed=training_seed,
            steps_per_epoch_override=train_cfg.get("steps_per_epoch_override"),
            balanced_sampling=train_cfg.get("balanced_sampling", False),
            grad_clip_norm=train_cfg.get("grad_clip_norm"),
            augment_fn=None,
            ignore_labels=(1, 0),
        )
        torch.save(
            {"model_state_dict": model.state_dict()},
            final_weights,
        )

    model.eval()

    from torch.utils.data import DataLoader
    from activation_research.metric_evaluator import MultiMetricHallucinationEvaluator

    train_ds_target = train_ds.slice_layers(target_layers)
    test_ds_target = test_ds.slice_layers(target_layers)

    train_loader = DataLoader(train_ds_target, batch_size=64, shuffle=False)
    eval_loader = DataLoader(test_ds_target, batch_size=64, shuffle=False)

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

    flip_auroc: bool = bool(eval_cfg.get("flip_auroc", False))
    effective_outlier_class = 0 if flip_auroc else dataset_cfg.get("outlier_class", 1)

    evaluator = MultiMetricHallucinationEvaluator(
        activation_parser_df=eval_ap.df,
        train_data_loader=train_loader,
        metrics=metrics_list,
        batch_size=eval_cfg.get("eval_batch_size", 256),
        sub_batch_size=eval_cfg.get("sub_batch_size", 64),
        device=train_device,
        num_workers=experiment_cfg.get("num_workers", 4),
        persistent_workers=False,
        outlier_class=effective_outlier_class,
    )

    ood_stats = evaluator.compute(eval_loader, model)

    eval_metrics: dict = {
        "method": method_cfg["name"],
        "dataset": dataset_cfg["name"],
        "seed": training_seed,
        "flip_auroc": flip_auroc,
        "split_seed": _resolve_run_split_seed(experiment_cfg, training_seed),
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "variant": variant,
    }
    eval_metrics.update(ood_stats)

    predictions: list[dict] = []
    return eval_metrics, predictions


def _build_classifier_predictions(classifier_scores, classifier_labels) -> list[dict]:
    """Convert aligned classifier arrays to the canonical prediction rows."""
    if len(classifier_scores) != len(classifier_labels):
        raise ValueError(
            "classifier score/label length mismatch: "
            f"{len(classifier_scores)} != {len(classifier_labels)}"
        )
    return [
        {"example_id": i, "score_halu": float(s), "label_halu": int(l)}
        for i, (s, l) in enumerate(zip(classifier_scores, classifier_labels))
    ]


def _clear_stale_run_error(run_error_path: str) -> bool:
    """Remove a superseded failure marker after all result writes succeed."""
    if not os.path.exists(run_error_path):
        return False
    os.remove(run_error_path)
    return True


def run_contrastive_logprob_recon_dualhead_fusion(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    training_seed: int,
    test_ap=None,
) -> tuple[dict, list[dict]]:
    """Train/evaluate models with standard and mirrored contrastive heads.

    The SS-1 fix over the D2 shared-trunk routine (#102/#128): the spine is protected
    (``spine_supcon_grad_scale`` on the model) and **eval scores each head's own
    embedding** (head A under ignore=1 → std convention; head B under ignore=0 →
    flip convention), fusing at the SCORE level via an a-priori equal-weight
    percentile-rank average — never the trunk, never concat (the #128 lesson).
    """
    import numpy as np
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader
    from activation_research.metric_evaluator import MultiMetricHallucinationEvaluator
    from activation_research.model import (
        DualConventionContrastiveClassifier,
        SharedSpineDualHeadCompressor,
        SharedStemDualBranchCompressor,
    )
    from activation_research.training import train_contrastive_logprob_recon_dualloss

    data_cfg = method_cfg["data"]
    train_cfg = method_cfg["training"]
    eval_cfg = method_cfg["evaluation"]
    model_params = method_cfg.get("model_params", {})

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

    train_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    model_class = method_cfg.get("model_class", "shared_spine_dual_head")
    if model_class == "dual_convention_contrastive_classifier":
        # Issue-149 scope expansion: two complete headline-size encoders under
        # one model, trained jointly with opposite label conventions and a
        # supervised classifier over their concatenated representations.
        model = DualConventionContrastiveClassifier(
            input_dim=dataset_cfg["input_dim"],
            final_dim=model_params.get("final_dim", 512),
            classifier_hidden_dim=model_params.get(
                "classifier_hidden_dim", 128
            ),
            classifier_dropout=model_params.get("classifier_dropout", 0.1),
            classifier_normalize_branches=model_params.get(
                "classifier_normalize_branches", True
            ),
            input_dropout=model_params.get("input_dropout", 0.3),
            normalize_input=model_params.get("normalize_input", False),
            recon_seq_len=model_params.get("recon_seq_len", 64),
            recon_hidden_dim=model_params.get("recon_hidden_dim", 256),
            recon_lambda=model_params.get("recon_lambda", 1.0),
            logprob_var_threshold=model_params.get(
                "logprob_var_threshold", 1e-4
            ),
            block_dims=model_params.get("block_dims"),
            pre_norm=model_params.get("pre_norm", False),
        )
    elif model_class == "shared_stem_dual_branch":
        # SS-1b: early split — shared input stem, two independent branch sub-encoders.
        model = SharedStemDualBranchCompressor(
            input_dim=dataset_cfg["input_dim"],
            stem_dim=model_params.get("stem_dim", 2048),
            stem_type=model_params.get("stem_type", "transformer"),
            branch_block_dims=model_params.get("branch_block_dims", [1024, 512]),
            final_dim=model_params.get("final_dim", 512),
            input_dropout=model_params.get("input_dropout", 0.3),
            recon_seq_len=model_params.get("recon_seq_len", 64),
            recon_hidden_dim=model_params.get("recon_hidden_dim", 256),
            recon_lambda=model_params.get("recon_lambda", 1.0),
            logprob_var_threshold=model_params.get("logprob_var_threshold", 1e-4),
        )
    else:
        # SS-1: shared spine + two heads, with protected-spine grad scale.
        model = SharedSpineDualHeadCompressor(
            input_dim=dataset_cfg["input_dim"],
            trunk_dim=model_params.get("trunk_dim", 512),
            head_dim=model_params.get("head_dim", 512),
            head_hidden_dim=model_params.get("head_hidden_dim", 512),
            head_depth=model_params.get("head_depth", 2),
            spine_supcon_grad_scale=model_params.get("spine_supcon_grad_scale", 0.0),
            input_dropout=model_params.get("input_dropout", 0.3),
            recon_seq_len=model_params.get("recon_seq_len", 64),
            recon_hidden_dim=model_params.get("recon_hidden_dim", 256),
            recon_lambda=model_params.get("recon_lambda", 1.0),
            logprob_var_threshold=model_params.get("logprob_var_threshold", 1e-4),
        )

    artifacts_dir = os.path.join(output_dir, "artifacts")
    final_weights = os.path.join(artifacts_dir, "final_weights.pt")
    if os.path.exists(final_weights):
        logger.info("final_weights.pt found — skipping training, loading weights for eval")
        model.load_state_dict(torch.load(final_weights, map_location=train_device)["model_state_dict"])
    else:
        os.makedirs(artifacts_dir, exist_ok=True)
        _classifier_pos_weight = train_cfg.get("classifier_pos_weight")
        if _classifier_pos_weight == "auto":
            _train_labels = ap.df.loc[ap.df["split"] == "train", "halu"]
            _n_pos = int((_train_labels == 1).sum())
            _n_neg = int((_train_labels == 0).sum())
            _classifier_pos_weight = _n_neg / max(_n_pos, 1)
            logger.info(
                f"dual-convention classifier pos_weight="
                f"{_classifier_pos_weight:.4f} ({_n_neg} truth/{_n_pos} halu)"
            )
        train_contrastive_logprob_recon_dualloss(
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
            checkpoint_dir=artifacts_dir,
            save_every=1,
            snapshot_every=10,
            snapshot_keep_last=3,
            persistent_workers=experiment_cfg.get("persistent_workers", True),
            recon_lambda=model_params.get("recon_lambda", 1.0),
            use_infinite_index_stream=train_cfg.get("use_infinite_index_stream", True),
            infinite_stream_shuffle=True,
            infinite_stream_seed=training_seed,
            steps_per_epoch_override=train_cfg.get("steps_per_epoch_override"),
            min_total_steps=train_cfg.get("min_total_steps"),
            balanced_sampling=train_cfg.get("balanced_sampling", False),
            grad_clip_norm=train_cfg.get("grad_clip_norm"),
            augment_fn=None,
            ignore_labels=tuple(train_cfg.get("ignore_labels", (1, 0))),
            classifier_lambda=train_cfg.get("classifier_lambda", 0.0),
            classifier_pos_weight=_classifier_pos_weight,
            prefix_view_mode=train_cfg.get("prefix_view_mode", "layer_only"),
            prefix_min_tokens=train_cfg.get("prefix_min_tokens", 8),
            prefix_min_gap=train_cfg.get("prefix_min_gap", 8),
            prefix_sampling_lengths=train_cfg.get("prefix_sampling_lengths"),
            prefix_seed=training_seed,
        )
        torch.save({"model_state_dict": model.state_dict()}, final_weights)

    model.eval().to(train_device)

    # --- Dual-head score-fusion eval ---------------------------------------
    # A thin view exposing one head as the eval embedding model, so we can reuse
    # the proven single-head evaluator (per-sample KNN) once per convention.
    class _HeadView(nn.Module):
        def __init__(self, base, which):
            super().__init__()
            self.base = base
            self.which = which

        def forward(self, x, **kwargs):
            # Works for both SS-1 (head on trunk) and SS-1b (branch on stem seq).
            return self.base.embed_head(x, self.which, **kwargs)

    train_loader = DataLoader(train_ds.slice_layers(target_layers), batch_size=64, shuffle=False)
    eval_loader = DataLoader(test_ds.slice_layers(target_layers), batch_size=64, shuffle=False)

    def _head_scores(which, outlier_class, flip, prefix_len=None):
        knn_params = dict(eval_cfg.get("knn_params", {}))
        knn_params["sample_seed"] = training_seed
        knn_params["include_per_sample"] = True
        head_model = _HeadView(model, which)
        if prefix_len is not None:
            from activation_research.prefix_views import PrefixEvalWrapper

            head_model = PrefixEvalWrapper(head_model, int(prefix_len))
        evaluator = MultiMetricHallucinationEvaluator(
            activation_parser_df=eval_ap.df,
            train_activation_parser_df=ap.df,
            train_data_loader=train_loader,
            metrics=[{"metric": "knn", "kwargs": knn_params, "train_selection": "all"}],
            batch_size=eval_cfg.get("eval_batch_size", 256),
            sub_batch_size=eval_cfg.get("sub_batch_size", 64),
            device=train_device,
            num_workers=experiment_cfg.get("num_workers", 4),
            persistent_workers=False,
            outlier_class=outlier_class,
        )
        stats = evaluator.compute(eval_loader, head_model)
        s = np.asarray(stats.pop("knn_scores"), dtype=np.float64)
        l = np.asarray(stats.pop("knn_labels"), dtype=np.int64)
        # Orient to "higher = more halu" (mirror the headline routine's flip rule).
        if flip:
            return -s, 1 - l, float(stats["knn_auroc"])
        return s, l, float(stats["knn_auroc"])

    # Prefix-trained models must mask zero-padded response rows even for their
    # full-response score. k=r_max retains every real token while excluding
    # padding, matching the training-time token-mask contract.
    _full_prefix = (
        int(getattr(test_ds, "_max_resp", None) or 64)
        if model_class == "dual_convention_contrastive_classifier"
        else None
    )
    s_std, lab, auroc_std = _head_scores(
        "A", outlier_class=1, flip=False, prefix_len=_full_prefix
    )
    s_flip, lab_b, auroc_flip = _head_scores(
        "B", outlier_class=0, flip=True, prefix_len=_full_prefix
    )

    # Both passes iterate the same shuffle=False eval_loader → index-aligned.
    assert np.array_equal(lab, lab_b), "head A/B eval ordering diverged — cannot fuse"

    def _pct_rank(a):
        order = np.empty(len(a), dtype=np.float64)
        order[np.argsort(a, kind="mergesort")] = np.arange(len(a))
        return order / max(1, len(a) - 1)

    pr_std, pr_flip = _pct_rank(s_std), _pct_rank(s_flip)
    fused = 0.5 * pr_std + 0.5 * pr_flip          # a-priori EQUAL weight (#127/#129)
    auroc_fused = float(roc_auc_score(lab, fused))
    # Diagnostics: how correlated are the two heads (the thing that determines
    # whether fusion can help), and would the test-optimal weight — UPPER BOUND
    # ONLY, not a usable result — beat equal weight?
    head_score_corr = float(np.corrcoef(pr_std, pr_flip)[0, 1])
    _ws = np.linspace(0.0, 1.0, 21)
    _oa = [roc_auc_score(lab, w * pr_std + (1.0 - w) * pr_flip) for w in _ws]
    _j = int(np.argmax(_oa))
    auroc_fused_oracle, fusion_oracle_weight_std = float(_oa[_j]), float(_ws[_j])

    eval_metrics: dict = {
        "method": method_cfg["name"],
        "dataset": dataset_cfg["name"],
        "seed": training_seed,
        "split_seed": _resolve_run_split_seed(experiment_cfg, training_seed),
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "model_class": model_class,
        "spine_supcon_grad_scale": float(model_params.get("spine_supcon_grad_scale", 0.0)),
        "knn_auroc_head_std": auroc_std,
        "knn_auroc_head_flip": auroc_flip,
        "knn_auroc_standard": auroc_std,
        "knn_auroc_mirrored": auroc_flip,
        # Headline = the fused score (the SS-1 method output).
        "knn_auroc": auroc_fused,
        "knn_auroc_fused": auroc_fused,
        "fusion_delta_vs_best": auroc_fused - max(auroc_std, auroc_flip),
        # head-correlation + best-weight diagnostics (the "why doesn't fusion help" probe)
        "head_score_corr": head_score_corr,
        "knn_auroc_fused_oracle": auroc_fused_oracle,
        "fusion_oracle_weight_std": fusion_oracle_weight_std,
        "fusion_oracle_delta_vs_best": auroc_fused_oracle - max(auroc_std, auroc_flip),
    }

    classifier_scores = None
    classifier_labels = None
    if model_class == "dual_convention_contrastive_classifier":
        from sklearn.metrics import average_precision_score

        def _classifier_scores(prefix_len=None):
            scores, labels = [], []
            with torch.no_grad():
                for batch in eval_loader:
                    views = batch["views_activations"].to(
                        train_device, non_blocking=True
                    )
                    bsz, num_views, seq_len, hidden_dim = views.shape
                    # Always exclude padded rows. prefix_len=None means every
                    # real response token, not every fixed-width storage row.
                    raw_valid = views.detach().ne(0).any(dim=-1)
                    response_lens = raw_valid.long().sum(dim=-1).clamp(
                        min=1, max=seq_len
                    )
                    if prefix_len is None:
                        effective = response_lens
                    else:
                        effective = torch.minimum(
                            response_lens,
                            torch.full_like(response_lens, int(prefix_len)),
                        )
                    positions = torch.arange(
                        seq_len, device=views.device
                    ).view(1, 1, -1)
                    token_mask = positions < effective.unsqueeze(-1)
                    views = views * token_mask.unsqueeze(-1).to(views.dtype)
                    x_flat = views.reshape(
                        bsz * num_views, seq_len, hidden_dim
                    )
                    mask_flat = (
                        token_mask.reshape(bsz * num_views, seq_len)
                        if token_mask is not None
                        else None
                    )
                    _, z_std, z_mirror, _ = model.forward_with_heads(
                        x_flat, token_mask=mask_flat
                    )
                    logits = model.classifier_logits_from_views(
                        z_std.reshape(bsz, num_views, -1),
                        z_mirror.reshape(bsz, num_views, -1),
                    )
                    scores.append(torch.sigmoid(logits).cpu())
                    labels.append(batch["halu"].reshape(-1).cpu())
            return (
                torch.cat(scores).numpy(),
                torch.cat(labels).numpy().astype(np.int64),
            )

        classifier_scores, classifier_labels = _classifier_scores()
        classifier_auroc = float(
            roc_auc_score(classifier_labels, classifier_scores)
        )
        classifier_auprc = float(
            average_precision_score(classifier_labels, classifier_scores)
        )
        eval_metrics.update(
            {
                "auroc": classifier_auroc,
                "auprc": classifier_auprc,
                "classifier_auroc": classifier_auroc,
                "classifier_auprc": classifier_auprc,
                "primary_scorer": "binary_classifier",
                "knn_fusion_role": "diagnostic",
                "classifier_input_dim": 2 * int(model.final_dim),
                "classifier_hidden_dim": int(
                    model.classifier_hidden_dim
                ),
                "model_parameter_count": int(
                    sum(p.numel() for p in model.parameters())
                ),
                "standard_encoder_parameter_count": int(
                    sum(p.numel() for p in model.standard_encoder.parameters())
                ),
                "mirrored_encoder_parameter_count": int(
                    sum(p.numel() for p in model.mirrored_encoder.parameters())
                ),
                "classifier_parameter_count": int(
                    sum(p.numel() for p in model.classifier.parameters())
                ),
            }
        )

        if eval_cfg.get("eval_prefix_curve", False):
            from activation_research.prefix_views import resolve_eval_prefixes

            _r_max = getattr(test_ds, "_max_resp", None) or 64
            for _k in resolve_eval_prefixes(
                eval_cfg.get("eval_prefix_lengths"), int(_r_max)
            ):
                _scores_k, _labels_k = _classifier_scores(prefix_len=_k)
                _, _, _std_k = _head_scores(
                    "A", outlier_class=1, flip=False, prefix_len=_k
                )
                _, _, _mirror_k = _head_scores(
                    "B", outlier_class=0, flip=True, prefix_len=_k
                )
                eval_metrics[f"k{_k}_classifier_auroc"] = float(
                    roc_auc_score(_labels_k, _scores_k)
                )
                eval_metrics[f"k{_k}_classifier_auprc"] = float(
                    average_precision_score(_labels_k, _scores_k)
                )
                eval_metrics[f"k{_k}_knn_auroc_head_std"] = _std_k
                eval_metrics[f"k{_k}_knn_auroc_head_flip"] = _mirror_k

    if classifier_scores is not None and classifier_labels is not None:
        predictions = _build_classifier_predictions(
            classifier_scores, classifier_labels
        )
    else:
        predictions = [
            {"example_id": i, "score_halu": float(f), "label_halu": int(l)}
            for i, (f, l) in enumerate(zip(fused, lab))
        ]
    logger.info(
        f"{model_class} dual-head: std={auroc_std:.4f} flip={auroc_flip:.4f} "
        f"fused={auroc_fused:.4f} (Δvs_best={auroc_fused-max(auroc_std,auroc_flip):+.4f}) "
        f"head_corr={head_score_corr:.3f} oracle={auroc_fused_oracle:.4f}@w_std={fusion_oracle_weight_std:.2f}"
    )
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
    eval_cfg = method_cfg.get("evaluation", {})

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
        pad_length=data_cfg.get("pad_length", 63),
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
        prefix_training=train_cfg.get("prefix_training", False),
        prefix_min_tokens=train_cfg.get("prefix_min_tokens", 8),
        prefix_min_gap=train_cfg.get("prefix_min_gap", 8),
        prefix_max_tokens=int(probe_train.cache.shape[-2]),
        prefix_seed=training_seed,
        device=device,
        num_workers=experiment_cfg.get("num_workers", 4),
        persistent_workers=experiment_cfg.get("persistent_workers", True),
        checkpoint_dir=checkpoint_dir,
        save_every=1,
    )

    trainer = LinearProbeTrainer(model, config=config)
    final_weights_path = os.path.join(
        output_dir, "artifacts", "final_weights.pt"
    )
    if os.path.exists(final_weights_path):
        logger.info(
            "[linear_probe] final_weights.pt found — loading checkpoint and "
            "skipping training"
        )
        checkpoint = torch.load(
            final_weights_path, map_location=trainer.device, weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        trainer.fit(train_dataset=probe_train, val_dataset=probe_val)
        torch.save(
            {"model_state_dict": model.state_dict()}, final_weights_path
        )

    # Evaluate
    from sklearn.metrics import roc_auc_score
    model.eval()
    eval_device = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(eval_device)

    all_preds_np, all_labels_np = _score_single_layer_classifier(
        model,
        probe_test,
        batch_size=train_cfg["batch_size"],
        eval_device=eval_device,
        length_aware=bool(train_cfg.get("prefix_training", False)),
    )
    if len(set(all_labels_np)) < 2:
        auroc = float("nan")
    else:
        auroc = roc_auc_score(all_labels_np, all_preds_np)


    prefix_curve, prefix_scores = _single_layer_prefix_curve(
        model,
        probe_test,
        batch_size=train_cfg["batch_size"],
        eval_device=eval_device,
        eval_cfg=eval_cfg,
    )

    eval_metrics = {
        "method": method_cfg["name"],
        "dataset": dataset_cfg["name"],
        "seed": training_seed,
        "split_seed": _resolve_run_split_seed(experiment_cfg, training_seed),
        "n_train": len(probe_train),
        "n_test": len(probe_test),
        "auroc": float(auroc),
        "probe_layer": probe_layer,
        "training_regime": (
            "multi_k" if train_cfg.get("prefix_training", False) else "full_length"
        ),
    }
    eval_metrics.update(prefix_curve)

    predictions = [
        {
            "example_id": i,
            "score_halu": float(s),
            "label_halu": int(l),
            **{
                f"k{k}_score_halu": float(scores[i])
                for k, scores in prefix_scores.items()
            },
        }
        for i, (s, l) in enumerate(zip(all_preds_np, all_labels_np))
    ]

    return eval_metrics, predictions


def run_saplma(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    training_seed: int,
    test_ap=None,
) -> tuple[dict, list[dict]]:
    """Train and evaluate SAPLMA (SimpleHaluClassifier) on a single layer."""
    data_cfg = method_cfg["data"]
    train_cfg = method_cfg["training"]
    eval_cfg = method_cfg.get("evaluation", {})

    relevant_layers = (
        parse_layer_range(data_cfg["relevant_layers"])
        if "relevant_layers" in data_cfg
        else list(range(14, 30))
    )
    probe_layer = data_cfg["probe_layer"]

    ds_kwargs = dict(
        relevant_layers=relevant_layers,
        num_views=2,
        preload=data_cfg.get("preload", True),
        include_response_logprobs=data_cfg.get("include_response_logprobs", False),
        response_logprobs_top_k=data_cfg.get("response_logprobs_top_k", 20),
        check_ram=False,
        pad_length=data_cfg.get("pad_length", 63),
    )

    train_ds = ap.get_dataset("train", **ds_kwargs)
    eval_ap = test_ap if test_ap is not None else ap
    test_ds = eval_ap.get_dataset("test", **ds_kwargs)

    has_val = ap.split_strategy == "three_way"
    val_ds = ap.get_dataset("val", **ds_kwargs) if has_val else test_ds

    probe_train = train_ds.get_single_layer_dataset(probe_layer)
    probe_test = test_ds.get_single_layer_dataset(probe_layer)
    probe_val = val_ds.get_single_layer_dataset(probe_layer) if has_val else probe_test

    from activation_research.model import SimpleHaluClassifier
    from activation_research.trainer import LinearProbeTrainer, LinearProbeTrainerConfig

    model_params = method_cfg.get("model_params", {})
    model = SimpleHaluClassifier(
        input_dim=dataset_cfg["input_dim"],
        hidden_dims=model_params.get("hidden_dims", [2048, 1024, 512]),
        dropout=model_params.get("dropout", 0.1),
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
        pooling="mean",
        prefix_training=train_cfg.get("prefix_training", False),
        prefix_min_tokens=train_cfg.get("prefix_min_tokens", 8),
        prefix_min_gap=train_cfg.get("prefix_min_gap", 8),
        prefix_max_tokens=int(probe_train.cache.shape[-2]),
        prefix_seed=training_seed,
        device=device,
        num_workers=experiment_cfg.get("num_workers", 4),
        persistent_workers=experiment_cfg.get("persistent_workers", True),
        checkpoint_dir=checkpoint_dir,
        save_every=1,
    )

    trainer = LinearProbeTrainer(model, config=config)
    final_weights_path = os.path.join(
        output_dir, "artifacts", "final_weights.pt"
    )
    if os.path.exists(final_weights_path):
        logger.info(
            "[saplma] final_weights.pt found — loading checkpoint and "
            "skipping training"
        )
        checkpoint = torch.load(
            final_weights_path, map_location=trainer.device, weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        trainer.fit(train_dataset=probe_train, val_dataset=probe_val)
        torch.save(
            {"model_state_dict": model.state_dict()}, final_weights_path
        )

    from sklearn.metrics import roc_auc_score
    model.eval()
    eval_device = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(eval_device)

    all_preds_np, all_labels_np = _score_single_layer_classifier(
        model,
        probe_test,
        batch_size=train_cfg["batch_size"],
        eval_device=eval_device,
        length_aware=bool(train_cfg.get("prefix_training", False)),
    )
    if len(set(all_labels_np)) < 2:
        auroc = float("nan")
    else:
        auroc = roc_auc_score(all_labels_np, all_preds_np)

    prefix_curve, prefix_scores = _single_layer_prefix_curve(
        model,
        probe_test,
        batch_size=train_cfg["batch_size"],
        eval_device=eval_device,
        eval_cfg=eval_cfg,
    )

    eval_metrics = {
        "method": method_cfg["name"],
        "dataset": dataset_cfg["name"],
        "seed": training_seed,
        "split_seed": _resolve_run_split_seed(experiment_cfg, training_seed),
        "n_train": len(probe_train),
        "n_test": len(probe_test),
        "auroc": float(auroc),
        "probe_layer": probe_layer,
        "training_regime": (
            "multi_k" if train_cfg.get("prefix_training", False) else "full_length"
        ),
    }
    eval_metrics.update(prefix_curve)

    predictions = [
        {
            "example_id": i,
            "score_halu": float(s),
            "label_halu": int(l),
            **{
                f"k{k}_score_halu": float(scores[i])
                for k, scores in prefix_scores.items()
            },
        }
        for i, (s, l) in enumerate(zip(all_preds_np, all_labels_np))
    ]

    return eval_metrics, predictions


def run_saplma_logprob_recon(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    training_seed: int,
    test_ap=None,
) -> tuple[dict, list[dict]]:
    """Train and evaluate SAPLMA + logprob recon auxiliary on a single layer.

    Ablation for issue #67 — isolates the contribution of the contrastive
    objective in ``contrastive_logprob_recon`` by giving SAPLMA the same
    auxiliary recon target. Inference path is identical to plain SAPLMA.
    """
    data_cfg = method_cfg["data"]
    train_cfg = method_cfg["training"]
    model_params = method_cfg.get("model_params", {})

    relevant_layers = (
        parse_layer_range(data_cfg["relevant_layers"])
        if "relevant_layers" in data_cfg
        else list(range(14, 30))
    )
    probe_layer = data_cfg["probe_layer"]
    if probe_layer not in relevant_layers:
        raise ValueError(
            f"probe_layer={probe_layer} not in relevant_layers={relevant_layers}"
        )
    probe_layer_pos = relevant_layers.index(probe_layer)

    # num_views=1 + fixed_layer=<pos> makes _select_view_indices return [pos]
    # deterministically, so views_activations is (1, T, H) — same shape that
    # LinearProbeTrainer expects, but with logprob fields attached.
    ds_kwargs = dict(
        relevant_layers=relevant_layers,
        num_views=1,
        fixed_layer=probe_layer_pos,
        min_target_layers=1,
        preload=data_cfg.get("preload", True),
        include_response_logprobs=True,
        response_logprobs_top_k=data_cfg.get("response_logprobs_top_k", 20),
        check_ram=False,
        pad_length=data_cfg.get("pad_length", 63),
    )

    train_ds = ap.get_dataset("train", **ds_kwargs)
    eval_ap = test_ap if test_ap is not None else ap
    test_ds = eval_ap.get_dataset("test", **ds_kwargs)

    has_val = ap.split_strategy == "three_way"
    val_ds = ap.get_dataset("val", **ds_kwargs) if has_val else test_ds

    from activation_research.model import SaplmaWithReconHead
    from activation_research.trainer import SaplmaReconTrainer, SaplmaReconTrainerConfig

    model = SaplmaWithReconHead(
        input_dim=dataset_cfg["input_dim"],
        hidden_dims=tuple(model_params.get("hidden_dims", [2048, 1024, 512])),
        dropout=model_params.get("dropout", 0.1),
        recon_seq_len=model_params.get("recon_seq_len", 64),
        recon_hidden_dim=model_params.get("recon_hidden_dim", 256),
        recon_lambda=model_params.get("recon_lambda", 1.0),
        logprob_var_threshold=model_params.get("logprob_var_threshold", 1e-4),
    )

    checkpoint_dir = os.path.join(output_dir, "artifacts")
    config = SaplmaReconTrainerConfig(
        max_epochs=train_cfg["max_epochs"],
        batch_size=train_cfg["batch_size"],
        lr=train_cfg["lr"],
        steps_per_epoch_override=train_cfg.get("steps_per_epoch_override"),
        min_total_steps=train_cfg.get("min_total_steps"),
        grad_clip_norm=train_cfg.get("grad_clip_norm"),
        balanced_sampling=train_cfg.get("balanced_sampling", True),
        use_infinite_index_stream=train_cfg.get("use_infinite_index_stream", False),
        infinite_stream_seed=training_seed,
        pooling="mean",
        device=device,
        num_workers=experiment_cfg.get("num_workers", 4),
        persistent_workers=experiment_cfg.get("persistent_workers", True),
        checkpoint_dir=checkpoint_dir,
        save_every=1,
        recon_lambda=model_params.get("recon_lambda", 1.0),
    )

    trainer = SaplmaReconTrainer(model, config=config)
    trainer.fit(train_dataset=train_ds, val_dataset=val_ds)

    torch.save(
        {"model_state_dict": model.state_dict()},
        os.path.join(output_dir, "artifacts", "final_weights.pt"),
    )

    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    model.eval()
    eval_device = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(eval_device)

    eval_loader = DataLoader(test_ds, batch_size=train_cfg["batch_size"], shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in eval_loader:
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
        "split_seed": _resolve_run_split_seed(experiment_cfg, training_seed),
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "auroc": float(auroc),
        "probe_layer": probe_layer,
        "recon_lambda": float(model_params.get("recon_lambda", 1.0)),
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
        pad_length=data_cfg.get("pad_length", 63),
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



def run_simclr_linear(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    training_seed: int,
    test_ap=None,
) -> tuple[dict, list[dict]]:
    """Train and evaluate a two-phase SimCLR + linear probe model.

    Phase 1: Unsupervised contrastive (SimCLR) on the ProgressiveCompressor.
    Phase 2: Linear probe on frozen encoder embeddings (BCE, AUROC).
    """
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

    # Build datasets for Phase 1 (contrastive, multi-view)
    train_ds = ap.get_dataset("train", **ds_kwargs)
    eval_ap = test_ap if test_ap is not None else ap
    test_ds = eval_ap.get_dataset("test", **ds_kwargs)

    has_val = ap.split_strategy == "three_way"
    val_ds = ap.get_dataset("val", **ds_kwargs) if has_val else test_ds

    # Build single-layer datasets for Phase 2 (linear probe on target layers)
    train_ds_target = train_ds.slice_layers(target_layers)
    test_ds_target = test_ds.slice_layers(target_layers)
    val_ds_target = val_ds.slice_layers(target_layers) if has_val else test_ds_target

    # Build model
    from activation_research.model import ProgressiveCompressor

    model = ProgressiveCompressor(
        input_dim=dataset_cfg["input_dim"],
        final_dim=method_cfg["model_params"].get("final_dim", 512),
        input_dropout=method_cfg["model_params"].get("input_dropout", 0.3),
    )

    # Build trainer
    from activation_research.trainer import SimCLRLinearTrainer, SimCLRLinearTrainerConfig

    checkpoint_dir = os.path.join(output_dir, "artifacts")
    config = SimCLRLinearTrainerConfig(
        batch_size=train_cfg["batch_size"],
        temperature=train_cfg.get("temperature", 0.25),
        contrastive_epochs=train_cfg.get("contrastive_epochs", 50),
        contrastive_lr=train_cfg.get("contrastive_lr", 1e-5),
        probe_epochs=train_cfg.get("probe_epochs", 100),
        probe_lr=train_cfg.get("probe_lr", 1e-3),
        probe_balanced_sampling=train_cfg.get("probe_balanced_sampling", True),
        probe_min_total_steps=train_cfg.get("probe_min_total_steps", 3000),
        grad_clip_norm=train_cfg.get("grad_clip_norm"),
        use_infinite_index_stream=train_cfg.get("use_infinite_index_stream", True),
        use_infinite_index_stream_eval=train_cfg.get(
            "use_infinite_index_stream_eval", True
        ),
        infinite_stream_seed=training_seed,
        infinite_eval_seed=training_seed,
        num_views=data_cfg.get("num_views", 2),
        device=device,
        num_workers=experiment_cfg.get("num_workers", 4),
        persistent_workers=experiment_cfg.get("persistent_workers", True),
        checkpoint_dir=checkpoint_dir,
        save_every=1,
        snapshot_every=10,
        snapshot_keep_last=3,
    )

    trainer = SimCLRLinearTrainer(model, config=config)
    trainer.fit(
        train_dataset=train_ds,
        val_dataset=val_ds,
        probe_train_dataset=train_ds_target,
        probe_val_dataset=val_ds_target,
    )

    # Save final encoder weights
    torch.save(
        {"model_state_dict": model.state_dict()},
        os.path.join(output_dir, "artifacts", "final_weights.pt"),
    )

    # Save linear head weights
    if trainer.linear_head is not None:
        torch.save(
            {"linear_head_state_dict": trainer.linear_head.state_dict()},
            os.path.join(output_dir, "artifacts", "final_linear_head.pt"),
        )

    # --- Evaluation ---
    # 1) Contrastive embedding quality (cosine, mds, knn) via OOD evaluator
    from torch.utils.data import DataLoader

    from activation_research.metric_evaluator import MultiMetricHallucinationEvaluator

    train_loader = DataLoader(train_ds_target, batch_size=64, shuffle=False)
    eval_loader = DataLoader(test_ds_target, batch_size=64, shuffle=False)

    model.eval()

    # Build metrics list from config (exclude "auroc" since that's probe-based)
    metrics_list: list = []
    for m in eval_cfg["metrics"]:
        if m == "auroc":
            continue  # handled separately via probe
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

    ood_stats = {}
    if metrics_list:
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

    # 2) Linear probe AUROC on test set
    from sklearn.metrics import roc_auc_score

    probe_auroc = float("nan")
    predictions: list[dict] = []
    if trainer.linear_head is not None:
        trainer.linear_head.eval()
        probe_eval_loader = DataLoader(
            test_ds_target, batch_size=train_cfg["batch_size"], shuffle=False
        )
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in probe_eval_loader:
                x = batch["views_activations"].to(
                    torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")),
                    non_blocking=True,
                )
                if x.dim() == 4:
                    x = x[:, 0]  # take first view
                z = model(x)
                preds = trainer.linear_head(z)
                all_preds.append(preds.cpu())
                all_labels.append(batch["halu"].cpu())

        all_preds_np = torch.cat(all_preds).squeeze().numpy()
        all_labels_np = torch.cat(all_labels).numpy()
        if len(set(all_labels_np)) >= 2:
            probe_auroc = float(roc_auc_score(all_labels_np, all_preds_np))

        predictions = [
            {"example_id": i, "score_halu": float(s), "label_halu": int(l)}
            for i, (s, l) in enumerate(zip(all_preds_np, all_labels_np))
        ]

    # Combine metrics
    eval_metrics: dict = {
        "method": method_cfg["name"],
        "dataset": dataset_cfg["name"],
        "seed": training_seed,
        "split_seed": experiment_cfg.get("split_seed", 42),
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "auroc": probe_auroc,
        "target_layers": target_layers,
    }
    eval_metrics.update(ood_stats)

    return eval_metrics, predictions



def run_simclr_cotrained(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    training_seed: int,
    test_ap=None,
) -> tuple[dict, list[dict]]:
    """Train and evaluate a SimCLR co-trained (joint contrastive + BCE) model."""
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
    from activation_research.model import SimCLRCotrainedModel

    model = SimCLRCotrainedModel(
        input_dim=dataset_cfg["input_dim"],
        final_dim=method_cfg["model_params"].get("final_dim", 512),
        input_dropout=method_cfg["model_params"].get("input_dropout", 0.3),
    )

    # Build trainer
    from activation_research.trainer import (
        SimCLRCotrainedTrainer,
        SimCLRCotrainedTrainerConfig,
    )

    checkpoint_dir = os.path.join(output_dir, "artifacts")
    config = SimCLRCotrainedTrainerConfig(
        max_epochs=train_cfg["max_epochs"],
        batch_size=train_cfg["batch_size"],
        lr=train_cfg["lr"],
        temperature=train_cfg.get("temperature", 0.25),
        simclr_weight=train_cfg.get("simclr_weight", 1.0),
        bce_grad_gate=train_cfg.get("bce_grad_gate", 1.0),
        steps_per_epoch_override=train_cfg.get("steps_per_epoch_override"),
        min_total_steps=train_cfg.get("min_total_steps"),
        grad_clip_norm=train_cfg.get("grad_clip_norm"),
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

    trainer = SimCLRCotrainedTrainer(model, config=config)
    trainer.fit(train_dataset=train_ds, val_dataset=val_ds)

    # Save final weights
    torch.save(
        {"model_state_dict": model.state_dict()},
        os.path.join(output_dir, "artifacts", "final_weights.pt"),
    )

    # OOD evaluation on target layers (contrastive embedding quality)
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

    # Also compute classification AUROC from the head on test set
    from sklearn.metrics import roc_auc_score

    eval_device = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(eval_device)

    head_eval_loader = DataLoader(test_ds_target, batch_size=256, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in head_eval_loader:
            x = batch["views_activations"].to(eval_device)
            if x.dim() == 4:
                x = x[:, 0]  # take first view
            _, pred = model.forward_with_head(x)
            all_preds.append(pred.cpu())
            all_labels.append(batch["halu"].cpu())

    all_preds_np = torch.cat(all_preds).squeeze().numpy()
    all_labels_np = torch.cat(all_labels).numpy()
    if len(set(all_labels_np)) < 2:
        head_auroc = float("nan")
    else:
        head_auroc = roc_auc_score(all_labels_np, all_preds_np)

    # Build eval_metrics
    eval_metrics: dict = {
        "method": method_cfg["name"],
        "dataset": dataset_cfg["name"],
        "seed": training_seed,
        "split_seed": experiment_cfg.get("split_seed", 42),
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "auroc": float(head_auroc),
    }
    eval_metrics.update(ood_stats)

    predictions: list[dict] = []
    return eval_metrics, predictions



def run_simclr_projection(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    training_seed: int,
    test_ap=None,
) -> tuple[dict, list[dict]]:
    """Train and evaluate a SimCLR projection head model.

    The contrastive loss operates on the projected embeddings p (after MLP
    projection head), while the classification head and downstream evaluation
    operate on the representation z (before the projection head).
    """
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
    from activation_research.model import SimCLRProjectionModel

    model_params = method_cfg.get("model_params", {})
    model = SimCLRProjectionModel(
        input_dim=dataset_cfg["input_dim"],
        final_dim=model_params.get("final_dim", 512),
        projection_dim=model_params.get("projection_dim", 128),
        input_dropout=model_params.get("input_dropout", 0.3),
    )

    # Build trainer
    from activation_research.trainer import (
        SimCLRProjectionTrainer,
        SimCLRProjectionTrainerConfig,
    )

    checkpoint_dir = os.path.join(output_dir, "artifacts")
    config = SimCLRProjectionTrainerConfig(
        max_epochs=train_cfg["max_epochs"],
        batch_size=train_cfg["batch_size"],
        lr=train_cfg["lr"],
        temperature=train_cfg["temperature"],
        simclr_weight=train_cfg.get("simclr_weight", 1.0),
        bce_weight=train_cfg.get("bce_weight", 1.0),
        projection_dim=model_params.get("projection_dim", 128),
        steps_per_epoch_override=train_cfg.get("steps_per_epoch_override"),
        min_total_steps=train_cfg.get("min_total_steps"),
        grad_clip_norm=train_cfg.get("grad_clip_norm"),
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

    trainer = SimCLRProjectionTrainer(model, config=config)
    trainer.fit(train_dataset=train_ds, val_dataset=val_ds)

    # Save final weights
    torch.save(
        {"model_state_dict": model.state_dict()},
        os.path.join(output_dir, "artifacts", "final_weights.pt"),
    )

    # OOD evaluation on target layers using z (encoder output, model.forward)
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
        elif m == "auroc":
            # AUROC from classification head -- compute separately below
            pass
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

    # Compute AUROC from classification head on test set
    eval_device = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(eval_device)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in eval_loader:
            x = batch["views_activations"].to(eval_device)
            if x.dim() == 4:
                x = x[:, 0]  # first view
            z = model(x)  # encoder output
            pred = torch.sigmoid(model.head(z))
            all_preds.append(pred.cpu())
            all_labels.append(batch["halu"].cpu())

    from sklearn.metrics import roc_auc_score

    all_preds_np = torch.cat(all_preds).squeeze().numpy()
    all_labels_np = torch.cat(all_labels).numpy()
    if len(set(all_labels_np)) < 2:
        head_auroc = float("nan")
    else:
        head_auroc = float(roc_auc_score(all_labels_np, all_preds_np))

    # Build eval_metrics
    eval_metrics: dict = {
        "method": method_cfg["name"],
        "dataset": dataset_cfg["name"],
        "seed": training_seed,
        "split_seed": experiment_cfg.get("split_seed", 42),
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "auroc": head_auroc,
    }
    eval_metrics.update(ood_stats)

    predictions: list[dict] = []
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
    eval_cfg = method_cfg.get("evaluation", {})
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

    if eval_cfg.get("eval_prefix_curve", False):
        from activation_research.prefix_views import resolve_eval_prefixes

        max_prefix = int(getattr(test_ds, "_max_resp", 64))
        for k in resolve_eval_prefixes(
            eval_cfg.get("eval_prefix_lengths"), max_prefix
        ):
            stats_k = detector.score(
                test_ds,
                batch_size=256,
                num_workers=experiment_cfg.get("num_workers", 4),
                prefix_len=k,
            )
            for metric_name, value in stats_k.items():
                eval_metrics[f"k{k}_{metric_name}"] = value

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
    eval_cfg = method_cfg.get("evaluation", {})
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

    if eval_cfg.get("eval_prefix_curve", False):
        from activation_research.prefix_views import resolve_eval_prefixes

        max_prefix = int(getattr(test_ds, "_max_resp", 64))
        for k in resolve_eval_prefixes(
            eval_cfg.get("eval_prefix_lengths"), max_prefix
        ):
            records_k = []
            for record in records:
                mask = record["response_logprob_mask"].clone().bool()
                mask[int(k):] = False
                records_k.append({**record, "response_logprob_mask": mask})
            stats_k = logprob_baseline_auroc(
                records_k, outlier_class=dataset_cfg.get("outlier_class", 1)
            )
            for metric_name, value in stats_k.items():
                eval_metrics[f"k{k}_{metric_name}"] = value

    return eval_metrics, []


def run_llmsknow_probe(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    training_seed: int,
    test_ap=None,
) -> tuple[dict, list[dict]]:
    """LLMsKnow Probe Baseline: sweep (layer, token) on dev subset, train final probe at best location."""
    import numpy as np
    from activation_research.llmsknow_probe import (
        _split_view,
        eval_probe,
        sweep_locations,
        train_final_probe,
    )

    data_cfg = method_cfg["data"]
    sweep_cfg = method_cfg.get("sweep", {})
    eval_cfg = method_cfg.get("evaluation", {})

    relevant_layers = parse_layer_range(data_cfg["relevant_layers"])
    pad_length = data_cfg.get("pad_length", 63)

    ds_kwargs = dict(
        relevant_layers=relevant_layers,
        num_views=1,
        pad_length=pad_length,
        preload=data_cfg.get("preload", True),
        # Request logprobs so the fingerprint matches the canonical caches
        # built by every other method; the loader will discard them.
        include_response_logprobs=True,
        response_logprobs_top_k=data_cfg.get("response_logprobs_top_k", 20),
        check_ram=False,
    )

    train_ds = ap.get_dataset("train", **ds_kwargs)
    eval_ap = test_ap if test_ap is not None else ap
    test_ds = eval_ap.get_dataset("test", **ds_kwargs)

    # Lazy views — no big read happens here, only metadata.
    train_full, train_rows, train_labels = _split_view(train_ds)
    test_full, test_rows, test_labels = _split_view(test_ds)

    dev_size = sweep_cfg.get("dev_size", 2000)
    val_size = sweep_cfg.get("val_size", 1000)
    C = sweep_cfg.get("C", 1.0)
    max_iter = sweep_cfg.get("max_iter", 100)

    # Phase 1: sweep (layer, token) on dev subset
    sweep_matrix, best_layer_idx, best_token_pos = sweep_locations(
        train_full, train_rows, train_labels, relevant_layers,
        dev_size=dev_size, val_size=val_size, seed=training_seed,
        C=C, max_iter=max_iter,
    )

    # Save sweep results
    artifacts_dir = os.path.join(output_dir, "artifacts")
    np.save(os.path.join(artifacts_dir, "sweep_auroc_matrix.npy"), sweep_matrix)
    # Why: best_layer_idx is in [0, L) where L = cache.shape[1]. The dataset
    # is supposed to have L == len(relevant_layers), but a slicing mismatch
    # upstream can yield L > len(relevant_layers), which crashed the run
    # mid-flight. Fall back to None when the layer id is unknown so the cell
    # still produces eval_metrics.json instead of silently failing.
    best_layer_id = (
        int(relevant_layers[best_layer_idx])
        if best_layer_idx < len(relevant_layers)
        else None
    )
    sweep_summary = {
        "relevant_layers": relevant_layers,
        "best_layer_idx": int(best_layer_idx),
        "best_layer": best_layer_id,
        "best_token_pos": int(best_token_pos),
        "best_dev_auroc": float(np.nanmax(sweep_matrix)) if not np.all(np.isnan(sweep_matrix)) else float("nan"),
    }
    with open(os.path.join(artifacts_dir, "sweep_summary.json"), "w") as f:
        json.dump(sweep_summary, f, indent=2)

    # Phase 2: train final probe on full training set
    probe = train_final_probe(
        train_full, train_rows, train_labels, best_layer_idx, best_token_pos,
        seed=training_seed, C=C, max_iter=max_iter,
    )

    # Evaluate on test set
    outlier_class = dataset_cfg.get("outlier_class", 1)
    auroc, scores = eval_probe(
        probe, test_full, test_rows, test_labels,
        best_layer_idx, best_token_pos, outlier_class=outlier_class,
    )

    prefix_metrics: dict = {}
    prefix_scores: dict[int, "np.ndarray"] = {}
    prefix_locations: dict[str, dict] = {}
    if eval_cfg.get("eval_prefix_curve", False):
        from activation_research.prefix_views import resolve_eval_prefixes

        ks = resolve_eval_prefixes(
            eval_cfg.get("eval_prefix_lengths"), int(sweep_matrix.shape[1])
        )
        fitted_by_location = {
            (int(best_layer_idx), int(best_token_pos)): (
                probe, float(auroc), scores
            )
        }
        for k in ks:
            constrained = sweep_matrix[:, : min(int(k), sweep_matrix.shape[1])]
            if np.all(np.isnan(constrained)):
                layer_idx_k, token_pos_k = 0, 0
            else:
                best_flat_k = int(np.nanargmax(constrained))
                layer_idx_k = best_flat_k // constrained.shape[1]
                token_pos_k = best_flat_k % constrained.shape[1]
            location = (int(layer_idx_k), int(token_pos_k))
            if location not in fitted_by_location:
                probe_k = train_final_probe(
                    train_full,
                    train_rows,
                    train_labels,
                    layer_idx_k,
                    token_pos_k,
                    seed=training_seed,
                    C=C,
                    max_iter=max_iter,
                )
                auroc_k, scores_k = eval_probe(
                    probe_k,
                    test_full,
                    test_rows,
                    test_labels,
                    layer_idx_k,
                    token_pos_k,
                    outlier_class=outlier_class,
                )
                fitted_by_location[location] = (
                    probe_k, float(auroc_k), scores_k
                )
            _, auroc_k, scores_k = fitted_by_location[location]
            prefix_metrics[f"k{k}_auroc"] = float(auroc_k)
            prefix_metrics[f"k{k}_selected_layer_idx"] = int(layer_idx_k)
            prefix_metrics[f"k{k}_selected_token_pos"] = int(token_pos_k)
            prefix_scores[int(k)] = scores_k
            prefix_locations[f"k{k}"] = {
                "selected_layer_idx": int(layer_idx_k),
                "selected_layer": (
                    int(relevant_layers[layer_idx_k])
                    if layer_idx_k < len(relevant_layers)
                    else None
                ),
                "selected_token_pos": int(token_pos_k),
                "dev_auroc": float(constrained[layer_idx_k, token_pos_k]),
                "test_auroc": float(auroc_k),
            }
        with open(
            os.path.join(artifacts_dir, "prefix_sweep_summary.json"), "w"
        ) as f:
            json.dump(prefix_locations, f, indent=2)

    eval_metrics = {
        "method": method_cfg["name"],
        "dataset": dataset_cfg["name"],
        "seed": training_seed,
        "split_seed": _resolve_run_split_seed(experiment_cfg, training_seed),
        "n_train": len(train_ds),
        "n_test": len(test_ds),
        "auroc": float(auroc),
        "selected_layer": best_layer_id,
        "selected_token_pos": int(best_token_pos),
        "sweep_best_dev_auroc": float(sweep_summary["best_dev_auroc"]),
        "training_regime": (
            "per_k" if eval_cfg.get("eval_prefix_curve", False) else "full_length"
        ),
    }
    eval_metrics.update(prefix_metrics)

    predictions = [
        {
            "example_id": i,
            "score_halu": float(s),
            "label_halu": int(l),
            **{
                f"k{k}_score_halu": float(scores_k[i])
                for k, scores_k in prefix_scores.items()
            },
        }
        for i, (s, l) in enumerate(zip(scores, test_labels))
    ]

    return eval_metrics, predictions


def run_icr_probe(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    training_seed: int,
    test_ap=None,
) -> tuple[dict, list[dict]]:
    """ICR Probe per Issue #70.

    Trains an ICRProbe MLP on precomputed icr_scores.npy from the capture
    directory, evaluates on the separate test cell, and returns eval_metrics
    + predictions in the standard schema.

    Dataset config must have an "icr_capture" block with "train_dir" and
    "test_dir" keys pointing to InferenceCaptureWriter output directories.
    """
    import torch
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    from activation_research.icr_dataset import ICRDataset
    from activation_research.icr_probe import ICRProbe
    from activation_research.icr_trainer import ICRProbeTrainer, ICRProbeTrainerConfig

    icr_cfg = dataset_cfg["icr_capture"]
    train_cfg = method_cfg["training"]
    data_cfg = method_cfg.get("data", {})
    eval_cfg = method_cfg.get("evaluation", {})

    split_seed = _resolve_run_split_seed(experiment_cfg, training_seed)
    val_fraction = data_cfg.get("val_fraction", 0.1)
    train_capture = Path(_resolve_shared(icr_cfg["train_dir"]))
    test_capture = Path(_resolve_shared(icr_cfg["test_dir"]))
    eval_device = torch.device(
        device
        if device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    prefix_eval = bool(eval_cfg.get("eval_prefix_curve", False))
    if prefix_eval:
        from activation_research.prefix_views import resolve_eval_prefixes

        cache_root_cfg = eval_cfg.get("prefix_icr_cache_root")
        if not cache_root_cfg:
            raise ValueError(
                "ICR prefix evaluation requires evaluation.prefix_icr_cache_root"
            )
        cache_root = Path(_resolve_shared(str(cache_root_cfg)))
        capture_cfg = json.loads((test_capture / "config.json").read_text())
        prefixes: list[int | None] = resolve_eval_prefixes(
            eval_cfg.get("eval_prefix_lengths"), int(capture_cfg["r_max"])
        )
    else:
        cache_root = None
        prefixes = [None]

    def _score_path(capture: Path, prefix: int | None) -> Path | None:
        if prefix is None:
            return None
        assert cache_root is not None
        return cache_root / capture.name / f"icr_scores_k{prefix}.npy"

    def _datasets(prefix: int | None):
        train_scores = _score_path(train_capture, prefix)
        test_scores = _score_path(test_capture, prefix)
        for score_path in (train_scores, test_scores):
            if score_path is not None and not score_path.exists():
                raise FileNotFoundError(
                    f"prefix ICR cache missing: {score_path}; run "
                    "scripts/build_prefix_icr_cache.py first"
                )
        train_ds = ICRDataset(
            train_capture,
            mode="memmap",
            split="train",
            val_fraction=val_fraction,
            random_seed=split_seed,
            scores_path=train_scores,
        )
        val_ds = ICRDataset(
            train_capture,
            mode="memmap",
            split="val",
            val_fraction=val_fraction,
            random_seed=split_seed,
            scores_path=train_scores,
        )
        test_ds = ICRDataset(
            test_capture,
            mode="memmap",
            split="all",
            random_seed=split_seed,
            scores_path=test_scores,
        )
        return train_ds, val_ds, test_ds

    prefix_scores: dict[int, object] = {}
    prefix_metrics: dict[str, float] = {}
    main_scores = main_labels = main_hashes = None
    main_train_n = main_val_n = main_test_n = num_layers = 0

    # Each k receives its own probe. This is deliberately more favorable to
    # ICR than a shared multi-k probe and serves as the matched-k oracle arm;
    # the expensive ICR representation itself is recomputed at that prefix.
    for prefix in prefixes:
        train_ds, val_ds, test_ds = _datasets(prefix)
        num_layers = int(train_ds[0]["icr_score"].shape[0])
        torch.manual_seed(training_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(training_seed)
        model = ICRProbe(input_dim=num_layers)

        artifact_dir = os.path.join(
            output_dir,
            "artifacts" if prefix is None else f"artifacts/k{prefix}",
        )
        checkpoint_path = os.path.join(artifact_dir, "linear_probe_last.pt")
        config = ICRProbeTrainerConfig(
            max_epochs=train_cfg["max_epochs"],
            batch_size=train_cfg["batch_size"],
            learning_rate=train_cfg.get("lr", 1e-3),
            lr=train_cfg.get("lr", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 0.0),
            plateau_patience=train_cfg.get("plateau_patience", 5),
            plateau_factor=train_cfg.get("plateau_factor", 0.5),
            early_stop_patience=train_cfg.get("early_stop_patience", 10),
            device=device,
            num_workers=experiment_cfg.get("num_workers", 4),
            persistent_workers=experiment_cfg.get("persistent_workers", True),
            checkpoint_dir=artifact_dir,
            resume_from=checkpoint_path if os.path.exists(checkpoint_path) else None,
            save_every=1,
        )
        trainer = ICRProbeTrainer(model, config=config)
        trainer.fit(train_dataset=train_ds, val_dataset=val_ds)

        model.eval()
        model.to(eval_device)
        loader = DataLoader(
            test_ds, batch_size=train_cfg["batch_size"], shuffle=False
        )
        all_logits, all_labels, all_hashes = [], [], []
        with torch.no_grad():
            for batch in loader:
                x = batch["icr_score"].to(eval_device)
                all_logits.append(torch.sigmoid(model(x)).cpu())
                all_labels.append(batch["halu"].cpu())
                all_hashes.extend(batch["hashkey"])

        scores = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()
        auroc = (
            float(roc_auc_score(labels, scores))
            if len(set(labels.tolist())) >= 2
            else float("nan")
        )
        if prefix is not None:
            prefix_scores[int(prefix)] = scores
            prefix_metrics[f"k{prefix}_auroc"] = auroc
            logger.info(f"[icr prefix oracle] k={prefix}: auroc={auroc}")

        # Standard score/auroc is the largest visible prefix (or legacy full run).
        if prefix == prefixes[-1]:
            main_scores, main_labels, main_hashes = scores, labels, all_hashes
            main_train_n, main_val_n, main_test_n = (
                len(train_ds),
                len(val_ds),
                len(test_ds),
            )

    assert main_scores is not None and main_labels is not None and main_hashes is not None
    main_auroc = (
        float(roc_auc_score(main_labels, main_scores))
        if len(set(main_labels.tolist())) >= 2
        else float("nan")
    )
    eval_metrics = {
        "method": method_cfg["name"],
        "dataset": dataset_cfg["name"],
        "seed": training_seed,
        "split_seed": split_seed,
        "n_train": main_train_n,
        "n_val": main_val_n,
        "n_test": main_test_n,
        "auroc": main_auroc,
        "selected_layer": None,
        "num_layers": num_layers,
        "training_regime": "per_k_oracle" if prefix_eval else "full_length",
        **prefix_metrics,
    }
    predictions = [
        {
            "example_id": h,
            "score_halu": float(s),
            "label_halu": int(l),
            **{
                f"k{k}_score_halu": float(scores_k[i])
                for k, scores_k in prefix_scores.items()
            },
        }
        for i, (h, s, l) in enumerate(
            zip(main_hashes, main_scores, main_labels)
        )
    ]
    return eval_metrics, predictions


def run_contrastive_actvit(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    training_seed: int,
    *,
    test_ap=None,
) -> tuple[dict, list[dict]]:
    """Contrastive ACT-ViT (issue #134).

    Trains the :class:`ACTViT` joint (layers × tokens) encoder with the existing
    SupCon + logprob-reconstruction objective (not act_vit's BCE classifier) and
    scores with the same KNN/Mahalanobis OOD evaluator as the b5 headline.

    Each grid view is flattened to ``(L*N, D)`` so the standard 4-D contrastive
    trainer/evaluator are reused unchanged; :class:`ContrastiveACTViT` reshapes
    back to the grid internally. View augmentation (CAV-0/1/2/3) is set via
    ``data.view_aug``.
    """
    import torch
    from torch.utils.data import DataLoader

    from activation_research.act_vit import ContrastiveACTViT
    from activation_research.contrastive_actvit_dataset import (
        ContrastiveACTViTDataset,
        make_cav_augment,
    )
    from activation_research.memmap_activation_parser import MemmapActivationParser
    from activation_research.metric_evaluator import MultiMetricHallucinationEvaluator
    from activation_research.training import train_contrastive_logprob_recon

    train_cfg = method_cfg["training"]
    data_cfg = method_cfg.get("data", {})
    eval_cfg = method_cfg["evaluation"]
    model_params = method_cfg.get("model_params", {})
    icr_cfg = dataset_cfg["icr_capture"]
    split_seed = _resolve_run_split_seed(experiment_cfg, training_seed)
    train_device = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # --- splits: identical pattern to run_act_vit / other runners ---
    train_parser = MemmapActivationParser(
        icr_cfg["train_dir"], random_seed=split_seed, split_strategy="three_way"
    )
    train_idx = train_parser.df[train_parser.df["split"] == "train"]["sample_index"].values
    val_idx = train_parser.df[train_parser.df["split"] == "val"]["sample_index"].values
    test_parser = MemmapActivationParser(
        icr_cfg["test_dir"], random_seed=split_seed, split_strategy="none"
    )
    test_idx = test_parser.df["sample_index"].values

    def _hmap(parser):
        return dict(zip(parser.df["sample_index"].astype(int), parser.df["prompt_hash"].astype(str)))

    relevant_layers = (
        parse_layer_range(data_cfg["relevant_layers"]) if data_cfg.get("relevant_layers") else None
    )
    design_aug = data_cfg.get("view_aug", "noise")     # CAV-0/1/2/3 — applied on GPU
    # The dataset stays lean (fp16, raw grid copies, no CPU augmentation, like
    # ACTViTDataset); augmentation runs batched on the GPU via augment_fn below.
    ds_kwargs = dict(relevant_layers=relevant_layers, view_aug="raw")
    num_views = data_cfg.get("num_views", 2)
    train_ds = ContrastiveACTViTDataset(
        icr_cfg["train_dir"], train_idx, num_views=num_views, seed=training_seed,
        hashkey_map=_hmap(train_parser), **ds_kwargs,
    )
    val_ds = ContrastiveACTViTDataset(
        icr_cfg["train_dir"], val_idx, num_views=num_views, seed=training_seed,
        hashkey_map=_hmap(train_parser), **ds_kwargs,
    )
    # Eval embeddings: a single clean (un-augmented) view.
    train_eval_ds = ContrastiveACTViTDataset(
        icr_cfg["train_dir"], train_idx, num_views=1, seed=training_seed,
        hashkey_map=_hmap(train_parser), **ds_kwargs,
    )
    test_eval_ds = ContrastiveACTViTDataset(
        icr_cfg["test_dir"], test_idx, num_views=1, seed=training_seed,
        hashkey_map=_hmap(test_parser), **ds_kwargs,
    )
    augment_fn = make_cav_augment(
        design_aug, train_ds.n_layers, train_ds.n_tokens,
        keep_frac=data_cfg.get("keep_frac", 0.6),
        mask_frac=data_cfg.get("mask_frac", 0.5),
        noise_scale=data_cfg.get("noise_scale", 0.1),
        patch_h=model_params.get("patch_h", 2),
        patch_w=model_params.get("patch_w", 10),
    )

    # --- model (n_layers/n_tokens from the dataset; target ~act_vit size, <20M) ---
    model = ContrastiveACTViT(
        n_layers=train_ds.n_layers, n_tokens=train_ds.n_tokens,
        input_dim=dataset_cfg["input_dim"],
        final_dim=model_params.get("final_dim", 256),
        recon_seq_len=model_params.get("recon_seq_len", 64),
        recon_hidden_dim=model_params.get("recon_hidden_dim", 256),
        recon_lambda=model_params.get("recon_lambda", 1.0),
        logprob_var_threshold=model_params.get("logprob_var_threshold", 1e-4),
        L_p=model_params.get("L_p", 8), N_p=model_params.get("N_p", 100),
        patch_h=model_params.get("patch_h", 2), patch_w=model_params.get("patch_w", 10),
        d_adapter=model_params.get("d_adapter", 256), d_model=model_params.get("d_model", 256),
        num_heads=model_params.get("num_heads", 8), depth=model_params.get("depth", 4),
        mlp_ratio=model_params.get("mlp_ratio", 4.0), dropout=model_params.get("dropout", 0.1),
        normalize_output=model_params.get("normalize_output", False),
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"ContrastiveACTViT: {n_params / 1e6:.2f}M params "
        f"(n_layers={train_ds.n_layers}, n_tokens={train_ds.n_tokens}, view_aug={design_aug})"
    )

    nw = experiment_cfg.get("num_workers", 12)   # grid loading is I/O-heavy
    checkpoint_dir = os.path.join(output_dir, "artifacts")
    train_contrastive_logprob_recon(
        model=model, train_dataset=train_ds, test_dataset=val_ds,
        epochs=train_cfg["max_epochs"], batch_size=train_cfg["batch_size"], lr=train_cfg["lr"],
        temperature=train_cfg["temperature"], device=str(train_device),
        num_workers=nw,
        sub_batch_size=train_cfg.get("sub_batch_size", train_cfg["batch_size"]),
        checkpoint_dir=checkpoint_dir, save_every=1, snapshot_every=10, snapshot_keep_last=3,
        use_labels=train_cfg.get("use_labels", True), ignore_label=train_cfg.get("ignore_label", 0),
        persistent_workers=experiment_cfg.get("persistent_workers", True),
        recon_lambda=model_params.get("recon_lambda", 1.0),
        use_infinite_index_stream=train_cfg.get("use_infinite_index_stream", True),
        infinite_stream_shuffle=True, infinite_stream_seed=training_seed,
        steps_per_epoch_override=train_cfg.get("steps_per_epoch_override"),
        min_total_steps=train_cfg.get("min_total_steps"),
        balanced_sampling=train_cfg.get("balanced_sampling", False),
        grad_clip_norm=train_cfg.get("grad_clip_norm"),
        augment_fn=augment_fn,
        optimizer_name=train_cfg.get("optimizer", "adam"),
        weight_decay=train_cfg.get("weight_decay", 0.0),
        lr_schedule=train_cfg.get("lr_schedule"),
        base_temperature=train_cfg.get("base_temperature", 0.07),
        select_on_val=train_cfg.get("select_on_val", False),
    )

    # --- eval: reuse the standard KNN/Mahalanobis evaluator (labels via prompt_hash) ---
    model.eval()
    ev_bs = eval_cfg.get("eval_batch_size", 64)
    train_loader = DataLoader(train_eval_ds, batch_size=ev_bs, shuffle=False, num_workers=nw)
    eval_loader = DataLoader(test_eval_ds, batch_size=ev_bs, shuffle=False, num_workers=nw)

    metrics_list: list = []
    for m in eval_cfg["metrics"]:
        if m == "knn":
            knn_params = dict(eval_cfg.get("knn_params", {}))
            knn_params["sample_seed"] = training_seed
            knn_params["include_per_sample"] = True
            metrics_list.append({"metric": "knn", "kwargs": knn_params, "train_selection": "all"})
        else:
            metrics_list.append(m)

    flip_auroc = bool(eval_cfg.get("flip_auroc", False))
    effective_outlier_class = 0 if flip_auroc else dataset_cfg.get("outlier_class", 1)
    # Label both the test embeddings AND the KNN train-reference embeddings: the
    # reference records carry train-split prompt_hashes, which only resolve if
    # the parser df also contains the train rows. Passing test_parser.df alone
    # leaves the reference unlabeled, which silently disables knn calibrate_k
    # (train_labels_binary stays None). train/test prompt sets are disjoint, so
    # the concat introduces no duplicate-hash ambiguity.
    import pandas as pd
    eval_parser_df = pd.concat([train_parser.df, test_parser.df], ignore_index=True)
    evaluator = MultiMetricHallucinationEvaluator(
        activation_parser_df=eval_parser_df, train_data_loader=train_loader, metrics=metrics_list,
        batch_size=ev_bs, sub_batch_size=eval_cfg.get("sub_batch_size", 64),
        device=str(train_device), num_workers=nw, persistent_workers=False,
        outlier_class=effective_outlier_class,
    )
    ood_stats = evaluator.compute(eval_loader, model)
    ood_stats.pop("knn_scores", None)
    ood_stats.pop("knn_labels", None)

    eval_metrics: dict = {
        "method": method_cfg["name"], "dataset": dataset_cfg["name"], "seed": training_seed,
        "flip_auroc": flip_auroc, "split_seed": split_seed,
        "n_train": len(train_eval_ds), "n_test": len(test_eval_ds),
        "n_params_M": round(n_params / 1e6, 3), "view_aug": design_aug,
    }
    eval_metrics.update(ood_stats)
    return eval_metrics, []


def run_act_vit(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    training_seed: int,
    *,
    test_ap=None,
) -> tuple[dict, list[dict]]:
    """Train and evaluate ACT-ViT (arXiv:2510.00296) on full activation tensors.

    Treats the full (layers × response-tokens × hidden-dim) activation tensor
    of each generation as an "image" and classifies it with a Vision Transformer.
    Reads directly from icr_capture memmap directories.
    """
    import math

    import torch
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader

    from activation_research.act_vit import ACTViT, ACTViTConfig
    from activation_research.act_vit_dataset import ACTViTDataset
    from activation_research.memmap_activation_parser import MemmapActivationParser

    train_cfg = method_cfg["training"]
    model_params = method_cfg.get("model_params", {})
    eval_cfg = method_cfg.get("evaluation", {})
    prefix_training = bool(train_cfg.get("prefix_training", False))

    icr_cfg = dataset_cfg["icr_capture"]
    split_seed = _resolve_run_split_seed(experiment_cfg, training_seed)
    label_source = dataset_cfg.get("label_source", "substring")  # substring | llm_judge (#145)

    # Build split indices from MemmapActivationParser (same pattern as other runners).
    train_parser = MemmapActivationParser(
        icr_cfg["train_dir"],
        random_seed=split_seed,
        split_strategy="three_way",
        label_source=label_source,
    )
    _p1_expected_train_n = _apply_train_prevalence(train_parser, experiment_cfg, run_seed=training_seed)  # P1 sweep (#140); no-op otherwise
    train_df = train_parser.df[train_parser.df["split"] == "train"]
    val_df = train_parser.df[train_parser.df["split"] == "val"]
    train_idx = train_df["sample_index"].values
    if _p1_expected_train_n is not None and len(train_idx) != _p1_expected_train_n:
        raise RuntimeError(
            f"[P1 guard] train_prevalence set (expected {_p1_expected_train_n} train rows) "
            f"but act_vit train split has {len(train_idx)} — subsample did not propagate."
        )
    val_idx = val_df["sample_index"].values

    if test_ap is not None:
        # test_ap is a MemmapActivationParser for the test capture
        test_parser = MemmapActivationParser(
            icr_cfg["test_dir"],
            random_seed=split_seed,
            split_strategy="none",
            label_source=label_source,
        )
    else:
        test_parser = MemmapActivationParser(
            icr_cfg["test_dir"],
            random_seed=split_seed,
            split_strategy="none",
            label_source=label_source,
        )
    test_df = test_parser.df
    test_idx = test_df["sample_index"].values

    train_ds = ACTViTDataset(icr_cfg["train_dir"], train_idx, label_source=label_source)
    val_ds = ACTViTDataset(icr_cfg["train_dir"], val_idx, label_source=label_source)
    test_ds = ACTViTDataset(icr_cfg["test_dir"], test_idx, label_source=label_source)

    prefix_sampler = None
    if prefix_training:
        from activation_research.prefix_views import PrefixPairSampler

        prefix_sampler = PrefixPairSampler(
            mode="prefix_only",
            num_views=2,
            max_prefix=int(train_ds.max_response_len),
            min_prefix=int(train_cfg.get("prefix_min_tokens", 8)),
            min_gap=int(train_cfg.get("prefix_min_gap", 8)),
            sampling_prefixes=train_cfg.get("prefix_sampling_lengths"),
            seed=training_seed,
        )
        logger.info(
            "[act_vit] multi-k training enabled: "
            f"min={train_cfg.get('prefix_min_tokens', 8)} "
            f"gap={train_cfg.get('prefix_min_gap', 8)} "
            f"max={train_ds.max_response_len} "
            f"support={train_cfg.get('prefix_sampling_lengths', 'continuous')} "
            f"seed={training_seed}"
        )

    num_workers = experiment_cfg.get("num_workers", 4)
    persistent_workers = experiment_cfg.get("persistent_workers", True) and num_workers > 0
    batch_size = train_cfg["batch_size"]

    prefetch_factor = 4 if num_workers > 0 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    # Build model
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

    eval_device = torch.device(
        device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(eval_device)

    # Optimizer + cosine LR schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-2),
    )
    max_epochs = train_cfg.get("max_epochs", 50)
    total_steps = max_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Artifact dir for checkpoints
    artifact_dir = os.path.join(output_dir, "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    # pos_weight knob (P1 sweep #140): "auto" = inverse-frequency from the
    # (possibly subsampled) train labels; a float sets it explicitly; absent =
    # plain unweighted BCE (the naive baseline arm).
    _pw_cfg = train_cfg.get("pos_weight", None)
    if _pw_cfg == "auto":
        _n_pos = int((train_df["halu"] == 1).sum())
        _n_neg = int((train_df["halu"] == 0).sum())
        _pw = _n_neg / max(_n_pos, 1)
    elif _pw_cfg is not None:
        _pw = float(_pw_cfg)
    else:
        _pw = None
    if _pw is not None:
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([_pw], dtype=torch.float32, device=eval_device))
        logger.info(f"[act_vit] pos_weight={_pw:.3f}")
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    best_val_auroc = -1.0
    best_epoch = -1

    # If final_weights.pt exists, training completed in a prior run that crashed
    # before writing eval_metrics.json (e.g. ACTViTDataset IndexError during test
    # eval).  Load best_checkpoint.pt and set max_epochs=0 so the training loop
    # is skipped entirely; fall through to test evaluation.
    # Guard on final_weights.pt (written after the loop) not best_checkpoint.pt
    # (written mid-training) to avoid silently evaluating an undertrained model
    # after a mid-epoch crash.
    best_ckpt_path = os.path.join(artifact_dir, "best_checkpoint.pt")
    final_weights_path = os.path.join(artifact_dir, "final_weights.pt")
    if os.path.exists(final_weights_path) and os.path.exists(best_ckpt_path):
        logger.info(
            "[act_vit] final_weights.pt found — training already complete, "
            "loading best_checkpoint.pt and skipping to test eval."
        )
        ckpt = torch.load(best_ckpt_path, map_location=eval_device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        best_epoch = int(ckpt.get("epoch", -1))
        max_epochs = 0

    log_every = 50
    for epoch in range(max_epochs):
        model.train()
        epoch_t0 = time.time()
        step_t0 = time.time()
        for step_idx, batch in enumerate(train_loader):
            x = batch["activations"].to(eval_device, non_blocking=True)       # (B, L, N, D)
            labels_b = batch["label"].float().to(eval_device, non_blocking=True)  # (B,)
            if prefix_sampler is None:
                logits = model(x).squeeze(1)               # (B,)
                loss = loss_fn(logits, labels_b)
            else:
                response_lens_b = batch["response_len"].to(
                    eval_device, non_blocking=True
                )
                spec = prefix_sampler.sample()
                prefix_losses = []
                for prefix_len in spec.prefix_lens:
                    logits_k = _act_vit_logits_at_prefix(
                        model, x, response_lens_b, int(prefix_len)
                    )
                    prefix_losses.append(loss_fn(logits_k, labels_b))
                loss = torch.stack(prefix_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (step_idx + 1) % log_every == 0:
                window_s = time.time() - step_t0
                ms_per_step = 1000.0 * window_s / log_every
                samples_per_sec = (log_every * batch_size) / window_s
                logger.info(
                    f"[act_vit] epoch={epoch+1}/{max_epochs} step={step_idx+1}/{len(train_loader)} "
                    f"ms/step={ms_per_step:.0f} samp/s={samples_per_sec:.0f} loss={loss.item():.4f}"
                )
                step_t0 = time.time()

        # Validation AUROC
        model.eval()
        if prefix_training:
            from activation_research.prefix_views import resolve_eval_prefixes

            val_ks = resolve_eval_prefixes(
                train_cfg.get("prefix_validation_lengths", [8, 16, 32, 64]),
                int(val_ds.max_response_len),
            )
        else:
            val_ks = [int(val_ds.max_response_len)]
        val_aurocs: dict[int, float] = {}
        for val_k in val_ks:
            val_preds, val_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["activations"].to(eval_device, non_blocking=True)
                    if prefix_training:
                        logits = _act_vit_logits_at_prefix(
                            model,
                            x,
                            batch["response_len"].to(
                                eval_device, non_blocking=True
                            ),
                            val_k,
                        )
                    else:
                        logits = model(x).squeeze(1)
                    val_preds.append(torch.sigmoid(logits).cpu())
                    val_labels.append(batch["label"].cpu())

            val_scores = torch.cat(val_preds).numpy()
            val_lbls = torch.cat(val_labels).numpy()
            val_aurocs[val_k] = (
                float(roc_auc_score(val_lbls, val_scores))
                if len(set(val_lbls.tolist())) >= 2
                else float("nan")
            )
        finite_val_aurocs = [
            v for v in val_aurocs.values() if not math.isnan(v)
        ]
        val_auroc = (
            float(sum(finite_val_aurocs) / len(finite_val_aurocs))
            if finite_val_aurocs
            else float("nan")
        )

        if not math.isnan(val_auroc) and val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "selection_auroc": val_auroc,
                    "per_k_val_auroc": val_aurocs,
                },
                os.path.join(artifact_dir, "best_checkpoint.pt"),
            )

        epoch_wall_s = time.time() - epoch_t0
        logger.info(
            f"[act_vit] epoch={epoch+1}/{max_epochs}  "
            f"selection_val_auroc={val_auroc:.4f} per_k={val_aurocs} "
            f"wall={epoch_wall_s:.0f}s"
        )

    # Load best checkpoint (if any was saved)
    best_ckpt = os.path.join(artifact_dir, "best_checkpoint.pt")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=eval_device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    torch.save(
        {"model_state_dict": model.state_dict()},
        os.path.join(artifact_dir, "final_weights.pt"),
    )

    # Test evaluation
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["activations"].to(eval_device)
            if prefix_training:
                logits = _act_vit_logits_at_prefix(
                    model,
                    x,
                    batch["response_len"].to(eval_device),
                    int(test_ds.max_response_len),
                )
            else:
                logits = model(x).squeeze(1)
            all_preds.append(torch.sigmoid(logits).cpu())
            all_labels.append(batch["label"].cpu())

    scores_np = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()
    if len(set(labels_np.tolist())) >= 2:
        auroc = float(roc_auc_score(labels_np, scores_np))
    else:
        auroc = float("nan")

    # ---- Issue #149: AUROC vs response prefix -----------------------------
    # Truncation here is a SLICE of the token axis of (B, L, N, D), not a mask:
    # act_vit adaptive-max-pools the (L, N) plane to a fixed (L_p, N_p) grid, so
    # a narrower N is absorbed by the pooling. With prefix_training=true this
    # is the main fair arm (one fixed-geometry ACT-ViT trained across sampled
    # widths); otherwise it remains the legacy full-length-to-prefix
    # deployability arm that can reuse an existing checkpoint.
    act_vit_prefix_curve: dict = {}
    act_vit_prefix_scores: dict[int, object] = {}
    if eval_cfg.get("eval_prefix_curve", False):
        from activation_research.prefix_views import resolve_eval_prefixes

        _n_tokens = int(getattr(test_ds, "max_response_len", 64))
        _ks = resolve_eval_prefixes(eval_cfg.get("eval_prefix_lengths"), _n_tokens)
        logger.info(
            f"[act_vit] prefix curve ({'multi-k' if prefix_training else 'eval-only'}): "
            f"k={_ks} over N={_n_tokens} tokens"
        )
        for _k in _ks:
            _p, _l = [], []
            with torch.no_grad():
                for batch in test_loader:
                    x = batch["activations"].to(eval_device)   # (B, L, N, D)
                    logits = _act_vit_logits_at_prefix(
                        model,
                        x,
                        batch["response_len"].to(eval_device),
                        _k,
                    )
                    _p.append(torch.sigmoid(logits).cpu())
                    _l.append(batch["label"].cpu())
            _sn = torch.cat(_p).numpy()
            _ln = torch.cat(_l).numpy()
            _a = (
                float(roc_auc_score(_ln, _sn))
                if len(set(_ln.tolist())) >= 2
                else float("nan")
            )
            act_vit_prefix_curve[f"k{_k}_auroc"] = _a
            act_vit_prefix_scores[int(_k)] = _sn
            logger.info(f"  [act_vit] k={_k}: auroc={_a}")

    eval_metrics = {
        **act_vit_prefix_curve,
        "method": method_cfg["name"],
        "dataset": dataset_cfg["name"],
        "seed": training_seed,
        "split_seed": split_seed,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "auroc": auroc,
        "best_val_auroc": best_val_auroc,
        "best_epoch": best_epoch,
        "training_regime": "multi_k" if prefix_training else "full_length",
    }
    predictions = [
        {
            "example_id": i,
            "score_halu": float(s),
            "label_halu": int(l),
            "split": "test",
            **{
                f"k{k}_score_halu": float(prefix_scores[i])
                for k, prefix_scores in act_vit_prefix_scores.items()
            },
        }
        for i, (s, l) in enumerate(zip(scores_np, labels_np))
    ]

    write_run_manifest(output_dir)
    return eval_metrics, predictions


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
        "--methods",
        type=str,
        default=None,
        help="Comma-separated method names to filter from experiment config (must be a subset of its methods list)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if eval_metrics.json already exists",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help=(
            "Require an existing artifacts/final_weights.pt checkpoint and run "
            "evaluation without permitting training."
        ),
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override max_epochs for all methods (useful for quick testing)",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="Override steps_per_epoch_override for all methods (useful for quick testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda, cpu, auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output_dir from the experiment config (useful for directing output to local NVMe)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run plan (expected/complete/failed/pending) and exit without executing",
    )
    parser.add_argument(
        "--smoketest-memmap-cache",
        action="store_true",
        help=(
            "For each (dataset, seed), build the train+test ActivationParsers and "
            "verify the canonical memmap cache exists for that fingerprint, then "
            "exit before any training. Useful to confirm seeds 1..N will reuse the "
            "seed-0 cache rather than rebuilding it (which can take many hours)."
        ),
    )
    return parser.parse_args()


def run_contrastive_logprob_recon_twin(
    ap,
    dataset_cfg: dict,
    method_cfg: dict,
    experiment_cfg: dict,
    output_dir: str,
    device: str,
    training_seed: int,
    test_ap=None,
) -> tuple[dict, list[dict]]:
    """Train two asymmetric LogprobReconProgressiveCompressor heads (H7-H10).

    head_a: trained with ignore_label=1  → truth forms the compact cluster.
    head_b: trained with ignore_label=0  → hallu forms the compact cluster.

    At eval time, per-head embeddings are saved to .npy files under
    {output_dir}/artifacts/embeddings/ and a suite of post-hoc combination
    strategies (H7-H10) is computed directly from those numpy arrays.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    from activation_research.evaluation import inference_embeddings
    from activation_research.metrics import (
        _binary_outlier_labels,
        _record_embedding,
        _safe_auroc,
        knn_ood_stats,
        mahalanobis_ood_stats,
    )
    from activation_research.model import (
        LogprobReconProgressiveCompressor,
        TwinConcatModel,
    )
    from activation_research.training import train_contrastive_logprob_recon

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

    model_params = method_cfg.get("model_params", {})
    train_device = device if device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    aug_cfg = data_cfg.get("augmentations", None)
    augment_fn = None
    if aug_cfg:
        from activation_research.augmentations import AugmentationComposer

        augment_fn = AugmentationComposer(
            augmentations=aug_cfg.get("ops", []),
            asymmetric=aug_cfg.get("asymmetric", False),
        )

    checkpoint_dir = os.path.join(output_dir, "artifacts")

    def _build_head():
        return LogprobReconProgressiveCompressor(
            input_dim=dataset_cfg["input_dim"],
            final_dim=model_params.get("final_dim", 512),
            input_dropout=model_params.get("input_dropout", 0.3),
            recon_seq_len=model_params.get("recon_seq_len", 64),
            recon_hidden_dim=model_params.get("recon_hidden_dim", 256),
            recon_lambda=model_params.get("recon_lambda", 1.0),
            logprob_var_threshold=model_params.get("logprob_var_threshold", 1e-4),
        )

    def _train_head(head, ignore_label: int, ckpt_suffix: str) -> None:
        ckpt_path = os.path.join(checkpoint_dir, f"final_weights_{ckpt_suffix}.pt")
        if os.path.exists(ckpt_path):
            logger.info(
                f"{ckpt_path} found — skipping training for head_{ckpt_suffix}"
            )
            ckpt = torch.load(ckpt_path, map_location=train_device)
            head.load_state_dict(ckpt["model_state_dict"])
            return
        train_contrastive_logprob_recon(
            model=head,
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
            ignore_label=ignore_label,
            persistent_workers=experiment_cfg.get("persistent_workers", True),
            recon_lambda=model_params.get("recon_lambda", 1.0),
            use_infinite_index_stream=train_cfg.get("use_infinite_index_stream", True),
            infinite_stream_shuffle=True,
            infinite_stream_seed=training_seed,
            steps_per_epoch_override=train_cfg.get("steps_per_epoch_override"),
            min_total_steps=train_cfg.get("min_total_steps"),
            balanced_sampling=train_cfg.get("balanced_sampling", False),
            grad_clip_norm=train_cfg.get("grad_clip_norm"),
            augment_fn=augment_fn,
            select_on_val=train_cfg.get("select_on_val", False),
        )
        torch.save(
            {"model_state_dict": head.state_dict()},
            ckpt_path,
        )

    head_a = _build_head()
    head_b = _build_head()

    _train_head(head_a, ignore_label=1, ckpt_suffix="a")
    _train_head(head_b, ignore_label=0, ckpt_suffix="b")

    # Build TwinConcatModel (kept for save/load compatibility).
    twin_model = TwinConcatModel(head_a, head_b)
    twin_model.eval()

    # ------------------------------------------------------------------
    # Per-head embedding inference + .npy cache
    # ------------------------------------------------------------------
    emb_dir = os.path.join(checkpoint_dir, "embeddings")
    os.makedirs(emb_dir, exist_ok=True)

    paths = {
        "train_za": os.path.join(emb_dir, "train_za.npy"),
        "train_zb": os.path.join(emb_dir, "train_zb.npy"),
        "test_za": os.path.join(emb_dir, "test_za.npy"),
        "test_zb": os.path.join(emb_dir, "test_zb.npy"),
        "train_labels": os.path.join(emb_dir, "train_labels.npy"),
        "test_labels": os.path.join(emb_dir, "test_labels.npy"),
    }
    all_cached = all(os.path.exists(p) for p in paths.values())

    if all_cached:
        logger.info("All .npy embedding files found — loading from cache.")
        train_za = np.load(paths["train_za"])
        train_zb = np.load(paths["train_zb"])
        test_za = np.load(paths["test_za"])
        test_zb = np.load(paths["test_zb"])
        train_labels = np.load(paths["train_labels"])
        test_labels = np.load(paths["test_labels"])
    else:
        logger.info("Running per-head inference to build embedding cache.")
        train_ds_target = train_ds.slice_layers(target_layers)
        test_ds_target = test_ds.slice_layers(target_layers)

        # Build label lookup dicts from the activation-parser dataframes.
        train_label_lookup = (
            ap.df.set_index("prompt_hash")["halu"].to_dict()
        )
        test_label_lookup = (
            eval_ap.df.set_index("prompt_hash")["halu"].to_dict()
        )

        def _run_inference(head, ds):
            head.eval()
            records = inference_embeddings(
                model=head,
                dataset=ds,
                batch_size=eval_cfg.get("eval_batch_size", 256),
                sub_batch_size=eval_cfg.get("sub_batch_size", 64),
                device=train_device,
                num_workers=experiment_cfg.get("num_workers", 4),
                persistent_workers=False,
            )
            return records

        def _to_arrays(records, label_lookup):
            zs, lbls = [], []
            for r in records:
                hk = r["hashkey"]
                if hk not in label_lookup:
                    continue
                z = _record_embedding(r).float().numpy()
                zs.append(z)
                lbls.append(int(label_lookup[hk]))
            return (
                np.stack(zs, axis=0).astype(np.float32),
                np.array(lbls, dtype=np.int32),
            )

        train_recs_a = _run_inference(head_a, train_ds_target)
        train_recs_b = _run_inference(head_b, train_ds_target)
        test_recs_a = _run_inference(head_a, test_ds_target)
        test_recs_b = _run_inference(head_b, test_ds_target)

        train_za, train_labels = _to_arrays(train_recs_a, train_label_lookup)
        train_zb, _ = _to_arrays(train_recs_b, train_label_lookup)
        test_za, test_labels = _to_arrays(test_recs_a, test_label_lookup)
        test_zb, _ = _to_arrays(test_recs_b, test_label_lookup)

        np.save(paths["train_za"], train_za)
        np.save(paths["train_zb"], train_zb)
        np.save(paths["test_za"], test_za)
        np.save(paths["test_zb"], test_zb)
        np.save(paths["train_labels"], train_labels)
        np.save(paths["test_labels"], test_labels)
        logger.info(
            f"Saved embeddings: train={train_za.shape}, test={test_za.shape}"
        )

    # ------------------------------------------------------------------
    # Build eval config
    # ------------------------------------------------------------------
    flip_auroc: bool = bool(eval_cfg.get("flip_auroc", False))
    effective_outlier_class = 0 if flip_auroc else dataset_cfg.get("outlier_class", 1)

    knn_params_cfg = dict(eval_cfg.get("knn_params", {}))
    knn_k = int(knn_params_cfg.get("k", 50))
    knn_metric = str(knn_params_cfg.get("metric", "euclidean"))
    knn_calibrate_k = bool(knn_params_cfg.get("calibrate_k", False))
    knn_k_candidates = knn_params_cfg.get("k_candidates", None)
    knn_max_train_size = int(knn_params_cfg.get("max_train_size", 200000))

    # ------------------------------------------------------------------
    # Main metrics: KNN + Mahalanobis on concat space via synthetic records
    # ------------------------------------------------------------------
    train_z_concat = np.concatenate([train_za, train_zb], axis=1)
    test_z_concat = np.concatenate([test_za, test_zb], axis=1)

    def _make_records(arr, labels):
        return [
            {"z1": torch.from_numpy(arr[i]).float(), "halu": int(labels[i])}
            for i in range(len(labels))
        ]

    train_records_concat = _make_records(train_z_concat, train_labels)
    test_records_concat = _make_records(test_z_concat, test_labels)

    knn_stats = knn_ood_stats(
        train_records=train_records_concat,
        test_records=test_records_concat,
        outlier_class=effective_outlier_class,
        k=knn_k,
        metric=knn_metric,
        train_label_filter="all",
        calibrate_k=knn_calibrate_k,
        k_candidates=knn_k_candidates,
        max_train_size=knn_max_train_size,
        sample_seed=training_seed,
    )

    maha_stats = mahalanobis_ood_stats(
        train_records=train_records_concat,
        test_records=test_records_concat,
        outlier_class=effective_outlier_class,
        train_label_filter="id_only",
    )

    # ------------------------------------------------------------------
    # H7: KNN distance ensemble (simple average of per-head distances)
    # ------------------------------------------------------------------
    binary_labels = (test_labels == int(effective_outlier_class)).astype(np.int32)

    n_neighbors_h7 = min(knn_k, len(train_za))
    nn_a = NearestNeighbors(n_neighbors=n_neighbors_h7, metric=knn_metric)
    nn_a.fit(train_za)
    dists_a, _ = nn_a.kneighbors(test_za)

    nn_b = NearestNeighbors(n_neighbors=n_neighbors_h7, metric=knn_metric)
    nn_b.fit(train_zb)
    dists_b, _ = nn_b.kneighbors(test_zb)

    ensemble_scores = (dists_a.mean(axis=1) + dists_b.mean(axis=1)) / 2.0
    h7_auroc = _safe_auroc(binary_labels, ensemble_scores)

    # ------------------------------------------------------------------
    # H8: per-class KNN recall in three spaces
    # ------------------------------------------------------------------
    def _knn_recall(train_arr, test_arr, train_lbls, test_lbls, k, metric):
        nn = NearestNeighbors(n_neighbors=min(k, len(train_arr)), metric=metric)
        nn.fit(train_arr)
        _, indices = nn.kneighbors(test_arr)
        # majority vote
        neighbor_labels = train_lbls[indices]  # (N_test, k)
        preds = (neighbor_labels.sum(axis=1) > (k / 2)).astype(np.int32)
        recalls = {}
        for cls in (0, 1):
            mask = test_lbls == cls
            if mask.sum() == 0:
                recalls[cls] = float("nan")
            else:
                recalls[cls] = float((preds[mask] == cls).mean())
        return recalls

    h8_concat = _knn_recall(
        train_z_concat, test_z_concat, train_labels, test_labels, knn_k, knn_metric
    )
    h8_head_a = _knn_recall(
        train_za, test_za, train_labels, test_labels, knn_k, knn_metric
    )
    h8_head_b = _knn_recall(
        train_zb, test_zb, train_labels, test_labels, knn_k, knn_metric
    )

    # ------------------------------------------------------------------
    # H9: whitened ensemble (per-head distances normalised by their std)
    # ------------------------------------------------------------------
    std_a = float(dists_a.mean(axis=1).std()) + 1e-8
    std_b = float(dists_b.mean(axis=1).std()) + 1e-8
    whitened_scores = dists_a.mean(axis=1) / std_a + dists_b.mean(axis=1) / std_b
    h9_auroc = _safe_auroc(binary_labels, whitened_scores)

    # ------------------------------------------------------------------
    # H10: element-wise sum space
    # ------------------------------------------------------------------
    train_z_sum = train_za + train_zb
    test_z_sum = test_za + test_zb

    train_records_sum = _make_records(train_z_sum, train_labels)
    test_records_sum = _make_records(test_z_sum, test_labels)

    knn_sum_stats = knn_ood_stats(
        train_records=train_records_sum,
        test_records=test_records_sum,
        outlier_class=effective_outlier_class,
        k=knn_k,
        metric=knn_metric,
        train_label_filter="all",
        calibrate_k=knn_calibrate_k,
        k_candidates=knn_k_candidates,
        max_train_size=knn_max_train_size,
        sample_seed=training_seed,
    )
    h10_knn_sum_auroc = float(knn_sum_stats.get("knn_auroc", float("nan")))

    try:
        maha_sum_stats = mahalanobis_ood_stats(
            train_records=train_records_sum,
            test_records=test_records_sum,
            outlier_class=effective_outlier_class,
            train_label_filter="id_only",
        )
        h10_maha_sum_auroc = float(maha_sum_stats.get("mahalanobis_auroc", float("nan")))
    except Exception:
        h10_maha_sum_auroc = float("nan")

    # ------------------------------------------------------------------
    # Assemble eval_metrics
    # ------------------------------------------------------------------
    eval_metrics: dict = {
        "method": method_cfg["name"],
        "dataset": dataset_cfg["name"],
        "seed": training_seed,
        "flip_auroc": flip_auroc,
        "split_seed": experiment_cfg.get("split_seed", 42),
        "n_train": len(train_labels),
        "n_test": len(test_labels),
    }
    eval_metrics.update(knn_stats)
    eval_metrics.update(maha_stats)

    eval_metrics["h7_knn_ensemble_auroc"] = float(h7_auroc)

    eval_metrics["h8_recall_class0_concat"] = h8_concat[0]
    eval_metrics["h8_recall_class1_concat"] = h8_concat[1]
    eval_metrics["h8_recall_class0_head_a"] = h8_head_a[0]
    eval_metrics["h8_recall_class1_head_a"] = h8_head_a[1]
    eval_metrics["h8_recall_class0_head_b"] = h8_head_b[0]
    eval_metrics["h8_recall_class1_head_b"] = h8_head_b[1]

    eval_metrics["h9_whitened_ensemble_auroc"] = float(h9_auroc)

    eval_metrics["h10_knn_sum_auroc"] = h10_knn_sum_auroc
    eval_metrics["h10_maha_sum_auroc"] = h10_maha_sum_auroc

    predictions: list[dict] = []
    return eval_metrics, predictions


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
        if args.methods is not None:
            requested = [m.strip() for m in args.methods.split(",") if m.strip()]
            unknown = [m for m in requested if m not in methods]
            if unknown:
                logger.error(
                    f"--methods filter contains names not in experiment config: {unknown}. "
                    f"Available: {methods}"
                )
                sys.exit(1)
            methods = requested

        if args.seeds is not None:
            training_seeds = [int(s) for s in args.seeds.split(",")]
        else:
            training_seeds = experiment_cfg.get("training_seeds", [42])
    elif args.dataset and args.method:
        with open(args.dataset) as f:
            single_dataset_cfg = json.load(f)
        single_method_cfg = load_method_config(
            args.method, project_root=str(project_root)
        )

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
    steps_per_epoch_override = args.steps_per_epoch

    # ---- Resolve device ----
    device = args.device or experiment_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")


    output_base = args.output_dir or experiment_cfg.get("output_dir", "runs")
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
            if args.methods is not None:
                exp_cfg_loaded["methods"] = methods
                if exp_cfg_loaded.get("method_configs"):
                    exp_cfg_loaded["method_configs"] = {
                        m: exp_cfg_loaded["method_configs"][m]
                        for m in methods
                        if m in exp_cfg_loaded["method_configs"]
                    }
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
                        method_configs[m] = load_method_config(
                            m, project_root=str(project_root)
                        )
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

    # ---- Smoketest helpers ----
    smoketest_results: list[dict] = []  # filled when --smoketest-memmap-cache

    def _canonical_preload_params(methods_list, preloaded_method_cfgs, project_root):
        """Derive (relevant_layers, pad_length, include_logprobs, top_k) from the
        first learned method in the experiment. The memmap cache fingerprint is
        identical across methods that share these data params, so any learned
        method's data block represents what the experiment will actually load."""
        for m in methods_list:
            if m in preloaded_method_cfgs:
                mcfg = preloaded_method_cfgs[m]
            else:
                mcfg_path = os.path.join(
                    str(project_root), "configs", "methods", f"{m}.json"
                )
                if not os.path.exists(mcfg_path):
                    continue
                mcfg = load_method_config(m, project_root=str(project_root))
            if mcfg.get("training") is None:
                continue  # non-learned method — doesn't preload
            data = mcfg.get("data", {})
            return (
                parse_layer_range(data.get("relevant_layers", "14-29")),
                int(data.get("pad_length", 63)),
                bool(data.get("include_response_logprobs", False)),
                int(data.get("response_logprobs_top_k", 20)),
            )
        return None

    def _check_memmap_cache(ap, params, label):
        """Compute the canonical fingerprint and report whether the cache exists.
        Returns a result dict and prints a one-line HIT/MISS summary."""
        relevant_layers, pad_length, include_logprobs, top_k = params
        fp = ap._memmap_cache_fingerprint(
            relevant_layers, pad_length, include_logprobs, top_k
        )
        cache_dir = ap._memmap_cache_dir(fp)
        manifest = cache_dir / "manifest.json"
        activations_npy = cache_dir / "activations.npy"
        hit = manifest.exists() and activations_npy.exists()
        result = {
            "label": label,
            "fingerprint": fp,
            "cache_dir": str(cache_dir),
            "hit": hit,
            "manifest_exists": manifest.exists(),
            "activations_npy_exists": activations_npy.exists(),
            "n_total": len(ap.df),
            "zarr_count": int(ap.activation_logger._response_activations.shape[0]),
        }
        status = "HIT " if hit else "MISS"
        logger.info(
            f"[smoketest] {status} {label}  fp={fp}  n_total={result['n_total']:,} "
            f"zarr_count={result['zarr_count']:,}  -> {cache_dir}"
        )
        return result

    # ---- Main loop ----
    # Track per-seed failures so the process can exit non-zero at the end.
    # Worker dispatch relies on exit code AND output-file existence to decide
    # done vs failed, and a multi-seed invocation must not silently succeed
    # when an earlier seed errored.
    had_failures = 0
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
        backend = dataset_cfg.get("backend", "zarr")

        # Resolve per-seed split seeds. split_seeds[i] is the split_seed used for
        # training_seeds[i] in the *full* config list. Falls back to the single
        # split_seed field if absent.
        split_seeds_list = experiment_cfg.get("split_seeds", None)
        global_split_seed = experiment_cfg.get("split_seed", 42)
        # Build a mapping from seed value → split_seed so that running a subset
        # of seeds via --seeds still picks the correct fold (not just index 0).
        full_training_seeds = experiment_cfg.get("training_seeds", list(training_seeds))
        _split_seed_map: dict = {}
        if split_seeds_list is not None:
            for _i, _s in enumerate(full_training_seeds):
                if _i < len(split_seeds_list):
                    _split_seed_map[_s] = split_seeds_list[_i]

        # Build the test ActivationParser once — the test set is constant across all
        # seeds (unified format: separate test zarr with split_strategy="none").
        test_ap = None
        if backend == "memmap":
            from activation_research.memmap_activation_parser import MemmapActivationParser
            logger.info(
                f"Loading test MemmapActivationParser from: "
                f"{dataset_cfg['icr_capture']['test_dir']}"
            )
            test_ap = MemmapActivationParser(
                capture_dir=_resolve_shared(dataset_cfg["icr_capture"]["test_dir"]),
                random_seed=global_split_seed,
                split_strategy="none",
                label_source=dataset_cfg.get("label_source", "substring"),  # #145
                verbose=True,
            )
        elif has_train_test and "test" in dataset_cfg and isinstance(dataset_cfg["test"], dict):
            test_cfg = {**dataset_cfg, **dataset_cfg["test"]}
            _resolve_paths(test_cfg, project_root)
            test_eval_json = _build_eval_json(test_cfg)
            logger.info(f"Loading test ActivationParser from: {test_cfg['activations_path']}")
            test_ap = ActivationParser(
                inference_json=test_cfg["inference_json"],
                eval_json=test_eval_json,
                activations_path=test_cfg["activations_path"],
                logger_type=dataset_cfg.get("backend", "zarr"),
                random_seed=global_split_seed,
                split_strategy="none",
                verbose=True,
            )

        # ---- Smoketest: check the test cache once per dataset ----
        smoketest_params = None
        if args.smoketest_memmap_cache:
            if backend == "memmap":
                logger.info(
                    f"[smoketest] {dataset_name}: backend=memmap — "
                    f"smoketest-memmap-cache not applicable, skipping"
                )
            else:
                smoketest_params = _canonical_preload_params(
                    methods, _preloaded_method_cfgs, project_root
                )
                if smoketest_params is None:
                    logger.warning(
                        f"[smoketest] {dataset_name}: no learned method found in "
                        f"experiment — nothing to check (non-learned methods don't preload)."
                    )
                elif test_ap is not None:
                    smoketest_results.append(
                        _check_memmap_cache(test_ap, smoketest_params, f"{dataset_name}/test")
                    )

        # Non-learned methods (token_entropy, logprob_baseline) run once regardless
        # of how many seeds are in the sweep.
        completed_nonlearned: set = set()

        # Smoketest fast-path: the memmap cache fingerprint is seed-agnostic by
        # design (uses zarr-path + n_total + cache params, not random_seed).  So
        # for the train zarr, build the parser once at the first seed, check the
        # cache, and replay the same HIT/MISS row for every other seed without
        # rebuilding the parser (which can take several minutes per JSONL load).
        if args.smoketest_memmap_cache and smoketest_params is not None and has_train_test:
            first_seed = training_seeds[0]
            first_split_seed = (
                _split_seed_map.get(first_seed, global_split_seed) if _split_seed_map else global_split_seed
            )
            train_cfg = {**dataset_cfg, **dataset_cfg["train"]}
            _resolve_paths(train_cfg, project_root)
            train_eval_json = _build_eval_json(train_cfg)
            logger.info(
                f"[smoketest] Loading train ActivationParser once "
                f"(split_seed={first_split_seed}) from: {train_cfg['activations_path']}"
            )
            ap = ActivationParser(
                inference_json=train_cfg["inference_json"],
                eval_json=train_eval_json,
                activations_path=train_cfg["activations_path"],
                logger_type=dataset_cfg.get("backend", "zarr"),
                random_seed=first_split_seed,
                split_strategy=split_strategy,
                verbose=True,
            )
            for seed in training_seeds:
                actual_split_seed = (
                    _split_seed_map.get(seed, global_split_seed) if _split_seed_map else global_split_seed
                )
                smoketest_results.append(
                    _check_memmap_cache(
                        ap,
                        smoketest_params,
                        f"{dataset_name}/train  seed={seed} split_seed={actual_split_seed}",
                    )
                )
            continue  # skip per-seed loop entirely for this dataset

        for seed_idx, seed in enumerate(training_seeds):
            actual_split_seed = (
                _split_seed_map.get(seed, global_split_seed) if _split_seed_map else global_split_seed
            )

            # Build a fresh train parser for this fold's split.
            if backend == "memmap":
                icr_cfg_block = dataset_cfg["icr_capture"]
                if "train_sources" in icr_cfg_block:
                    from activation_research.composite_memmap_parser import (
                        CompositeMemmapActivationParser,
                    )
                    resolved_sources = [
                        {**s, "dir": _resolve_shared(s["dir"])}
                        for s in icr_cfg_block["train_sources"]
                    ]
                    logger.info(
                        f"Loading train CompositeMemmapActivationParser "
                        f"(split_seed={actual_split_seed}) from {len(resolved_sources)} sources"
                    )
                    ap = CompositeMemmapActivationParser(
                        sources=resolved_sources,
                        random_seed=actual_split_seed,
                        split_strategy="three_way",
                        source_sample_seed=icr_cfg_block.get("train_sources_seed", 0),
                        verbose=True,
                    )
                else:
                    from activation_research.memmap_activation_parser import MemmapActivationParser
                    logger.info(
                        f"Loading train MemmapActivationParser (split_seed={actual_split_seed}) "
                        f"from: {icr_cfg_block['train_dir']}"
                    )
                    ap = MemmapActivationParser(
                        capture_dir=_resolve_shared(icr_cfg_block["train_dir"]),
                        random_seed=actual_split_seed,
                        split_strategy="three_way",
                        label_source=dataset_cfg.get("label_source", "substring"),  # #145
                        verbose=True,
                    )
            elif has_train_test:
                train_cfg = {**dataset_cfg, **dataset_cfg["train"]}
                _resolve_paths(train_cfg, project_root)
                train_eval_json = _build_eval_json(train_cfg)
                logger.info(
                    f"Loading train ActivationParser (split_seed={actual_split_seed}) "
                    f"from: {train_cfg['activations_path']}"
                )
                ap = ActivationParser(
                    inference_json=train_cfg["inference_json"],
                    eval_json=train_eval_json,
                    activations_path=train_cfg["activations_path"],
                    logger_type=dataset_cfg.get("backend", "zarr"),
                    random_seed=actual_split_seed,
                    split_strategy=split_strategy,
                    verbose=True,
                )
            else:
                # Legacy flat config: single zarr, split_seed controls train/test boundary.
                _resolve_paths(dataset_cfg, project_root)
                eval_json_path = _build_eval_json(dataset_cfg)
                ap = ActivationParser(
                    inference_json=dataset_cfg["inference_json"],
                    eval_json=eval_json_path,
                    activations_path=dataset_cfg["activations_path"],
                    logger_type=dataset_cfg.get("backend", "zarr"),
                    random_seed=actual_split_seed,
                    split_strategy=split_strategy,
                    verbose=True,
                )

            # ---- Smoketest: check the train cache for this seed and skip training ----
            if args.smoketest_memmap_cache:
                if smoketest_params is not None:
                    smoketest_results.append(
                        _check_memmap_cache(
                            ap,
                            smoketest_params,
                            f"{dataset_name}/train  seed={seed} split_seed={actual_split_seed}",
                        )
                    )
                continue  # skip method loop entirely

            for method_name in methods:
                # Load method config (use pre-loaded if available)
                if method_name in _preloaded_method_cfgs:
                    method_cfg = _preloaded_method_cfgs[method_name]
                else:
                    method_cfg = load_method_config(
                        method_name, project_root=str(project_root)
                    )

                # Apply max_epochs / steps_per_epoch overrides
                if (max_epochs_override is not None or steps_per_epoch_override is not None) and method_cfg.get("training"):
                    method_cfg = dict(method_cfg)  # shallow copy to avoid mutating cached
                    method_cfg["training"] = dict(method_cfg["training"])
                    if max_epochs_override is not None:
                        method_cfg["training"]["max_epochs"] = max_epochs_override
                    if steps_per_epoch_override is not None:
                        method_cfg["training"]["steps_per_epoch_override"] = steps_per_epoch_override

                from scripts.experiment_utils import is_seeded_method

                is_learned = is_seeded_method(method_cfg)

                if not is_learned:
                    # Non-seeded methods run exactly once across the entire seed sweep.
                    if method_name in completed_nonlearned:
                        continue
                    effective_seed = None
                    run_dir = os.path.join(output_base, exp_name, dataset_name, method_name)
                else:
                    effective_seed = seed
                    run_dir = os.path.join(
                        output_base, exp_name, dataset_name, method_name, f"seed_{seed}"
                    )

                # Resume check — skip only when eval_metrics.json exists AND there is
                # no run_error.json (which would indicate the prior result was degenerate).
                # For methods that also produce per-sample predictions.csv, require both
                # files; otherwise a prior run that wrote only eval_metrics.json (issue #107)
                # would be falsely treated as complete.
                eval_metrics_path = os.path.join(run_dir, "eval_metrics.json")
                pred_path = os.path.join(run_dir, "predictions.csv")
                run_error_path = os.path.join(run_dir, "run_error.json")
                prior_error = os.path.exists(run_error_path)
                routine_for_skip = method_cfg.get("routine", method_cfg["name"])
                needs_predictions = routine_for_skip in {
                    "contrastive_logprob_recon",
                    "contrastive_logprob_recon_twin",
                    "contrastive_logprob_recon_dualhead_fusion",
                    "tokenwise_projection_rescore",
                }
                have_predictions = (not needs_predictions) or os.path.exists(pred_path)
                if os.path.exists(eval_metrics_path) and have_predictions and not args.force:
                    if prior_error:
                        logger.warning(
                            f"Found both eval_metrics.json and run_error.json for "
                            f"{method_name} seed={effective_seed} — re-running to fix "
                            f"degenerate prior result. Use --force to suppress this check."
                        )
                    else:
                        logger.info(
                            f"Skipping {method_name} seed={effective_seed} (already complete)"
                        )
                        if not is_learned:
                            completed_nonlearned.add(method_name)
                        continue
                elif os.path.exists(eval_metrics_path) and needs_predictions and not have_predictions:
                    logger.info(
                        f"{method_name} seed={effective_seed}: eval_metrics.json present but "
                        f"predictions.csv missing — re-running eval to write predictions."
                    )

                os.makedirs(run_dir, exist_ok=True)
                os.makedirs(os.path.join(run_dir, "artifacts"), exist_ok=True)

                logger.info(f"Running {method_name} seed={effective_seed} -> {run_dir}")

                try:
                    if args.eval_only:
                        required_checkpoint = os.path.join(
                            run_dir, "artifacts", "final_weights.pt"
                        )
                        if not os.path.isfile(required_checkpoint) or os.path.getsize(
                            required_checkpoint
                        ) == 0:
                            raise FileNotFoundError(
                                "--eval-only requires a nonempty checkpoint at "
                                f"{required_checkpoint}"
                            )

                    if effective_seed is not None:
                        from utils.seeding import seed_everything

                        seed_everything(effective_seed)

                    # Write merged config
                    merged_config = {
                        "dataset": dataset_cfg,
                        "method": method_cfg,
                        "experiment": {k: v for k, v in experiment_cfg.items()},
                        "training_seed": effective_seed,
                        "split_seed": actual_split_seed,
                    }
                    with open(os.path.join(run_dir, "config.json"), "w") as f:
                        json.dump(merged_config, f, indent=2)

                    # Write manifest
                    write_run_manifest(run_dir)

                    # Dispatch to method runner
                    routine = method_cfg.get("routine", method_cfg["name"])
                    if routine == "contrastive":
                        eval_metrics, predictions = run_contrastive(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, effective_seed,
                            test_ap=test_ap,
                        )
                    elif routine == "contrastive_logprob_recon":
                        eval_metrics, predictions = run_contrastive_logprob_recon(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, effective_seed,
                            test_ap=test_ap,
                        )
                    elif routine == "tokenwise_contrastive_logprob_recon":
                        eval_metrics, predictions = run_tokenwise_contrastive_logprob_recon(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, effective_seed,
                            test_ap=test_ap,
                        )
                    elif routine == "tokenwise_projection_rescore":
                        eval_metrics, predictions = run_tokenwise_projection_rescore(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, effective_seed,
                            test_ap=test_ap,
                        )
                    elif routine == "contrastive_logprob_recon_twin":
                        eval_metrics, predictions = run_contrastive_logprob_recon_twin(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, effective_seed,
                            test_ap=test_ap,
                        )
                    elif routine == "contrastive_logprob_recon_shared_trunk":
                        eval_metrics, predictions = run_contrastive_logprob_recon_shared_trunk(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, effective_seed,
                            test_ap=test_ap,
                        )
                    elif routine == "contrastive_logprob_recon_dualhead_fusion":
                        eval_metrics, predictions = run_contrastive_logprob_recon_dualhead_fusion(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, effective_seed,
                            test_ap=test_ap,
                        )
                    elif routine == "linear_probe":
                        eval_metrics, predictions = run_linear_probe(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, effective_seed,
                            test_ap=test_ap,
                        )
                    elif routine == "multi_layer_linear_probe":
                        eval_metrics, predictions = run_multi_layer_linear_probe(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, effective_seed,
                            test_ap=test_ap,
                        )
                    elif routine == "simclr_linear":
                        eval_metrics, predictions = run_simclr_linear(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, effective_seed,
                            test_ap=test_ap,
                        )
                    elif routine == "simclr_cotrained":
                        eval_metrics, predictions = run_simclr_cotrained(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, effective_seed,
                            test_ap=test_ap,
                        )
                    elif routine == "simclr_projection":
                        eval_metrics, predictions = run_simclr_projection(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, effective_seed,
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
                    elif routine == "saplma":
                        eval_metrics, predictions = run_saplma(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, effective_seed,
                            test_ap=test_ap,
                        )
                    elif routine == "saplma_logprob_recon":
                        eval_metrics, predictions = run_saplma_logprob_recon(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, effective_seed,
                            test_ap=test_ap,
                        )
                    elif routine == "llmsknow_probe":
                        eval_metrics, predictions = run_llmsknow_probe(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, effective_seed,
                            test_ap=test_ap,
                        )
                    elif routine == "icr_probe":
                        eval_metrics, predictions = run_icr_probe(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, effective_seed,
                            test_ap=test_ap,
                        )
                    elif routine == "act_vit":
                        eval_metrics, predictions = run_act_vit(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, effective_seed,
                            test_ap=test_ap,
                        )
                    elif routine == "contrastive_actvit":
                        eval_metrics, predictions = run_contrastive_actvit(
                            ap, dataset_cfg, method_cfg, experiment_cfg, run_dir, device, effective_seed,
                            test_ap=test_ap,
                        )
                    else:
                        logger.warning(f"Unknown routine: {routine}, skipping")
                        continue

                    # Write eval_metrics.json — preserve original mtime when re-running
                    # only to backfill predictions.csv (values would be identical).
                    if not os.path.exists(eval_metrics_path) or args.force:
                        with open(eval_metrics_path, "w") as f:
                            json.dump(eval_metrics, f, indent=2)

                    # Write predictions.csv
                    if predictions:
                        pred_path = os.path.join(run_dir, "predictions.csv")
                        with open(pred_path, "w", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
                            writer.writeheader()
                            writer.writerows(predictions)

                    # A successful recovery supersedes a previous per-seed error.
                    # Leaving this marker behind makes status tooling classify valid
                    # eval outputs as failed and causes subsequent runs to repeat.
                    if _clear_stale_run_error(run_error_path):
                        logger.info(
                            f"Removed stale run_error.json after successful {method_name} recovery"
                        )

                    logger.info(f"Completed {method_name} seed={effective_seed}: {eval_metrics}")

                    if not is_learned:
                        completed_nonlearned.add(method_name)

                    # Free per-seed GPU/CPU state before the next iteration —
                    # multi-seed invocations load a different checkpoint each pass
                    # and stale CUDA allocations can pile up.
                    import gc as _gc

                    _gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception:
                    import traceback

                    tb = traceback.format_exc()
                    logger.error(
                        f"Failed {method_name} seed={effective_seed}: {tb}"
                    )
                    # Write error record so we know this run failed
                    error_path = os.path.join(run_dir, "run_error.json")
                    with open(error_path, "w") as f:
                        json.dump(
                            {"method": method_name, "seed": effective_seed, "error": tb},
                            f,
                            indent=2,
                        )
                    had_failures += 1

    if args.smoketest_memmap_cache:
        n_total = len(smoketest_results)
        n_hit = sum(1 for r in smoketest_results if r["hit"])
        n_miss = n_total - n_hit
        logger.info("")
        logger.info(f"[smoketest] Summary: {n_hit}/{n_total} cache hits, {n_miss} miss")
        if n_miss:
            for r in smoketest_results:
                if not r["hit"]:
                    logger.info(f"[smoketest] MISS  {r['label']}  -> {r['cache_dir']}")
        sys.exit(0 if n_miss == 0 else 1)

    logger.info("Experiment complete.")
    if had_failures:
        logger.error(
            f"Experiment finished with {had_failures} per-seed failure(s) — exiting with code 1."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
