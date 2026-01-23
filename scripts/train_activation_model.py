#!/usr/bin/env python3
"""Train and evaluate lightweight models on logged LLM activations.

This script is the CLI/script equivalent of the prototype notebook
`a_preparing_training_precise_qa_halu_as_outlier.ipynb`.

It trains either:
- a supervised-contrastive embedding model (e.g., `ProgressiveCompressor`) on
  pairs of layer activations, or
- a hallucination classifier on a chosen layer activation sequence.

It can also run an OOD-style evaluation by computing embeddings for a baseline
(ID) subset and scoring the full test set with Mahalanobis distance and cosine
similarity statistics.

Run from repo root:
    python scripts/train_activation_model.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from loguru import logger

# Add project root to import path (matches existing scripts/* convention)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from activation_logging.activation_parser import ActivationParser
from activation_research.evaluation import inference_embeddings
from activation_research.metrics import cosine_similarity_ood_stats, mahalanobis_ood_stats
from activation_research.model import (  # noqa: E402
    HallucinationClassifier,
    LastLayerHaluClassifier,
    ProgressiveCompressor,
    SimpleHaluClassifier,
)
from activation_research.training import train_contrastive, train_halu_classifier


@dataclass(frozen=True)
class RunConfig:
    experiment_name: str
    inference_json: str
    eval_json: str
    activations_path: str
    logger_type: str
    seed: int

    routine: str
    model_name: str

    train_relevant_layers: List[int]
    eval_relevant_layers: List[int]
    fixed_layer: Optional[int]
    pad_length: int

    # training hyperparams
    epochs: int
    batch_size: int
    sub_batch_size: int
    lr: float
    temperature: float
    num_workers: int
    device: str

    # OOD evaluation config
    do_ood_eval: bool
    id_label: int

    # model hyperparams
    final_dim: int
    input_dropout: float


def _parse_layers(s: str) -> List[int]:
    """Parse layer spec like '14-29' or '22,26' or '16-29,31'."""
    s = s.strip()
    if not s:
        raise ValueError("Empty layer spec")

    parts = [p.strip() for p in s.split(",") if p.strip()]
    layers: List[int] = []
    for part in parts:
        if "-" in part:
            lo_s, hi_s = [x.strip() for x in part.split("-", 1)]
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"Invalid range '{part}'")
            layers.extend(list(range(lo, hi + 1)))
        else:
            layers.append(int(part))

    # stable unique
    seen = set()
    out: List[int] = []
    for x in layers:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _resolve_fixed_layer_index(relevant_layers: List[int], fixed_layer: Optional[int]) -> Optional[int]:
    """Map a fixed layer *number* to the dataset's fixed layer *index*.

    ActivationDataset.fixed_layer is defined as the index within relevant_layers.
    This helper lets the CLI accept an actual layer number (e.g., 28).
    """
    if fixed_layer is None:
        return None
    if fixed_layer in range(len(relevant_layers)):
        # Ambiguous: user might pass index directly.
        # If the value is also a valid layer number, prefer mapping as layer number.
        if fixed_layer in relevant_layers and relevant_layers.index(fixed_layer) != fixed_layer:
            return relevant_layers.index(fixed_layer)
        return fixed_layer
    if fixed_layer not in relevant_layers:
        raise ValueError(
            f"fixed_layer={fixed_layer} is not in relevant_layers={relevant_layers}. "
            "Pass either a layer number in --*-layers, or an index within those layers."
        )
    return relevant_layers.index(fixed_layer)


def _infer_activation_dim(ap: ActivationParser, candidate_layers: List[int]) -> int:
    """Infer the activation hidden dimension D from the first available sample."""
    row, _result, activations, _input_len = ap.get_activations(0)
    for layer_idx in candidate_layers:
        if layer_idx < 0 or layer_idx >= len(activations):
            continue
        act = activations[layer_idx]
        if act is None:
            continue
        # act shape is (B=1, L, D)
        return int(act.shape[-1])
    raise RuntimeError(
        "Unable to infer activation dimension. "
        "No activations found in candidate layers for the first sample."
    )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _default_checkpoint_name(routine: str) -> str:
    if routine == "contrastive":
        return "contrastive_last.pt"
    if routine == "classifier":
        return "halu_classifier_last.pt"
    raise ValueError(f"Unknown routine: {routine}")


def _auto_resume_checkpoint_path(checkpoint_dir: Path, routine: str) -> Optional[str]:
    """Return absolute checkpoint path to resume from, if present."""
    last_name = _default_checkpoint_name(routine)
    candidate = checkpoint_dir / last_name
    if candidate.exists():
        return str(candidate)
    return None


def _attach_labels_from_df(records: List[dict], df) -> List[dict]:
    """Attach halu labels to inference embeddings via prompt hash."""
    if "prompt_hash" not in df.columns or "halu" not in df.columns:
        raise ValueError("ActivationParser.df must contain 'prompt_hash' and 'halu' columns")

    label_by_hash = dict(zip(df["prompt_hash"].tolist(), df["halu"].tolist()))

    missing = 0
    for r in records:
        hk = r.get("hashkey")
        if hk is None:
            continue
        if hk not in label_by_hash:
            missing += 1
            continue
        r["halu"] = int(bool(label_by_hash[hk]))

    if missing:
        logger.warning(f"Missing labels for {missing} embedding records")
    return records


def _filter_df_by_label(df, label: int):
    return df[df["halu"].astype(int) == int(label)].copy()


def _build_model(
    routine: str,
    model_name: str,
    input_dim: int,
    final_dim: int,
    input_dropout: float,
) -> torch.nn.Module:
    if routine == "contrastive":
        if model_name != "progressive_compressor":
            raise ValueError(f"Unsupported contrastive model: {model_name}")
        return ProgressiveCompressor(input_dim=input_dim, final_dim=final_dim, input_dropout=input_dropout)

    if routine == "classifier":
        if model_name == "simple_halu_classifier":
            return SimpleHaluClassifier(input_dim=input_dim)
        if model_name == "last_layer_transformer":
            return LastLayerHaluClassifier(input_dim=input_dim)
        if model_name == "hallucination_classifier":
            return HallucinationClassifier(dim=input_dim, layer_index=0)
        raise ValueError(f"Unsupported classifier model: {model_name}")

    raise ValueError(f"Unknown routine: {routine}")


def _train(
    model: torch.nn.Module,
    routine: str,
    train_dataset,
    test_dataset,
    *,
    epochs: int,
    batch_size: int,
    sub_batch_size: int,
    lr: float,
    temperature: float,
    device: str,
    num_workers: int,
    checkpoint_dir: str,
    resume_from: Optional[str],
) -> None:
    if routine == "contrastive":
        train_contrastive(
            model,
            train_dataset,
            test_dataset=test_dataset,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            temperature=temperature,
            device=device,
            num_workers=num_workers,
            sub_batch_size=sub_batch_size,
            use_labels=True,
            ignore_label=-1,
            checkpoint_dir=checkpoint_dir,
            resume_from=resume_from,
        )
        return

    if routine == "classifier":
        train_halu_classifier(
            model,
            train_dataset,
            test_dataset=test_dataset,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            num_workers=num_workers,
            sub_batch_size=sub_batch_size,
            checkpoint_dir=checkpoint_dir,
            resume_from=resume_from,
        )
        return

    raise ValueError(f"Unknown routine: {routine}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate activation-based detectors")

    parser.add_argument(
        "--experiment-name",
        required=True,
        help=(
            "Unique name for this experiment. All outputs/checkpoints/logs go under "
            "<output-dir>/<experiment-name>/ so re-running the same command can auto-resume."
        ),
    )

    parser.add_argument("--inference-json", required=True, help="Path to generation.jsonl")
    parser.add_argument("--eval-json", required=True, help="Path to eval_results.json")
    parser.add_argument("--activations-path", required=True, help="LMDB path or JSON activations directory")
    parser.add_argument("--logger-type", choices=["lmdb", "json"], default="json")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--routine", choices=["contrastive", "classifier"], default="contrastive")
    parser.add_argument(
        "--model",
        dest="model_name",
        default="progressive_compressor",
        choices=[
            "progressive_compressor",
            "simple_halu_classifier",
            "last_layer_transformer",
            "hallucination_classifier",
        ],
    )

    parser.add_argument("--train-layers", default="14-29", help="Layers used to sample training pairs")
    parser.add_argument("--eval-layers", default="22,26", help="Layers used for embedding OOD eval")
    parser.add_argument(
        "--fixed-layer",
        type=int,
        default=None,
        help="Optional fixed layer (layer number or index within --*-layers)",
    )
    parser.add_argument("--pad-length", type=int, default=63)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--sub-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--final-dim", type=int, default=512)
    parser.add_argument("--input-dropout", type=float, default=0.3)

    parser.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Checkpoint file (relative to checkpoint dir) or absolute. "
            "If omitted, the script auto-resumes from the latest <routine>_last.pt if present."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable auto-resume behavior (starts training from scratch).",
    )

    parser.add_argument("--do-ood-eval", action="store_true")
    parser.add_argument(
        "--id-label",
        type=int,
        default=0,
        choices=[0, 1],
        help="Label treated as in-distribution for baseline embeddings",
    )

    parser.add_argument(
        "--run-name",
        default="default",
        help=(
            "Optional sub-run name under the experiment directory. "
            "Use a stable value if you want reruns to auto-resume. Default: 'default'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "output" / "activation_training"),
        help="Base output directory",
    )

    args = parser.parse_args(argv)

    run_name = (args.run_name or "default").strip()
    if not run_name:
        run_name = "default"

    out_base = Path(args.output_dir) / args.experiment_name / run_name
    ckpt_dir = out_base / "checkpoints"
    logs_dir = out_base / "logs"

    _ensure_dir(out_base)
    _ensure_dir(ckpt_dir)
    _ensure_dir(logs_dir)

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(str(logs_dir / "train.log"), level="INFO")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_layers = _parse_layers(args.train_layers)
    eval_layers = _parse_layers(args.eval_layers)

    fixed_layer_train = _resolve_fixed_layer_index(train_layers, args.fixed_layer)
    fixed_layer_eval = _resolve_fixed_layer_index(eval_layers, args.fixed_layer)

    logger.info(f"Experiment: {args.experiment_name} / run: {run_name}")
    logger.info(f"Output directory: {out_base}")

    ap = ActivationParser(
        inference_json=args.inference_json,
        eval_json=args.eval_json,
        activations_path=args.activations_path,
        logger_type=args.logger_type,
        random_seed=args.seed,
        verbose=True,
    )

    input_dim = _infer_activation_dim(ap, train_layers)
    logger.info(f"Inferred activation dim: {input_dim}")

    # Datasets
    min_target_layers = 2 if args.routine == "contrastive" else 1

    train_dataset = ap.get_dataset(
        "train",
        relevant_layers=train_layers,
        fixed_layer=fixed_layer_train,
        pad_length=args.pad_length,
        min_target_layers=min_target_layers,
    )
    test_dataset = ap.get_dataset(
        "test",
        relevant_layers=train_layers,
        fixed_layer=fixed_layer_train,
        pad_length=args.pad_length,
        min_target_layers=min_target_layers,
    )

    # Model
    model = _build_model(
        routine=args.routine,
        model_name=args.model_name,
        input_dim=input_dim,
        final_dim=args.final_dim,
        input_dropout=args.input_dropout,
    )

    cfg = RunConfig(
        experiment_name=args.experiment_name,
        inference_json=str(Path(args.inference_json)),
        eval_json=str(Path(args.eval_json)),
        activations_path=str(Path(args.activations_path)),
        logger_type=args.logger_type,
        seed=args.seed,
        routine=args.routine,
        model_name=args.model_name,
        train_relevant_layers=train_layers,
        eval_relevant_layers=eval_layers,
        fixed_layer=args.fixed_layer,
        pad_length=args.pad_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sub_batch_size=args.sub_batch_size,
        lr=args.lr,
        temperature=args.temperature,
        num_workers=args.num_workers,
        device=device,
        do_ood_eval=bool(args.do_ood_eval),
        id_label=int(args.id_label),
        final_dim=args.final_dim,
        input_dropout=args.input_dropout,
    )

    (out_base / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    logger.info(f"Training: routine={args.routine} model={args.model_name} device={device}")

    resume_from = args.resume_from
    if resume_from is None and not args.no_resume:
        auto_path = _auto_resume_checkpoint_path(ckpt_dir, args.routine)
        if auto_path is not None:
            resume_from = auto_path
            logger.info(f"Auto-resuming from: {resume_from}")
        else:
            logger.info("No existing last-checkpoint found; starting from scratch")

    _train(
        model,
        args.routine,
        train_dataset,
        test_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sub_batch_size=args.sub_batch_size,
        lr=args.lr,
        temperature=args.temperature,
        device=device,
        num_workers=args.num_workers,
        checkpoint_dir=str(ckpt_dir),
        resume_from=resume_from,
    )

    if not args.do_ood_eval:
        logger.info("Skipping OOD evaluation (--do-ood-eval not set)")
        return 0

    # OOD evaluation (not applicable for pure classifier routines yet)
    if args.routine != "contrastive":
        logger.warning("OOD embedding eval currently runs for contrastive models only")
        return 0

    # Baseline embeddings are computed from ID-only training split.
    ap_id = ActivationParser(
        inference_json=args.inference_json,
        eval_json=args.eval_json,
        activations_path=args.activations_path,
        logger_type=args.logger_type,
        random_seed=args.seed,
        verbose=False,
    )
    ap_id.df = _filter_df_by_label(ap_id.df, cfg.id_label)

    baseline_ds = ap_id.get_dataset(
        "train",
        relevant_layers=eval_layers,
        fixed_layer=fixed_layer_eval,
        pad_length=args.pad_length,
        min_target_layers=2,
    )

    full_ap = ActivationParser(
        inference_json=args.inference_json,
        eval_json=args.eval_json,
        activations_path=args.activations_path,
        logger_type=args.logger_type,
        random_seed=args.seed,
        verbose=False,
    )
    eval_ds = full_ap.get_dataset(
        "test",
        relevant_layers=eval_layers,
        fixed_layer=fixed_layer_eval,
        pad_length=args.pad_length,
        min_target_layers=2,
    )

    logger.info("Computing baseline embeddings")
    baseline = inference_embeddings(
        model,
        baseline_ds,
        batch_size=256,
        sub_batch_size=args.sub_batch_size,
        device=device,
        num_workers=args.num_workers,
        layers=None,
        persistent_workers=False,
    )

    logger.info("Computing evaluation embeddings")
    embeddings = inference_embeddings(
        model,
        eval_ds,
        batch_size=256,
        sub_batch_size=args.sub_batch_size,
        device=device,
        num_workers=args.num_workers,
        layers=None,
        persistent_workers=False,
    )

    baseline = _attach_labels_from_df(baseline, full_ap.df)
    embeddings = _attach_labels_from_df(embeddings, full_ap.df)

    # outlier label is the opposite of id label
    outlier_label = 1 - cfg.id_label

    metrics = {
        "mahalanobis": mahalanobis_ood_stats(baseline, embeddings, outlier_class=outlier_label),
        "cosine": cosine_similarity_ood_stats(baseline, embeddings, outlier_class=outlier_label),
        "n_baseline": len(baseline),
        "n_eval": len(embeddings),
    }

    (out_base / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info(f"Saved metrics to: {out_base / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
