"""Smoketest for Issue #102: shared-trunk twin-head variants (D1 + D2).

End-to-end on real captured data — verifies that both new model classes
(``SharedTrunkSplitOutputCompressor`` and ``SharedTrunkProjectionHeadCompressor``)
plus ``train_contrastive_logprob_recon_dualloss`` run cleanly on a single GPU
node before kicking off the full experiment grid.

Default capture: ``shared/icr_capture/hotpotqa_test_Llama-3.1-8B-Instruct``
(same as smoketest_issue_75.py).

Per variant we check:
- Model instantiates and forward shapes match the spec (eval surface is full
  2D output for D1, trunk for D2).
- ``forward_slices`` / ``forward_with_heads`` returns a 4-tuple with correct
  shapes including ``logprob_pred``.
- ``train_contrastive_logprob_recon_dualloss`` completes a handful of capped
  steps with finite loss.
- After training, ``model.eval(); model(x)`` produces a finite embedding of
  the expected dim.
- A checkpoint is written to disk.

Usage (on a GPU node)::

    python scripts/smoketest_issue_102.py \\
        --capture-dir shared/icr_capture/hotpotqa_test_Llama-3.1-8B-Instruct \\
        --output-dir /tmp/smoketest_issue_102 \\
        --epochs 1 --steps-per-epoch 10 --variants d1,d2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from activation_research.memmap_contrastive_dataset import MemmapContrastiveDataset
from activation_research.model import (
    SharedTrunkProjectionHeadCompressor,
    SharedTrunkSplitOutputCompressor,
)
from activation_research.training import train_contrastive_logprob_recon_dualloss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--capture-dir",
        default="shared/icr_capture/hotpotqa_test_Llama-3.1-8B-Instruct",
        help="Issue #72 capture directory to read.",
    )
    p.add_argument("--output-dir", default="/tmp/smoketest_issue_102")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--steps-per-epoch", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--sub-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--relevant-layers", default="14-29")
    p.add_argument(
        "--variants",
        default="d1,d2",
        help="Comma-separated variants to test: d1 (split-output), d2 (projection-head).",
    )
    p.add_argument("--half-dim", type=int, default=256, help="D1: per-half dim (total = 2*half_dim).")
    p.add_argument("--trunk-dim", type=int, default=512, help="D2: trunk dim (eval surface).")
    p.add_argument("--head-dim", type=int, default=256, help="D2: projection head output dim.")
    p.add_argument("--head-hidden-dim", type=int, default=256, help="D2: head MLP hidden dim.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def parse_layers(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def build_datasets(capture_dir: Path, relevant_layers: list[int], max_resp: int, seed: int):
    train_ds = MemmapContrastiveDataset(
        capture_dir,
        split="train",
        num_views=2,
        relevant_layers=relevant_layers,
        include_response_logprobs=True,
        response_logprobs_top_k=20,
        pad_length=max_resp - 1,
        random_seed=seed,
    )
    val_ds = MemmapContrastiveDataset(
        capture_dir,
        split="val",
        num_views=2,
        relevant_layers=relevant_layers,
        include_response_logprobs=True,
        response_logprobs_top_k=20,
        pad_length=max_resp - 1,
        random_seed=seed,
    )
    return train_ds, val_ds


def smoketest_variant(
    variant: str,
    args: argparse.Namespace,
    input_dim: int,
    max_resp: int,
    train_ds,
    val_ds,
    sample,
    device: str,
    output_dir: Path,
) -> None:
    logger.info(f"=== variant={variant} ===")

    if variant == "d1":
        model = SharedTrunkSplitOutputCompressor(
            input_dim=input_dim,
            half_dim=args.half_dim,
            input_dropout=0.3,
            recon_seq_len=max_resp - 1,
            recon_hidden_dim=256,
            recon_lambda=1.0,
            logprob_var_threshold=1e-4,
        )
        expected_eval_dim = 2 * args.half_dim
    elif variant == "d2":
        model = SharedTrunkProjectionHeadCompressor(
            input_dim=input_dim,
            trunk_dim=args.trunk_dim,
            head_dim=args.head_dim,
            head_hidden_dim=args.head_hidden_dim,
            input_dropout=0.3,
            recon_seq_len=max_resp - 1,
            recon_hidden_dim=256,
            recon_lambda=1.0,
            logprob_var_threshold=1e-4,
        )
        expected_eval_dim = args.trunk_dim
    else:
        raise ValueError(f"unknown variant {variant!r}; expected 'd1' or 'd2'")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{variant}: {model.__class__.__name__} params={n_params:,} eval_dim={expected_eval_dim}")

    # --- Helper-method shape check on CPU before training ---
    x = sample["views_activations"][0].unsqueeze(0)  # (1, L, D)
    model_cpu = model  # still on CPU here
    if variant == "d1":
        z, z_A, z_B, logprob_pred = model_cpu.forward_slices(x)
        assert z.shape == (1, 2 * args.half_dim), f"D1 z shape {z.shape}"
        assert z_A.shape == (1, args.half_dim), f"D1 z_A shape {z_A.shape}"
        assert z_B.shape == (1, args.half_dim), f"D1 z_B shape {z_B.shape}"
    else:
        z, z_A, z_B, logprob_pred = model_cpu.forward_with_heads(x)
        assert z.shape == (1, args.trunk_dim), f"D2 z (trunk) shape {z.shape}"
        assert z_A.shape == (1, args.head_dim), f"D2 z_A shape {z_A.shape}"
        assert z_B.shape == (1, args.head_dim), f"D2 z_B shape {z_B.shape}"
    assert logprob_pred.shape == (1, max_resp - 1), f"logprob_pred shape {logprob_pred.shape}"
    logger.info(
        f"{variant}: helper shapes ok — z={tuple(z.shape)} z_A={tuple(z_A.shape)} "
        f"z_B={tuple(z_B.shape)} logprob_pred={tuple(logprob_pred.shape)}"
    )

    # --- Train ---
    checkpoint_dir = output_dir / variant / "artifacts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train_contrastive_logprob_recon_dualloss(
        model=model,
        train_dataset=train_ds,
        test_dataset=val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sub_batch_size=args.sub_batch_size,
        lr=1e-5,
        temperature=0.25,
        device=device,
        num_workers=args.num_workers,
        checkpoint_dir=str(checkpoint_dir),
        save_every=1,
        use_infinite_index_stream=True,
        infinite_stream_shuffle=True,
        infinite_stream_seed=args.seed,
        steps_per_epoch_override=args.steps_per_epoch,
        balanced_sampling=False,
        grad_clip_norm=1.0,
        ignore_labels=(1, 0),
    )
    train_secs = time.time() - t0
    logger.info(f"{variant}: train {train_secs:.1f}s for {args.epochs}ep x {args.steps_per_epoch}st")

    # --- Eval-surface forward check ---
    model.eval()
    with torch.no_grad():
        batch_x = sample["views_activations"][0].unsqueeze(0).to(device)
        z = model(batch_x)
    assert torch.isfinite(z).all(), f"{variant}: non-finite values in embedding"
    assert z.shape == (1, expected_eval_dim), (
        f"{variant}: unexpected eval-surface shape {z.shape}, expected (1, {expected_eval_dim})"
    )
    logger.info(f"{variant}: inference z.shape={tuple(z.shape)} norm={z.norm().item():.3f}")

    # --- Checkpoint sanity ---
    ckpts = sorted(checkpoint_dir.glob("*.pt"))
    assert ckpts, f"{variant}: no checkpoint files written"
    logger.info(f"{variant}: checkpoints={[c.name for c in ckpts]}")

    logger.success(f"PASS: variant {variant}")


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    capture_dir = (PROJECT_ROOT / args.capture_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device}, capture={capture_dir}, output={output_dir}")
    if device == "cuda":
        logger.info(f"gpu={torch.cuda.get_device_name(0)}")

    with (capture_dir / "config.json").open() as fh:
        capture_cfg = json.load(fh)
    input_dim = int(capture_cfg["hidden_dim"])
    max_resp = int(capture_cfg["max_response_len"])
    logger.info(
        f"capture: n={capture_cfg['n_samples']}, num_layers={capture_cfg['num_layers']}, "
        f"hidden={input_dim}, r_max={capture_cfg['r_max']}"
    )

    relevant_layers = parse_layers(args.relevant_layers)
    t0 = time.time()
    train_ds, val_ds = build_datasets(capture_dir, relevant_layers, max_resp, args.seed)
    logger.info(f"dataset load: {time.time()-t0:.1f}s; train={len(train_ds)}, val={len(val_ds)}")

    sample = train_ds[0]
    assert sample["views_activations"].shape == (2, max_resp, input_dim), (
        f"unexpected views shape {sample['views_activations'].shape}"
    )

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        smoketest_variant(
            variant=v,
            args=args,
            input_dim=input_dim,
            max_resp=max_resp,
            train_ds=train_ds,
            val_ds=val_ds,
            sample=sample,
            device=device,
            output_dir=output_dir,
        )

    logger.success(f"PASS: issue #102 smoketest ({','.join(variants)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
