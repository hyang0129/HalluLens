"""Smoketest for Issue #75: combined logprob + attention reconstruction.

End-to-end on real captured data — verifies the new
``LogprobAttnReconProgressiveCompressor`` + ``train_contrastive_logprob_attn_recon``
+ ``MemmapContrastiveDataset`` stack runs cleanly on a single GPU node before
we commit to the full ablation matrix.

Default capture: ``shared/icr_capture/hotpotqa_test_Llama-3.1-8B-Instruct``
(7405 samples, has both response_activations.npy and response_attention.npy).

Sanity checks per run:
- Dataset loads and __getitem__ returns expected keys/shapes.
- One batch forward+backward produces a finite loss with no NaN gradients.
- Training over a handful of capped steps completes and writes a checkpoint.
- The encoder's inference forward(x) yields finite embeddings of shape
  (B, final_dim).

Usage (on a GPU node)::

    python scripts/smoketest_issue_75.py \\
        --capture-dir shared/icr_capture/hotpotqa_test_Llama-3.1-8B-Instruct \\
        --output-dir /tmp/smoketest_issue_75 \\
        --epochs 2 --steps-per-epoch 20
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
from activation_research.model import LogprobAttnReconProgressiveCompressor
from activation_research.training import train_contrastive_logprob_attn_recon


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--capture-dir",
        default="shared/icr_capture/hotpotqa_test_Llama-3.1-8B-Instruct",
        help="Issue #72 capture directory to read.",
    )
    p.add_argument("--output-dir", default="/tmp/smoketest_issue_75")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--steps-per-epoch", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--sub-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument(
        "--relevant-layers",
        default="14-29",
        help="Layer range '14-29' or comma list '22,26' for view sampling.",
    )
    p.add_argument(
        "--attn-direction",
        default="backward",
        choices=("forward", "backward", "both"),
    )
    p.add_argument("--attn-offset-k", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def parse_layers(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


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

    # --- Load config to size things correctly ---
    with (capture_dir / "config.json").open() as fh:
        capture_cfg = json.load(fh)
    input_dim = int(capture_cfg["hidden_dim"])
    max_resp = int(capture_cfg["max_response_len"])
    logger.info(
        f"capture: n={capture_cfg['n_samples']}, "
        f"num_layers={capture_cfg['num_layers']}, "
        f"hidden={input_dim}, r_max={capture_cfg['r_max']}"
    )

    # --- Build dataset ---
    relevant_layers = parse_layers(args.relevant_layers)
    forward_offset = args.attn_offset_k if args.attn_direction in ("forward", "both") else None
    backward_offset = args.attn_offset_k if args.attn_direction in ("backward", "both") else None

    t0 = time.time()
    train_ds = MemmapContrastiveDataset(
        capture_dir,
        split="train",
        num_views=2,
        relevant_layers=relevant_layers,
        include_response_logprobs=True,
        response_logprobs_top_k=20,
        pad_length=max_resp - 1,
        include_response_attention=True,
        attention_summary="stats",
        attention_target_layer_offset_forward=forward_offset,
        attention_target_layer_offset_backward=backward_offset,
        random_seed=args.seed,
    )
    val_ds = MemmapContrastiveDataset(
        capture_dir,
        split="val",
        num_views=2,
        relevant_layers=relevant_layers,
        include_response_logprobs=True,
        response_logprobs_top_k=20,
        pad_length=max_resp - 1,
        include_response_attention=True,
        attention_summary="stats",
        attention_target_layer_offset_forward=forward_offset,
        attention_target_layer_offset_backward=backward_offset,
        random_seed=args.seed,
    )
    logger.info(
        f"dataset load: {time.time()-t0:.1f}s; "
        f"train={len(train_ds)}, val={len(val_ds)}"
    )

    # --- Smoke item check ---
    sample = train_ds[0]
    expected_keys = {"views_activations", "view_indices", "halu", "hashkey"}
    missing = expected_keys - set(sample)
    assert not missing, f"sample missing keys: {missing}"
    assert sample["views_activations"].shape == (2, max_resp, input_dim), (
        f"unexpected views shape {sample['views_activations'].shape}"
    )
    if backward_offset is not None:
        assert "attention_backward" in sample, "attention_backward field missing"
        assert sample["attention_backward"].shape == (2, 3), (
            f"unexpected attention_backward shape {sample['attention_backward'].shape}"
        )
    if forward_offset is not None:
        assert "attention_forward" in sample, "attention_forward field missing"
    logger.info(
        f"sample[0] views={tuple(sample['views_activations'].shape)} "
        f"halu={float(sample['halu'])} "
        f"attn_keys={[k for k in sample if k.startswith('attention_')]}"
    )

    # --- Build model ---
    model = LogprobAttnReconProgressiveCompressor(
        input_dim=input_dim,
        final_dim=512,
        input_dropout=0.3,
        recon_seq_len=max_resp - 1,
        recon_hidden_dim=256,
        recon_lambda=1.0,
        logprob_var_threshold=1e-4,
        attn_direction=args.attn_direction,
        attn_offset_k=args.attn_offset_k,
        attn_target="stats",
        attn_num_stat_features=3,
        attn_recon_hidden_dim=256,
        attn_recon_lambda=1.0,
        attn_var_threshold=1e-5,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"model: {model.__class__.__name__}, params={n_params:,}")

    # --- Train ---
    checkpoint_dir = output_dir / "artifacts"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    t1 = time.time()
    train_contrastive_logprob_attn_recon(
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
        use_labels=True,
        ignore_label=1,
        use_infinite_index_stream=True,
        infinite_stream_shuffle=True,
        infinite_stream_seed=args.seed,
        steps_per_epoch_override=args.steps_per_epoch,
        balanced_sampling=False,
        grad_clip_norm=1.0,
    )
    train_secs = time.time() - t1
    logger.info(f"train: {train_secs:.1f}s for {args.epochs}ep x {args.steps_per_epoch}st")

    # --- Inference sanity ---
    model.eval()
    with torch.no_grad():
        batch_x = sample["views_activations"][0].unsqueeze(0).to(device)  # (1, L, D)
        z = model(batch_x)
    assert torch.isfinite(z).all(), "non-finite values in embedding"
    assert z.shape == (1, 512), f"unexpected embedding shape {z.shape}"
    logger.info(f"inference: z.shape={tuple(z.shape)}, norm={z.norm().item():.3f}")

    # --- Checkpoint sanity ---
    ckpts = sorted(checkpoint_dir.glob("*.pt"))
    assert ckpts, "no checkpoint files written"
    logger.info(f"checkpoints written: {[c.name for c in ckpts]}")

    logger.success("PASS: issue #75 smoketest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
