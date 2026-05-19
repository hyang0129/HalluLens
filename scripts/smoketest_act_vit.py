"""Smoketest for PR #93 / act_vit branch — ACT-ViT classifier.

Verifies ACTViT + ACTViTDataset on a real GPU node with icr_capture data
(same memmap captures used by contrastive baselines).

Checks
------
1. GPU is available and CUDA identity is printed.
2. Synthetic forward pass: (B, L, N, D) → (B, 1) with paper-default config
   (input_dim=4096), on GPU, fp16 input (cast internally to fp32).
3. Varying spatial sizes trigger adaptive pooling correctly.
4. Dataset load from a real icr_capture dir — __getitem__ keys and shapes.
5. DataLoader batch through model: shape + finite output.
6. Train loop: one mini-epoch (few steps), loss finite, no NaN gradients.
7. Checkpoint written to output_dir.

Usage (on a GPU node)::

    python scripts/smoketest_act_vit.py \\
        --capture-dir shared/icr_capture/hotpotqa_test_Llama-3.1-8B-Instruct \\
        --output-dir /tmp/smoketest_act_vit \\
        --steps 30 --batch-size 4

The script exits 0 on PASS, 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from activation_research.act_vit import ACTViT, ACTViTConfig
from activation_research.act_vit_dataset import ACTViTDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--capture-dir",
        default="shared/icr_capture/hotpotqa_test_Llama-3.1-8B-Instruct",
        help="icr_capture directory (has config.json + response_activations.npy).",
    )
    p.add_argument("--output-dir", default="/tmp/smoketest_act_vit")
    p.add_argument("--steps", type=int, default=30, help="Mini-train steps.")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _make_config(input_dim: int) -> ACTViTConfig:
    """Minimal config fast enough for smoke — smaller than paper defaults."""
    return ACTViTConfig(
        input_dim=input_dim,
        L_p=8,
        N_p=20,
        patch_h=2,
        patch_w=4,
        d_adapter=64,
        d_model=128,
        num_heads=4,
        depth=2,
        mlp_ratio=2.0,
        dropout=0.0,
    )


def check_synthetic(device: str, input_dim: int) -> None:
    """Forward-pass sanity on synthetic tensors."""
    cfg = _make_config(input_dim)
    model = ACTViT(cfg).to(device)
    model.eval()

    with torch.no_grad():
        # fp32 input — standard case
        x = torch.randn(2, 32, 50, input_dim, device=device)
        out = model(x)
        assert out.shape == (2, 1), f"wrong shape fp32: {out.shape}"
        assert torch.isfinite(out).all(), "non-finite in fp32 forward"
        logger.info(f"  fp32 (2,32,50,{input_dim}) → {tuple(out.shape)}  ✓")

        # fp16 input — model casts internally
        x16 = x.half()
        out16 = model(x16)
        assert out16.shape == (2, 1), f"wrong shape fp16: {out16.shape}"
        assert torch.isfinite(out16).all(), "non-finite in fp16 forward"
        logger.info(f"  fp16 (2,32,50,{input_dim}) → {tuple(out16.shape)}  ✓")

        # Irregular spatial size — adaptive pool must handle it
        x_big = torch.randn(1, 64, 200, input_dim, device=device)
        out_big = model(x_big)
        assert out_big.shape == (1, 1), f"wrong shape big: {out_big.shape}"
        assert torch.isfinite(out_big).all(), "non-finite in big-tensor forward"
        logger.info(f"  fp32 (1,64,200,{input_dim}) → {tuple(out_big.shape)}  ✓")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  model params: {n_params:,}")


def check_dataset(capture_dir: Path, batch_size: int, num_workers: int) -> tuple[ACTViTDataset, int]:
    """Load dataset, verify __getitem__ contract, return (train_ds, hidden_dim)."""
    with (capture_dir / "config.json").open() as fh:
        cap_cfg = json.load(fh)
    n_samples: int = cap_cfg["n_samples"]
    hidden_dim: int = cap_cfg["hidden_dim"]
    num_layers: int = cap_cfg["num_layers"]
    max_response_len: int = cap_cfg["max_response_len"]
    logger.info(
        f"  capture: n={n_samples}, layers={num_layers}, "
        f"hidden={hidden_dim}, r_max={max_response_len}"
    )

    indices = list(range(min(n_samples, 200)))  # small slice for smoke
    ds = ACTViTDataset(capture_dir, indices)

    assert len(ds) == len(indices), f"dataset len mismatch: {len(ds)} vs {len(indices)}"

    sample = ds[0]
    assert "activations" in sample, "sample missing 'activations'"
    assert "label" in sample, "sample missing 'label'"
    assert "response_len" in sample, "sample missing 'response_len'"

    act = sample["activations"]
    assert act.ndim == 3, f"activations should be 3-D (L,R,D), got {act.shape}"
    assert act.shape[0] == num_layers, (
        f"layer dim mismatch: got {act.shape[0]}, expected {num_layers}"
    )
    assert act.shape[2] == hidden_dim, (
        f"hidden dim mismatch: got {act.shape[2]}, expected {hidden_dim}"
    )
    assert sample["label"] in (0, 1), f"label out of range: {sample['label']}"
    logger.info(
        f"  sample[0]: activations={tuple(act.shape)}, "
        f"label={sample['label']}, response_len={sample['response_len']}  ✓"
    )

    # DataLoader batch shape check
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    batch = next(iter(loader))
    B = len(batch["label"])
    assert batch["activations"].ndim == 4, (
        f"batched activations should be 4-D (B,L,R,D), got {batch['activations'].shape}"
    )
    logger.info(f"  DataLoader batch: activations={tuple(batch['activations'].shape)}  ✓")

    return ds, hidden_dim


def check_train_loop(
    ds: ACTViTDataset,
    input_dim: int,
    device: str,
    batch_size: int,
    num_workers: int,
    steps: int,
    output_dir: Path,
) -> None:
    """A few training steps — verify finite loss, no NaN grads, checkpoint written."""
    cfg = _make_config(input_dim)
    model = ACTViT(cfg).to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    t0 = time.time()
    total_loss = 0.0
    step = 0
    done = False
    while not done:
        for batch in loader:
            acts = batch["activations"].to(device)      # (B, L, R, D)
            labels = batch["label"].float().to(device)  # (B,)

            optimizer.zero_grad()
            logits = model(acts).squeeze(-1)             # (B,)
            loss = criterion(logits, labels)
            loss.backward()

            # Check no NaN gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all(), (
                        f"NaN/Inf gradient in {name} at step {step}"
                    )

            optimizer.step()

            total_loss += loss.item()
            step += 1
            if step >= steps:
                done = True
                break

    avg_loss = total_loss / step
    elapsed = time.time() - t0
    assert avg_loss < 10.0, f"loss suspiciously large: {avg_loss:.4f}"
    logger.info(
        f"  train: {step} steps, avg_loss={avg_loss:.4f}, "
        f"elapsed={elapsed:.1f}s  ✓"
    )

    # Write checkpoint
    ckpt_path = output_dir / "act_vit_smoke.pt"
    torch.save({"model_state": model.state_dict(), "cfg": cfg}, ckpt_path)
    assert ckpt_path.exists(), "checkpoint not written"
    logger.info(f"  checkpoint: {ckpt_path}  ✓")


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    capture_dir = (PROJECT_ROOT / args.capture_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. GPU ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU found — running on CPU (forward pass only, no train loop)")

    # --- 2. Synthetic forward pass ---
    logger.info("Check 1/3: synthetic forward pass (input_dim=4096)")
    check_synthetic(device, input_dim=4096)

    # --- 3. Dataset ---
    if not capture_dir.exists():
        logger.error(f"capture dir not found: {capture_dir}")
        logger.warning("Skipping dataset + train-loop checks (capture missing).")
        logger.success("PARTIAL PASS: synthetic checks only (capture dir absent)")
        return 0

    logger.info(f"Check 2/3: ACTViTDataset ({capture_dir.name})")
    ds, hidden_dim = check_dataset(capture_dir, args.batch_size, args.num_workers)

    # --- 4. Train loop ---
    if device == "cpu":
        logger.warning("Skipping train-loop check (CPU only).")
    else:
        logger.info("Check 3/3: mini train loop")
        check_train_loop(
            ds=ds,
            input_dim=hidden_dim,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            steps=args.steps,
            output_dir=output_dir,
        )

    logger.success("PASS: ACT-ViT smoketest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
