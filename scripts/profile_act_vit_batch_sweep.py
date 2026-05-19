"""Profile ACT-ViT training throughput at a sweep of batch sizes.

Loads the dataset + model once, then for each batch size:
  - constructs a DataLoader with num_workers=4, pin_memory=True, persistent_workers=True
  - runs N_WARMUP warmup steps
  - times N_MEASURE steps (fwd + bwd + opt.step), reporting ms/batch, samples/sec
  - records peak VRAM (torch.cuda.max_memory_allocated / reserved)
  - polls nvidia-smi via subprocess for steady-state GPU utilization

Designed for the post-fp16-fix path (act_vit_dataset returns fp16 tensors;
ACTViT.forward casts to fp32 on GPU). Mirrors run_act_vit's training step.

Usage::
    python scripts/profile_act_vit_batch_sweep.py \\
        --capture-dir shared/icr_capture/hotpotqa_train_Llama-3.1-8B-Instruct_0-50000 \\
        --batch-sizes 64,128,256 \\
        --n-warmup 5 --n-measure 30
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from activation_research.act_vit import ACTViT, ACTViTConfig
from activation_research.act_vit_dataset import ACTViTDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--capture-dir",
        default="shared/icr_capture/hotpotqa_train_Llama-3.1-8B-Instruct_0-50000",
    )
    p.add_argument(
        "--batch-sizes",
        default="64,128,256",
        help="Comma-separated list of batch sizes to sweep.",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-measure", type=int, default=30)
    p.add_argument(
        "--n-samples",
        type=int,
        default=4096,
        help="How many dataset indices to expose (>= max batch_size * (n_warmup+n_measure)).",
    )
    return p.parse_args()


def gpu_util_pct() -> int | None:
    """Query nvidia-smi for current GPU 0 utilization. Returns None on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-i", "0"],
            timeout=2,
        )
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return None


def fmt_bytes(n: int) -> str:
    return f"{n / 1024**3:.2f} GiB"


def profile_batch_size(
    capture_dir: Path,
    bs: int,
    indices: list[int],
    num_workers: int,
    n_warmup: int,
    n_measure: int,
    input_dim: int,
) -> dict:
    """Run warmup + timed loop at one batch size; return stats dict."""
    print(f"\n=== batch_size={bs} ===", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    ds = ACTViTDataset(capture_dir, indices)
    loader = DataLoader(
        ds,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True,
    )

    cfg = ACTViTConfig(
        input_dim=input_dim,
        L_p=8, N_p=100, patch_h=2, patch_w=10,
        d_adapter=256, d_model=256,
        num_heads=8, depth=4, mlp_ratio=4.0, dropout=0.1,
    )
    model = ACTViT(cfg).cuda()
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    it = iter(loader)
    util_samples: list[int] = []

    # warmup
    print(f"  warmup x{n_warmup}...", flush=True)
    for _ in range(n_warmup):
        batch = next(it)
        acts = batch["activations"].cuda(non_blocking=True)
        labels = batch["label"].float().cuda(non_blocking=True)
        opt.zero_grad()
        logits = model(acts).squeeze(-1)
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()

    # reset peak after warmup so we measure steady-state
    torch.cuda.reset_peak_memory_stats()

    # measured
    print(f"  measure x{n_measure}...", flush=True)
    step_times: list[float] = []
    torch.cuda.synchronize()
    for i in range(n_measure):
        t0 = time.time()
        batch = next(it)
        acts = batch["activations"].cuda(non_blocking=True)
        labels = batch["label"].float().cuda(non_blocking=True)
        opt.zero_grad()
        logits = model(acts).squeeze(-1)
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        step_times.append(time.time() - t0)
        # Sample GPU util mid-stride (skip first/last)
        if 5 <= i < n_measure - 2:
            u = gpu_util_pct()
            if u is not None:
                util_samples.append(u)

    ms_per_step = sum(step_times) / len(step_times) * 1000
    samples_per_sec = bs / (ms_per_step / 1000)
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    util_mean = sum(util_samples) / len(util_samples) if util_samples else None

    stats = dict(
        batch_size=bs,
        ms_per_step=ms_per_step,
        samples_per_sec=samples_per_sec,
        peak_alloc_bytes=peak_alloc,
        peak_reserved_bytes=peak_reserved,
        gpu_util_mean_pct=util_mean,
        n_samples_util=len(util_samples),
    )
    print(
        f"  bs={bs}: {ms_per_step:7.1f} ms/step  "
        f"{samples_per_sec:7.1f} samp/s  "
        f"VRAM_alloc={fmt_bytes(peak_alloc)} reserved={fmt_bytes(peak_reserved)}  "
        f"GPU_util={util_mean if util_mean is None else f'{util_mean:.0f}%'}",
        flush=True,
    )

    # cleanup before next iter
    del model, opt, loader, it, ds
    torch.cuda.empty_cache()
    return stats


def main() -> int:
    args = parse_args()
    capture_dir = (PROJECT_ROOT / args.capture_dir).resolve()
    assert capture_dir.exists(), f"capture dir not found: {capture_dir}"

    with (capture_dir / "config.json").open() as fh:
        cap_cfg = json.load(fh)
    n_total = cap_cfg["n_samples"]
    hidden_dim = cap_cfg["hidden_dim"]
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"capture: n_samples={n_total}, hidden_dim={hidden_dim}, "
          f"layers={cap_cfg['num_layers']}, R={cap_cfg['max_response_len']}")

    n_expose = min(n_total, args.n_samples)
    indices = list(range(n_expose))

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    print(f"sweep: batch_sizes={batch_sizes}, n_warmup={args.n_warmup}, "
          f"n_measure={args.n_measure}, num_workers={args.num_workers}")

    results = []
    for bs in batch_sizes:
        if bs * (args.n_warmup + args.n_measure) > n_expose:
            print(f"\nSKIP bs={bs}: needs {bs*(args.n_warmup+args.n_measure)} samples, have {n_expose}")
            continue
        try:
            stats = profile_batch_size(
                capture_dir=capture_dir,
                bs=bs,
                indices=indices,
                num_workers=args.num_workers,
                n_warmup=args.n_warmup,
                n_measure=args.n_measure,
                input_dim=hidden_dim,
            )
            results.append(stats)
        except torch.cuda.OutOfMemoryError as e:
            print(f"\nOOM at bs={bs}: {e}")
            torch.cuda.empty_cache()
            results.append({"batch_size": bs, "oom": True, "error": str(e)})
            # Don't try larger batch sizes after OOM
            break

    # Summary table
    print("\n" + "=" * 78)
    print(f"{'bs':>6}  {'ms/step':>9}  {'samp/s':>9}  {'VRAM alloc':>12}  {'VRAM resv':>12}  {'util':>6}")
    print("-" * 78)
    for r in results:
        if r.get("oom"):
            print(f"{r['batch_size']:>6}  OOM")
            continue
        util = "—" if r["gpu_util_mean_pct"] is None else f"{r['gpu_util_mean_pct']:.0f}%"
        print(
            f"{r['batch_size']:>6}  "
            f"{r['ms_per_step']:>9.1f}  "
            f"{r['samples_per_sec']:>9.1f}  "
            f"{fmt_bytes(r['peak_alloc_bytes']):>12}  "
            f"{fmt_bytes(r['peak_reserved_bytes']):>12}  "
            f"{util:>6}"
        )
    print("=" * 78)

    # Save JSON for downstream
    out_path = PROJECT_ROOT / "runs" / "profile_act_vit_batch_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nresults saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
