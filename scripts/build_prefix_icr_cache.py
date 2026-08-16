"""Build response-prefix-specific ICR score caches from an existing capture.

ICR is an aggregate over response query tokens, so a full-response
``icr_scores.npy`` cannot be truncated after the fact. This script recomputes
the score at every requested response prefix directly from the stored
attention and activation memmaps, using the batched GPU kernel.

Example:
    python scripts/build_prefix_icr_cache.py \
        shared/icr_capture/hotpotqa_train \
        --output-dir shared/prefix149_icr/hotpotqa_train \
        --prefixes 1,4,8,16,32,48,64 --device cuda --batch-size 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from activation_research.icr_score_gpu import (  # noqa: E402
    compute_icr_per_layer_batched_gpu,
)


def _parse_prefixes(raw: str, r_max: int) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or values[0] < 1:
        raise ValueError("prefixes must contain positive integers")
    if values[-1] > r_max:
        raise ValueError(f"prefix {values[-1]} exceeds capture r_max={r_max}")
    return values


def _atomic_save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.save(fh, array)
    tmp.replace(path)


def build_prefix_caches(
    capture_dir: Path,
    output_dir: Path,
    *,
    prefixes: list[int],
    batch_size: int,
    device: torch.device,
    top_p: float = 0.1,
    limit: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    cfg = json.loads((capture_dir / "config.json").read_text())
    n_alloc = int(cfg["n_samples"])
    num_layers = int(cfg["num_layers"])
    hidden_dim = int(cfg["hidden_dim"])
    r_max = int(cfg["r_max"])
    max_response_len = int(cfg["max_response_len"])

    for prefix in prefixes:
        if prefix < 1 or prefix > r_max:
            raise ValueError(f"prefix must be in [1, {r_max}], got {prefix}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    meta = [
        json.loads(line)
        for line in (capture_dir / "meta.jsonl").read_text().splitlines()
        if line.strip()
    ]
    sample_indices = sorted({int(row["sample_index"]) for row in meta})
    if limit is not None:
        sample_indices = sample_indices[:limit]
    if not sample_indices:
        raise ValueError(f"no committed samples found in {capture_dir / 'meta.jsonl'}")

    # A limited smoke cache intentionally exposes only the computed prefix of
    # sample indices. Full caches use n_alloc so ICRDataset can index directly
    # by the capture's sample_index.
    score_rows = n_alloc if limit is None else max(sample_indices) + 1
    output_paths = {k: output_dir / f"icr_scores_k{k}.npy" for k in prefixes}
    pending = [k for k in prefixes if overwrite or not output_paths[k].exists()]
    if not pending:
        return [output_paths[k] for k in prefixes]

    response_attn = np.memmap(
        capture_dir / "response_attention.npy",
        dtype=np.float16,
        mode="r",
        shape=(n_alloc, num_layers, r_max, r_max),
    )
    response_act = np.memmap(
        capture_dir / "response_activations.npy",
        dtype=np.float16,
        mode="r",
        shape=(n_alloc, num_layers + 1, max_response_len, hidden_dim),
    )
    response_lens = np.memmap(
        capture_dir / "response_len.npy",
        dtype=np.int32,
        mode="r",
        shape=(n_alloc,),
    )
    prompt_lens = np.memmap(
        capture_dir / "prompt_len.npy",
        dtype=np.int32,
        mode="r",
        shape=(n_alloc,),
    )

    scores = {
        k: np.zeros((score_rows, num_layers), dtype=np.float32) for k in pending
    }
    started = time.perf_counter()
    for offset in range(0, len(sample_indices), batch_size):
        idx = np.asarray(sample_indices[offset : offset + batch_size], dtype=np.int64)

        # Materialize memmap slices before transfer; the first r_max response
        # positions are the complete window available to the ICR kernel.
        attn = torch.from_numpy(np.asarray(response_attn[idx], dtype=np.float16)).to(
            device, non_blocking=True
        )
        acts = torch.from_numpy(
            np.asarray(response_act[idx, :, :r_max, :], dtype=np.float16)
        ).to(device, non_blocking=True)
        h_in = acts[:, :-1].float()
        delta_h = acts[:, 1:].float() - h_in
        raw_rlens = torch.from_numpy(
            np.asarray(response_lens[idx], dtype=np.int64)
        ).to(device)
        plens = torch.from_numpy(
            np.asarray(prompt_lens[idx], dtype=np.int64)
        ).to(device)

        for prefix in pending:
            visible_lens = raw_rlens.clamp(max=min(prefix, r_max))
            batch_scores = compute_icr_per_layer_batched_gpu(
                attn,
                h_in,
                delta_h,
                visible_lens,
                top_p=top_p,
                prompt_lens=plens,
            )
            scores[prefix][idx] = batch_scores.cpu().numpy()

        done = min(offset + batch_size, len(sample_indices))
        if done == len(sample_indices) or done % max(100, batch_size) == 0:
            elapsed = time.perf_counter() - started
            rate = done / max(elapsed, 1e-9)
            print(
                f"[{done}/{len(sample_indices)}] {rate:.2f} samples/s "
                f"prefixes={pending}",
                file=sys.stderr,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    for prefix in pending:
        _atomic_save(output_paths[prefix], scores[prefix])

    manifest = {
        "capture_dir": str(capture_dir.resolve()),
        "prefixes": prefixes,
        "computed_prefixes": pending,
        "top_p": top_p,
        "num_committed_samples": len(sample_indices),
        "score_rows": score_rows,
        "num_layers": num_layers,
        "r_max": r_max,
        "device": str(device),
        "limited": limit is not None,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return [output_paths[k] for k in prefixes]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefixes", default="1,4,8,16,32,48,64")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top-p", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = json.loads((args.capture_dir / "config.json").read_text())
    prefixes = _parse_prefixes(args.prefixes, int(cfg["r_max"]))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    paths = build_prefix_caches(
        args.capture_dir,
        args.output_dir,
        prefixes=prefixes,
        batch_size=args.batch_size,
        device=torch.device(args.device),
        top_p=args.top_p,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
