"""Dump per-layer contrastive embeddings for train/val/test, once.

Loads a trained contrastive run (config.json + artifacts/final_weights.pt) and
runs the encoder over every layer in `relevant_layers` for the train, val, and
test splits. Saves one .npz per split with arrays:

    z_per_layer    : (N, L, D)  float16
    halu           : (N,)       int8
    prompt_hash    : (N,)       <U64
    layer_indices  : (L,)       int32
    split_name     : ()         <U16

Plus a `meta.json` describing the run, dataset paths, and split parameters.

Convention (matches the actual experiment, which uses split_strategy="two_way"):
    train.npz : train zarr, internal "train" split (80%). Encoder trained on these.
    val.npz   : train zarr, internal "test" split (20%). Encoder never saw these —
                under two_way this 20% partition is unused by run_experiment and
                serves as a clean held-out val from the encoder's perspective.
    test.npz  : test zarr, all rows (split_strategy="none"). True held-out test.

Usage:
    python scripts/dump_contrastive_embeddings.py \\
        --run-dir runs/baseline_comparison_nq_qwen3/nq_qwen3/contrastive_logprob_recon/seed_0 \\
        --output-dir shared/knn_eval_dumps/qwen3_nq_seed0
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Ensure repo root is importable when invoked as `python scripts/...`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from activation_logging.activation_parser import ActivationParser  # noqa: E402
from activation_research.evaluation import _call_model  # noqa: E402
from utils.progress import tqdm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("dump_contrastive_embeddings")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_layers(s: str) -> List[int]:
    """Parse '14-29' or '22,26' or '14-20,25' into a list of ints (stable-unique)."""
    s = s.strip()
    if not s:
        raise ValueError("Empty layer spec")
    out: List[int] = []
    seen = set()
    for part in (p.strip() for p in s.split(",") if p.strip()):
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"Invalid range '{part}'")
            xs = range(lo, hi + 1)
        else:
            xs = [int(part)]
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
    return out


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _build_model(model_class: str, input_dim: int, model_params: dict) -> torch.nn.Module:
    """Instantiate one of the supported encoder classes by string name.

    Extend this dispatch as new model classes are introduced for contrastive runs.
    """
    name = model_class.strip().lower()
    if name == "progressive_compressor":
        from activation_research.model import ProgressiveCompressor

        return ProgressiveCompressor(
            input_dim=input_dim,
            final_dim=model_params.get("final_dim", 512),
            input_dropout=model_params.get("input_dropout", 0.3),
        )
    if name == "logprob_recon_progressive_compressor":
        from activation_research.model import LogprobReconProgressiveCompressor

        return LogprobReconProgressiveCompressor(
            input_dim=input_dim,
            final_dim=model_params.get("final_dim", 512),
            input_dropout=model_params.get("input_dropout", 0.3),
            recon_seq_len=model_params.get("recon_seq_len", 64),
            recon_hidden_dim=model_params.get("recon_hidden_dim", 256),
            recon_lambda=model_params.get("recon_lambda", 1.0),
            logprob_var_threshold=model_params.get("logprob_var_threshold", 1e-4),
            block_dims=model_params.get("block_dims"),
        )
    if name == "logprob_recon_adapter_vit_compressor":
        from activation_research.model import LogprobReconAdapterViTCompressor

        return LogprobReconAdapterViTCompressor(
            input_dim=input_dim,
            d_model=model_params.get("d_model", 256),
            depth=model_params.get("depth", 4),
            num_heads=model_params.get("num_heads", 8),
            mlp_ratio=model_params.get("mlp_ratio", 4),
            dropout=model_params.get("dropout", 0.1),
            input_dropout=model_params.get("input_dropout", 0.2),
            pool=model_params.get("pool", "mean"),
            recon_seq_len=model_params.get("recon_seq_len", 64),
            recon_hidden_dim=model_params.get("recon_hidden_dim", 256),
            recon_lambda=model_params.get("recon_lambda", 1.0),
            logprob_var_threshold=model_params.get("logprob_var_threshold", 1e-4),
        )
    raise ValueError(
        f"Unsupported model_class='{model_class}'. "
        "Add a branch in _build_model() to support it."
    )


def _load_weights(model: torch.nn.Module, ckpt_path: Path) -> None:
    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing keys when loading weights: %s", missing[:8])
    if unexpected:
        logger.warning("Unexpected keys when loading weights: %s", unexpected[:8])


def _build_parser(
    inference_json: str,
    eval_json: str,
    activations_path: str,
    *,
    split_strategy: str,
    split_seed: int,
) -> ActivationParser:
    return ActivationParser(
        inference_json=inference_json,
        eval_json=eval_json,
        activations_path=activations_path,
        logger_type="zarr",
        random_seed=split_seed,
        split_strategy=split_strategy,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Per-layer inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_split(
    *,
    model: torch.nn.Module,
    parser: ActivationParser,
    split: str,
    layers: List[int],
    pad_length: int,
    device: str,
    batch_size: int,
    num_workers: int,
    preload: bool,
    check_ram: bool,
) -> Dict[str, np.ndarray]:
    """Run the encoder over every layer for one split and return stacked arrays.

    Output keys:
        z_per_layer:  (N, L, D)  float16
        halu:         (N,)       int8
        prompt_hash:  (N,)       <U64
    """
    from torch.utils.data import DataLoader

    ds = parser.get_dataset(
        split,
        relevant_layers=layers,
        num_views=2,  # required >=2; we ignore views and use all_activations
        pad_length=pad_length,
        return_all_activations=True,
        preload=preload,
        check_ram=check_ram,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.startswith("cuda")),
        persistent_workers=False,
    )

    n = len(ds)
    L = len(layers)
    final_dim: Optional[int] = None
    z_buf: Optional[np.ndarray] = None
    halu_buf = np.empty(n, dtype=np.int8)
    hash_buf: List[str] = [""] * n

    cursor = 0
    df = parser.df.set_index("prompt_hash")["halu"]
    halu_lookup = df.to_dict()

    model.eval()

    for batch in tqdm(loader, desc=f"encode[{split}]"):
        all_acts = batch["all_activations"]  # list of len L, each (B, 1, T, D) or (B, T, D)
        if len(all_acts) != L:
            raise RuntimeError(
                f"Expected len(all_activations)={L} layers, got {len(all_acts)}"
            )
        # Resolve batch size from the first layer tensor
        first = all_acts[0]
        if first.dim() == 4 and first.shape[1] == 1:
            first = first.squeeze(1)
        bsz = first.shape[0]

        z_layers: List[torch.Tensor] = []
        for layer_pos, layer_acts in enumerate(all_acts):
            x = layer_acts
            if x.dim() == 4 and x.shape[1] == 1:
                x = x.squeeze(1)  # (B, T, D)
            x = x.to(device, non_blocking=True)
            layer_idx_tensor = torch.full(
                (x.shape[0],), int(layers[layer_pos]), dtype=torch.long, device=x.device
            )
            z = _call_model(model, x, layer_idx=layer_idx_tensor)  # (B, final_dim)
            z_layers.append(z.detach().to("cpu"))

        z_stack = torch.stack(z_layers, dim=1)  # (B, L, final_dim)
        if final_dim is None:
            final_dim = z_stack.shape[-1]
            z_buf = np.empty((n, L, final_dim), dtype=np.float16)
        z_buf[cursor : cursor + bsz] = z_stack.to(torch.float16).numpy()

        hashkeys = batch["hashkey"]
        # hashkey may be a list[str] (default collate) — handle both list and tensor cases.
        if isinstance(hashkeys, torch.Tensor):
            hashkeys = [str(h) for h in hashkeys.tolist()]
        for i, h in enumerate(hashkeys):
            hash_buf[cursor + i] = str(h)
            halu_buf[cursor + i] = int(halu_lookup[str(h)])

        cursor += bsz

    if cursor != n:
        raise RuntimeError(f"Expected to encode {n} samples, got {cursor}")

    assert z_buf is not None
    return {
        "z_per_layer": z_buf,
        "halu": halu_buf,
        "prompt_hash": np.array(hash_buf, dtype=np.str_),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        required=True,
        help="Trained run directory (must contain config.json and artifacts/<checkpoint>).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for {train,val,test}.npz and meta.json.",
    )
    p.add_argument(
        "--checkpoint-name",
        default="final_weights.pt",
        help="File inside <run-dir>/artifacts/ to load. Default: final_weights.pt.",
    )
    p.add_argument(
        "--layers",
        default=None,
        help="Layer spec to dump (e.g. '14-29' or '22,26'). "
        "Default: use the run's relevant_layers.",
    )
    p.add_argument(
        "--split-strategy",
        default="two_way",
        choices=["two_way", "three_way"],
        help="Split strategy for the train zarr. Default 'two_way' matches the "
        "actual training experiment (run_experiment.py): under two_way, the "
        "train zarr's 'test' partition (20%%) was unused during training and "
        "is emitted here as our clean 'val' (encoder never saw it). 'three_way' "
        "is supported but its 'val' partition would have been inside the "
        "encoder's training set — not recommended for this investigation.",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="random_seed for the train zarr split. Default: read split_seed from run config.",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument(
        "--no-preload",
        action="store_true",
        help="Disable RAM preload of the dataset (default: preload on).",
    )
    p.add_argument(
        "--no-check-ram",
        action="store_true",
        help="Skip the preload RAM check.",
    )
    p.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated subset of {train,val,test} to dump. Default: all three.",
    )
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "config.json").exists():
        raise FileNotFoundError(f"{run_dir / 'config.json'} not found")

    with open(run_dir / "config.json") as f:
        run_cfg = json.load(f)

    method_cfg = run_cfg["method"]
    dataset_cfg = run_cfg["dataset"]
    data_cfg = method_cfg["data"]

    train_dataset_cfg = dataset_cfg["train"]
    test_dataset_cfg = dataset_cfg["test"]
    input_dim = int(dataset_cfg["input_dim"])

    # Resolve layers
    if args.layers is not None:
        layers = _parse_layers(args.layers)
    else:
        relevant = data_cfg["relevant_layers"]
        if isinstance(relevant, list):
            layers = list(map(int, relevant))
        else:
            layers = _parse_layers(str(relevant))
    pad_length = int(data_cfg.get("pad_length", 63))

    # Resolve split seed
    split_seed = (
        args.split_seed if args.split_seed is not None
        else int(run_cfg.get("split_seed", 42))
    )

    # Resolve splits to emit. Names below are the *output* labels.
    # The mapping from output label -> (parser, internal split) is:
    #   train -> train_ap, "train"
    #   val   -> train_ap, "val" (three_way) | "test" (two_way; encoder never saw these)
    #   test  -> test_ap,  "test"
    splits_requested = [s.strip() for s in args.splits.split(",") if s.strip()]
    for s in splits_requested:
        if s not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split '{s}'")

    # Build & load model
    model_class = method_cfg["model_class"]
    model_params = method_cfg.get("model_params", {})
    model = _build_model(model_class, input_dim=input_dim, model_params=model_params)
    ckpt_path = run_dir / "artifacts" / args.checkpoint_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    _load_weights(model, ckpt_path)
    device = _resolve_device(args.device)
    model = model.to(device)

    logger.info("Loaded %s from %s", model_class, ckpt_path)
    logger.info("Layers to dump: %s", layers)
    logger.info("split_strategy=%s split_seed=%d", args.split_strategy, split_seed)

    # Build parsers
    train_ap = _build_parser(
        train_dataset_cfg["inference_json"],
        train_dataset_cfg["eval_json"],
        train_dataset_cfg["activations_path"],
        split_strategy=args.split_strategy,
        split_seed=split_seed,
    )
    test_ap = _build_parser(
        test_dataset_cfg["inference_json"],
        test_dataset_cfg["eval_json"],
        test_dataset_cfg["activations_path"],
        split_strategy="none",
        split_seed=split_seed,
    )

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    preload = not args.no_preload
    check_ram = not args.no_check_ram

    layer_indices_arr = np.array(layers, dtype=np.int32)

    # Map output split label -> (parser, internal split name on that parser).
    if args.split_strategy == "three_way":
        split_routing = {
            "train": (train_ap, "train"),
            "val": (train_ap, "val"),
            "test": (test_ap, "test"),
        }
    else:  # two_way
        split_routing = {
            "train": (train_ap, "train"),
            "val": (train_ap, "test"),  # 20% partition unused by the actual run
            "test": (test_ap, "test"),
        }

    summary: Dict[str, Dict[str, int]] = {}
    for split in splits_requested:
        parser, internal_split = split_routing[split]
        result = _encode_split(
            model=model,
            parser=parser,
            split=internal_split,
            layers=layers,
            pad_length=pad_length,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            preload=preload,
            check_ram=check_ram,
        )
        npz_path = out_dir / f"{split}.npz"
        np.savez(
            npz_path,
            z_per_layer=result["z_per_layer"],
            halu=result["halu"],
            prompt_hash=result["prompt_hash"],
            layer_indices=layer_indices_arr,
            split_name=np.array(split, dtype=np.str_),
        )
        n = int(result["z_per_layer"].shape[0])
        n_pos = int(result["halu"].sum())
        summary[split] = {
            "n": n,
            "n_halu": n_pos,
            "n_clean": n - n_pos,
            "shape": list(result["z_per_layer"].shape),
        }
        logger.info(
            "Wrote %s  N=%d  halu=%d  shape=%s",
            npz_path, n, n_pos, list(result["z_per_layer"].shape),
        )

    meta = {
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "model_class": model_class,
        "model_params": model_params,
        "input_dim": input_dim,
        "pad_length": pad_length,
        "layers": layers,
        "split_strategy_train_zarr": args.split_strategy,
        "split_seed": split_seed,
        "train_dataset": train_dataset_cfg,
        "test_dataset": test_dataset_cfg,
        "splits": summary,
        "split_routing": {
            label: {"zarr": ("test_zarr" if parser is test_ap else "train_zarr"),
                    "internal_split": internal}
            for label, (parser, internal) in split_routing.items()
        },
        "device": device,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "command": " ".join(sys.argv),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info("Wrote %s", out_dir / "meta.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
