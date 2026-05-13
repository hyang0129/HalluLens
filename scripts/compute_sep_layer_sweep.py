"""Phase 5a: SEP layer sweep for Qwen3-8B.

Fits SEP-binary for each candidate layer on a single dataset's val split.
Prints an AUROC table; the best layer is used in compute_sep.py.
Llama uses layer 22 (matching existing linear probe baseline) — no sweep needed.

Usage:
    python scripts/compute_sep_layer_sweep.py \\
        --dataset hotpotqa \\
        --model Qwen/Qwen3-8B \\
        --layers 16 18 20 22 24 26

Results are printed to stdout and saved to:
    output/sampling_baselines/sep/{model_name}/layer_sweep_{dataset}.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.sampling_baselines.paths import (
    DATASETS,
    eval_results_json,
    generation_jsonl,
    model_name,
    zarr_path,
)
from tasks.sampling_baselines.sep import layer_sweep_sep_binary

DEFAULT_LAYERS = [16, 18, 20, 22, 24, 26]


def main():
    parser = argparse.ArgumentParser(description="SEP layer sweep.")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=DEFAULT_LAYERS,
        help="Layer indices to sweep",
    )
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gen = generation_jsonl(args.dataset, args.model, "test")
    zarr = zarr_path(args.dataset, args.model, "test")
    eval_j = eval_results_json(args.dataset, args.model, "test")

    for p, label in [(gen, "generation.jsonl"), (zarr, "zarr store"), (eval_j, "eval_results_for_training.json")]:
        if not Path(p).exists():
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    print(
        f"SEP layer sweep | {args.dataset} | {model_name(args.model)}\n"
        f"Layers: {args.layers}"
    )

    results = layer_sweep_sep_binary(
        generation_jsonl=str(gen),
        zarr_path=str(zarr),
        eval_json=str(eval_j),
        layers=args.layers,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    print("\nLayer sweep results:")
    print(f"{'Layer':>6}  {'Val AUROC':>10}")
    print("-" * 20)
    for layer in sorted(results):
        marker = " <-- best" if layer == max(results, key=results.get) else ""
        print(f"{layer:>6}  {results[layer]:>10.4f}{marker}")

    best_layer = max(results, key=results.get)
    print(f"\nBest layer: {best_layer} (AUROC={results[best_layer]:.4f})")
    print(f"Pass --layer {best_layer} to compute_sep.py for {model_name(args.model)}")

    out_path = (
        Path("output")
        / "sampling_baselines"
        / "sep"
        / model_name(args.model)
        / f"layer_sweep_{args.dataset}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "model": args.model,
                "layers_swept": args.layers,
                "auroc_by_layer": results,
                "best_layer": best_layer,
                "seed": args.seed,
            },
            f,
            indent=2,
        )
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
