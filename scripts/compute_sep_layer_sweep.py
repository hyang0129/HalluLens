"""Phase 5a: SEP-SE layer sweep.

Fits SEP-SE per layer on 80% of the train subset and reports val AUROC on the
held-out 20% (binary halu labels). Pick the best layer for compute_sep.py.
Llama-3.1-8B uses layer 22 by project convention; sweep Qwen3-8B explicitly.

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
    SAMPLING_DATASETS,
    eval_results_json,
    generation_jsonl,
    model_name,
    se_labels_path,
    subset_index_path,
    zarr_path,
)
from tasks.sampling_baselines.sep import layer_sweep_sep_se

DEFAULT_LAYERS = [16, 18, 20, 22, 24, 26]


def main():
    parser = argparse.ArgumentParser(description="SEP-SE layer sweep.")
    parser.add_argument("--dataset", required=True, choices=SAMPLING_DATASETS)
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument(
        "--layers", nargs="+", type=int, default=DEFAULT_LAYERS, help="Layer indices to sweep"
    )
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gen_train = generation_jsonl(args.dataset, args.model, "train")
    zarr_train = zarr_path(args.dataset, args.model, "train")
    eval_train = eval_results_json(args.dataset, args.model, "train")
    se_train = se_labels_path(args.dataset, args.model, "train")
    subset_path = subset_index_path(args.dataset, args.model)

    for p, label in [
        (gen_train, "generation.jsonl (train)"),
        (zarr_train, "zarr store (train)"),
        (eval_train, "eval_results_for_training.json (train)"),
        (se_train, "se_labels.jsonl (train)"),
        (subset_path, "subset index"),
    ]:
        if not Path(p).exists():
            print(f"ERROR: {label} not found: {p}")
            sys.exit(1)

    with open(subset_path) as f:
        subset = json.load(f)["question_ids"]

    print(
        f"SEP-SE layer sweep | {args.dataset} | {model_name(args.model)}\n"
        f"Layers: {args.layers} | subset_size: {len(subset)} | val_frac: {args.val_frac}"
    )

    results = layer_sweep_sep_se(
        generation_jsonl_train=str(gen_train),
        zarr_path_train=str(zarr_train),
        eval_json_train=str(eval_train),
        se_labels_train_path=str(se_train),
        train_subset_indices=subset,
        layers=args.layers,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    if not results:
        print("ERROR: no layers produced AUROC values.")
        sys.exit(1)

    print("\nLayer sweep results:")
    print(f"{'Layer':>6}  {'Val AUROC':>10}")
    print("-" * 20)
    best_layer = max(results, key=results.get)
    for layer in sorted(results):
        marker = " <-- best" if layer == best_layer else ""
        print(f"{layer:>6}  {results[layer]:>10.4f}{marker}")

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
                "val_frac": args.val_frac,
            },
            f,
            indent=2,
        )
    print(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
