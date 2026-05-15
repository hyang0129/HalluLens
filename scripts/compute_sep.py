"""Phase 5: Fit SEP-SE probes (Kossen et al. 2024).

SEP-SE = Ridge probe predicting length-normalized semantic entropy from a
single greedy-pass hidden state. AUROC is reported against binary halu labels
on the test split. MMLU is skipped (no SE labels).

Layer is picked per-dataset from compute_sep_layer_sweep.py output by default
(matching the Kossen paper's per-task selection); pass --layer to override
with a single fixed layer across all datasets.

Usage:
    # Auto-pick layer per dataset from layer_sweep_{ds}.json (run sweep first)
    python scripts/compute_sep.py --model Qwen/Qwen3-8B

    # Pin a single layer across all datasets
    python scripts/compute_sep.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --layer 22

    # Single dataset
    python scripts/compute_sep.py \\
        --dataset hotpotqa \\
        --model Qwen/Qwen3-8B
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasks.sampling_baselines.paths import (
    SAMPLING_DATASETS,
    model_name,
    se_labels_path,
    searchqa_test_cap_path,
    sep_results_path,
    subset_index_path,
)
from tasks.sampling_baselines.sep import run_sep


def main():
    parser = argparse.ArgumentParser(description="Fit SEP-SE probes.")
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=SAMPLING_DATASETS,
        choices=SAMPLING_DATASETS,
        help="Datasets to process (default: all sampling datasets; MMLU is excluded)",
    )
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Override layer for all datasets. Default: per-dataset best from layer sweep.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing sep_results.json files",
    )
    args = parser.parse_args()

    datasets = args.dataset if isinstance(args.dataset, list) else [args.dataset]

    mode = f"layer={args.layer} (fixed)" if args.layer is not None else "layer=auto (from sweep)"
    print(f"SEP-SE | {model_name(args.model)} | {mode}")

    sweep_dir = Path("output") / "sampling_baselines" / "sep" / model_name(args.model)

    for ds in datasets:
        print(f"\n--- {ds} ---")
        out_path = sep_results_path(ds, args.model)

        if out_path.exists() and not args.force:
            print(f"  Already exists — skipping. Pass --force to rerun: {out_path}")
            continue

        # Resolve layer: --layer override, else best_layer from sweep file.
        if args.layer is not None:
            layer = args.layer
            layer_source = "cli"
        else:
            sweep_path = sweep_dir / f"layer_sweep_{ds}.json"
            if not sweep_path.exists():
                print(
                    f"  SKIP: no --layer and sweep file missing — {sweep_path}\n"
                    f"        Run compute_sep_layer_sweep.py --dataset {ds} --model {args.model} first."
                )
                continue
            with open(sweep_path) as f:
                sweep_data = json.load(f)
            layer = int(sweep_data["best_layer"])
            best_auroc = sweep_data["auroc_by_layer"].get(str(layer))
            print(f"  layer={layer} (val AUROC={best_auroc:.4f}, from {sweep_path.name})")
            layer_source = str(sweep_path)

        subset_path = subset_index_path(ds, args.model)
        if not subset_path.exists():
            print(f"  SKIP: subset index missing — {subset_path}")
            continue
        with open(subset_path) as f:
            subset = json.load(f)["question_ids"]

        se_train = se_labels_path(ds, args.model, "train")
        if not se_train.exists():
            print(f"  SKIP: SE labels missing — {se_train}")
            continue

        # Apply searchqa test cap to match other baselines' evaluation set.
        test_subset = None
        if ds == "searchqa":
            cap_path = searchqa_test_cap_path(args.model)
            if cap_path.exists():
                with open(cap_path) as f:
                    test_subset = json.load(f)["question_ids"]
                print(f"  test cap: {len(test_subset)} rows from {cap_path.name}")
            else:
                print(f"  WARN: searchqa cap not found ({cap_path}) — using full test split")

        try:
            result = run_sep(
                dataset=ds,
                model_id=args.model,
                layer_idx=layer,
                train_subset_indices=subset,
                se_labels_train_path=str(se_train),
                output_path=str(out_path),
                test_subset_indices=test_subset,
            )
            # Annotate where the layer choice came from.
            result["layer_source"] = layer_source
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"  SKIP: {e}")

    print("\nSEP done.")


if __name__ == "__main__":
    main()
