#!/usr/bin/env python3
"""Audit memmap cache status for the canonical baseline experiment params.

For each (model, dataset) pair, both the train and test zarr stores need a
memmap cache built with the parameters the learned baseline methods use
(layers 14-29, pad_length=63, include_logprobs=True, top_k=20). Building one
cache can take ~12h, but the fingerprint is seed-agnostic so all training
seeds reuse it.

This audit scans every cache manifest in each zarr's _memmap_cache/ directory
and reports HIT iff at least one matches the canonical params AND has the
activations.npy payload (not just an empty/abandoned hash dir). It does NOT
instantiate ActivationParser, so it's instant even on 90K-sample stores.

Usage:
    python scripts/audit_memmap_cache.py
    python scripts/audit_memmap_cache.py --model Qwen/Qwen3-8B
    python scripts/audit_memmap_cache.py --datasets hotpotqa,mmlu
    python scripts/audit_memmap_cache.py --json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

DATASETS = ["hotpotqa", "mmlu", "nq", "popqa", "sciq", "searchqa"]

# Llama uses default suffix; Qwen uses _qwen3 suffix in dataset config filenames.
MODELS = {
    "Llama-3.1-8B-Instruct": "",
    "Qwen3-8B": "_qwen3",
}

# Canonical params shared by both learned methods (contrastive_logprob_recon,
# linear_probe) in the baseline_comparison_* experiment configs.
CANONICAL_RELEVANT_LAYERS = list(range(14, 30))
CANONICAL_PAD_LENGTH = 63
CANONICAL_INCLUDE_LOGPROBS = True
CANONICAL_TOP_K = 20


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (ROOT / pp).resolve()


def find_canonical_cache(
    zarr_path: Path,
    expected_split_strategy: str,
) -> dict:
    """Search a zarr's _memmap_cache/ for a manifest matching canonical params.

    Returns dict with status ∈ {hit, payload_missing, params_only_no_match,
    no_cache_dir, no_zarr} and details about what was found.
    """
    if not zarr_path.exists():
        return {"status": "no_zarr", "zarr_path": str(zarr_path)}

    cache_root = zarr_path / "_memmap_cache"
    if not cache_root.exists():
        return {"status": "no_cache_dir", "cache_root": str(cache_root)}

    matches = []
    other_manifests = []
    for sub in sorted(cache_root.iterdir()):
        if not sub.is_dir():
            continue
        manifest_path = sub / "manifest.json"
        if not manifest_path.exists():
            other_manifests.append({"dir": str(sub), "reason": "no manifest.json"})
            continue
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            other_manifests.append({"dir": str(sub), "reason": f"bad manifest: {e}"})
            continue

        layers_match = sorted(m.get("relevant_layers", [])) == sorted(CANONICAL_RELEVANT_LAYERS)
        pad_match = m.get("pad_length") == CANONICAL_PAD_LENGTH
        lp_match = bool(m.get("include_logprobs", False)) == CANONICAL_INCLUDE_LOGPROBS
        top_k_match = m.get("response_logprobs_top_k", 20) == CANONICAL_TOP_K
        # split_strategy in manifest defaults to "two_way" if missing in older caches.
        ss_match = m.get("split_strategy", "two_way") == expected_split_strategy

        all_match = layers_match and pad_match and lp_match and top_k_match and ss_match
        info = {
            "dir": str(sub),
            "fingerprint": m.get("fingerprint"),
            "n_total": m.get("n_total"),
            "n_train": m.get("n_train"),
            "n_test": m.get("n_test"),
            "zarr_sample_count": m.get("zarr_sample_count"),
            "include_logprobs": m.get("include_logprobs"),
            "split_strategy": m.get("split_strategy", "two_way"),
            "pad_length": m.get("pad_length"),
            "has_activations_npy": (sub / "activations.npy").exists(),
            "match": all_match,
        }
        if all_match:
            matches.append(info)
        else:
            other_manifests.append(info)

    if matches:
        # Prefer one with activations.npy payload.
        with_payload = [m for m in matches if m["has_activations_npy"]]
        if with_payload:
            return {
                "status": "hit",
                "match": with_payload[0],
                "other_manifests": other_manifests,
                "extra_matches": with_payload[1:],
            }
        return {
            "status": "payload_missing",
            "match": matches[0],
            "other_manifests": other_manifests,
        }

    return {
        "status": "params_only_no_match",
        "other_manifests": other_manifests,
    }


def audit_dataset(model_name: str, suffix: str, dataset: str) -> dict:
    cfg_path = ROOT / "configs" / "datasets" / f"{dataset}{suffix}.json"
    if not cfg_path.exists():
        return {"status": "no_config", "config_path": str(cfg_path)}
    with open(cfg_path) as f:
        dataset_cfg = json.load(f)
    if "train" not in dataset_cfg or "test" not in dataset_cfg:
        return {"status": "legacy_config", "config_path": str(cfg_path)}

    train_zarr = _resolve(dataset_cfg["train"]["activations_path"])
    test_zarr = _resolve(dataset_cfg["test"]["activations_path"])

    return {
        "config_path": str(cfg_path),
        "train": {"zarr": str(train_zarr),
                  **find_canonical_cache(train_zarr, expected_split_strategy="two_way")},
        "test": {"zarr": str(test_zarr),
                 **find_canonical_cache(test_zarr, expected_split_strategy="none")},
    }


def fmt_one(info: dict) -> str:
    s = info["status"]
    if s == "hit":
        m = info["match"]
        return (f"HIT  fp={m['fingerprint']}  n_total={m['n_total']:,} "
                f"zarr_n={m['zarr_sample_count']:,}")
    if s == "payload_missing":
        m = info["match"]
        return (f"MISS (manifest only, no activations.npy)  fp={m['fingerprint']} "
                f"-> {m['dir']}")
    if s == "params_only_no_match":
        n = len(info.get("other_manifests", []))
        if n == 0:
            return "MISS (cache dir is empty)"
        return f"MISS (no cache matches canonical params; {n} other manifest(s) present)"
    if s == "no_cache_dir":
        return f"MISS (no _memmap_cache/ subdir)"
    if s == "no_zarr":
        return f"NO_ZARR  {info['zarr_path']}"
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--datasets", default=None,
                   help="Comma-separated subset (default: all 6)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--show-others", action="store_true",
                   help="List other (non-matching) manifests too")
    args = p.parse_args()

    if args.model is None:
        models_to_audit = list(MODELS.items())
    else:
        short = args.model.split("/")[-1]
        if short not in MODELS:
            print(f"Unknown model: {short}. Known: {list(MODELS)}", file=sys.stderr)
            sys.exit(2)
        models_to_audit = [(short, MODELS[short])]

    datasets = args.datasets.split(",") if args.datasets else DATASETS

    report: dict = {}
    for model_name, suffix in models_to_audit:
        report[model_name] = {ds: audit_dataset(model_name, suffix, ds)
                              for ds in datasets}

    if args.json:
        print(json.dumps(report, indent=2))
        return

    n_total = n_hit = n_miss = n_other = 0
    for model_name, per_ds in report.items():
        print(f"\n=== {model_name} ===")
        print(f"  canonical params: layers={CANONICAL_RELEVANT_LAYERS[0]}-"
              f"{CANONICAL_RELEVANT_LAYERS[-1]}, pad={CANONICAL_PAD_LENGTH}, "
              f"include_logprobs={CANONICAL_INCLUDE_LOGPROBS}, top_k={CANONICAL_TOP_K}")
        for dataset, info in per_ds.items():
            if "status" in info:  # whole-dataset error
                print(f"  {dataset:<10} {info['status']}")
                n_other += 1
                continue
            for split in ("train", "test"):
                s = info[split]
                n_total += 1
                if s["status"] == "hit":
                    n_hit += 1
                elif s["status"] in ("payload_missing", "params_only_no_match",
                                      "no_cache_dir"):
                    n_miss += 1
                else:
                    n_other += 1
                print(f"  {dataset:<10} {split:<5} {fmt_one(s)}")
                if args.show_others and s.get("other_manifests"):
                    for o in s["other_manifests"]:
                        print(f"             other: {json.dumps(o)}")

    print()
    print(f"Summary: {n_hit}/{n_total} HIT, {n_miss} MISS, {n_other} other")
    sys.exit(0 if n_miss == 0 and n_other == 0 else 1)


if __name__ == "__main__":
    main()
