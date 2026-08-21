"""Validate that issue #151 cells are safe to submit without claiming them.

Checks the 25-cell manifest and, by default, opens one real Empire capture to
prove that pair-training and token-zero evaluation adapters share the ordinary
contrastive cache.  This script performs no training and starts no workers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from activation_research.memmap_activation_parser import (  # noqa: E402
    MemmapActivationParser,
)
from activation_research.tokenwise_contrastive_dataset import (  # noqa: E402
    TokenwiseContrastiveDataset,
)
from scripts.dispatch.claim import count_status  # noqa: E402

_ARCHITECTURE = "tokenwise_shared_contrastive_cache_v2"
_METHOD = "tokenwise_contrastive_first_anchored"


def validate_cells(dispatch_root: Path) -> dict:
    status = count_status(dispatch_root)
    cell_paths = sorted((dispatch_root / "pending").glob("*.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in cell_paths]
    if len(payloads) != 25:
        raise RuntimeError(f"expected 25 pending cells, found {len(payloads)}")
    if status["claimed"] or status["failed"]:
        raise RuntimeError(f"queue is not submission-clean: {status}")
    if any(payload.get("architecture") != _ARCHITECTURE for payload in payloads):
        raise RuntimeError("one or more cells use a stale architecture revision")
    if any(payload.get("method") != _METHOD for payload in payloads):
        raise RuntimeError("one or more cells are not first-token anchored")
    if any("mmlu" in json.dumps(payload).lower() for payload in payloads):
        raise RuntimeError("MMLU unexpectedly appears in the issue #151 queue")
    for payload in payloads:
        for key in ("experiment_config", "worker_script"):
            if not (_PROJECT_ROOT / payload[key]).is_file():
                raise FileNotFoundError(payload[key])
    return status


def validate_shared_cache_smoke(dataset_name: str) -> dict:
    dataset_cfg = json.loads(
        (_PROJECT_ROOT / "configs" / "datasets" / f"{dataset_name}.json").read_text(
            encoding="utf-8"
        )
    )
    method_cfg = json.loads(
        (
            _PROJECT_ROOT
            / "configs"
            / "methods"
            / "tokenwise_contrastive_first_anchored.json"
        ).read_text(encoding="utf-8")
    )
    capture_dir = _PROJECT_ROOT / dataset_cfg["icr_capture"]["train_dir"]
    parser = MemmapActivationParser(
        capture_dir,
        random_seed=42,
        split_strategy="three_way",
        label_source=dataset_cfg.get("label_source", "substring"),
    )
    layers = list(range(1, 33))
    base = parser.get_dataset(
        "train",
        relevant_layers=layers,
        num_views=2,
        pad_length=method_cfg["data"]["pad_length"],
        include_response_logprobs=True,
        response_logprobs_top_k=method_cfg["data"]["response_logprobs_top_k"],
        check_ram=False,
    )
    pairs = TokenwiseContrastiveDataset(
        base,
        layer_positions=layers,
        num_views=2,
        token_pair_mode="first_anchored",
        min_response_tokens=2,
    )
    token0 = TokenwiseContrastiveDataset(
        base,
        layer_positions=layers,
        num_views=1,
        token_pair_mode="first_anchored",
        fixed_token=0,
        min_response_tokens=1,
    )
    pair_item = pairs[0]
    token0_item = token0[0]
    if pairs.cache is not token0.cache or pairs.cache is not base.cache:
        raise RuntimeError("token-wise adapters did not reuse the base cache")
    if tuple(pair_item["views_activations"].shape[:2]) != (2, 32):
        raise RuntimeError("pair view does not have shape (2, 32, hidden_dim)")
    if pair_item["view_token_indices"][0].item() != 0:
        raise RuntimeError("first-anchored pair did not start at token zero")
    if token0_item["view_token_indices"].tolist() != [0]:
        raise RuntimeError("primary evaluation view is not fixed at token zero")
    return {
        "dataset": dataset_name,
        "base_rows": len(base),
        "pair_rows": len(pairs),
        "token0_rows": len(token0),
        "pair_shape": list(pair_item["views_activations"].shape),
        "token0_shape": list(token0_item["views_activations"].shape),
        "shared_cache": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dispatch-root",
        default="shared/issue_151_tokenwise_dispatch",
    )
    parser.add_argument("--dataset", default="hotpotqa_memmap")
    parser.add_argument("--skip-data-smoke", action="store_true")
    args = parser.parse_args()
    dispatch_root = Path(args.dispatch_root)
    if not dispatch_root.is_absolute():
        dispatch_root = _PROJECT_ROOT / dispatch_root

    result = {"queue": validate_cells(dispatch_root)}
    if not args.skip_data_smoke:
        result["data_smoke"] = validate_shared_cache_smoke(args.dataset)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
