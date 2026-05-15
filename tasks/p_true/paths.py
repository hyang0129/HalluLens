"""Path resolution for P(true) artefacts.

Source-data paths are resolved from the canonical dataset configs in
``configs/datasets/{ds}{_qwen3}.json`` rather than from on-disk directory names.
This is required because the Issue #60 searchqa train/test flip changed the
split semantics in the configs but did NOT rename the directories — so the
directory-name convention used by ``tasks.sampling_baselines.paths`` points
P(true) at the wrong split for searchqa.
"""
import json
from pathlib import Path

from tasks.sampling_baselines.paths import _dir_stem, model_name

DATASETS = ["hotpotqa", "nq", "popqa", "sciq", "searchqa", "mmlu"]
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
]

# Per-model suffix for the dataset-config filename.
# Llama uses the bare {ds}.json; Qwen3 uses {ds}_qwen3.json.
_MODEL_CONFIG_SUFFIX = {
    "meta-llama/Llama-3.1-8B-Instruct": "",
    "Qwen/Qwen3-8B": "_qwen3",
}


def dataset_config_path(dataset: str, model_id: str) -> Path:
    if model_id not in _MODEL_CONFIG_SUFFIX:
        raise ValueError(
            f"Unknown model for P(true) config lookup: {model_id}. "
            f"Add to tasks/p_true/paths._MODEL_CONFIG_SUFFIX."
        )
    suffix = _MODEL_CONFIG_SUFFIX[model_id]
    return Path("configs") / "datasets" / f"{dataset}{suffix}.json"


def resolve_split_paths(dataset: str, model_id: str, split: str) -> dict:
    """Return ``{'generation_jsonl', 'eval_json'}`` Paths for a (ds, model, split) cell.

    Reads the canonical dataset config — necessary so the post-Issue-#60
    searchqa flip (and any future flips) is honored without hardcoding the
    on-disk directory naming.
    """
    cfg_path = dataset_config_path(dataset, model_id)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {cfg_path}")
    cfg = json.loads(cfg_path.read_text())
    if split not in cfg:
        raise KeyError(
            f"Split '{split}' not in {cfg_path}; available: {list(cfg.keys())}"
        )
    block = cfg[split]
    return {
        "generation_jsonl": Path(block["inference_json"]),
        "eval_json": Path(block["eval_json"]),
    }


def ptrue_output_dir(dataset: str, model_id: str, split: str = "test") -> Path:
    stem = _dir_stem(dataset)
    ds_dir = f"{stem}_train" if split == "train" else stem
    return Path("output") / "p_true" / ds_dir / model_name(model_id)


def ptrue_scores_path(dataset: str, model_id: str, split: str = "test") -> Path:
    return ptrue_output_dir(dataset, model_id, split) / "ptrue.jsonl"
