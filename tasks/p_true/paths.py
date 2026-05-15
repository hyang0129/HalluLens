"""Path resolution for P(true) artefacts."""
from pathlib import Path

from tasks.sampling_baselines.paths import _dir_stem, model_name

DATASETS = ["hotpotqa", "nq", "popqa", "sciq", "searchqa", "mmlu"]
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
]


def ptrue_output_dir(dataset: str, model_id: str, split: str = "test") -> Path:
    stem = _dir_stem(dataset)
    ds_dir = f"{stem}_train" if split == "train" else stem
    return Path("output") / "p_true" / ds_dir / model_name(model_id)


def ptrue_scores_path(dataset: str, model_id: str, split: str = "test") -> Path:
    return ptrue_output_dir(dataset, model_id, split) / "ptrue.jsonl"
