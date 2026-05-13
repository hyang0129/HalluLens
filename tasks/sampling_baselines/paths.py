"""Path resolution for sampling-baseline artefacts."""
from pathlib import Path

DATASETS = ["hotpotqa", "nq", "popqa", "sciq", "searchqa", "mmlu"]
SAMPLING_DATASETS = ["hotpotqa", "nq", "popqa", "sciq", "searchqa"]  # MMLU skips sampling
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
]


def model_name(model_id: str) -> str:
    """Last component of HuggingFace model ID, matching output path convention."""
    return model_id.split("/")[-1]


def model_slug(model_id: str) -> str:
    """Lowercase underscored slug used in shared/ zarr path names."""
    return model_name(model_id).lower().replace("-", "_").replace(".", "_")


def generation_jsonl(dataset: str, model_id: str, split: str) -> Path:
    ds_dir = f"{dataset}_train" if split == "train" else dataset
    return Path("output") / ds_dir / model_name(model_id) / "generation.jsonl"


def eval_results_json(dataset: str, model_id: str, split: str) -> Path:
    ds_dir = f"{dataset}_train" if split == "train" else dataset
    return Path("output") / ds_dir / model_name(model_id) / "eval_results_for_training.json"


def zarr_path(dataset: str, model_id: str, split: str) -> Path:
    slug = model_slug(model_id)
    if split == "train":
        return Path("shared") / f"{dataset}_train_{slug}" / "activations.zarr"
    return Path("shared") / f"{dataset}_{slug}" / "activations.zarr"


def sampling_output_dir(dataset: str, model_id: str, split: str) -> Path:
    ds_dir = f"{dataset}_train" if split == "train" else dataset
    return Path("output") / "sampling_baselines" / ds_dir / model_name(model_id)


def selfcheck_samples_path(dataset: str, model_id: str, split: str) -> Path:
    return sampling_output_dir(dataset, model_id, split) / "selfcheck_samples.jsonl"


def nli_matrix_path(dataset: str, model_id: str, split: str) -> Path:
    return sampling_output_dir(dataset, model_id, split) / "nli_matrix.jsonl"


def se_labels_path(dataset: str, model_id: str, split: str) -> Path:
    return sampling_output_dir(dataset, model_id, split) / "se_labels.jsonl"


def selfcheck_scores_path(dataset: str, model_id: str, split: str) -> Path:
    return sampling_output_dir(dataset, model_id, split) / "selfcheck_scores.jsonl"


def sep_results_path(dataset: str, model_id: str) -> Path:
    return (
        Path("output")
        / "sampling_baselines"
        / "sep"
        / model_name(model_id)
        / f"{dataset}_sep_results.json"
    )


def subset_index_path(dataset: str, model_id: str) -> Path:
    return Path("output") / f"sep_subset_{dataset}_{model_name(model_id)}_seed42.json"


def searchqa_test_cap_path(model_id: str) -> Path:
    return Path("output") / f"searchqa_test_cap_{model_name(model_id)}_seed42.json"
