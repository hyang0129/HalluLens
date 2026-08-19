"""Contracts for the chatv1 chat-template re-capture method matrix cells."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.dispatch.build_chatv1_matrix_cells import build
from scripts.experiment_utils import load_method_config

_ROOT = Path(__file__).resolve().parent.parent

_TASKS = ["hotpotqa", "nq", "popqa", "sciq", "searchqa"]

_LLAMA_METHODS = [
    "tokenwise_contrastive_first_anchored",
    "tokenwise_arch_v1_input_norm_only",
    "act_vit",
    "token_zero_mlp_probe",
    "logprob_baseline",
    "token_entropy",
]
_QWEN_METHODS = [
    "tokenwise_contrastive_first_anchored_qwen3",
    "tokenwise_arch_v1_input_norm_only_qwen3",
    "act_vit",
    "token_zero_mlp_probe_qwen3",
    "logprob_baseline",
    "token_entropy",
]
_TOKENWISE_METHODS = {
    "tokenwise_contrastive_first_anchored",
    "tokenwise_arch_v1_input_norm_only",
    "tokenwise_contrastive_first_anchored_qwen3",
    "tokenwise_arch_v1_input_norm_only_qwen3",
}

# Fields worker_experiment.sh actually parses from a cell JSON (see
# scripts/dispatch/worker_experiment.sh): kind, experiment_config, method,
# seed, output_check are required; eval_only, checkpoint_check,
# embedding_backfill/preserve_existing_eval_outputs are optional.
_REQUIRED_WORKER_FIELDS = {"kind", "experiment_config", "method", "seed", "output_check"}


def test_all_method_configs_resolve():
    for method in set(_LLAMA_METHODS) | set(_QWEN_METHODS):
        cfg = load_method_config(method, project_root=str(_ROOT))
        assert cfg["name"] or method  # loads without raising


def test_qwen3_specific_methods_widen_layers_to_36():
    for method in ("tokenwise_contrastive_first_anchored_qwen3", "tokenwise_arch_v1_input_norm_only_qwen3", "token_zero_mlp_probe_qwen3"):
        cfg = load_method_config(method, project_root=str(_ROOT))
        assert cfg["data"]["relevant_layers"] == "1-36", method


def test_experiment_configs_cover_five_tasks_two_models_without_mmlu():
    payloads = []
    for task in _TASKS:
        llama_path = _ROOT / "configs/experiments" / f"chatv1_{task}.json"
        llama = json.loads(llama_path.read_text(encoding="utf-8"))
        payloads.append(llama)
        assert llama["experiment_name"] == f"chatv1_{task}"
        assert llama["dataset"] == f"{task}_chat_memmap"
        assert llama["methods"] == _LLAMA_METHODS
        assert llama["training_seeds"] == [0]
        assert "mmlu" not in json.dumps(llama).lower()

        qwen_path = _ROOT / "configs/experiments" / f"chatv1_qwen3_{task}.json"
        qwen = json.loads(qwen_path.read_text(encoding="utf-8"))
        payloads.append(qwen)
        assert qwen["experiment_name"] == f"chatv1_qwen3_{task}"
        assert qwen["dataset"] == f"{task}_qwen3_chat_memmap"
        assert qwen["methods"] == _QWEN_METHODS
        assert qwen["training_seeds"] == [0]
        assert "mmlu" not in json.dumps(qwen).lower()
    assert len(payloads) == 10


def test_builder_writes_60_cells_and_is_idempotent(tmp_path):
    dispatch_root = tmp_path / "dispatch"

    assert build(dispatch_root, project_root=_ROOT) == 60
    # Re-running must not duplicate cells.
    assert build(dispatch_root, project_root=_ROOT) == 0

    cells = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((dispatch_root / "pending").glob("*.json"))
    ]
    assert len(cells) == 60

    # cell_id uniqueness.
    ids = [c["cell_id"] for c in cells]
    assert len(ids) == len(set(ids))

    # Every worker-consumed field present with correct types.
    for cell in cells:
        missing = _REQUIRED_WORKER_FIELDS - set(cell)
        assert not missing, f"cell {cell['cell_id']} missing {missing}"
        assert cell["kind"] == "experiment"
        assert isinstance(cell["method"], str)
        assert isinstance(cell["seed"], int)
        assert isinstance(cell["output_check"], str)
        if cell["method"] in ("logprob_baseline", "token_entropy"):
            # Why: run_experiment treats these as unseeded — no seed_N/ level,
            # completion signalled by eval_metrics.json, not predictions.csv.
            assert cell["seeded"] is False
            assert cell["output_check"].endswith(f"{cell['method']}/eval_metrics.json")
            assert "/seed_" not in cell["output_check"]
        else:
            assert cell["seeded"] is True
            assert cell["output_check"].endswith("seed_0/predictions.csv")
        assert cell["worker_script"] == "scripts/dispatch/worker_experiment.sh"
        # experiment_config must resolve to a real file relative to project root.
        assert (_ROOT / cell["experiment_config"]).is_file()
        assert "mmlu" not in json.dumps(cell).lower()
        # No eval_only cell in this matrix (fresh seed-0 training runs only).
        assert cell.get("eval_only", False) is False

    # 5 tasks x 2 models x 6 methods = 60.
    by_dataset = {}
    for c in cells:
        by_dataset.setdefault(c["dataset"], []).append(c)
    assert len(by_dataset) == 10
    assert all(len(v) == 6 for v in by_dataset.values())

    # Tokenwise cells sort first: every tokenwise cell_id must precede every
    # non-tokenwise cell_id lexicographically for the SAME dataset_order
    # prefix, and the global claim order (a flat sorted glob) puts the whole
    # "0_high" prefix block before "5_mid"/"8_low".
    tokenwise_ids = sorted(c["cell_id"] for c in cells if c["method"] in _TOKENWISE_METHODS)
    other_ids = sorted(c["cell_id"] for c in cells if c["method"] not in _TOKENWISE_METHODS)
    assert all(cid.startswith("0_high_") for cid in tokenwise_ids)
    assert max(tokenwise_ids) < min(other_ids)

    by_method = {}
    for c in cells:
        by_method.setdefault(c["method"], []).append(c)
    assert set(by_method) == set(_LLAMA_METHODS) | set(_QWEN_METHODS)
    for method in ("tokenwise_contrastive_first_anchored", "tokenwise_arch_v1_input_norm_only",
                   "tokenwise_contrastive_first_anchored_qwen3", "tokenwise_arch_v1_input_norm_only_qwen3",
                   "token_zero_mlp_probe", "token_zero_mlp_probe_qwen3"):
        assert len(by_method[method]) == 5, method
    for method in ("act_vit", "logprob_baseline", "token_entropy"):
        assert len(by_method[method]) == 10, method

    assert all(c["seed"] == 0 for c in cells)
    assert all(c["split_seed"] == 42 for c in cells)
    assert sum(c["seeded"] is False for c in cells) == 20  # 2 unseeded methods x 10 (task, model)
