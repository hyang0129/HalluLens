"""Contracts for the >=10M-parameter token-zero supervised baseline."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from activation_research.model import TokenZeroMLPProbe
from activation_research.trainer import LinearProbeTrainer, LinearProbeTrainerConfig
from scripts.dispatch.build_issue_151_token_zero_mlp_cells import build

_ROOT = Path(__file__).resolve().parent.parent
_METHOD = "token_zero_mlp_probe"
_DATASETS = {
    "hotpotqa_memmap": "hotpotqa",
    "nq_memmap": "nq",
    "popqa_memmap": "popqa",
    "sciq_memmap": "sciq",
    "searchqa_memmap": "searchqa",
}


def test_token_zero_mlp_has_expected_capacity_and_geometry():
    model = TokenZeroMLPProbe(
        input_dim=4096,
        num_layers=32,
        hidden_dim=2048,
        output_dim=1024,
        dropout=0.1,
        normalize_input=True,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 10_531_842
    assert sum(parameter.numel() for parameter in model.parameters()) >= 10_000_000

    # Use a small same-structure model for the forward contract.
    small = TokenZeroMLPProbe(
        input_dim=8,
        num_layers=4,
        hidden_dim=6,
        output_dim=3,
        dropout=0.0,
    )
    scores = small(torch.randn(5, 4, 8))
    assert scores.shape == (5, 1)
    assert torch.all((0.0 <= scores) & (scores <= 1.0))
    with pytest.raises(ValueError, match="input geometry mismatch"):
        small(torch.randn(5, 3, 8))


def test_validation_auroc_selection_restores_best_epoch(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Sigmoid())
    trainer = LinearProbeTrainer(
        model,
        config=LinearProbeTrainerConfig(
            max_epochs=2,
            batch_size=1,
            lr=1e-3,
            device="cpu",
            num_workers=0,
            checkpoint_dir=str(tmp_path),
            save_every=1,
            select_on_val=True,
            cleanup_legacy_checkpoints=True,
        ),
    )

    validation_aurocs = (0.8, 0.7)

    def train_epoch(*, epoch, train_dataset):
        with torch.no_grad():
            trainer.model[0].weight.fill_(float(epoch + 1))
        return {"train_loss": float(epoch)}

    def validate(*, epoch, val_dataset):
        auroc = validation_aurocs[epoch]
        trainer.best_auroc = max(trainer.best_auroc, auroc)
        return {"val_loss": 1.0 - auroc, "val_auroc": auroc, "best_auroc": trainer.best_auroc}

    trainer.train_epoch = train_epoch
    trainer.validate = validate
    trainer.fit(train_dataset=[0], val_dataset=[0])

    assert trainer.selected_epoch == 0
    assert trainer.selected_val_auroc == pytest.approx(0.8)
    assert torch.allclose(trainer.model[0].weight, torch.ones_like(trainer.model[0].weight))
    assert (tmp_path / "linear_probe_best.pt").is_file()


def test_method_and_experiments_are_token_zero_only_and_paired():
    method = json.loads(
        (_ROOT / "configs/methods/token_zero_mlp_probe.json").read_text(
            encoding="utf-8"
        )
    )
    assert method["model_params"]["expected_total_params"] == 10_531_842
    assert method["model_params"]["parameter_floor"] >= 10_000_000
    assert method["data"]["fixed_token"] == 0
    assert method["data"]["num_views"] == 1
    assert method["data"]["include_response_logprobs"] is False
    assert method["training"]["checkpoint_selection_metric"] == "validation_auroc"

    payloads = []
    for slug in _DATASETS.values():
        path = _ROOT / "configs/experiments" / f"issue151_token_zero_mlp_{slug}.json"
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    assert {payload["dataset"] for payload in payloads} == set(_DATASETS)
    assert all(payload["methods"] == [_METHOD] for payload in payloads)
    assert all(payload["training_seeds"] == [0, 1, 2, 3, 4] for payload in payloads)
    assert all(payload["split_seeds"] == [42, 1, 2, 3, 4] for payload in payloads)
    assert all("mmlu" not in json.dumps(payload).lower() for payload in payloads)


def test_builder_adds_exactly_25_idempotent_low_priority_cells(tmp_path):
    dispatch_root = tmp_path / "dispatch"
    assert build(dispatch_root, project_root=_ROOT) == 25
    assert build(dispatch_root, project_root=_ROOT) == 0

    paths = sorted((dispatch_root / "pending").glob("*.json"))
    cells = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert len(cells) == 25
    assert {cell["dataset"] for cell in cells} == set(_DATASETS)
    assert {cell["method"] for cell in cells} == {_METHOD}
    assert {cell["seed"] for cell in cells} == {0, 1, 2, 3, 4}
    assert {(cell["seed"], cell["split_seed"]) for cell in cells} == {
        (0, 42),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
    }
    assert all(cell["priority"] == "low" for cell in cells)
    assert all(cell["cell_id"].startswith("9_low_") for cell in cells)
    assert all(cell["model_total_params"] == 10_531_842 for cell in cells)
    assert all(cell["primary_eval_token"] == 0 for cell in cells)
    assert all("mmlu" not in json.dumps(cell).lower() for cell in cells)

    source = paths[0]
    claimed = dispatch_root / "claimed" / "worker-0" / source.name
    claimed.parent.mkdir(parents=True)
    source.rename(claimed)
    assert build(dispatch_root, project_root=_ROOT) == 0
    assert claimed.is_file()
