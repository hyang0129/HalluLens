"""CPU-only contracts for the issue #149 dual-convention scope expansion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from activation_research.model import (
    DualConventionContrastiveClassifier,
    LogprobReconProgressiveCompressor,
)


def _make_model() -> DualConventionContrastiveClassifier:
    torch.manual_seed(149)
    return DualConventionContrastiveClassifier(
        input_dim=128,
        final_dim=64,
        classifier_hidden_dim=16,
        recon_seq_len=10,
        recon_hidden_dim=16,
        input_dropout=0.0,
        classifier_dropout=0.0,
    )


def test_two_full_branches_plus_classifier_parameter_contract():
    base = LogprobReconProgressiveCompressor(
        input_dim=128,
        final_dim=64,
        recon_seq_len=10,
        recon_hidden_dim=16,
        input_dropout=0.0,
    )
    model = _make_model()
    base_count = sum(p.numel() for p in base.parameters())
    classifier_count = sum(p.numel() for p in model.classifier.parameters())
    assert sum(p.numel() for p in model.standard_encoder.parameters()) == base_count
    assert sum(p.numel() for p in model.mirrored_encoder.parameters()) == base_count
    assert sum(p.numel() for p in model.parameters()) == 2 * base_count + classifier_count
    assert next(model.standard_encoder.parameters()).data_ptr() != next(
        model.mirrored_encoder.parameters()
    ).data_ptr()


def test_forward_exposes_two_64d_representations_and_128d_concat():
    model = _make_model().eval()
    x = torch.randn(8, 10, 128)
    with torch.no_grad():
        concat, standard, mirrored, recon = model.forward_with_heads(x)
        logits = model.classifier_logits_from_views(
            standard.reshape(4, 2, -1), mirrored.reshape(4, 2, -1)
        )
    assert concat.shape == (8, 128)
    assert standard.shape == mirrored.shape == (8, 64)
    assert recon[0].shape == recon[1].shape == (8, 10)
    assert logits.shape == (4,)


def test_classifier_loss_backpropagates_into_both_encoders():
    model = _make_model().train()
    x = torch.randn(8, 10, 128)
    _, standard, mirrored, _ = model.forward_with_heads(x)
    logits = model.classifier_logits_from_views(
        standard.reshape(4, 2, -1), mirrored.reshape(4, 2, -1)
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([0.0, 1.0, 0.0, 1.0])
    )
    loss.backward()
    assert all(
        p.grad is not None for p in model.standard_encoder.encoder.parameters()
    )
    assert all(
        p.grad is not None for p in model.mirrored_encoder.encoder.parameters()
    )
    assert all(p.grad is not None for p in model.classifier.parameters())


class _TinyDataset(torch.utils.data.Dataset):
    _max_resp = 10
    _num_views = 2

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(17)
        self.views = torch.randn(8, 2, 10, 128, generator=generator)
        self.labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        self.logprobs = torch.randn(8, 10, generator=generator)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {
            "views_activations": self.views[index],
            "halu": self.labels[index],
            "logprob": self.logprobs[index],
            "response_len": 10,
            "hashkey": f"tiny-{index}",
        }


def test_joint_trainer_updates_branches_decoders_and_classifier(tmp_path: Path):
    from activation_research.training import train_contrastive_logprob_recon_dualloss

    model = _make_model()
    before = {
        "standard": next(model.standard_encoder.encoder.parameters()).detach().clone(),
        "mirrored": next(model.mirrored_encoder.encoder.parameters()).detach().clone(),
        "decoder": next(model.standard_encoder.decoder.parameters()).detach().clone(),
        "classifier": next(model.classifier.parameters()).detach().clone(),
    }
    train_contrastive_logprob_recon_dualloss(
        model=model,
        train_dataset=_TinyDataset(),
        test_dataset=None,
        epochs=1,
        batch_size=8,
        lr=1e-3,
        temperature=0.1,
        device="cpu",
        num_workers=0,
        sub_batch_size=8,
        checkpoint_dir=str(tmp_path),
        persistent_workers=False,
        use_infinite_index_stream=False,
        recon_lambda=1.0,
        classifier_lambda=1.0,
        classifier_pos_weight=1.0,
        ignore_labels=(1, 0),
        prefix_view_mode="mixed",
        prefix_min_tokens=1,
        prefix_min_gap=1,
        prefix_sampling_lengths=[1, 4, 8, 10],
        prefix_seed=149,
    )
    after = {
        "standard": next(model.standard_encoder.encoder.parameters()).detach(),
        "mirrored": next(model.mirrored_encoder.encoder.parameters()).detach(),
        "decoder": next(model.standard_encoder.decoder.parameters()).detach(),
        "classifier": next(model.classifier.parameters()).detach(),
    }
    assert all(not torch.equal(before[name], after[name]) for name in before)
    checkpoint = torch.load(tmp_path / "contrastive_last.pt", map_location="cpu")
    assert checkpoint["train_classifier"] > 0
    assert checkpoint["classifier_lambda"] == 1.0


def test_issue149_config_keeps_full_size_branches_and_excludes_mmlu():
    root = Path(__file__).resolve().parents[1]
    method_path = (
        root
        / "configs/methods/dual_convention_contrastive_classifier_prefix_mixed_lowk.json"
    )
    method = json.loads(method_path.read_text())
    assert method["model_params"]["final_dim"] == 512
    assert method["model_params"]["classifier_hidden_dim"] == 128
    assert method["training"]["ignore_labels"] == [1, 0]
    assert method["training"]["classifier_lambda"] == 1.0
    assert method["evaluation"]["eval_prefix_lengths"] == [1, 4, 8, 16, 32, 48, 64]

    experiment_paths = sorted(
        (root / "configs/experiments").glob("prefix149_dual_convention_*.json")
    )
    datasets = {json.loads(path.read_text())["dataset"] for path in experiment_paths}
    assert datasets == {
        "hotpotqa_memmap",
        "nq_memmap",
        "popqa_memmap",
        "sciq_memmap",
        "searchqa_memmap",
    }
    assert all("mmlu" not in path.name.lower() for path in experiment_paths)


def test_dual_convention_cells_are_per_seed_idempotent_and_exclude_mmlu(
    tmp_path: Path,
):
    from scripts.dispatch.build_issue_149_dual_convention_cells import build

    dispatch_root = tmp_path / "dual-convention-dispatch"
    assert build(dispatch_root) == 25
    assert build(dispatch_root) == 0

    cells = sorted((dispatch_root / "pending").glob("*.json"))
    payloads = [json.loads(path.read_text()) for path in cells]
    assert len(payloads) == 25
    assert {cell["seed"] for cell in payloads} == {0, 1, 2, 3, 4}
    assert {cell["dataset"] for cell in payloads} == {
        "hotpotqa_memmap",
        "nq_memmap",
        "popqa_memmap",
        "sciq_memmap",
        "searchqa_memmap",
    }
    assert {cell["method"] for cell in payloads} == {
        "dual_convention_contrastive_classifier_prefix_mixed_lowk"
    }
    assert len(
        {(cell["dataset"], cell["seed"]) for cell in payloads}
    ) == 25
    assert all(cell["output_check"].endswith("predictions.csv") for cell in payloads)
    assert all("mmlu" not in path.name.lower() for path in cells)


def test_dual_convention_cell_builder_skips_complete_seed(tmp_path: Path):
    from scripts.dispatch.build_issue_149_dual_convention_cells import build

    project_root = tmp_path / "project"
    source_root = Path(__file__).resolve().parents[1]
    method_rel = Path(
        "configs/methods/dual_convention_contrastive_classifier_prefix_mixed_lowk.json"
    )
    method_path = project_root / method_rel
    method_path.parent.mkdir(parents=True)
    method_path.write_text((source_root / method_rel).read_text())
    for experiment in source_root.glob(
        "configs/experiments/prefix149_dual_convention_*.json"
    ):
        target = project_root / "configs/experiments" / experiment.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(experiment.read_text())

    completed = (
        project_root
        / "runs/prefix149_dual_convention_hotpotqa/hotpotqa_memmap"
        / "dual_convention_contrastive_classifier_prefix_mixed_lowk/seed_0"
    )
    completed.mkdir(parents=True)
    (completed / "eval_metrics.json").write_text("{}")
    (completed / "predictions.csv").write_text("score_halu\n")

    dispatch_root = tmp_path / "dispatch"
    assert build(dispatch_root, project_root=project_root) == 24
    assert not any("hotpotqa_memmap__seed_0" in path.name for path in (
        dispatch_root / "pending"
    ).glob("*.json"))


def test_classifier_prediction_rows_zip_scores_and_labels():
    from scripts.run_experiment import _build_classifier_predictions

    rows = _build_classifier_predictions(
        np.asarray([0.25, 0.75]), np.asarray([0, 1])
    )
    assert rows == [
        {"example_id": 0, "score_halu": 0.25, "label_halu": 0},
        {"example_id": 1, "score_halu": 0.75, "label_halu": 1},
    ]


def test_successful_recovery_clears_stale_run_error(tmp_path: Path):
    from scripts.run_experiment import _clear_stale_run_error

    marker = tmp_path / "run_error.json"
    marker.write_text('{"error": "old failure"}')
    assert _clear_stale_run_error(str(marker)) is True
    assert not marker.exists()
    assert _clear_stale_run_error(str(marker)) is False


def test_eval_recovery_cells_are_high_priority_checkpoint_guarded_and_idempotent(
    tmp_path: Path,
):
    from scripts.dispatch.build_issue_149_dual_convention_eval_cells import build

    project_root = tmp_path / "project"
    dispatch_root = project_root / "shared/issue_149_dual_convention_dispatch"
    failed_root = dispatch_root / "failed"
    failed_root.mkdir(parents=True)

    run_rel = Path(
        "runs/prefix149_dual_convention_hotpotqa/hotpotqa_memmap/"
        "dual_convention_contrastive_classifier_prefix_mixed_lowk/seed_0"
    )
    checkpoint = project_root / run_rel / "artifacts/final_weights.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    failed_cell = {
        "cell_id": "00_issue149_dual_convention__hotpotqa_memmap__seed_0",
        "kind": "experiment",
        "experiment_config": (
            "configs/experiments/prefix149_dual_convention_hotpotqa.json"
        ),
        "dataset": "hotpotqa_memmap",
        "method": "dual_convention_contrastive_classifier_prefix_mixed_lowk",
        "seed": 0,
        "seeded": True,
        "output_check": str(run_rel / "predictions.csv"),
    }
    source = failed_root / f"{failed_cell['cell_id']}.json"
    source.write_text(json.dumps(failed_cell))
    (failed_root / f"{source.name}.err").write_text("serialization failure")

    assert build(dispatch_root, project_root=project_root) == 1
    assert build(dispatch_root, project_root=project_root) == 0

    pending = list((dispatch_root / "pending").glob("*.json"))
    assert len(pending) == 1
    assert pending[0].name.startswith("000_eval_")
    recovery = json.loads(pending[0].read_text())
    assert recovery["kind"] == "evaluation"
    assert recovery["eval_only"] is True
    assert recovery["checkpoint_check"] == str(
        run_rel / "artifacts/final_weights.pt"
    )
    assert recovery["recovery_of"] == failed_cell["cell_id"]
    assert not list(failed_root.glob("*.json"))
    assert (dispatch_root / "recovery_history" / source.name).exists()
    assert (dispatch_root / "recovery_history" / f"{source.name}.err").exists()


def test_eval_recovery_builder_refuses_missing_checkpoint(tmp_path: Path):
    from scripts.dispatch.build_issue_149_dual_convention_eval_cells import build

    project_root = tmp_path / "project"
    dispatch_root = project_root / "shared/issue_149_dual_convention_dispatch"
    failed_root = dispatch_root / "failed"
    failed_root.mkdir(parents=True)
    failed_cell = {
        "cell_id": "20_issue149_dual_convention__nq_memmap__seed_0",
        "dataset": "nq_memmap",
        "seed": 0,
        "output_check": "runs/missing/seed_0/predictions.csv",
    }
    source = failed_root / f"{failed_cell['cell_id']}.json"
    source.write_text(json.dumps(failed_cell))

    assert build(dispatch_root, project_root=project_root) == 0
    assert source.exists()
    assert not list((dispatch_root / "pending").glob("*.json"))


def test_running_worker_auto_promotes_checkpoint_backed_failure(
    tmp_path: Path, monkeypatch
):
    from argparse import Namespace
    from scripts.dispatch import _claim_cli

    project_root = tmp_path / "project"
    dispatch_root = project_root / "shared/issue_149_dual_convention_dispatch"
    claimed_root = dispatch_root / "claimed/worker-1"
    claimed_root.mkdir(parents=True)

    run_rel = Path(
        "runs/prefix149_dual_convention_sciq/sciq_memmap/"
        "dual_convention_contrastive_classifier_prefix_mixed_lowk/seed_3"
    )
    checkpoint = project_root / run_rel / "artifacts/final_weights.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    cell = {
        "cell_id": "40_issue149_dual_convention__sciq_memmap__seed_3",
        "kind": "experiment",
        "experiment_config": (
            "configs/experiments/prefix149_dual_convention_sciq.json"
        ),
        "dataset": "sciq_memmap",
        "method": "dual_convention_contrastive_classifier_prefix_mixed_lowk",
        "seed": 3,
        "seeded": True,
        "output_check": str(run_rel / "predictions.csv"),
    }
    cell_path = claimed_root / f"{cell['cell_id']}.json"
    cell_path.write_text(json.dumps(cell))
    err_path = tmp_path / "worker.log"
    err_path.write_text("old process hit serializer bug")
    monkeypatch.setattr(_claim_cli, "_PROJECT_ROOT", project_root)

    args = Namespace(
        root=str(dispatch_root),
        worker_id="worker-1",
        cell=str(cell_path),
        err_file=str(err_path),
    )
    assert _claim_cli.cmd_fail(args) == 0

    pending = list((dispatch_root / "pending").glob("000_eval_*.json"))
    assert len(pending) == 1
    recovery = json.loads(pending[0].read_text())
    assert recovery["eval_only"] is True
    assert recovery["recovery_of"] == cell["cell_id"]
    assert not list((dispatch_root / "failed").glob("*.json"))
