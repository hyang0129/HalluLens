"""Contracts for the Issue #153 token-wise v2 model and pilot."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from activation_research.evaluation import evaluate
from activation_research.model import (
    AttentionPoolProgressiveCompressor,
    LogprobReconProgressiveCompressor,
    LogprobReconProjectedProgressiveCompressor,
    ProgressiveCompressor,
)
from activation_research.training import (
    SupConLoss,
    _contrastive_collate_with_logprob,
    train_contrastive_logprob_recon,
)
from scripts.dispatch.build_issue_153_v2_cells import build

_ROOT = Path(__file__).resolve().parent.parent
_METHOD = "tokenwise_contrastive_v2_depthnorm_projection"
_DATASETS = {
    "hotpotqa_memmap",
    "nq_memmap",
    "popqa_memmap",
    "sciq_memmap",
    "searchqa_memmap",
}
_DATASET_SUFFIXES = ("hotpotqa", "nq", "popqa", "sciq", "searchqa")


def _make_model(**overrides) -> LogprobReconProjectedProgressiveCompressor:
    kwargs = {
        "input_dim": 128,
        "final_dim": 64,
        "projection_hidden_dim": 32,
        "projection_dim": 16,
        "depth_pooling": "attention",
        "pool_num_queries": 1,
        "normalize_input": True,
        "pre_norm": True,
        "block_dims": [64],
        "input_dropout": 0.0,
        "recon_seq_len": 8,
        "recon_hidden_dim": 16,
    }
    kwargs.update(overrides)
    return LogprobReconProjectedProgressiveCompressor(**kwargs)


class _TinyTwoViewDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 8) -> None:
        generator = torch.Generator().manual_seed(153)
        self.views = torch.randn(n, 2, 4, 128, generator=generator)
        self.labels = torch.tensor([0, 0, 1, 1] * (n // 4))
        self.logprobs = torch.randn(n, 8, generator=generator)

    def __len__(self) -> int:
        return len(self.views)

    def __getitem__(self, index: int) -> dict:
        return {
            "views_activations": self.views[index],
            "halu": self.labels[index],
            "logprob": self.logprobs[index],
            "hashkey": f"issue153_{index}",
        }


def test_v2_returns_trunk_for_deployment_and_normalized_projection_for_supcon():
    model = _make_model().eval()
    x = torch.randn(3, 4, 128)
    projection_calls = []
    handle = model.projection_head.register_forward_hook(
        lambda *_args: projection_calls.append(True)
    )
    with torch.no_grad():
        deployment = model(x)
        assert projection_calls == []
        trunk, projection, reconstruction = model.forward_with_contrastive_recon(x)
    handle.remove()

    assert deployment.shape == (3, 64)
    assert trunk.shape == (3, 64)
    assert projection.shape == (3, 16)
    assert reconstruction.shape == (3, 8)
    torch.testing.assert_close(deployment, trunk)
    torch.testing.assert_close(
        projection.norm(dim=-1), torch.ones(3), rtol=1e-5, atol=1e-5
    )


def test_v2_components_are_independently_ablatable():
    mean_model = _make_model(
        depth_pooling="mean", normalize_input=False, pre_norm=False
    )
    assert isinstance(mean_model.encoder, ProgressiveCompressor)
    assert not hasattr(mean_model.encoder, "input_norm")
    assert mean_model.encoder.blocks[0].encoder.norm_first is False

    attention_model = _make_model(
        depth_pooling="attention", normalize_input=True, pre_norm=True
    )
    assert isinstance(attention_model.encoder, AttentionPoolProgressiveCompressor)
    assert hasattr(attention_model.encoder, "input_norm")
    assert attention_model.encoder.blocks[0].encoder.norm_first is True
    assert attention_model.encoder.pool.num_queries == 1


def test_v2_supcon_and_reconstruction_reach_all_training_components():
    model = _make_model()
    views = torch.randn(4, 2, 4, 128)
    flat = views.reshape(8, 4, 128)
    trunk, projection, reconstruction = model.forward_with_contrastive_recon(flat)
    projection_views = projection.reshape(4, 2, 16)
    labels = torch.tensor([0, 0, 1, 1])
    supcon = SupConLoss(temperature=0.25, ignore_label=1)(
        projection_views, labels=labels
    )
    recon, _ = model.recon_loss(reconstruction, torch.randn(8, 8))
    (supcon + recon).backward()

    assert trunk.shape == (8, 64)
    assert model.encoder.input_norm.weight.grad.abs().sum() > 0
    assert model.encoder.pool.query.grad.abs().sum() > 0
    assert model.projection_head[0].weight.grad.abs().sum() > 0
    assert model.decoder[0].weight.grad.abs().sum() > 0


def test_shared_trainer_and_validation_loss_use_projection(tmp_path):
    model = _make_model()
    before = model.projection_head[0].weight.detach().clone()
    dataset = _TinyTwoViewDataset()
    train_contrastive_logprob_recon(
        model,
        train_dataset=dataset,
        test_dataset=None,
        epochs=1,
        batch_size=8,
        sub_batch_size=8,
        lr=1e-3,
        temperature=0.25,
        device="cpu",
        num_workers=0,
        checkpoint_dir=tmp_path,
        persistent_workers=False,
        use_labels=True,
        ignore_label=1,
    )
    assert not torch.equal(before, model.projection_head[0].weight.detach())

    calls = []
    handle = model.projection_head.register_forward_hook(
        lambda *_args: calls.append(True)
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        collate_fn=_contrastive_collate_with_logprob,
    )
    loss, _, _ = evaluate(
        model,
        loader,
        batch_size=8,
        sub_batch_size=8,
        loss_fn=SupConLoss(temperature=0.25, ignore_label=1),
        device="cpu",
        use_labels=True,
    )
    handle.remove()
    assert calls
    assert torch.isfinite(torch.tensor(loss))


def test_v2_checkpoint_round_trip_preserves_deployment_embedding(tmp_path):
    model = _make_model().eval()
    x = torch.randn(2, 4, 128)
    with torch.no_grad():
        expected = model(x)
    path = tmp_path / "v2.pt"
    torch.save(model.state_dict(), path)
    restored = _make_model().eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        actual = restored(x)
    torch.testing.assert_close(actual, expected)


def test_v2_full_parameter_budget_matches_config_on_meta_device():
    config = json.loads(
        (_ROOT / "configs" / "methods" / f"{_METHOD}.json").read_text()
    )
    params = config["model_params"]
    with torch.device("meta"):
        v1 = LogprobReconProgressiveCompressor(
            input_dim=4096,
            final_dim=512,
            input_dropout=0.3,
            recon_seq_len=64,
            recon_hidden_dim=256,
        )
        v2 = LogprobReconProjectedProgressiveCompressor(
            input_dim=4096,
            final_dim=params["final_dim"],
            projection_hidden_dim=params["projection_hidden_dim"],
            projection_dim=params["projection_dim"],
            projection_l2_normalize=params["projection_l2_normalize"],
            depth_pooling=params["depth_pooling"],
            pool_num_queries=params["pool_num_queries"],
            normalize_input=params["normalize_input"],
            pre_norm=params["pre_norm"],
            input_dropout=params["input_dropout"],
            recon_seq_len=params["recon_seq_len"],
            recon_hidden_dim=params["recon_hidden_dim"],
        )
    v1_count = sum(p.numel() for p in v1.parameters())
    v2_count = sum(p.numel() for p in v2.parameters())
    assert v1_count == params["reference_total_params"] == 77_538_112
    assert v2_count == params["expected_total_params"] == 77_875_136
    assert (v2_count - v1_count) / v1_count < 0.005


def test_v2_method_is_matched_to_corrected_v1_outside_the_model():
    v1 = json.loads(
        (
            _ROOT
            / "configs"
            / "methods"
            / "tokenwise_contrastive_first_anchored.json"
        ).read_text()
    )
    v2 = json.loads(
        (_ROOT / "configs" / "methods" / f"{_METHOD}.json").read_text()
    )
    assert v2["routine"] == v1["routine"]
    assert v2["training"] == v1["training"]
    assert v2["data"] == v1["data"]
    assert v2["evaluation"] == v1["evaluation"]


def test_issue153_builder_appends_exact_five_seed_zero_cells(tmp_path):
    dispatch_root = tmp_path / "dispatch"
    assert build(dispatch_root, project_root=_ROOT) == 5
    assert build(dispatch_root, project_root=_ROOT) == 0

    paths = sorted((dispatch_root / "pending").glob("*.json"))
    payloads = [json.loads(path.read_text()) for path in paths]
    assert len(payloads) == 5
    assert {cell["dataset"] for cell in payloads} == _DATASETS
    assert {cell["seed"] for cell in payloads} == {0}
    assert {cell["method"] for cell in payloads} == {_METHOD}
    assert all(cell["issue"] == 153 for cell in payloads)
    assert all(cell["kind"] == "experiment" for cell in payloads)
    assert all(path.name.startswith("0_high_") for path in paths)
    assert all("mmlu" not in json.dumps(cell).lower() for cell in payloads)
    assert all(cell["baseline_run"].endswith("eval_metrics.json") for cell in payloads)


def test_issue153_analysis_plan_is_predeclared_and_identical():
    plans = []
    for suffix in _DATASET_SUFFIXES:
        payload = json.loads(
            (
                _ROOT
                / "configs"
                / "experiments"
                / f"issue153_v2_{suffix}.json"
            ).read_text()
        )
        plans.append(payload["analysis_plan"])
    assert all(plan == plans[0] for plan in plans)
    assert plans[0] == {
        "primary_metric": "knn_auroc",
        "aggregate": "unweighted_macro_of_dataset_seed_means",
        "baseline_experiment_prefix": "issue151_knnval",
        "expansion_threshold": "v2_macro_delta_gt_0_then_run_seeds_1_2",
    }
    for suffix in _DATASET_SUFFIXES:
        payload = json.loads(
            (
                _ROOT
                / "configs"
                / "experiments"
                / f"issue153_v2_{suffix}.json"
            ).read_text()
        )
        assert payload["training_seeds"] == [0]
        assert payload["split_seeds"] == [42]
