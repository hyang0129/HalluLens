"""CPU-only contracts for the issue #149 matched baseline queues."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def test_split_seed_is_paired_with_training_seed():
    from scripts.run_experiment import _resolve_run_split_seed

    cfg = {
        "split_seed": 42,
        "training_seeds": [0, 1, 2, 3, 4],
        "split_seeds": [42, 11, 12, 13, 14],
    }
    assert _resolve_run_split_seed(cfg, 3) == 13
    assert _resolve_run_split_seed(cfg, 99) == 42


def test_act_vit_prefix_helper_groups_early_eos_and_preserves_order():
    from scripts.run_experiment import _act_vit_logits_at_prefix

    class LastVisible(torch.nn.Module):
        def forward(self, x):
            return x[:, 0, -1, 0].unsqueeze(1)

    x = torch.zeros(3, 1, 5, 1, requires_grad=True)
    with torch.no_grad():
        for row in range(3):
            x[row, 0, :, 0] = row * 10 + torch.arange(5)
    logits = _act_vit_logits_at_prefix(
        LastVisible(), x, torch.tensor([4, 1, 3]), prefix_len=4
    )
    assert logits.tolist() == [3.0, 10.0, 22.0]
    logits.sum().backward()
    assert x.grad is not None
    assert x.grad[:, 0, :, 0].sum(dim=1).tolist() == [1.0, 1.0, 1.0]


def test_linear_probe_trainer_runs_two_prefix_losses(tmp_path: Path):
    from activation_research.model import LinearProbe
    from activation_research.trainer import (
        LinearProbeTrainer,
        LinearProbeTrainerConfig,
    )

    model = LinearProbe(input_dim=3, pooling="mean")
    trainer = LinearProbeTrainer(
        model,
        config=LinearProbeTrainerConfig(
            max_epochs=1,
            batch_size=2,
            device="cpu",
            checkpoint_dir=str(tmp_path),
            prefix_training=True,
            prefix_min_tokens=1,
            prefix_min_gap=1,
            prefix_max_tokens=4,
            prefix_seed=7,
        ),
    )
    loss, metrics = trainer.training_step(
        {
            "views_activations": torch.randn(2, 1, 4, 3),
            "halu": torch.tensor([0.0, 1.0]),
            "response_len": torch.tensor([2, 4]),
        }
    )
    assert torch.isfinite(loss)
    assert 0.0 <= metrics["acc"] <= 1.0
    loss.backward()
    assert model.linear.weight.grad is not None


def _write_tiny_capture(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    path.mkdir(parents=True)
    n, layers, r_max, hidden = 2, 2, 4, 5
    cfg = {
        "n_samples": n,
        "num_layers": layers,
        "hidden_dim": hidden,
        "r_max": r_max,
        "max_response_len": r_max,
        "max_prompt_len": 8,
    }
    (path / "config.json").write_text(json.dumps(cfg))
    (path / "meta.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_index": i,
                    "prompt_hash": f"h{i}",
                    "hallucinated": bool(i),
                }
            )
            for i in range(n)
        )
        + "\n"
    )

    raw_attn = rng.random((n, layers, r_max, r_max)).astype(np.float32)
    raw_attn /= raw_attn.sum(axis=-1, keepdims=True)
    acts = rng.standard_normal((n, layers + 1, r_max, hidden)).astype(np.float16)
    response_lens = np.array([2, 4], dtype=np.int32)
    prompt_lens = np.array([8, 3], dtype=np.int32)

    for filename, array in (
        ("response_attention.npy", raw_attn.astype(np.float16)),
        ("response_activations.npy", acts),
        ("response_len.npy", response_lens),
        ("prompt_len.npy", prompt_lens),
    ):
        mm = np.memmap(path / filename, dtype=array.dtype, mode="w+", shape=array.shape)
        mm[:] = array
        mm.flush()
    return raw_attn, acts.astype(np.float32), response_lens


def test_prefix_icr_cache_recomputes_each_visible_window(tmp_path: Path):
    from activation_research.icr_score import compute_icr_score
    from scripts.build_prefix_icr_cache import build_prefix_caches

    capture = tmp_path / "capture"
    raw_attn, acts, response_lens = _write_tiny_capture(capture)
    output = tmp_path / "prefix-cache"
    paths = build_prefix_caches(
        capture,
        output,
        prefixes=[1, 2, 4],
        batch_size=2,
        device=torch.device("cpu"),
        top_p=0.5,
    )
    assert paths == [
        output / "icr_scores_k1.npy",
        output / "icr_scores_k2.npy",
        output / "icr_scores_k4.npy",
    ]

    prompt_lens = [8, 3]
    for prefix in (1, 2, 4):
        cached = np.load(output / f"icr_scores_k{prefix}.npy")
        expected = np.zeros_like(cached)
        for sample in range(2):
            visible = min(prefix, int(response_lens[sample]))
            for layer in range(2):
                expected[sample, layer] = compute_icr_score(
                    raw_attn[sample, layer],
                    acts[sample, layer],
                    acts[sample, layer + 1] - acts[sample, layer],
                    visible,
                    top_p=0.5,
                    prompt_len=prompt_lens[sample],
                )
        np.testing.assert_allclose(cached, expected, atol=5e-4, rtol=5e-4)


def test_matched_cell_builders_are_isolated_and_idempotent(tmp_path: Path):
    from scripts.dispatch.build_issue_149_icr_cache_cells import build as build_cache
    from scripts.dispatch.build_issue_149_matched_cells import build as build_matched

    cache_root = tmp_path / "cache-dispatch"
    matched_root = tmp_path / "matched-dispatch"
    assert build_cache(cache_root) == 4
    assert build_cache(cache_root) == 0
    assert build_matched(matched_root) == 14
    assert build_matched(matched_root) == 0

    cache_cells = list((cache_root / "pending").glob("*.json"))
    matched_cells = list((matched_root / "pending").glob("*.json"))
    assert len(cache_cells) == 4
    assert len(matched_cells) == 14
    assert all("issue_149_dispatch" not in str(path) for path in cache_cells + matched_cells)

    learned = [
        json.loads(path.read_text())
        for path in matched_cells
        if "act_vit_prefix_multik" in path.name
    ]
    assert len(learned) == 2
    assert all(cell["seed"] == "0,1,2,3,4" for cell in learned)
    assert all(cell["output_check"].endswith("seed_4/eval_metrics.json") for cell in learned)


def test_lowk_cells_cover_benchmark_datasets_and_prioritize_hotpotqa(tmp_path: Path):
    from scripts.dispatch.build_issue_149_lowk_cells import build

    root = tmp_path / "lowk-dispatch"
    assert build(root) == 50
    assert build(root) == 0

    cells = sorted((root / "pending").glob("*.json"))
    assert len(cells) == 50
    assert all("hotpotqa_memmap" in path.name for path in cells[:10])
    assert {
        json.loads(path.read_text())["dataset"] for path in cells
    } == {
        "hotpotqa_memmap",
        "nq_memmap",
        "popqa_memmap",
        "sciq_memmap",
        "searchqa_memmap",
    }
    assert {
        json.loads(path.read_text())["method"] for path in cells
    } == {
        "act_vit_prefix_multik_lowk",
        "contrastive_logprob_recon_prefix_mixed_lowk",
    }
    payloads = [json.loads(path.read_text()) for path in cells]
    assert {cell["seed"] for cell in payloads} == {0, 1, 2, 3, 4}
    assert len(
        {
            (cell["dataset"], cell["method"], cell["seed"])
            for cell in payloads
        }
    ) == 50
    assert all(
        f"seed_{cell['seed']}" in cell["output_check"] for cell in payloads
    )


def test_lowk_builder_preserves_claimed_bundle_and_deletes_only_nonrunning(
    tmp_path: Path,
):
    from scripts.dispatch.build_issue_149_lowk_cells import (
        build,
        remove_nonrunning_bundled_cells,
    )
    from scripts.dispatch.claim import init_dispatch_dirs

    root = tmp_path / "lowk-dispatch"
    init_dispatch_dirs(root)
    (root / "cancelled").mkdir()
    bundled = {
        "dataset": "hotpotqa_memmap",
        "method": "act_vit_prefix_multik_lowk",
        "seed": "0,1,2,3,4",
    }
    for state in ("pending", "done", "failed", "cancelled"):
        (root / state / f"legacy_{state}.json").write_text(
            json.dumps(bundled)
        )
    claimed = root / "claimed" / "live-worker"
    claimed.mkdir()
    active_path = claimed / "legacy_active.json"
    active_path.write_text(json.dumps(bundled))

    removed = remove_nonrunning_bundled_cells(root)
    assert len(removed) == 4
    assert active_path.exists()
    # Five active HotpotQA ACT-ViT seeds are suppressed; every other
    # dataset/method/seed cell is emitted.
    assert build(root) == 45
    assert not any(
        "hotpotqa_memmap__act_vit_prefix_multik_lowk" in path.name
        for path in (root / "pending").glob("*.json")
    )


def test_lowk_seed_completion_is_method_specific(tmp_path: Path):
    from scripts.dispatch.build_issue_149_lowk_cells import _seed_is_complete

    act_run = tmp_path / "runs" / "act" / "seed_0"
    act_run.mkdir(parents=True)
    (act_run / "eval_metrics.json").write_text("{}")
    assert _seed_is_complete(
        tmp_path,
        Path("runs/act/seed_0"),
        "act_vit_prefix_multik_lowk",
    )

    contrastive_run = tmp_path / "runs" / "contrastive" / "seed_0"
    contrastive_run.mkdir(parents=True)
    (contrastive_run / "eval_metrics.json").write_text("{}")
    method = "contrastive_logprob_recon_prefix_mixed_lowk"
    assert not _seed_is_complete(
        tmp_path, Path("runs/contrastive/seed_0"), method
    )
    (contrastive_run / "predictions.csv").write_text("score\n")
    assert _seed_is_complete(
        tmp_path, Path("runs/contrastive/seed_0"), method
    )
    (contrastive_run / "run_error.json").write_text("{}")
    assert not _seed_is_complete(
        tmp_path, Path("runs/contrastive/seed_0"), method
    )


def test_lowk_methods_share_exact_prefix_training_support():
    root = Path(__file__).resolve().parents[1]
    methods = (
        "act_vit_prefix_multik_lowk",
        "contrastive_logprob_recon_prefix_mixed_lowk",
    )
    configs = [
        json.loads((root / "configs" / "methods" / f"{name}.json").read_text())
        for name in methods
    ]
    expected = [1, 4, 8, 16, 32, 48, 64]
    assert all(cfg["training"]["prefix_min_tokens"] == 1 for cfg in configs)
    assert all(cfg["training"]["prefix_min_gap"] == 1 for cfg in configs)
    assert all(
        cfg["training"]["prefix_sampling_lengths"] == expected for cfg in configs
    )
    assert all(
        cfg["evaluation"]["eval_prefix_lengths"] == expected for cfg in configs
    )


def test_standard_lowk_selects_fixed_k1_validation_knn_auroc():
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads(
        (
            root
            / "configs/methods/contrastive_logprob_recon_prefix_mixed_lowk.json"
        ).read_text()
    )
    training = cfg["training"]
    assert training["select_on_val"] is True
    assert training["checkpoint_selection_metric"] == "knn_auroc"
    assert training["validation_knn"] == {
        "k": 50,
        "metric": "euclidean",
        "calibrate_k": False,
        "train_selection": "all",
        "max_train_size": 200000,
        "prefix_length": 1,
    }
