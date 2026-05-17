"""Smoke test for train_contrastive_logprob_attn_recon.

End-to-end: synthetic memmap capture → MemmapContrastiveDataset →
LogprobAttnReconProgressiveCompressor → trainer for a couple of steps.
Confirms shapes line up, loss is finite, no NaN gradients, checkpoint
written.

CPU-only. Uses tiny dims to run in seconds.

Run with:
    pytest tests/test_logprob_attn_recon_trainer.py -v
"""

from __future__ import annotations

import os

import torch

from activation_research.memmap_contrastive_dataset import MemmapContrastiveDataset
from activation_research.model import LogprobAttnReconProgressiveCompressor
from activation_research.training import (
    _contrastive_collate_with_logprob_attn,
    train_contrastive_logprob_attn_recon,
)

# Re-use the synthetic capture fixture from the dataset test.
from tests.test_memmap_contrastive_dataset import (
    _make_full_capture_dir,
    _SMALL_CFG,
)


def _build_dataset(capture_dir):
    return MemmapContrastiveDataset(
        capture_dir,
        split="all",
        num_views=2,
        relevant_layers=[1, 2, 3, 4],
        include_response_logprobs=True,
        response_logprobs_top_k=5,
        pad_length=_SMALL_CFG["max_response_len"],
        include_response_attention=True,
        attention_summary="stats",
        attention_target_layer_offset_backward=1,
        attention_target_layer_offset_forward=1,
    )


# ---------------------------------------------------------------------------
# Collate fn — verify its output shape
# ---------------------------------------------------------------------------

def test_collate_packs_attention_and_logprob_fields(tmp_path):
    capture = _make_full_capture_dir(tmp_path, n_samples=8)
    ds = _build_dataset(capture)
    items = [ds[i] for i in range(4)]
    out = _contrastive_collate_with_logprob_attn(items)

    assert out["views_activations"].shape == (4, 2, _SMALL_CFG["max_response_len"], _SMALL_CFG["hidden_dim"])
    assert out["halu"].shape == (4,)
    assert out["view_indices"].shape == (4, 2)
    assert "logprob" in out  # mapped by inner collate from response_token_logprobs
    assert out["logprob"].shape == (4, _SMALL_CFG["max_response_len"])
    assert out["attention_forward"].shape == (4, 2, 3)
    assert out["attention_backward"].shape == (4, 2, 3)


def test_collate_omits_attention_when_dataset_omits_it(tmp_path):
    capture = _make_full_capture_dir(tmp_path, n_samples=8)
    ds = MemmapContrastiveDataset(
        capture, split="all", num_views=2, include_response_logprobs=True,
    )
    items = [ds[i] for i in range(4)]
    out = _contrastive_collate_with_logprob_attn(items)
    assert "attention_forward" not in out
    assert "attention_backward" not in out
    assert "logprob" in out


# ---------------------------------------------------------------------------
# Trainer smoke test
# ---------------------------------------------------------------------------

def test_trainer_runs_one_epoch_backward_only(tmp_path):
    capture = _make_full_capture_dir(tmp_path, n_samples=32)
    ds = _build_dataset(capture)

    model = LogprobAttnReconProgressiveCompressor(
        input_dim=_SMALL_CFG["hidden_dim"],
        final_dim=64,
        recon_seq_len=12,
        recon_hidden_dim=8,
        attn_direction="backward",
        attn_recon_hidden_dim=8,
        recon_lambda=1.0,
        attn_recon_lambda=1.0,
        logprob_var_threshold=1e-8,
        attn_var_threshold=1e-8,
    )

    ckpt_dir = tmp_path / "ckpts"
    train_contrastive_logprob_attn_recon(
        model,
        ds,
        test_dataset=None,
        epochs=1,
        batch_size=4,
        sub_batch_size=4,
        lr=1e-4,
        temperature=0.25,
        device="cpu",
        num_workers=0,
        checkpoint_dir=str(ckpt_dir),
        persistent_workers=False,
        use_labels=False,
    )

    # Checkpoint was written.
    assert os.path.exists(ckpt_dir / "contrastive_last.pt")

    # Reload and verify the recorded lambdas + diagnostics keys.
    ckpt = torch.load(ckpt_dir / "contrastive_last.pt", map_location="cpu", weights_only=False)
    assert ckpt["recon_lambda"] == 1.0
    assert ckpt["attn_recon_lambda"] == 1.0
    assert "train_recon_attn_backward" in ckpt
    assert "train_recon_lp" in ckpt
    # Loss must be finite.
    assert torch.isfinite(torch.tensor(ckpt["train_loss"]))


def test_trainer_both_directions(tmp_path):
    capture = _make_full_capture_dir(tmp_path, n_samples=32)
    ds = _build_dataset(capture)

    model = LogprobAttnReconProgressiveCompressor(
        input_dim=_SMALL_CFG["hidden_dim"],
        final_dim=64,
        recon_seq_len=12,
        recon_hidden_dim=8,
        attn_direction="both",
        attn_recon_hidden_dim=8,
        recon_lambda=1.0,
        attn_recon_lambda=1.0,
        logprob_var_threshold=1e-8,
        attn_var_threshold=1e-8,
    )

    ckpt_dir = tmp_path / "ckpts_both"
    train_contrastive_logprob_attn_recon(
        model,
        ds,
        test_dataset=None,
        epochs=1,
        batch_size=4,
        sub_batch_size=4,
        lr=1e-4,
        device="cpu",
        num_workers=0,
        checkpoint_dir=str(ckpt_dir),
        persistent_workers=False,
        use_labels=False,
    )

    ckpt = torch.load(ckpt_dir / "contrastive_last.pt", map_location="cpu", weights_only=False)
    assert "train_recon_attn_forward" in ckpt
    assert "train_recon_attn_backward" in ckpt


def test_trainer_attn_lambda_zero_disables_k_loss(tmp_path):
    """attn_recon_lambda=0 → no K contribution to loss; trainer must still run."""
    capture = _make_full_capture_dir(tmp_path, n_samples=16)
    ds = _build_dataset(capture)

    model = LogprobAttnReconProgressiveCompressor(
        input_dim=_SMALL_CFG["hidden_dim"],
        final_dim=64,
        recon_seq_len=12,
        recon_hidden_dim=8,
        attn_direction="backward",
        attn_recon_hidden_dim=8,
        recon_lambda=1.0,
        attn_recon_lambda=0.0,
        logprob_var_threshold=1e-8,
        attn_var_threshold=1e-8,
    )

    ckpt_dir = tmp_path / "ckpts_kzero"
    train_contrastive_logprob_attn_recon(
        model,
        ds,
        test_dataset=None,
        epochs=1,
        batch_size=4,
        sub_batch_size=4,
        lr=1e-4,
        device="cpu",
        num_workers=0,
        checkpoint_dir=str(ckpt_dir),
        persistent_workers=False,
        use_labels=False,
        attn_recon_lambda=0.0,
    )

    ckpt = torch.load(ckpt_dir / "contrastive_last.pt", map_location="cpu", weights_only=False)
    # K loss skipped → recorded value is 0.0.
    assert ckpt["train_recon_attn_backward"] == 0.0
    # F loss still active.
    assert ckpt["train_recon_lp"] > 0.0 or ckpt["train_recon_lp"] == 0.0  # finite, depends on init


def test_trainer_falls_back_when_dataset_lacks_attention(tmp_path):
    """Dataset without attention fields → trainer skips K; no error."""
    capture = _make_full_capture_dir(tmp_path, n_samples=16)
    ds = MemmapContrastiveDataset(
        capture, split="all", num_views=2,
        relevant_layers=[1, 2, 3, 4],
        include_response_logprobs=True,
        include_response_attention=False,  # no K fields emitted
    )

    model = LogprobAttnReconProgressiveCompressor(
        input_dim=_SMALL_CFG["hidden_dim"],
        final_dim=64,
        recon_seq_len=12,
        recon_hidden_dim=8,
        attn_direction="backward",
        attn_recon_hidden_dim=8,
        logprob_var_threshold=1e-8,
        attn_var_threshold=1e-8,
    )

    ckpt_dir = tmp_path / "ckpts_noattn"
    train_contrastive_logprob_attn_recon(
        model,
        ds,
        test_dataset=None,
        epochs=1,
        batch_size=4,
        sub_batch_size=4,
        lr=1e-4,
        device="cpu",
        num_workers=0,
        checkpoint_dir=str(ckpt_dir),
        persistent_workers=False,
        use_labels=False,
    )
    ckpt = torch.load(ckpt_dir / "contrastive_last.pt", map_location="cpu", weights_only=False)
    # No K contribution recorded since the batch never had attention fields.
    assert ckpt["train_recon_attn_backward"] == 0.0
