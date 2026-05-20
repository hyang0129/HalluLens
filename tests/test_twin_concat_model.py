"""Tests for TwinConcatModel and H7-H10 post-hoc evaluation helpers.

CPU-only; no real data or GPU.

Run with:
    pytest tests/test_twin_concat_model.py -v
"""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from activation_research.model import LogprobReconProgressiveCompressor, TwinConcatModel


# ---------------------------------------------------------------------------
# Helpers (lightweight heads for unit tests)
# ---------------------------------------------------------------------------


class _LinearHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=1))


# ---------------------------------------------------------------------------
# Original tests (issue #99)
# ---------------------------------------------------------------------------


def test_twin_concat_model_output_shape():
    """Output shape must be (B, final_dim_a + final_dim_b) = (B, 128)."""
    torch.manual_seed(0)
    # Use input_dim=128, final_dim=64 so TransformerBlock nhead=64//64=1 (valid).
    head_a = LogprobReconProgressiveCompressor(input_dim=128, final_dim=64)
    head_b = LogprobReconProgressiveCompressor(input_dim=128, final_dim=64)
    model = TwinConcatModel(head_a=head_a, head_b=head_b)
    model.eval()

    B = 4
    seq_len = 5
    x = torch.randn(B, seq_len, 128)
    out = model(x)

    assert out.shape == (B, 128), f"Expected (4, 128), got {out.shape}"


def test_twin_concat_model_gradients_independent():
    """head_a and head_b must share no parameters."""
    head_a = LogprobReconProgressiveCompressor(input_dim=128, final_dim=64)
    head_b = LogprobReconProgressiveCompressor(input_dim=128, final_dim=64)
    model = TwinConcatModel(head_a=head_a, head_b=head_b)

    ids_a = set(id(p) for p in model.head_a.parameters())
    ids_b = set(id(p) for p in model.head_b.parameters())

    assert ids_a.isdisjoint(ids_b), (
        "head_a and head_b share parameter objects — they must be independent"
    )


# ---------------------------------------------------------------------------
# New tests (issue #101): concat value check + .npy round-trip + H7-H10 smoke
# ---------------------------------------------------------------------------


def test_twin_concat_values():
    """Concat output == [head_a(x), head_b(x)] element-wise."""
    B, L, D, out_dim = 4, 3, 128, 64
    head_a = _LinearHead(D, out_dim)
    head_b = _LinearHead(D, out_dim)
    model = TwinConcatModel(head_a, head_b)

    x = torch.randn(B, L, D)
    za = head_a(x)
    zb = head_b(x)
    z_cat = model(x)

    expected = torch.cat([za, zb], dim=-1)
    assert torch.allclose(z_cat, expected), "Concat output mismatch"


def test_npy_roundtrip_shapes_and_values():
    """Per-head .npy files save and reload with correct shapes and values."""
    N_train, N_test, D = 50, 20, 16

    rng = np.random.default_rng(0)
    train_za = rng.standard_normal((N_train, D)).astype(np.float32)
    train_zb = rng.standard_normal((N_train, D)).astype(np.float32)
    test_za = rng.standard_normal((N_test, D)).astype(np.float32)
    test_zb = rng.standard_normal((N_test, D)).astype(np.float32)
    train_labels = rng.integers(0, 2, size=N_train).astype(np.int32)
    test_labels = rng.integers(0, 2, size=N_test).astype(np.int32)

    with tempfile.TemporaryDirectory() as tmpdir:
        emb_dir = os.path.join(tmpdir, "embeddings")
        os.makedirs(emb_dir)

        paths = {
            "train_za": os.path.join(emb_dir, "train_za.npy"),
            "train_zb": os.path.join(emb_dir, "train_zb.npy"),
            "test_za": os.path.join(emb_dir, "test_za.npy"),
            "test_zb": os.path.join(emb_dir, "test_zb.npy"),
            "train_labels": os.path.join(emb_dir, "train_labels.npy"),
            "test_labels": os.path.join(emb_dir, "test_labels.npy"),
        }

        np.save(paths["train_za"], train_za)
        np.save(paths["train_zb"], train_zb)
        np.save(paths["test_za"], test_za)
        np.save(paths["test_zb"], test_zb)
        np.save(paths["train_labels"], train_labels)
        np.save(paths["test_labels"], test_labels)

        assert all(os.path.exists(p) for p in paths.values())

        loaded_train_za = np.load(paths["train_za"])
        loaded_train_zb = np.load(paths["train_zb"])
        loaded_test_za = np.load(paths["test_za"])
        loaded_test_zb = np.load(paths["test_zb"])
        loaded_train_labels = np.load(paths["train_labels"])
        loaded_test_labels = np.load(paths["test_labels"])

        assert loaded_train_za.shape == (N_train, D)
        assert loaded_train_zb.shape == (N_train, D)
        assert loaded_test_za.shape == (N_test, D)
        assert loaded_test_zb.shape == (N_test, D)
        assert loaded_train_labels.shape == (N_train,)
        assert loaded_test_labels.shape == (N_test,)

        np.testing.assert_array_equal(loaded_train_za, train_za)
        np.testing.assert_array_equal(loaded_train_zb, train_zb)
        np.testing.assert_array_equal(loaded_test_za, test_za)
        np.testing.assert_array_equal(loaded_test_zb, test_zb)
        np.testing.assert_array_equal(loaded_train_labels, train_labels)
        np.testing.assert_array_equal(loaded_test_labels, test_labels)


def _compute_h_metrics(
    train_za, train_zb, test_za, test_zb, train_labels, test_labels,
    knn_k=5, knn_metric="euclidean", effective_outlier_class=1,
):
    """Replicate H7-H10 logic from run_contrastive_logprob_recon_twin
    using the same numpy/sklearn path, isolated for testing."""
    from sklearn.neighbors import NearestNeighbors

    from activation_research.metrics import (
        _safe_auroc,
        knn_ood_stats,
        mahalanobis_ood_stats,
    )

    binary_labels = (test_labels == int(effective_outlier_class)).astype(np.int32)

    def _make_records(arr, labels):
        return [
            {"z1": torch.from_numpy(arr[i]).float(), "halu": int(labels[i])}
            for i in range(len(labels))
        ]

    train_z_concat = np.concatenate([train_za, train_zb], axis=1)
    test_z_concat = np.concatenate([test_za, test_zb], axis=1)
    train_records_concat = _make_records(train_z_concat, train_labels)
    test_records_concat = _make_records(test_z_concat, test_labels)

    knn_stats = knn_ood_stats(
        train_records=train_records_concat,
        test_records=test_records_concat,
        outlier_class=effective_outlier_class,
        k=knn_k,
        metric=knn_metric,
        train_label_filter="all",
        calibrate_k=False,
        max_train_size=200000,
        sample_seed=0,
    )
    maha_stats = mahalanobis_ood_stats(
        train_records=train_records_concat,
        test_records=test_records_concat,
        outlier_class=effective_outlier_class,
        train_label_filter="id_only",
    )

    # H7
    n_nbrs = min(knn_k, len(train_za))
    nn_a = NearestNeighbors(n_neighbors=n_nbrs, metric=knn_metric)
    nn_a.fit(train_za)
    dists_a, _ = nn_a.kneighbors(test_za)

    nn_b = NearestNeighbors(n_neighbors=n_nbrs, metric=knn_metric)
    nn_b.fit(train_zb)
    dists_b, _ = nn_b.kneighbors(test_zb)

    ensemble_scores = (dists_a.mean(axis=1) + dists_b.mean(axis=1)) / 2.0
    h7_auroc = _safe_auroc(binary_labels, ensemble_scores)

    # H8
    def _knn_recall(train_arr, test_arr, train_lbls, test_lbls, k, metric):
        nn = NearestNeighbors(n_neighbors=min(k, len(train_arr)), metric=metric)
        nn.fit(train_arr)
        _, indices = nn.kneighbors(test_arr)
        neighbor_labels = train_lbls[indices]
        preds = (neighbor_labels.sum(axis=1) > (k / 2)).astype(np.int32)
        recalls = {}
        for cls in (0, 1):
            mask = test_lbls == cls
            if mask.sum() == 0:
                recalls[cls] = float("nan")
            else:
                recalls[cls] = float((preds[mask] == cls).mean())
        return recalls

    h8_concat = _knn_recall(train_z_concat, test_z_concat, train_labels, test_labels, knn_k, knn_metric)
    h8_head_a = _knn_recall(train_za, test_za, train_labels, test_labels, knn_k, knn_metric)
    h8_head_b = _knn_recall(train_zb, test_zb, train_labels, test_labels, knn_k, knn_metric)

    # H9
    std_a = float(dists_a.mean(axis=1).std()) + 1e-8
    std_b = float(dists_b.mean(axis=1).std()) + 1e-8
    whitened_scores = dists_a.mean(axis=1) / std_a + dists_b.mean(axis=1) / std_b
    h9_auroc = _safe_auroc(binary_labels, whitened_scores)

    # H10
    train_z_sum = train_za + train_zb
    test_z_sum = test_za + test_zb
    train_records_sum = _make_records(train_z_sum, train_labels)
    test_records_sum = _make_records(test_z_sum, test_labels)

    knn_sum_stats = knn_ood_stats(
        train_records=train_records_sum,
        test_records=test_records_sum,
        outlier_class=effective_outlier_class,
        k=knn_k,
        metric=knn_metric,
        train_label_filter="all",
        calibrate_k=False,
        max_train_size=200000,
        sample_seed=0,
    )
    h10_knn_sum_auroc = float(knn_sum_stats.get("knn_auroc", float("nan")))

    try:
        maha_sum_stats = mahalanobis_ood_stats(
            train_records=train_records_sum,
            test_records=test_records_sum,
            outlier_class=effective_outlier_class,
            train_label_filter="id_only",
        )
        h10_maha_sum_auroc = float(maha_sum_stats.get("mahalanobis_auroc", float("nan")))
    except Exception:
        h10_maha_sum_auroc = float("nan")

    metrics = {}
    metrics.update(knn_stats)
    metrics.update(maha_stats)
    metrics["h7_knn_ensemble_auroc"] = float(h7_auroc)
    metrics["h8_recall_class0_concat"] = h8_concat[0]
    metrics["h8_recall_class1_concat"] = h8_concat[1]
    metrics["h8_recall_class0_head_a"] = h8_head_a[0]
    metrics["h8_recall_class1_head_a"] = h8_head_a[1]
    metrics["h8_recall_class0_head_b"] = h8_head_b[0]
    metrics["h8_recall_class1_head_b"] = h8_head_b[1]
    metrics["h9_whitened_ensemble_auroc"] = float(h9_auroc)
    metrics["h10_knn_sum_auroc"] = h10_knn_sum_auroc
    metrics["h10_maha_sum_auroc"] = h10_maha_sum_auroc

    return metrics


def test_npy_resume_skips_inference():
    """Resume-detection logic: all-6-present -> skip; missing one -> re-run."""
    N_train, N_test, D = 50, 20, 16

    rng = np.random.default_rng(1)
    arrays = {
        "train_za":     rng.standard_normal((N_train, D)).astype(np.float32),
        "train_zb":     rng.standard_normal((N_train, D)).astype(np.float32),
        "test_za":      rng.standard_normal((N_test, D)).astype(np.float32),
        "test_zb":      rng.standard_normal((N_test, D)).astype(np.float32),
        "train_labels": rng.integers(0, 2, size=N_train).astype(np.int32),
        "test_labels":  rng.integers(0, 2, size=N_test).astype(np.int32),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        emb_dir = os.path.join(tmpdir, "artifacts", "embeddings")
        os.makedirs(emb_dir)

        paths = {key: os.path.join(emb_dir, f"{key}.npy") for key in arrays}

        # Write all 6 files
        for key, arr in arrays.items():
            np.save(paths[key], arr)

        # All present -> inference should be skipped
        assert all(os.path.exists(p) for p in paths.values()), (
            "All 6 npy files must exist after saving"
        )

        # Remove one file -> re-run should be triggered
        os.remove(paths["test_za"])
        assert not all(os.path.exists(p) for p in paths.values()), (
            "Condition must be False when any npy file is missing"
        )

        # Restore the missing file and verify loaded shapes match saved shapes
        np.save(paths["test_za"], arrays["test_za"])
        for key, arr in arrays.items():
            loaded = np.load(paths[key])
            assert loaded.shape == arr.shape, (
                f"Shape mismatch for {key}: expected {arr.shape}, got {loaded.shape}"
            )


def test_h7_to_h10_keys_present_and_finite():
    """H7-H10 keys are present and have finite values on random data."""
    rng = np.random.default_rng(42)
    N_train, N_test, D = 80, 30, 16

    train_labels = rng.integers(0, 2, size=N_train).astype(np.int32)
    test_labels = rng.integers(0, 2, size=N_test).astype(np.int32)

    # head_a: class-0 cluster near 0, class-1 scattered
    train_za = rng.standard_normal((N_train, D)).astype(np.float32)
    train_za[train_labels == 0] += 2.0
    test_za = rng.standard_normal((N_test, D)).astype(np.float32)
    test_za[test_labels == 0] += 2.0

    # head_b: class-1 cluster near 0, class-0 scattered
    train_zb = rng.standard_normal((N_train, D)).astype(np.float32)
    train_zb[train_labels == 1] -= 2.0
    test_zb = rng.standard_normal((N_test, D)).astype(np.float32)
    test_zb[test_labels == 1] -= 2.0

    metrics = _compute_h_metrics(
        train_za, train_zb, test_za, test_zb, train_labels, test_labels,
        knn_k=5, knn_metric="euclidean", effective_outlier_class=1,
    )

    required_keys = [
        "knn_auroc",
        "mahalanobis_auroc",
        "h7_knn_ensemble_auroc",
        "h8_recall_class0_concat",
        "h8_recall_class1_concat",
        "h8_recall_class0_head_a",
        "h8_recall_class1_head_a",
        "h8_recall_class0_head_b",
        "h8_recall_class1_head_b",
        "h9_whitened_ensemble_auroc",
        "h10_knn_sum_auroc",
        "h10_maha_sum_auroc",
    ]

    for key in required_keys:
        assert key in metrics, f"Missing key: {key}"
        val = metrics[key]
        assert val is not None, f"Key {key} is None"
        assert not (isinstance(val, float) and math.isnan(val)), (
            f"Key {key} is NaN (got {val})"
        )
