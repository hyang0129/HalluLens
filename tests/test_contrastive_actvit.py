"""Unit tests for Contrastive ACT-ViT (issue #134).

Covers the model (param budget <20M, shapes, recon head, gradient flow) and the
grid dataset (flattened views, four augmentations) via a tiny synthetic capture.
Run: ``pytest tests/test_contrastive_actvit.py`` (needs torch + numpy).
"""
import json
from pathlib import Path

import numpy as np
import torch

from activation_research.act_vit import ContrastiveACTViT
from activation_research.contrastive_actvit_dataset import (
    ContrastiveACTViTDataset,
    make_cav_augment,
)


# --------------------------------------------------------------------------- model


def test_param_budget_default_dims():
    """Default (act_vit-size) config must be ~act_vit size and well under 20M."""
    m = ContrastiveACTViT(n_layers=32, n_tokens=64, input_dim=4096)
    n = sum(p.numel() for p in m.parameters())
    assert n < 20_000_000, f"{n/1e6:.2f}M exceeds 20M budget"
    assert 3_000_000 < n < 10_000_000, f"{n/1e6:.2f}M not near act_vit's ~5.5M"


def test_shapes_grad_and_recon():
    torch.manual_seed(0)
    L, N, D, B, FINAL, RSEQ = 4, 8, 32, 3, 16, 6
    m = ContrastiveACTViT(
        n_layers=L, n_tokens=N, input_dim=D, final_dim=FINAL,
        recon_seq_len=RSEQ, recon_hidden_dim=16,
        L_p=4, N_p=20, patch_h=2, patch_w=5, d_adapter=16, d_model=64, num_heads=4, depth=1,
    )
    x = torch.randn(B, L * N, D)               # flattened grid
    z = m(x)
    assert z.shape == (B, FINAL)
    z2, lp = m.forward_with_recon(x)
    assert z2.shape == (B, FINAL) and lp.shape == (B, RSEQ)
    loss, diag = m.recon_loss(lp, torch.randn(B, 12))   # arbitrary target len → resampled
    assert torch.isfinite(loss) and isinstance(diag, dict)
    # gradient reaches encoder (adapter), proj head, and decoder
    m.zero_grad(set_to_none=True)
    (m(x).sum()).backward()
    g = lambda ps: float(sum((p.grad.norm() ** 2) for p in ps if p.grad is not None) ** 0.5)
    assert g(m.encoder.adapter.parameters()) > 0, "encoder got no gradient"
    assert g(m.proj.parameters()) > 0, "proj head got no gradient"


def test_forward_accepts_layer_idx_kwarg():
    """Trainer/eval call sites may pass layer_idx=None; forward must accept+ignore it."""
    m = ContrastiveACTViT(n_layers=4, n_tokens=8, input_dim=32, final_dim=16,
                          d_model=64, d_adapter=16, depth=1, L_p=4, N_p=20, patch_h=2, patch_w=5)
    x = torch.randn(2, 32, 32)
    assert m(x, layer_idx=None).shape == (2, 16)


def test_normalize_output_unit_sphere():
    """normalize_output=True must return L2-unit embeddings; default keeps raw norms."""
    kw = dict(n_layers=4, n_tokens=8, input_dim=32, final_dim=16,
              d_model=64, d_adapter=16, depth=1, L_p=4, N_p=20, patch_h=2, patch_w=5)
    x = torch.randn(5, 32, 32)
    m_norm = ContrastiveACTViT(normalize_output=True, **kw)
    z = m_norm(x)
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # recon path uses the same (normalized) embedding without crashing
    z2, recon = m_norm.forward_with_recon(x)
    assert torch.allclose(z2.norm(dim=-1), torch.ones_like(norms), atol=1e-5)
    # default (legacy) does NOT normalize — at least one norm clearly off unit
    m_raw = ContrastiveACTViT(**kw)
    assert not torch.allclose(m_raw(x).norm(dim=-1), torch.ones(5), atol=1e-2)


# --------------------------------------------------------------------------- dataset


def _make_capture(tmp: Path, n=8, n_layers=3, max_resp=4, hidden=8):
    # Raw binary (no .npy header) — the dataset opens these via np.memmap(shape=...).
    n_layers_p1 = n_layers + 1
    np.random.randn(n, n_layers_p1, max_resp, hidden).astype(np.float16).tofile(
        tmp / "response_activations.npy")
    np.array([max_resp] * n, dtype=np.int32).tofile(tmp / "response_len.npy")
    np.random.randn(n, max_resp).astype(np.float32).tofile(tmp / "response_token_logprobs.npy")
    with (tmp / "config.json").open("w") as fh:
        json.dump({"n_samples": n, "num_layers": n_layers,
                   "max_response_len": max_resp, "hidden_dim": hidden}, fh)
    with (tmp / "meta.jsonl").open("w") as fh:
        for i in range(n):
            fh.write(json.dumps({"hallucinated": bool(i % 2), "prompt_hash": f"h{i}"}) + "\n")


def test_dataset_views_and_augmentations(tmp_path):
    _make_capture(tmp_path)
    for aug in ("noise", "token_crop", "layer_band", "patch_mask"):
        ds = ContrastiveACTViTDataset(
            tmp_path, indices=list(range(8)), num_views=2, view_aug=aug,
            relevant_layers=None, seed=0, patch_h=2, patch_w=2,
        )
        assert ds.n_layers == 3 and ds.n_tokens == 4
        item = ds[0]
        v = item["views_activations"]
        assert v.shape == (2, ds.n_layers * ds.n_tokens, ds.hidden_dim), aug
        assert item["halu"].item() in (0, 1)
        assert item["logprob"].shape == (4,)
        # Augmentation must introduce view variation. Discrete-choice augs
        # (layer_band) can coincide on a single sample in a tiny synthetic
        # space, so require variation across at least one of the samples.
        any_diff = any(
            not torch.equal(ds[i]["views_activations"][0], ds[i]["views_activations"][1])
            for i in range(len(ds))
        )
        assert any_diff, f"{aug}: all views identical across samples"


def test_gpu_augment_factory():
    """make_cav_augment: preserves shape and diversifies the two views per design."""
    B, V, L, N, D = 4, 2, 6, 8, 5
    base = torch.randn(B, V, L * N, D)
    for aug in ("noise", "token_crop", "layer_band", "patch_mask"):
        fn = make_cav_augment(aug, L, N, keep_frac=0.6, mask_frac=0.5, patch_h=2, patch_w=2)
        # feed identical views (as the dataset's raw mode does) → must come out different
        x = base.clone()
        out = fn(x, torch.zeros(B, dtype=torch.long))
        assert out.shape == (B, V, L * N, D), aug
        assert not torch.equal(out[:, 0], out[:, 1]), f"{aug}: views not diversified"
    # raw is identity
    raw = make_cav_augment("raw", L, N)
    x = base.clone()
    assert torch.equal(raw(x, None), x)


def test_dataset_raw_mode_returns_copies(tmp_path):
    _make_capture(tmp_path)
    ds = ContrastiveACTViTDataset(tmp_path, indices=list(range(8)), num_views=2, view_aug="raw", seed=0)
    v = ds[0]["views_activations"]
    assert v.shape == (2, ds.n_layers * ds.n_tokens, ds.hidden_dim)
    assert torch.equal(v[0], v[1]), "raw mode should return identical copies (GPU augment diversifies)"


def test_dataset_hashkey_map(tmp_path):
    _make_capture(tmp_path)
    hmap = {i: f"h{i}" for i in range(8)}
    ds = ContrastiveACTViTDataset(tmp_path, indices=list(range(8)), num_views=1,
                                  hashkey_map=hmap, seed=0)
    assert ds[3]["hashkey"] == "h3"


if __name__ == "__main__":
    test_param_budget_default_dims()
    test_shapes_grad_and_recon()
    test_forward_accepts_layer_idx_kwarg()
    print("model tests passed")
