"""Tests for activation_research/prefix_views.py (issue #149).

Covers the two invariants the experiment's validity rests on:
  1. sampled k never pins to the evaluated grid (non-circular evaluation),
  2. slicing preserves prefix content exactly and never leaks future tokens.
"""

from __future__ import annotations

import pytest
import torch

from activation_research.prefix_views import (
    EVAL_PREFIX_LENGTHS,
    PrefixPairSampler,
    PrefixViewSpec,
    apply_prefix_views,
    resolve_eval_prefixes,
)


# --------------------------------------------------------------------- #
# PrefixViewSpec
# --------------------------------------------------------------------- #
def test_spec_rejects_empty_and_negative():
    with pytest.raises(ValueError):
        PrefixViewSpec(())
    with pytest.raises(ValueError):
        PrefixViewSpec((16, -1))


def test_spec_uniformity_flag():
    assert PrefixViewSpec((64, 64)).is_uniform
    assert not PrefixViewSpec((16, 32)).is_uniform
    assert PrefixViewSpec((16, 32)).num_views == 2


# --------------------------------------------------------------------- #
# Sampler modes
# --------------------------------------------------------------------- #
def test_layer_only_mode_uses_full_prefix_everywhere():
    """layer_only must reproduce current behaviour exactly — full sequence."""
    s = PrefixPairSampler(mode="layer_only", num_views=2, max_prefix=64, seed=0)
    for _ in range(50):
        spec = s.sample()
        assert spec.prefix_lens == (64, 64)
        assert spec.is_uniform


def test_mixed_mode_respects_bounds_and_min_gap():
    s = PrefixPairSampler(
        mode="mixed", num_views=2, max_prefix=64, min_prefix=8, min_gap=8, seed=0
    )
    for _ in range(500):
        lo, hi = s.sample().prefix_lens
        assert 8 <= lo < hi <= 64
        assert hi - lo >= 8


def test_sampled_k_is_not_pinned_to_eval_grid():
    """The whole point of sampling k: training must not privilege 16/32/48/64.

    If every sampled pair landed on the eval grid, an AUROC-at-k=16 result
    could not be distinguished from overfitting to that k.
    """
    s = PrefixPairSampler(mode="mixed", num_views=2, max_prefix=64, seed=7)
    off_grid = 0
    for _ in range(500):
        lens = s.sample().prefix_lens
        if any(k not in EVAL_PREFIX_LENGTHS for k in lens):
            off_grid += 1
    # Overwhelmingly off-grid; the grid is 5 of 64 possible values.
    assert off_grid > 450


def test_gap_is_independent_of_base_prefix():
    """gap must not be predictable from k_lo, else gap size is a shortcut."""
    s = PrefixPairSampler(
        mode="mixed", num_views=2, max_prefix=64, min_prefix=8, min_gap=4, seed=3
    )
    pairs = [s.sample().prefix_lens for _ in range(2000)]
    los = torch.tensor([p[0] for p in pairs], dtype=torch.float64)
    gaps = torch.tensor([p[1] - p[0] for p in pairs], dtype=torch.float64)
    # Bounded sampling induces a mild negative dependence (a large k_lo leaves
    # less room for a large gap); assert it stays weak rather than absent.
    corr = torch.corrcoef(torch.stack([los, gaps]))[0, 1].abs().item()
    assert corr < 0.75


def test_multi_view_interpolates_between_extremes():
    s = PrefixPairSampler(mode="mixed", num_views=4, max_prefix=64, seed=1)
    lens = s.sample().prefix_lens
    assert len(lens) == 4
    assert list(lens) == sorted(lens)
    assert lens[0] >= 8 and lens[-1] <= 64


def test_prefix_only_mode_still_varies_k():
    s = PrefixPairSampler(mode="prefix_only", num_views=2, max_prefix=64, seed=2)
    specs = {s.sample().prefix_lens for _ in range(100)}
    assert len(specs) > 1
    assert all(not PrefixViewSpec(p).is_uniform for p in specs)


def test_sampler_is_reproducible_by_seed():
    a = PrefixPairSampler(mode="mixed", max_prefix=64, seed=42)
    b = PrefixPairSampler(mode="mixed", max_prefix=64, seed=42)
    assert [a.sample().prefix_lens for _ in range(20)] == [
        b.sample().prefix_lens for _ in range(20)
    ]


def test_sampler_does_not_consume_global_random_state():
    """View geometry must not perturb the layer sampler's global RNG stream."""
    import random as _random

    _random.seed(123)
    before = [_random.random() for _ in range(3)]

    _random.seed(123)
    s = PrefixPairSampler(mode="mixed", max_prefix=64, seed=9)
    for _ in range(50):
        s.sample()
    after = [_random.random() for _ in range(3)]
    assert before == after


def test_sampler_rejects_impossible_geometry():
    with pytest.raises(ValueError):
        PrefixPairSampler(mode="mixed", max_prefix=16, min_prefix=12, min_gap=8)
    with pytest.raises(ValueError):
        PrefixPairSampler(mode="bogus", max_prefix=64)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PrefixPairSampler(mode="mixed", max_prefix=64, min_prefix=0)


def test_eval_spec_is_uniform():
    s = PrefixPairSampler(mode="mixed", num_views=2, max_prefix=64, seed=0)
    spec = s.eval_spec(16)
    assert spec.prefix_lens == (16, 16)
    assert spec.is_uniform


# --------------------------------------------------------------------- #
# apply_prefix_views
# --------------------------------------------------------------------- #
def _ramp(b=3, k=2, seq=64, d=5) -> torch.Tensor:
    """Views whose values encode their token position, so leakage is visible."""
    t = torch.arange(1, seq + 1, dtype=torch.float32).view(1, 1, seq, 1)
    return t.expand(b, k, seq, d).clone()


def test_shapes_are_preserved_for_the_fused_encode():
    """Shape preservation is the whole point — the trainer's fused
    reshape(B*K, L, D) must keep working untouched."""
    views = _ramp()
    out, mask = apply_prefix_views(views, PrefixViewSpec((16, 48)))
    assert tuple(out.shape) == (3, 2, 64, 5)
    assert tuple(mask.shape) == (3, 2, 64)
    assert mask.dtype == torch.bool


def test_mask_reshapes_alongside_activations():
    """(B,K,L) must flatten under the same reshape as (B,K,L,D)."""
    views = _ramp()
    out, mask = apply_prefix_views(views, PrefixViewSpec((16, 48)))
    b, k, l, d = out.shape
    assert tuple(out.reshape(b * k, l, d).shape) == (6, 64, 5)
    assert tuple(mask.reshape(b * k, l).shape) == (6, 64)


def test_mask_marks_exactly_the_prefix():
    views = _ramp()
    _, mask = apply_prefix_views(views, PrefixViewSpec((16, 32)))
    assert mask[:, 0].sum(dim=-1).unique().tolist() == [16]
    assert mask[:, 1].sum(dim=-1).unique().tolist() == [32]
    assert mask[:, 0, :16].all() and not mask[:, 0, 16:].any()


def test_prefix_content_is_preserved_exactly():
    views = _ramp()
    out, _ = apply_prefix_views(views, PrefixViewSpec((16, 32)))
    assert torch.equal(out[:, 0, :16], views[:, 0, :16])
    assert torch.equal(out[:, 1, :32], views[:, 1, :32])


def test_never_leaks_future_tokens():
    """The fairness invariant: no view may retain a token beyond its own k."""
    views = _ramp(seq=64)
    for prefix in (1, 16, 32, 48, 64):
        out, _ = apply_prefix_views(views[:, :1], PrefixViewSpec((prefix,)))
        assert out.max().item() == float(prefix)
        assert (out[:, 0, prefix:] == 0).all()


def test_zero_prefix_falls_back_to_single_token():
    """A fully-masked row has no mean and NaNs the attention softmax."""
    views = _ramp()
    _, mask = apply_prefix_views(views, PrefixViewSpec((0, 32)))
    assert mask[:, 0].sum(dim=-1).unique().tolist() == [1]


def test_zero_prefix_can_be_made_hard_error():
    views = _ramp()
    with pytest.raises(ValueError, match="prompt_fallback"):
        apply_prefix_views(views, PrefixViewSpec((0, 32)), prompt_fallback=False)


def test_rejects_prefix_longer_than_sequence():
    views = _ramp(seq=32)
    with pytest.raises(ValueError, match="only 32 tokens"):
        apply_prefix_views(views, PrefixViewSpec((48, 16)))


def test_rejects_view_count_mismatch():
    views = _ramp(k=2)
    with pytest.raises(ValueError, match="K=2"):
        apply_prefix_views(views, PrefixViewSpec((16, 32, 48)))


def test_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\(B, K, L, D\)"):
        apply_prefix_views(torch.zeros(3, 64, 5), PrefixViewSpec((16,)))


def test_full_prefix_is_identity_with_all_true_mask():
    """k=max must reproduce the current post-hoc setup bit-for-bit."""
    views = _ramp(seq=64)
    out, mask = apply_prefix_views(views, PrefixViewSpec((64, 64)))
    assert torch.equal(out, views)
    assert mask.all()


# --------------------------------------------------------------------- #
# masked pooling / attention (activation_research.model)
# --------------------------------------------------------------------- #
def test_masked_mean_matches_plain_mean_when_all_valid():
    from activation_research.model import masked_mean

    x = torch.randn(4, 16, 8)
    mask = torch.ones(4, 16, dtype=torch.bool)
    assert torch.allclose(masked_mean(x, mask), x.mean(dim=1), atol=1e-6)
    assert torch.equal(masked_mean(x, None), x.mean(dim=1))


def test_masked_mean_ignores_padding():
    """The k-leak guard: pooled value must not be diluted by pad positions."""
    from activation_research.model import masked_mean

    x = torch.randn(4, 64, 8)
    x_padded = x.clone()
    x_padded[:, 16:] = 0.0
    mask = torch.zeros(4, 64, dtype=torch.bool)
    mask[:, :16] = True
    assert torch.allclose(masked_mean(x_padded, mask), x[:, :16].mean(dim=1), atol=1e-6)


def test_unmasked_mean_would_have_leaked_k():
    """Documents the failure mode masking exists to prevent: the plain mean
    scales with k/L, putting the prefix length into the embedding norm."""
    x = torch.randn(4, 64, 8)
    x_padded = x.clone()
    x_padded[:, 16:] = 0.0
    naive = x_padded.mean(dim=1)
    true_mean = x[:, :16].mean(dim=1)
    assert torch.allclose(naive, true_mean * (16 / 64), atol=1e-6)


def test_masked_mean_survives_fully_masked_row():
    from activation_research.model import masked_mean

    x = torch.randn(2, 8, 4)
    mask = torch.zeros(2, 8, dtype=torch.bool)
    out = masked_mean(x, mask)
    assert torch.isfinite(out).all()


def test_compressor_ignores_padded_tokens_end_to_end():
    """Padding must not change the embedding: a 16-token input and the same
    16 tokens zero-padded to 64 with a mask must encode identically.

    This is the real guarantee — it covers attention contamination, not just
    the pool.
    """
    from activation_research.model import ProgressiveCompressor

    torch.manual_seed(0)
    m = ProgressiveCompressor(input_dim=32, final_dim=16).eval()

    short = torch.randn(2, 16, 32)
    padded = torch.zeros(2, 64, 32)
    padded[:, :16] = short
    mask = torch.zeros(2, 64, dtype=torch.bool)
    mask[:, :16] = True

    with torch.no_grad():
        z_short = m(short)
        z_padded = m(padded, token_mask=mask)
    assert torch.allclose(z_short, z_padded, atol=1e-5)


def test_compressor_default_path_is_unchanged():
    """token_mask=None must reproduce pre-masking behaviour bit-for-bit, so
    the layer_only arm remains a true control."""
    from activation_research.model import ProgressiveCompressor

    torch.manual_seed(0)
    m = ProgressiveCompressor(input_dim=32, final_dim=16).eval()
    x = torch.randn(2, 64, 32)
    with torch.no_grad():
        a = m(x)
        b = m(x, token_mask=torch.ones(2, 64, dtype=torch.bool))
    assert torch.allclose(a, b, atol=1e-6)


def test_padding_without_mask_changes_the_embedding():
    """Control for the test above: without the mask, padding leaks."""
    from activation_research.model import ProgressiveCompressor

    torch.manual_seed(0)
    m = ProgressiveCompressor(input_dim=32, final_dim=16).eval()
    short = torch.randn(2, 16, 32)
    padded = torch.zeros(2, 64, 32)
    padded[:, :16] = short
    with torch.no_grad():
        assert not torch.allclose(m(short), m(padded), atol=1e-5)


# --------------------------------------------------------------------- #
# resolve_eval_prefixes
# --------------------------------------------------------------------- #
def test_resolve_defaults_to_eval_grid():
    assert resolve_eval_prefixes(None, 64) == [0, 16, 32, 48, 64]


def test_resolve_drops_prefixes_above_capture_width():
    assert resolve_eval_prefixes(None, 32) == [0, 16, 32]


def test_resolve_dedupes_and_preserves_order():
    assert resolve_eval_prefixes([32, 16, 32, 8], 64) == [32, 16, 8]


def test_resolve_rejects_negative_and_empty_result():
    with pytest.raises(ValueError):
        resolve_eval_prefixes([-1], 64)
    with pytest.raises(ValueError, match="no evaluation prefixes"):
        resolve_eval_prefixes([128], 64)
