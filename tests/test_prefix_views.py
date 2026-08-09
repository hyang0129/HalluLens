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
    resolve_eval_prefixes,
    slice_views,
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
# slice_views
# --------------------------------------------------------------------- #
def _ramp(b=3, k=2, seq=64, d=5) -> torch.Tensor:
    """Views whose values encode their token position, so leakage is visible."""
    t = torch.arange(seq, dtype=torch.float32).view(1, 1, seq, 1)
    return t.expand(b, k, seq, d).clone()


def test_slice_shapes_match_spec():
    views = _ramp()
    out = slice_views(views, PrefixViewSpec((16, 48)))
    assert [tuple(o.shape) for o in out] == [(3, 16, 5), (3, 48, 5)]


def test_slice_preserves_prefix_content_exactly():
    views = _ramp()
    out = slice_views(views, PrefixViewSpec((16, 32)))
    assert torch.equal(out[0], views[:, 0, :16, :])
    assert torch.equal(out[1], views[:, 1, :32, :])


def test_slice_never_leaks_future_tokens():
    """The fairness invariant: no view may contain a token beyond its own k."""
    views = _ramp(seq=64)
    for prefix in (1, 16, 32, 48, 64):
        (out,) = slice_views(views[:, :1], PrefixViewSpec((prefix,)))
        assert out.max().item() == float(prefix - 1)


def test_zero_prefix_falls_back_to_single_token():
    views = _ramp()
    out = slice_views(views, PrefixViewSpec((0, 32)))
    assert tuple(out[0].shape) == (3, 1, 5)


def test_zero_prefix_can_be_made_hard_error():
    views = _ramp()
    with pytest.raises(ValueError, match="prompt_fallback"):
        slice_views(views, PrefixViewSpec((0, 32)), prompt_fallback=False)


def test_slice_rejects_prefix_longer_than_sequence():
    views = _ramp(seq=32)
    with pytest.raises(ValueError, match="only 32 tokens"):
        slice_views(views, PrefixViewSpec((48, 16)))


def test_slice_rejects_view_count_mismatch():
    views = _ramp(k=2)
    with pytest.raises(ValueError, match="K=2"):
        slice_views(views, PrefixViewSpec((16, 32, 48)))


def test_slice_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\(B, K, L, D\)"):
        slice_views(torch.zeros(3, 64, 5), PrefixViewSpec((16,)))


def test_full_prefix_slice_is_identity():
    """k=max must reproduce the current post-hoc setup bit-for-bit."""
    views = _ramp(seq=64)
    out = slice_views(views, PrefixViewSpec((64, 64)))
    assert torch.equal(out[0], views[:, 0])
    assert torch.equal(out[1], views[:, 1])


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
