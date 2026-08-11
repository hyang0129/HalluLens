"""prefix_views.py — mixed layer x prefix view construction for issue #149.

Motivation
----------
Every activation probe in the grid is *post-hoc*: it consumes the finished
64-token generation. Issue #149 asks the predictive question — can the
hallucination be caught at token 16, before the model finishes saying it?

Causal attention makes this free: the hidden state at response position ``i``
depends only on positions ``<= i``, so ``h[:k]`` sliced out of an existing
64-token capture is bit-identical to what a k-token generation would have
produced. The whole study is a slice of the token axis on existing memmaps.

Design decision — zero-pad plus an explicit mask, keeping the fused encode
--------------------------------------------------------------------------
The trainer fuses the batch and view axes into a single encoder call
(``views_full.reshape(bsz * num_views, seq_len, hidden_dim)`` in
``activation_research/training.py``). Keeping that fused path is worth more than
per-view slicing, so a prefix view is represented as the full ``max_response_len``
tensor with positions ``>= k`` zeroed, carried alongside a boolean token mask.

The mask must be explicit — padding cannot be recovered by testing for zeros
downstream. ``TransformerBlock.input_proj`` has a bias and ``PositionalEncoding``
adds to every slot, so a zero-filled pad row is non-zero by the first block.

The mask is threaded to two places (see ``activation_research/model.py``):

* ``nn.TransformerEncoderLayer(src_key_padding_mask=...)`` — without this, pad
  slots act as attention keys/values and contaminate the real tokens. Note the
  inverted convention: torch marks positions to *ignore*, we mark real tokens.
* ``masked_mean`` in place of ``x.mean(dim=1)`` — an unmasked mean would scale
  the pooled vector by ``k / max_response_len``, leaking k straight into the
  embedding norm as a shortcut.

A ``(B, K, L)`` mask flattens under the same reshape as the activations, so the
fused call is untouched. ``token_mask=None`` reproduces the pre-masking
behaviour bit-for-bit, which is what keeps the ``layer_only`` control arm a true
control.

Prefix pairs are still drawn **once per batch** rather than per item. Nothing in
the masked path requires it, but it keeps k constant across a gradient
accumulation window (``torch.cat`` of buffered sub-batches in the trainer) and
makes the geometry of each step reportable.

Sampling contract
-----------------
``k`` is sampled, never fixed at the evaluated values. Training at exactly
16/32 and then reporting "AUROC at k=16" cannot distinguish the objective from
overfitting to the evaluated k. The gap ``k_2 - k_1`` is sampled independently
of ``k_1`` so the encoder cannot learn gap size as a shortcut.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

import torch

# Prefix lengths the evaluation reports on. Training never pins to these.
#
# The low end (1, 4, 8) is where the curve is actually informative: measurements
# on HotpotQA show it is already flat from k=16 upward, so the question is how
# far down detection survives, not what happens between 16 and 64.
#
# k=0 is not a value here and never will be. With zero response tokens the
# encoder receives nothing — it is degenerate, not the k->0 limit of this curve.
# A "prompt only" number would require feeding prompt_activations.npy, i.e. a
# different input space and therefore a different method, which belongs as its
# own baseline rather than as a point on this curve.
#
# Note also that prompt conditioning is not a confound for these measurements.
# The encoder sees response activations only, but those are hidden states of a
# causal LM that attended over the full prompt, so prompt information is
# legitimately present in every response token's representation. That is the
# model's own state being read, not information smuggled in from outside it.
EVAL_PREFIX_LENGTHS: tuple[int, ...] = (1, 4, 8, 16, 32, 48, 64)

ViewMode = Literal["layer_only", "prefix_only", "mixed"]


@dataclass(frozen=True)
class PrefixViewSpec:
    """Resolved per-batch view geometry.

    Attributes
    ----------
    prefix_lens : tuple[int, ...]
        One prefix length per view slot. ``len == num_views``.
    """

    prefix_lens: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.prefix_lens:
            raise ValueError("prefix_lens must be non-empty")
        for k in self.prefix_lens:
            if k < 0:
                raise ValueError(f"prefix length must be >= 0, got {k}")

    @property
    def num_views(self) -> int:
        return len(self.prefix_lens)

    @property
    def is_uniform(self) -> bool:
        """True when every view slot uses the same prefix (pure layer-pair)."""
        return len(set(self.prefix_lens)) == 1


class PrefixPairSampler:
    """Samples per-batch prefix lengths for mixed layer x prefix training.

    Parameters
    ----------
    mode : {"layer_only", "prefix_only", "mixed"}
        ``layer_only`` reproduces current behaviour — all views share the full
        prefix, so positives differ only in layer. ``prefix_only`` holds the
        layer fixed and varies the prefix. ``mixed`` varies both, and is the
        headline configuration for #149.
    num_views : int
        View slots per item (the contrastive K).
    max_prefix : int
        Upper bound on a prefix, normally ``r_max`` (64 for our captures).
    min_prefix : int
        Lower bound on a sampled prefix. Guards against degenerate 1-token
        views whose mean-pool is a single activation.
    min_gap : int
        Minimum ``k_max - k_min`` when more than one distinct prefix is drawn.
        Prevents near-identical positives that make the InfoNCE task trivial.
    seed : int | None
        Seeds a private RNG so view geometry is reproducible per run and
        independent of global ``random`` state (which the layer sampler uses).

    Notes
    -----
    ``sample()`` is called once per batch by the collate function. The gap is
    drawn independently of the base prefix — see the module docstring.
    """

    def __init__(
        self,
        *,
        mode: ViewMode = "mixed",
        num_views: int = 2,
        max_prefix: int = 64,
        min_prefix: int = 8,
        min_gap: int = 8,
        seed: Optional[int] = None,
    ) -> None:
        if mode not in ("layer_only", "prefix_only", "mixed"):
            raise ValueError(f"unknown mode {mode!r}")
        if num_views < 1:
            raise ValueError(f"num_views must be >= 1, got {num_views}")
        if max_prefix < 1:
            raise ValueError(f"max_prefix must be >= 1, got {max_prefix}")
        if min_prefix < 1 or min_prefix > max_prefix:
            raise ValueError(
                f"min_prefix must be in [1, max_prefix]; got {min_prefix} "
                f"with max_prefix={max_prefix}"
            )
        if min_gap < 0:
            raise ValueError(f"min_gap must be >= 0, got {min_gap}")
        if mode != "layer_only" and min_prefix + min_gap > max_prefix:
            raise ValueError(
                f"min_prefix({min_prefix}) + min_gap({min_gap}) exceeds "
                f"max_prefix({max_prefix}) — no valid prefix pair exists"
            )

        self.mode = mode
        self.num_views = int(num_views)
        self.max_prefix = int(max_prefix)
        self.min_prefix = int(min_prefix)
        self.min_gap = int(min_gap)
        self._seed = seed
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    def reseed_for_worker(self, worker_id: int) -> None:
        """Re-seed this sampler inside a forked DataLoader worker.

        ``DataLoader`` forks workers, so every worker inherits a *copy* of this
        object with an identical RNG state and would emit the same k sequence in
        lockstep — cutting effective prefix diversity by ``num_workers``.
        Mixing the worker id into the seed decorrelates the streams while
        keeping the run reproducible.
        """
        base = 0 if self._seed is None else int(self._seed)
        self._rng = random.Random((base + 1) * 100003 + int(worker_id))

    # ------------------------------------------------------------------ #
    def sample(self) -> PrefixViewSpec:
        """Draw one prefix length per view slot for the next batch."""
        if self.mode == "layer_only":
            return PrefixViewSpec(tuple([self.max_prefix] * self.num_views))

        if self.num_views == 1:
            k = self._rng.randint(self.min_prefix, self.max_prefix)
            return PrefixViewSpec((k,))

        # Draw the short prefix, then the gap independently of it, so gap size
        # carries no information about k_1 (shortcut avoidance).
        k_lo = self._rng.randint(self.min_prefix, self.max_prefix - self.min_gap)
        gap = self._rng.randint(self.min_gap, self.max_prefix - k_lo)
        k_hi = k_lo + gap

        if self.num_views == 2:
            lens = (k_lo, k_hi)
        else:
            # Interpolate intermediate slots, keeping the extremes pinned.
            step = (k_hi - k_lo) / (self.num_views - 1)
            lens = tuple(
                int(round(k_lo + step * i)) for i in range(self.num_views)
            )
        return PrefixViewSpec(lens)

    # ------------------------------------------------------------------ #
    def eval_spec(self, k: int) -> PrefixViewSpec:
        """Fixed-k geometry for evaluation: every view slot sees the same k.

        At eval time we are measuring "what can be known from the first k
        tokens", so all views share k. k=0 is represented as a 1-token prefix
        floor by the caller (see ``slice_views``) since a zero-length sequence
        has no mean.
        """
        return PrefixViewSpec(tuple([int(k)] * self.num_views))


# ---------------------------------------------------------------------- #
def apply_prefix_views(
    views: torch.Tensor,
    spec: PrefixViewSpec,
    *,
    response_lens: Optional[torch.Tensor] = None,
    prompt_fallback: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero out post-prefix tokens and build the matching token mask.

    Shapes are preserved so the trainer's fused
    ``reshape(B * K, L, D)`` encode continues to work untouched — the mask
    reshapes to ``(B * K, L)`` under exactly the same call.

    Parameters
    ----------
    views : Tensor (B, K, L, D)
        Stacked activations as produced by the contrastive collate.
    spec : PrefixViewSpec
        Per-slot prefix lengths. ``spec.num_views`` must equal ``K``.
    prompt_fallback : bool
        A prefix of 0 (the "prompt only, before any response token" arm) would
        leave a row with no valid tokens, which has no mean and produces NaN
        under attention softmax. With this flag such a slot keeps a single
        token; the caller is responsible for substituting prompt-side
        activations for a true k=0 arm.
    response_lens : Tensor (B,), optional
        Number of real response tokens in each sample.  When present, the
        valid region is ``min(prefix, response_len[b])`` rather than the
        requested prefix alone.  This prevents early-EOS padding from being
        treated as generated content.

    Returns
    -------
    (masked_views, token_mask)
        ``masked_views`` is ``(B, K, L, D)`` with positions ``>= k`` zeroed.
        ``token_mask`` is ``(B, K, L)`` bool, ``True`` at real tokens.

    Raises
    ------
    ValueError
        If ``views`` is not 4-D, K disagrees with the spec, or a requested
        prefix exceeds the available sequence length.
    """
    if views.dim() != 4:
        raise ValueError(
            f"expected views of shape (B, K, L, D), got {tuple(views.shape)}"
        )
    bsz, k_slots, seq_len, _ = views.shape
    if k_slots != spec.num_views:
        raise ValueError(
            f"view tensor has K={k_slots} but spec declares {spec.num_views}"
        )

    if response_lens is not None:
        response_lens = torch.as_tensor(response_lens, device=views.device)
        if response_lens.ndim != 1 or int(response_lens.shape[0]) != bsz:
            raise ValueError(
                f"response_lens must have shape ({bsz},), got "
                f"{tuple(response_lens.shape)}"
            )
        response_lens = response_lens.to(dtype=torch.long).clamp(
            min=0, max=seq_len
        )

    token_mask = torch.zeros(
        (bsz, k_slots, seq_len), dtype=torch.bool, device=views.device
    )
    positions = torch.arange(seq_len, device=views.device).unsqueeze(0)
    for slot, prefix in enumerate(spec.prefix_lens):
        if prefix > seq_len:
            raise ValueError(
                f"view slot {slot} requests prefix {prefix} but only "
                f"{seq_len} tokens are available"
            )
        eff = int(prefix)
        if eff == 0:
            if not prompt_fallback:
                raise ValueError(
                    f"view slot {slot} has prefix 0 and prompt_fallback=False"
                )
            eff = 1
        if response_lens is None:
            token_mask[:, slot, :eff] = True
        else:
            effective_lens = torch.minimum(
                response_lens, torch.full_like(response_lens, eff)
            )
            if prompt_fallback:
                effective_lens = effective_lens.clamp(min=1)
            token_mask[:, slot] = positions < effective_lens.unsqueeze(1)

    masked_views = views * token_mask.unsqueeze(-1).to(views.dtype)
    return masked_views, token_mask


class PrefixEvalWrapper(torch.nn.Module):
    """Wrap a trained encoder so it only ever sees the first ``k`` tokens.

    At evaluation time the prefix is *uniform* — every view of every item is cut
    at the same k, because the question being asked is "what is knowable from
    the first k tokens?". That makes the token mask a pure function of the input
    shape, so it can be built inside the wrapper instead of threaded through the
    dataloaders, the collate, and the metric evaluator. The whole eval stack
    keeps calling ``model(x)`` unchanged.

    Post-prefix activations are additionally zeroed. The mask alone is
    sufficient — masked attention and masked pooling already ignore them — but
    zeroing makes the eval-time input byte-identical in construction to what
    training produced, so a discrepancy shows up as a shape/mask bug rather than
    as a silent distribution shift.

    Parameters
    ----------
    model : nn.Module
        A trained encoder accepting ``token_mask``.
    prefix_len : int
        Tokens to retain. Must be >= 1; see ``resolve_eval_prefixes`` for why
        k=0 is not expressible here (it needs prompt-side activations, which
        live in a different memmap).
    """

    def __init__(self, model: torch.nn.Module, prefix_len: int) -> None:
        super().__init__()
        if int(prefix_len) < 1:
            raise ValueError(
                f"prefix_len must be >= 1, got {prefix_len}. A true k=0 arm "
                "requires prompt-side activations (prompt_activations.npy), "
                "not a zero-length response slice."
            )
        self.model = model
        self.prefix_len = int(prefix_len)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        seq_len = x.shape[1]
        k = min(self.prefix_len, seq_len)
        response_lens = kwargs.pop("response_lens", None)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        if response_lens is None:
            # Memmap captures use exact zero rows after response_len.  Infer the
            # raw-input validity before projection/positional encodings make
            # those rows non-zero.  Callers with explicit lengths can pass
            # response_lens to avoid relying on this storage invariant.
            raw_valid = x.detach().ne(0).any(dim=-1)
            inferred = raw_valid.long().sum(dim=-1).clamp(min=1, max=seq_len)
            response_lens = inferred
        else:
            response_lens = torch.as_tensor(
                response_lens, dtype=torch.long, device=x.device
            ).clamp(min=1, max=seq_len)
        effective = torch.minimum(
            response_lens, torch.full_like(response_lens, k)
        )
        mask = positions < effective.unsqueeze(1)
        x = x * mask.unsqueeze(-1).to(x.dtype)
        return self.model(x, *args, token_mask=mask, **kwargs)


def build_prefix_mask(
    response_lens: torch.Tensor,
    *,
    seq_len: int,
    prefix_len: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return a response-length-aware ``(B, seq_len)`` prefix mask.

    The helper is shared by supervised baselines whose batches retain the
    fixed capture width.  Every row keeps exactly
    ``min(response_len[b], prefix_len, seq_len)`` positions, with a one-token
    floor because all response-prefix experiments require ``k >= 1``.
    """
    if int(seq_len) < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if int(prefix_len) < 1:
        raise ValueError(f"prefix_len must be >= 1, got {prefix_len}")
    lens = torch.as_tensor(response_lens, dtype=torch.long, device=device)
    if lens.ndim != 1:
        raise ValueError(
            f"response_lens must be one-dimensional, got {tuple(lens.shape)}"
        )
    cap = min(int(prefix_len), int(seq_len))
    lens = lens.clamp(min=1, max=cap)
    positions = torch.arange(int(seq_len), device=lens.device).unsqueeze(0)
    return positions < lens.unsqueeze(1)


def resolve_eval_prefixes(
    requested: Optional[Sequence[int]], max_prefix: int
) -> List[int]:
    """Validate and normalise the evaluation k grid.

    Drops values above ``max_prefix`` rather than erroring, so a capture with a
    shorter ``r_max`` than the default grid still evaluates on the k values it
    can support. Order is preserved and duplicates removed.
    """
    grid = list(EVAL_PREFIX_LENGTHS) if requested is None else [int(k) for k in requested]
    seen: set[int] = set()
    out: List[int] = []
    for k in grid:
        if k < 1:
            raise ValueError(
                f"evaluation prefix must be >= 1, got {k}. k=0 is a separate "
                "prompt-only arm (prompt_activations.npy), not a response slice."
            )
        if k > max_prefix or k in seen:
            continue
        seen.add(k)
        out.append(k)
    if not out:
        raise ValueError(
            f"no evaluation prefixes survive max_prefix={max_prefix} "
            f"(requested {grid})"
        )
    return out
