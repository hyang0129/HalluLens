"""
memmap_contrastive_dataset.py — memmap-backed contrastive dataset for Issue #72's
capture layout, emitting the same per-item dict shape as the zarr-backed
:class:`activation_logging.activation_parser.PreloadedActivationDataset`.

Reads from ``shared/icr_capture/{dataset}_{model_slug}/`` (or any directory
produced by :class:`activation_logging.inference_capture_writer.InferenceCaptureWriter`):

    config.json
    meta.jsonl
    response_activations.npy      memmap (N, num_layers+1, max_response_len, hidden_dim) fp16
    response_attention.npy        memmap (N, num_layers, r_max, r_max) fp16
    response_token_logprobs.npy   memmap (N, max_response_len) float32 (NaN-padded)
    response_topk_token_ids.npy   memmap (N, max_response_len, top_k) int32 (-1 padded)
    response_topk_logprobs.npy    memmap (N, max_response_len, top_k) float32 (NaN-padded)
    response_token_ids.npy        memmap (N, max_response_len) int32 (-1 padded)
    response_len.npy              memmap (N,) int32
    prompt_len.npy                memmap (N,) int32

Designed as a drop-in source for ``train_contrastive`` / ``train_contrastive_logprob_recon``
/ ``train_contrastive_logprob_attn_recon`` — same dict keys, same shapes.  The
default view axis is layers.  ``view_axis="token"`` transposes the experimental
geometry: each view is one response step and its sequence is the selected model
depths.

For Mechanism K (Issue #75), the dataset also emits per-layer attention summary
statistics aligned to the sampled view layers (target layer = view_layer ± offset).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from activation_research.icr_dataset import _make_split_indices
from activation_research.labels import load_meta


# ---------------------------------------------------------------------------
# Attention summary
# ---------------------------------------------------------------------------

_ATTN_STAT_DIM = 3  # (entropy, focal_frac, self_mass)


def _compute_attn_stats(
    attn_block: np.ndarray, response_len: int, *, eps: float = 1e-9
) -> np.ndarray:
    """Compute per-layer attention summary stats from a single sample's slice.

    Parameters
    ----------
    attn_block : ndarray (r_max, r_max), fp16/fp32
        Head-averaged response-to-response attention for ONE layer.
    response_len : int
        Number of real response tokens; rows past this are excluded.

    Returns
    -------
    ndarray (_ATTN_STAT_DIM,) fp32
        (entropy, focal_frac, self_mass). Returns NaN-filled array when
        response_len == 0 (no valid rows to summarize).
    """
    out = np.full((_ATTN_STAT_DIM,), np.nan, dtype=np.float32)
    if response_len <= 0:
        return out

    a = attn_block.astype(np.float32)
    valid_rows = a[:response_len, :response_len]
    if valid_rows.size == 0:
        return out

    # Renormalize per row to handle zero-padded key positions beyond response_len.
    row_sums = valid_rows.sum(axis=-1, keepdims=True)
    row_sums = np.where(row_sums > eps, row_sums, 1.0)
    p = valid_rows / row_sums

    # Entropy per query row, averaged.
    entropy = -np.sum(p * np.log(p + eps), axis=-1).mean()
    # Peak attention mass per row, averaged.
    focal_frac = p.max(axis=-1).mean()
    # Self-mass: diagonal entry per row, averaged.
    diag_idx = np.arange(p.shape[0])
    self_mass = p[diag_idx, diag_idx].mean()

    out[0] = entropy
    out[1] = focal_frac
    out[2] = self_mass
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class MemmapContrastiveDataset(Dataset):
    """Memmap-backed contrastive dataset over Issue #72 captures.

    Emits the same per-item dict shape as
    :class:`activation_logging.activation_parser.PreloadedActivationDataset`,
    so the existing contrastive trainer + collate consume it unchanged. Adds
    optional ``attention_forward`` and ``attention_backward`` fields for
    Mechanism K (Issue #75).

    Parameters
    ----------
    capture_dir : str | Path
        Directory containing the Issue #72 capture files.
    split : {"train", "val", "test", "all"}
        Partition to expose. Shared stratified-split helper with
        :class:`ICRDataset`.
    val_fraction : float | None
        Fraction of *total* samples for val. ``None`` uses the
        ``ActivationParser`` three-way default (~10%). Ignored when
        ``split="all"``.
    random_seed : int
        RNG seed for stratified split.
    num_views : int
        Number of layer views to sample per item (contrastive view count).
    relevant_layers : list[int] | None
        Subset of model-layer IDs (positions into the memmap's layer axis)
        to sample views from. Default: all ``num_layers+1`` available.
    fixed_layer : int | None
        If set, first view is always this model-layer ID. Remaining views
        are sampled from ``relevant_layers \\ {fixed_layer}``. Matches
        :meth:`PreloadedActivationDataset._select_view_indices` semantics.
    view_sampling_with_replacement : bool
        Sample views with replacement.
    view_axis : {"layer", "token"}
        Axis from which contrastive views are drawn.  ``"layer"`` preserves the
        original dataset exactly.  ``"token"`` emits tensors shaped
        ``(num_views, num_selected_layers, hidden_dim)``.
    token_pair_mode : {"first_anchored", "random_distinct"}
        Token sampling rule when ``view_axis="token"``.  The first-anchored rule
        emits token 0 plus later response tokens; the comparison rule samples
        distinct response tokens uniformly.
    fixed_token : int | None
        Freeze token-wise views to one response step.  Intended for evaluation;
        use ``num_views=1`` to obtain a single representation per example.
    min_response_tokens : int
        Drop split rows shorter than this many captured response steps.  The
        token-pair training runner sets this to two while token-0 evaluation sets
        it to one, retaining one-token generations in the primary KNN surface.
    include_response_logprobs : bool
        Emit per-token logprob fields (for Mechanism F).
    response_logprobs_top_k : int
        Top-K alternative count to emit (capped at the stored top-K).
    pad_length : int | None
        Target length for logprob sequences. Defaults to
        ``config.max_response_len``.
    include_response_attention : bool
        Emit attention summary fields (for Mechanism K).
    attention_summary : {"stats"}
        Summary form. Only ``"stats"`` is implemented in this PR.
    attention_target_layer_offset_forward : int | None
        If set, emits ``attention_forward[k]`` as the summary at model
        layer ``view_model_layer[k] + offset``. Clamped — out-of-range
        layers yield NaN rows (suppressed by trainer's variance gate).
    attention_target_layer_offset_backward : int | None
        Symmetric: emits ``attention_backward[k]`` at ``view_layer - offset``.
    """

    def __init__(
        self,
        capture_dir: str | Path,
        *,
        # Split
        split: Literal["train", "val", "test", "all"] = "train",
        val_fraction: Optional[float] = None,
        random_seed: int = 42,
        label_source: str = "substring",
        # Contrastive view sampling
        num_views: int = 2,
        relevant_layers: Optional[List[int]] = None,
        fixed_layer: Optional[int] = None,
        view_sampling_with_replacement: bool = False,
        view_axis: Literal["layer", "token"] = "layer",
        token_pair_mode: Literal["first_anchored", "random_distinct"] = "first_anchored",
        fixed_token: Optional[int] = None,
        min_response_tokens: int = 0,
        # Logprob recon (Mechanism F)
        include_response_logprobs: bool = False,
        response_logprobs_top_k: int = 20,
        pad_length: Optional[int] = None,
        # Attention recon (Mechanism K)
        include_response_attention: bool = False,
        attention_summary: Literal["stats", "coarse", "full"] = "stats",
        attention_target_layer_offset_forward: Optional[int] = None,
        attention_target_layer_offset_backward: Optional[int] = None,
        # Private overrides — used by MemmapActivationParser to inject pre-computed splits
        _override_indices: Optional[np.ndarray] = None,
        _override_split_name: Optional[str] = None,
    ) -> None:
        capture_dir = Path(capture_dir)
        if not capture_dir.exists():
            raise FileNotFoundError(f"capture_dir not found: {capture_dir}")
        self._capture_dir = capture_dir
        self._label_source = str(label_source)

        if view_axis not in ("layer", "token"):
            raise ValueError("view_axis must be either 'layer' or 'token'")
        if token_pair_mode not in ("first_anchored", "random_distinct"):
            raise ValueError(
                "token_pair_mode must be 'first_anchored' or 'random_distinct'"
            )
        if int(min_response_tokens) < 0:
            raise ValueError("min_response_tokens must be non-negative")
        if fixed_token is not None and int(fixed_token) < 0:
            raise ValueError("fixed_token must be non-negative")
        if view_axis == "layer" and fixed_token is not None:
            raise ValueError("fixed_token is only valid when view_axis='token'")
        if view_axis == "token" and fixed_layer is not None:
            raise ValueError("fixed_layer is only valid when view_axis='layer'")

        if attention_summary not in ("stats", "full"):
            # 'coarse' is reserved — no plans to implement it (issue #82 comment).
            raise NotImplementedError(
                f"attention_summary={attention_summary!r} is not implemented; "
                "supported values: 'stats', 'full'."
            )

        # --- Config ---
        with (capture_dir / "config.json").open() as fh:
            cfg = json.load(fh)
        self._cfg = cfg
        self._n_samples: int = cfg["n_samples"]
        self._num_layers: int = cfg["num_layers"]
        self._hidden_dim: int = cfg["hidden_dim"]
        self._r_max: int = cfg["r_max"]
        self._max_resp: int = cfg["max_response_len"]
        self._max_prompt: int = cfg["max_prompt_len"]
        stored_top_k: int = cfg.get("response_logprobs_top_k", response_logprobs_top_k)

        # --- meta.jsonl (authoritative valid-rows list) ---
        # hallucinated set per label_source (substring | llm_judge); see
        # activation_research/labels.py and issue #145.
        meta_rows = load_meta(capture_dir, label_source)
        if not meta_rows:
            raise ValueError(f"meta.jsonl in {capture_dir} contains no rows")
        self._meta = meta_rows
        # sample_index → memmap row (the writer pre-allocates by sample_index
        # but only meta-listed indices were committed).
        self._valid_sample_indices = np.array(
            [r["sample_index"] for r in meta_rows], dtype=np.int64
        )

        # --- Layer selection ---
        # response_activations.npy's second axis is num_layers+1 (layer 0 is
        # the embeddings, 1..num_layers are post-block residuals).
        total_layer_slots = self._num_layers + 1
        if relevant_layers is None:
            self._relevant_layers = list(range(total_layer_slots))
        else:
            for lid in relevant_layers:
                if not 0 <= lid < total_layer_slots:
                    raise ValueError(
                        f"relevant_layers entry {lid} outside [0, {total_layer_slots})"
                    )
            self._relevant_layers = list(relevant_layers)

        self._num_views = int(num_views)
        if self._num_views < 1:
            raise ValueError("num_views must be at least one")
        self._fixed_layer = fixed_layer
        if fixed_layer is not None and fixed_layer not in self._relevant_layers:
            raise ValueError(
                f"fixed_layer={fixed_layer} not in relevant_layers "
                f"{self._relevant_layers}"
            )
        self._view_with_replacement = bool(view_sampling_with_replacement)
        self._view_axis = str(view_axis)
        self._token_pair_mode = str(token_pair_mode)
        self._fixed_token = int(fixed_token) if fixed_token is not None else None
        if self._fixed_token is not None and self._fixed_token >= self._max_resp:
            raise ValueError(
                f"fixed_token={self._fixed_token} outside captured response width "
                f"[0, {self._max_resp})"
            )
        self._min_response_tokens = int(min_response_tokens)

        # --- Logprob options ---
        self._include_lp = bool(include_response_logprobs)
        self._pad_length = int(pad_length) if pad_length is not None else self._max_resp
        self._target_top_k = int(min(response_logprobs_top_k, stored_top_k))

        # --- Attention options ---
        self._include_attn = bool(include_response_attention)
        if self._view_axis == "token" and self._include_attn:
            raise NotImplementedError(
                "response-attention reconstruction is layer-view-specific and is "
                "not supported for token-wise views"
            )
        self._attn_summary = str(attention_summary)
        self._attn_offset_fwd = (
            int(attention_target_layer_offset_forward)
            if attention_target_layer_offset_forward is not None
            else None
        )
        self._attn_offset_bwd = (
            int(attention_target_layer_offset_backward)
            if attention_target_layer_offset_backward is not None
            else None
        )

        # --- Open memmaps ---
        self._resp_act = self._open_memmap(
            capture_dir / "response_activations.npy",
            (self._n_samples, total_layer_slots, self._max_resp, self._hidden_dim),
            np.float16,
        )
        self._resp_attn = self._open_memmap(
            capture_dir / "response_attention.npy",
            (self._n_samples, self._num_layers, self._r_max, self._r_max),
            np.float16,
        ) if self._include_attn else None
        self._resp_len = self._open_memmap(
            capture_dir / "response_len.npy",
            (self._n_samples,),
            np.int32,
        )
        self._prompt_len = self._open_memmap(
            capture_dir / "prompt_len.npy",
            (self._n_samples,),
            np.int32,
        )

        if self._include_lp:
            self._lp_token_ids = self._open_memmap(
                capture_dir / "response_token_ids.npy",
                (self._n_samples, self._max_resp),
                np.int32,
            )
            self._lp_token_logprobs = self._open_memmap(
                capture_dir / "response_token_logprobs.npy",
                (self._n_samples, self._max_resp),
                np.float32,
            )
            self._lp_topk_ids = self._open_memmap(
                capture_dir / "response_topk_token_ids.npy",
                (self._n_samples, self._max_resp, stored_top_k),
                np.int32,
            )
            self._lp_topk_logprobs = self._open_memmap(
                capture_dir / "response_topk_logprobs.npy",
                (self._n_samples, self._max_resp, stored_top_k),
                np.float32,
            )
        else:
            self._lp_token_ids = None
            self._lp_token_logprobs = None
            self._lp_topk_ids = None
            self._lp_topk_logprobs = None

        # --- Split ---
        labels = np.array(
            [int(bool(r["hallucinated"])) for r in self._meta], dtype=np.int32
        )
        if split == "all":
            self._split_indices = np.arange(len(labels), dtype=np.int64)
        else:
            self._split_indices = _make_split_indices(
                labels, split, val_fraction, random_seed
            )

        # Allow the parser to inject a pre-computed split (e.g. 90/10 train/val).
        if _override_indices is not None:
            self._split_indices = np.asarray(_override_indices, dtype=np.int64)

        # Split name for .df property — prefer explicit override, fall back to the split arg.
        self._split_name: str = _override_split_name if _override_split_name is not None else split

        # Pair training excludes rows that cannot supply two real response
        # steps.  Fixed-token evaluation applies its own minimum so t=0 keeps
        # every generation with at least one captured response token.
        required_tokens = self._min_response_tokens
        if self._fixed_token is not None:
            required_tokens = max(required_tokens, self._fixed_token + 1)
        if required_tokens:
            sample_rows = self._valid_sample_indices[self._split_indices]
            keep = np.asarray(self._resp_len[sample_rows] >= required_tokens)
            self._split_indices = self._split_indices[keep]
            if len(self._split_indices) == 0:
                raise ValueError(
                    f"No {self._split_name} rows have at least "
                    f"{required_tokens} captured response tokens"
                )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _open_memmap(path: Path, shape: tuple, dtype) -> np.memmap:
        if not path.exists():
            raise FileNotFoundError(f"memmap file not found: {path}")
        return np.memmap(str(path), dtype=dtype, mode="r", shape=shape)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return int(len(self._split_indices))

    # ------------------------------------------------------------------ #
    @property
    def df(self) -> "pd.DataFrame":
        """DataFrame view of this split's samples.

        Columns: prompt_hash (str), halu (int), split (str),
                 sample_index (int), prompt_len (int), response_len (int).
        len(dataset.df) == len(dataset).
        """
        import pandas as pd
        prompt_hashes = []
        halu_vals = []
        sample_indices = []
        prompt_lens = []
        response_lens = []
        for i in self._split_indices:
            meta = self._meta[int(i)]
            sample_row = int(self._valid_sample_indices[int(i)])
            prompt_hashes.append(meta["prompt_hash"])
            halu_vals.append(int(bool(meta["hallucinated"])))
            sample_indices.append(sample_row)
            prompt_lens.append(int(self._prompt_len[sample_row]))
            response_lens.append(int(self._resp_len[sample_row]))
        return pd.DataFrame({
            "prompt_hash": prompt_hashes,
            "halu": halu_vals,
            "split": [self._split_name] * len(self._split_indices),
            "sample_index": sample_indices,
            "prompt_len": prompt_lens,
            "response_len": response_lens,
        })

    @property
    def cache(self) -> np.memmap:
        """Full activation memmap (N_total, num_layers+1, max_resp, hidden_dim).
        Exposed for _split_view compatibility — consumers must use _row_indices to locate split rows."""
        return self._resp_act

    @property
    def labels(self) -> np.ndarray:
        """int32 label array for this split's samples (halu=1, non-halu=0)."""
        return np.array(
            [int(bool(self._meta[int(i)]["hallucinated"])) for i in self._split_indices],
            dtype=np.int32,
        )

    @property
    def _row_indices(self) -> np.ndarray:
        """Memmap row indices (into self.cache) for this split's samples.
        Exposed for _split_view compatibility."""
        return self._valid_sample_indices[self._split_indices]

    def get_auxiliary_fields(self, idx: int) -> Dict[str, Any]:
        """Return non-activation fields for a logical split row.

        Token-wise adapters use this hook to reuse logprob targets and metadata
        without calling ``__getitem__`` and triggering layer-view sampling.
        """
        meta_idx = int(self._split_indices[idx])
        sample_row = int(self._valid_sample_indices[meta_idx])
        fields: Dict[str, Any] = {
            "input_length": int(self._prompt_len[sample_row]),
            "response_len": int(self._resp_len[sample_row]),
        }
        if self._include_lp:
            fields.update(self._get_logprob_fields(sample_row))
        return fields

    # ------------------------------------------------------------------ #
    def get_single_layer_dataset(self, layer_id: int):
        """Return a SingleLayerDataset for one fixed layer (for linear_probe / saplma).

        Reuses the existing SingleLayerDataset class backed by this memmap's
        response_activations. layer_id is a model-layer index (position in dim-1
        of response_activations.npy).
        """
        from activation_logging.activation_parser import SingleLayerDataset
        labels_arr = self.labels
        hashes = [self._meta[int(i)]["prompt_hash"] for i in self._split_indices]
        row_idx = self._valid_sample_indices[self._split_indices]
        return SingleLayerDataset(
            cache=self._resp_act,
            labels=labels_arr,
            prompt_hashes=hashes,
            layer_pos=layer_id,
            layer_id=layer_id,
            _row_indices=row_idx,
            response_lens=np.asarray(self._resp_len[row_idx], dtype=np.int32),
        )

    # ------------------------------------------------------------------ #
    def slice_layers(
        self,
        layers: List[int],
        num_views: Optional[int] = None,
    ) -> "MemmapContrastiveDataset":
        """Return a new dataset restricted to a subset of model layers.

        Shares the underlying memmaps — no data is copied. The returned
        dataset exposes the same split as the caller (injected via
        ``_override_indices``), so train/test alignment is preserved.

        Args:
            layers: Model-layer indices to keep (must be within
                ``[0, num_layers+1)``).
            num_views: Views to sample per item. Defaults to
                ``len(layers)`` so all selected layers are used.
        """
        if num_views is None:
            # Layer-wise evaluation historically averages one view per selected
            # layer.  In token-wise mode the selected layers are the sequence,
            # not the view axis, so preserve the caller's token-view count.
            num_views = (
                self._num_views if self._view_axis == "token" else len(layers)
            )
        return MemmapContrastiveDataset(
            self._capture_dir,
            split="all",  # _override_indices below replaces this
            num_views=num_views,
            relevant_layers=layers,
            fixed_layer=None,
            view_sampling_with_replacement=self._view_with_replacement,
            view_axis=self._view_axis,
            token_pair_mode=self._token_pair_mode,
            fixed_token=self._fixed_token,
            min_response_tokens=self._min_response_tokens,
            include_response_logprobs=self._include_lp,
            response_logprobs_top_k=self._target_top_k,
            pad_length=self._pad_length,
            include_response_attention=self._include_attn,
            attention_summary=self._attn_summary,
            attention_target_layer_offset_forward=self._attn_offset_fwd,
            attention_target_layer_offset_backward=self._attn_offset_bwd,
            label_source=self._label_source,
            _override_indices=self._split_indices,
            _override_split_name=self._split_name,
        )

    # ------------------------------------------------------------------ #
    def fixed_token_view(self, token_index: int) -> "MemmapContrastiveDataset":
        """Return this split as one token-wise view across selected layers.

        The new dataset shares all memmaps with the caller and filters only
        rows for which ``token_index`` is a real captured response step.
        """
        token_index = int(token_index)
        return MemmapContrastiveDataset(
            self._capture_dir,
            split="all",
            num_views=1,
            relevant_layers=self._relevant_layers,
            view_sampling_with_replacement=False,
            view_axis="token",
            token_pair_mode=self._token_pair_mode,
            fixed_token=token_index,
            min_response_tokens=token_index + 1,
            include_response_logprobs=self._include_lp,
            response_logprobs_top_k=self._target_top_k,
            pad_length=self._pad_length,
            label_source=self._label_source,
            _override_indices=self._split_indices,
            _override_split_name=self._split_name,
        )

    # ------------------------------------------------------------------ #
    def _select_view_positions(self) -> List[int]:
        """Return positions (indices into self._relevant_layers) for K views."""
        relevant = self._relevant_layers
        if self._fixed_layer is not None:
            fixed_pos = relevant.index(self._fixed_layer)
            other_positions = [i for i in range(len(relevant)) if i != fixed_pos]
            need = self._num_views - 1
            if self._view_with_replacement:
                if need > 0 and not other_positions:
                    raise ValueError(
                        f"num_views > 1 but no other layers besides fixed_layer={self._fixed_layer}"
                    )
                sampled = random.choices(other_positions, k=need) if need > 0 else []
            else:
                if len(other_positions) < need:
                    raise ValueError(
                        f"Not enough non-fixed layers for {self._num_views} unique views"
                    )
                sampled = random.sample(other_positions, need) if need > 0 else []
            return [fixed_pos, *sampled]

        if self._view_with_replacement:
            if not relevant:
                raise ValueError("relevant_layers is empty")
            return random.choices(range(len(relevant)), k=self._num_views)

        if len(relevant) < self._num_views:
            raise ValueError(
                f"Not enough layers for {self._num_views} unique views "
                f"(have {len(relevant)})"
            )
        return random.sample(range(len(relevant)), self._num_views)

    # ------------------------------------------------------------------ #
    def _select_token_positions(self, sample_row: int) -> List[int]:
        """Return real response-token positions for token-wise views."""
        response_len = max(0, min(int(self._resp_len[sample_row]), self._max_resp))
        if self._fixed_token is not None:
            if self._fixed_token >= response_len:
                raise ValueError(
                    f"fixed_token={self._fixed_token} is unavailable for sample row "
                    f"{sample_row} with response_len={response_len}"
                )
            return [self._fixed_token] * self._num_views

        if response_len <= 0:
            raise ValueError(
                f"sample row {sample_row} has no captured response tokens"
            )

        if self._token_pair_mode == "first_anchored":
            if self._num_views == 1:
                return [0]
            later = list(range(1, response_len))
            need = self._num_views - 1
            if self._view_with_replacement:
                if not later:
                    raise ValueError(
                        "first_anchored token views require response_len >= 2"
                    )
                sampled = random.choices(later, k=need)
            else:
                if len(later) < need:
                    raise ValueError(
                        f"first_anchored token views require at least "
                        f"{self._num_views} response tokens; got {response_len}"
                    )
                sampled = random.sample(later, need)
            return [0, *sampled]

        positions = list(range(response_len))
        if self._view_with_replacement:
            return random.choices(positions, k=self._num_views)
        if len(positions) < self._num_views:
            raise ValueError(
                f"random_distinct token views require at least {self._num_views} "
                f"response tokens; got {response_len}"
            )
        return random.sample(positions, self._num_views)

    # ------------------------------------------------------------------ #
    def _attn_stats_for_layer(self, sample_row: int, model_layer: int) -> np.ndarray:
        """Read and summarise attention at one model-layer for one sample.

        model_layer is indexed into response_attention's first axis (range
        [0, num_layers)). Out-of-range layer → NaN row.
        """
        if model_layer < 0 or model_layer >= self._num_layers:
            return np.full((_ATTN_STAT_DIM,), np.nan, dtype=np.float32)
        attn_block = self._resp_attn[sample_row, model_layer]  # (r_max, r_max) fp16
        # Cap response_len to r_max — the attention block was sliced to r_max
        # by the capture writer (stitching contract §"Upstream stitching" in #72).
        rlen = int(self._resp_len[sample_row])
        rlen = min(rlen, self._r_max)
        return _compute_attn_stats(np.asarray(attn_block), rlen)

    # ------------------------------------------------------------------ #
    def _attn_full_for_layer(self, sample_row: int, model_layer: int) -> torch.Tensor:
        """Return full attention at one target layer with prompt-sink augmentation.

        The stored response_attention rows sum to 1 − (mass on prompt keys)
        because only the response-to-response slice is captured. We recover the
        lost prompt mass as a single extra "sink" column so each valid query row
        sums to exactly 1 over r_eff response keys + 1 prompt-sink cell.

        Returns
        -------
        Tensor (r_max, r_max + 1) float32
            Rows 0..r_eff-1 have finite values in columns [0, r_eff) (response
            keys) and column r_max (prompt sink). Columns [r_eff, r_max) within
            valid rows are NaN (key padding). Rows r_eff..r_max-1 are entirely
            NaN. Out-of-range model_layer → full-NaN tensor.
        """
        r_max = self._r_max
        out = torch.full((r_max, r_max + 1), float("nan"), dtype=torch.float32)

        if model_layer < 0 or model_layer >= self._num_layers:
            return out  # out-of-range layer → full NaN, contributes 0 to loss

        rlen = int(self._resp_len[sample_row])
        r_eff = min(rlen, r_max)
        if r_eff <= 0:
            return out

        # fp16 → fp32 to avoid accumulation error when summing for the sink column
        raw = torch.from_numpy(
            np.array(self._resp_attn[sample_row, model_layer], dtype=np.float32)
        )  # (r_max, r_max)

        out[:r_eff, :r_eff] = raw[:r_eff, :r_eff]
        # Prompt sink: mass not captured in the response-to-response slice.
        out[:r_eff, r_max] = (1.0 - out[:r_eff, :r_eff].nansum(dim=-1)).clamp_(min=0.0)
        return out

    # ------------------------------------------------------------------ #
    def _get_logprob_fields(self, sample_row: int) -> Dict[str, torch.Tensor]:
        target_len = self._pad_length
        target_top_k = self._target_top_k

        token_ids_out = torch.full((target_len,), -1, dtype=torch.int32)
        token_logprobs_out = torch.full((target_len,), float("nan"), dtype=torch.float32)
        topk_ids_out = torch.full((target_len, target_top_k), -1, dtype=torch.int32)
        topk_logprobs_out = torch.full(
            (target_len, target_top_k), float("nan"), dtype=torch.float32
        )
        token_mask = torch.zeros((target_len,), dtype=torch.bool)

        # The stored arrays are length max_response_len; we copy up to
        # min(pad_length, response_len) into the output.
        rlen = int(self._resp_len[sample_row])
        copied_len = max(0, min(target_len, rlen, self._max_resp))
        copied_top_k = min(target_top_k, self._lp_topk_ids.shape[-1])

        if copied_len > 0 and copied_top_k > 0:
            # np.array() copies the memmap slice into a writable buffer; the
            # raw memmap is read-only and torch.from_numpy warns + can produce
            # undefined behaviour on writable target tensors.
            token_ids_out[:copied_len] = torch.from_numpy(
                np.array(self._lp_token_ids[sample_row, :copied_len], dtype=np.int32)
            )
            token_logprobs_out[:copied_len] = torch.from_numpy(
                np.array(
                    self._lp_token_logprobs[sample_row, :copied_len], dtype=np.float32
                )
            )
            topk_ids_out[:copied_len, :copied_top_k] = torch.from_numpy(
                np.array(
                    self._lp_topk_ids[sample_row, :copied_len, :copied_top_k],
                    dtype=np.int32,
                )
            )
            topk_logprobs_out[:copied_len, :copied_top_k] = torch.from_numpy(
                np.array(
                    self._lp_topk_logprobs[sample_row, :copied_len, :copied_top_k],
                    dtype=np.float32,
                )
            )
            token_mask[:copied_len] = True

        return {
            "response_token_ids": token_ids_out,
            "response_token_logprobs": token_logprobs_out,
            "response_topk_token_ids": topk_ids_out,
            "response_topk_logprobs": topk_logprobs_out,
            "response_logprob_mask": token_mask,
            "response_logprob_len": torch.tensor(copied_len, dtype=torch.long),
            "response_logprobs_top_k": torch.tensor(target_top_k, dtype=torch.long),
            "response_logprobs_available_top_k": torch.tensor(
                copied_top_k, dtype=torch.long
            ),
        }

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta_idx = int(self._split_indices[idx])
        sample_row = int(self._valid_sample_indices[meta_idx])
        row_meta = self._meta[meta_idx]

        # self._resp_act[sample_row] is
        # (num_layers+1, max_response_len, hidden_dim). Layer-wise views keep
        # one layer and its token sequence; token-wise views keep one response
        # step and its depth sequence.
        if self._view_axis == "layer":
            view_positions = self._select_view_positions()
            view_model_layers = [self._relevant_layers[p] for p in view_positions]
            acts_slice = np.array(
                self._resp_act[sample_row, view_model_layers, :, :],
                dtype=np.float32,
            )  # (K, max_resp, hidden_dim)
            view_token_positions = None
        else:
            view_token_positions = self._select_token_positions(sample_row)
            view_model_layers = self._relevant_layers
            # Stack per token explicitly: NumPy's multi-axis advanced indexing
            # otherwise changes axis order depending on list shapes.
            acts_slice = np.stack(
                [
                    np.array(
                        self._resp_act[
                            sample_row, self._relevant_layers, token_position, :
                        ],
                        dtype=np.float32,
                    )
                    for token_position in view_token_positions
                ],
                axis=0,
            )  # (K, selected_layers, hidden_dim)
            view_positions = view_token_positions

        # np.array()/np.stack ensure writable owned memory; raw memmap slices
        # are read-only and torch.from_numpy warns about them.
        views_activations = torch.from_numpy(acts_slice)

        sample: Dict[str, Any] = {
            "hashkey": row_meta["prompt_hash"],
            "halu": torch.tensor(
                float(bool(row_meta["hallucinated"])), dtype=torch.float32
            ),
            "views_activations": views_activations,
            "view_indices": torch.tensor(view_positions, dtype=torch.long),
            "input_length": int(self._prompt_len[sample_row]),
            "response_len": int(self._resp_len[sample_row]),
        }
        if view_token_positions is not None:
            sample["view_token_indices"] = torch.tensor(
                view_token_positions, dtype=torch.long
            )

        # --- Logprob fields (Mechanism F) ---
        if self._include_lp:
            sample.update(self._get_logprob_fields(sample_row))

        # --- Attention summary fields (Mechanism K) ---
        if self._include_attn:
            if self._attn_summary == "stats":
                if self._attn_offset_fwd is not None:
                    stats_fwd = np.stack(
                        [
                            self._attn_stats_for_layer(sample_row, m + self._attn_offset_fwd)
                            for m in view_model_layers
                        ],
                        axis=0,
                    )  # (K, _ATTN_STAT_DIM)
                    sample["attention_forward"] = torch.from_numpy(stats_fwd)

                if self._attn_offset_bwd is not None:
                    stats_bwd = np.stack(
                        [
                            self._attn_stats_for_layer(sample_row, m - self._attn_offset_bwd)
                            for m in view_model_layers
                        ],
                        axis=0,
                    )  # (K, _ATTN_STAT_DIM)
                    sample["attention_backward"] = torch.from_numpy(stats_bwd)

            else:  # "full"
                # Single target layer per (view k, direction d): view_indices[k] ± offset.
                # Shape per view: (r_max, r_max + 1). Stacked: (K, r_max, r_max + 1).
                if self._attn_offset_fwd is not None:
                    full_fwd = torch.stack(
                        [
                            self._attn_full_for_layer(sample_row, m + self._attn_offset_fwd)
                            for m in view_model_layers
                        ],
                        dim=0,
                    )  # (K, r_max, r_max + 1)
                    sample["attention_forward"] = full_fwd

                if self._attn_offset_bwd is not None:
                    full_bwd = torch.stack(
                        [
                            self._attn_full_for_layer(sample_row, m - self._attn_offset_bwd)
                            for m in view_model_layers
                        ],
                        dim=0,
                    )  # (K, r_max, r_max + 1)
                    sample["attention_backward"] = full_bwd

        return sample
