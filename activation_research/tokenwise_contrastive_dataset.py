"""Token-wise views over an existing contrastive activation dataset.

The base contrastive datasets already own the expensive activation cache and
split metadata.  This adapter changes only the final indexing operation:
layer-wise views ``cache[row, layer, :, :]`` become token-wise views
``cache[row, selected_layers, token, :]``.  Pair training, token-zero KNN
evaluation, and later-token diagnostics can therefore share one base dataset
per split without reopening or reloading activations.
"""
from __future__ import annotations

import random
from typing import Any, Dict, Literal, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenwiseContrastiveDataset(Dataset):
    """Slice token trajectories from an existing contrastive dataset cache.

    Parameters
    ----------
    base_dataset:
        Existing contrastive dataset exposing ``cache``, ``labels``,
        ``_row_indices``, and either ``df`` or ``prompt_hashes``.  No activation
        arrays are copied by this adapter.
    layer_positions:
        Positions on the base cache's layer axis, in the fixed order presented
        to the encoder.  For Issue #151 these are post-block rows ``1..32``.
    token_pair_mode:
        ``first_anchored`` emits token zero followed by sampled later tokens;
        ``first_same`` emits token zero twice so model stochasticity supplies
        the augmentation control; ``shuffled_later`` pairs token zero with a
        later token from a different, label/length-matched response;
        ``random_distinct`` samples all positions without replacement.
    fixed_token:
        When set, emit only this token position.  The primary KNN surface uses
        ``fixed_token=0`` and ``num_views=1``.
    min_response_tokens:
        Filter logical split rows by real captured response length.  Pair
        datasets use two; token-zero evaluation uses one.
    later_token_sampling:
        ``uniform`` samples later positions directly. ``stratified`` first
        divides the available later positions into early/middle/late bins,
        samples bins uniformly, and then samples within each selected bin.
    source_indices:
        Optional logical base-dataset indices to inherit.  Used by
        :meth:`fixed_token_view` so later-token diagnostics stay within the
        caller's current evaluation population.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        *,
        layer_positions: Sequence[int],
        num_views: int = 2,
        token_pair_mode: Literal[
            "first_anchored", "first_same", "shuffled_later", "random_distinct"
        ] = "first_anchored",
        fixed_token: Optional[int] = None,
        min_response_tokens: int = 0,
        view_sampling_with_replacement: bool = False,
        shuffle_length_bucket_size: int = 8,
        later_token_sampling: Literal["uniform", "stratified"] = "uniform",
        emit_view_logprob_targets: bool = False,
        source_indices: Optional[Sequence[int]] = None,
    ) -> None:
        required = ("cache", "labels", "_row_indices")
        missing = [name for name in required if not hasattr(base_dataset, name)]
        if missing:
            raise TypeError(
                "base_dataset does not satisfy the contrastive cache contract; "
                f"missing {missing}"
            )

        self.base_dataset = base_dataset
        self.cache = base_dataset.cache
        if getattr(self.cache, "ndim", None) != 4:
            raise ValueError("base contrastive cache must have shape (N, L, T, H)")

        self.layer_positions = [int(position) for position in layer_positions]
        if not self.layer_positions:
            raise ValueError("layer_positions cannot be empty")
        for position in self.layer_positions:
            if position < 0 or position >= self.cache.shape[1]:
                raise ValueError(
                    f"layer position {position} outside cache layer axis "
                    f"[0, {self.cache.shape[1]})"
                )

        self.num_views = int(num_views)
        if self.num_views < 1:
            raise ValueError("num_views must be at least one")
        self.fixed_token = int(fixed_token) if fixed_token is not None else None
        if token_pair_mode not in (
            "first_anchored",
            "first_same",
            "shuffled_later",
            "random_distinct",
        ):
            raise ValueError(
                "token_pair_mode must be one of {'first_anchored', "
                "'first_same', 'shuffled_later', 'random_distinct'}"
            )
        if (
            self.fixed_token is None
            and token_pair_mode in ("first_same", "shuffled_later")
            and self.num_views != 2
        ):
            raise ValueError(f"{token_pair_mode} requires exactly two views")
        self.token_pair_mode = str(token_pair_mode)
        if self.fixed_token is not None and not (
            0 <= self.fixed_token < self.cache.shape[2]
        ):
            raise ValueError(
                f"fixed_token={self.fixed_token} outside cache token axis "
                f"[0, {self.cache.shape[2]})"
            )
        self.min_response_tokens = int(min_response_tokens)
        if self.min_response_tokens < 0:
            raise ValueError("min_response_tokens must be non-negative")
        self.view_sampling_with_replacement = bool(view_sampling_with_replacement)
        self.shuffle_length_bucket_size = int(shuffle_length_bucket_size)
        if self.shuffle_length_bucket_size < 1:
            raise ValueError("shuffle_length_bucket_size must be positive")
        self.later_token_sampling = str(later_token_sampling).strip().lower()
        if self.later_token_sampling not in {"uniform", "stratified"}:
            raise ValueError(
                "later_token_sampling must be one of {'uniform', 'stratified'}"
            )
        self.emit_view_logprob_targets = bool(emit_view_logprob_targets)

        base_labels = np.asarray(base_dataset.labels)
        self._base_labels = base_labels
        base_row_indices = base_dataset._row_indices
        self._base_rows = (
            np.arange(len(base_labels), dtype=np.int64)
            if base_row_indices is None
            else np.asarray(base_row_indices, dtype=np.int64)
        )
        if len(base_labels) != len(self._base_rows):
            raise ValueError("base labels and row indices must have equal length")

        if hasattr(base_dataset, "df"):
            base_df = base_dataset.df.reset_index(drop=True)
            if len(base_df) != len(base_labels):
                raise ValueError("base df and labels must have equal length")
            self._base_df = base_df
            self._response_lengths = base_df["response_len"].to_numpy(
                dtype=np.int32
            )
            self._prompt_hashes = base_df["prompt_hash"].astype(str).tolist()
        else:
            self._base_df = None
            if not hasattr(base_dataset, "response_lengths"):
                raise TypeError(
                    "base dataset without df must expose response_lengths"
                )
            response_lengths = np.asarray(base_dataset.response_lengths)
            self._response_lengths = response_lengths[self._base_rows].astype(
                np.int32, copy=False
            )
            self._prompt_hashes = list(base_dataset.prompt_hashes)

        candidates = (
            np.arange(len(base_labels), dtype=np.int64)
            if source_indices is None
            else np.asarray(source_indices, dtype=np.int64)
        )
        if candidates.ndim != 1:
            raise ValueError("source_indices must be one-dimensional")
        if len(candidates) and (
            int(candidates.min()) < 0 or int(candidates.max()) >= len(base_labels)
        ):
            raise IndexError("source_indices contain an out-of-range logical row")

        required_tokens = self.min_response_tokens
        if self.fixed_token is not None:
            required_tokens = max(required_tokens, self.fixed_token + 1)
        if required_tokens:
            candidates = candidates[
                self._response_lengths[candidates] >= required_tokens
            ]
        if len(candidates) == 0:
            raise ValueError(
                f"No base rows have at least {required_tokens} captured tokens"
            )
        self._valid_indices = candidates

        # The shuffled control keeps the anchor cohort identical to the other
        # two arms. Every anchor must have a different same-label partner; we
        # fail instead of silently filtering rows. Prefer the exact response-
        # length bucket and fall back to the nearest same-label response length
        # only when a bucket contains the anchor alone.
        self._shuffle_partner_candidates: dict[int, tuple[int, ...]] = {}
        if self.token_pair_mode == "shuffled_later":
            valid = [int(value) for value in self._valid_indices]
            groups: dict[tuple[int, int], list[int]] = {}
            for logical_idx in valid:
                label = int(self._base_labels[logical_idx])
                bucket = self._length_bucket(self._response_lengths[logical_idx])
                groups.setdefault((label, bucket), []).append(logical_idx)

            for logical_idx in valid:
                label = int(self._base_labels[logical_idx])
                bucket = self._length_bucket(self._response_lengths[logical_idx])
                exact = [
                    partner
                    for partner in groups[(label, bucket)]
                    if partner != logical_idx
                ]
                if exact:
                    partners = exact
                else:
                    same_label = [
                        partner
                        for partner in valid
                        if partner != logical_idx
                        and int(self._base_labels[partner]) == label
                    ]
                    if not same_label:
                        raise ValueError(
                            "shuffled_later requires at least two eligible rows "
                            f"for label {label}"
                        )
                    anchor_len = int(self._response_lengths[logical_idx])
                    best_gap = min(
                        abs(int(self._response_lengths[partner]) - anchor_len)
                        for partner in same_label
                    )
                    partners = [
                        partner
                        for partner in same_label
                        if abs(int(self._response_lengths[partner]) - anchor_len)
                        == best_gap
                    ]
                self._shuffle_partner_candidates[logical_idx] = tuple(partners)

        # Compatibility attributes consumed by the existing trainer.
        self._num_views = self.num_views
        self._max_resp = int(self.cache.shape[2])

    def __len__(self) -> int:
        return int(len(self._valid_indices))

    @property
    def labels(self) -> np.ndarray:
        return self._base_labels[self._valid_indices]

    @property
    def _row_indices(self) -> np.ndarray:
        return self._base_rows[self._valid_indices]

    @property
    def df(self):
        if self._base_df is None:
            raise AttributeError("base dataset does not expose df")
        return self._base_df.iloc[self._valid_indices].reset_index(drop=True)

    def fixed_token_view(self, token_index: int) -> "TokenwiseContrastiveDataset":
        """Return a one-view fixed-token adapter sharing the same base cache."""
        token_index = int(token_index)
        return TokenwiseContrastiveDataset(
            self.base_dataset,
            layer_positions=self.layer_positions,
            num_views=1,
            token_pair_mode=self.token_pair_mode,
            fixed_token=token_index,
            min_response_tokens=token_index + 1,
            view_sampling_with_replacement=False,
            shuffle_length_bucket_size=self.shuffle_length_bucket_size,
            later_token_sampling=self.later_token_sampling,
            emit_view_logprob_targets=self.emit_view_logprob_targets,
            source_indices=self._valid_indices,
        )

    def _length_bucket(self, response_len: int) -> int:
        return max(0, (int(response_len) - 2) // self.shuffle_length_bucket_size)

    def _sample_later_positions(
        self, later: list[int], needed: int
    ) -> list[int]:
        if self.view_sampling_with_replacement:
            if not later:
                raise ValueError(
                    "first_anchored views require response_len >= 2"
                )
        elif len(later) < needed:
            raise ValueError(
                f"first_anchored {self.num_views}-view sampling requires "
                f"response_len >= {self.num_views}; got {len(later) + 1}"
            )

        if self.later_token_sampling == "uniform":
            return (
                random.choices(later, k=needed)
                if self.view_sampling_with_replacement
                else random.sample(later, needed)
            )

        # Equal-probability position regimes prevent long responses from
        # making late positions dominate. np.array_split keeps adjacent token
        # positions together and handles short responses without empty middle
        # regimes after filtering below.
        bins = [
            [int(position) for position in bucket]
            for bucket in np.array_split(later, 3)
            if len(bucket)
        ]
        if needed <= len(bins):
            selected_bins = random.sample(bins, needed)
            return [random.choice(bucket) for bucket in selected_bins]
        if self.view_sampling_with_replacement:
            return [random.choice(random.choice(bins)) for _ in range(needed)]
        # More keys than non-empty regimes: retain distinct positions while
        # degrading gracefully to uniform sampling for the excess keys.
        return random.sample(later, needed)

    def _select_view_sources(
        self, logical_idx: int
    ) -> tuple[list[int], list[int]]:
        response_len = max(
            0,
            min(
                int(self._response_lengths[logical_idx]),
                int(self.cache.shape[2]),
            ),
        )
        if self.fixed_token is not None:
            if self.fixed_token >= response_len:
                raise ValueError(
                    f"fixed_token={self.fixed_token} unavailable for "
                    f"response_len={response_len}"
                )
            return (
                [logical_idx] * self.num_views,
                [self.fixed_token] * self.num_views,
            )

        if self.token_pair_mode == "first_same":
            return [logical_idx, logical_idx], [0, 0]

        if self.token_pair_mode == "shuffled_later":
            partner = random.choice(self._shuffle_partner_candidates[logical_idx])
            partner_len = max(
                0,
                min(
                    int(self._response_lengths[partner]),
                    int(self.cache.shape[2]),
                ),
            )
            if partner_len < 2:
                raise RuntimeError("shuffled_later selected an ineligible partner")
            return [logical_idx, partner], [0, random.randrange(1, partner_len)]

        if self.token_pair_mode == "first_anchored":
            if self.num_views == 1:
                return [logical_idx], [0]
            later = list(range(1, response_len))
            needed = self.num_views - 1
            sampled = self._sample_later_positions(later, needed)
            return [logical_idx] * self.num_views, [0, *sampled]

        positions = list(range(response_len))
        if self.view_sampling_with_replacement:
            return (
                [logical_idx] * self.num_views,
                random.choices(positions, k=self.num_views),
            )
        if len(positions) < self.num_views:
            raise ValueError(
                f"random_distinct {self.num_views}-view sampling requires "
                f"response_len >= {self.num_views}; got {response_len}"
            )
        return (
            [logical_idx] * self.num_views,
            random.sample(positions, self.num_views),
        )

    def _auxiliary_fields(self, logical_idx: int) -> Dict[str, Any]:
        getter = getattr(self.base_dataset, "get_auxiliary_fields", None)
        if getter is not None:
            return dict(getter(logical_idx))

        # Compatibility with the RAM-resident PreloadedActivationDataset.
        if getattr(self.base_dataset, "include_response_logprobs", False):
            cache_idx = int(self._base_rows[logical_idx])
            return dict(self.base_dataset._get_logprobs(logical_idx, cache_idx))
        return {}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        logical_idx = int(self._valid_indices[idx])
        source_indices, token_positions = self._select_view_sources(logical_idx)
        source_cache_indices = [
            int(self._base_rows[source_idx]) for source_idx in source_indices
        ]

        trajectories = np.stack(
            [
                np.array(
                    self.cache[
                        source_cache_idx,
                        self.layer_positions,
                        token_position,
                        :,
                    ],
                    dtype=np.float32,
                )
                for source_cache_idx, token_position in zip(
                    source_cache_indices, token_positions
                )
            ],
            axis=0,
        )
        sample: Dict[str, Any] = {
            "views_activations": torch.from_numpy(trajectories),
            "view_indices": torch.tensor(token_positions, dtype=torch.long),
            "view_token_indices": torch.tensor(
                token_positions, dtype=torch.long
            ),
            "view_source_indices": torch.tensor(
                source_indices, dtype=torch.long
            ),
            "view_source_response_lens": torch.tensor(
                [int(self._response_lengths[source_idx]) for source_idx in source_indices],
                dtype=torch.long,
            ),
            "temporal_same_response": torch.tensor(
                source_indices[0] == source_indices[1]
                if len(source_indices) > 1
                else True,
                dtype=torch.bool,
            ),
            "halu": torch.tensor(
                float(self._base_labels[logical_idx]), dtype=torch.float32
            ),
            "hashkey": self._prompt_hashes[logical_idx],
            "response_len": int(self._response_lengths[logical_idx]),
        }
        auxiliary_by_source: dict[int, Dict[str, Any]] = {}
        for source_idx in set(source_indices):
            auxiliary_by_source[source_idx] = self._auxiliary_fields(source_idx)
        sample.update(auxiliary_by_source[logical_idx])
        if self.emit_view_logprob_targets:
            targets = [
                auxiliary_by_source[source_idx].get("response_token_logprobs")
                for source_idx in source_indices
            ]
            if all(target is not None for target in targets):
                sample["view_response_token_logprobs"] = torch.stack(
                    [target.float() for target in targets], dim=0
                )
        return sample
