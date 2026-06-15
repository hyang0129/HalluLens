"""
composite_memmap_parser.py — issue #124

Parser that exposes a virtual "merged" capture stitched from multiple icr_capture
directories. No data is copied — each source's memmaps are opened in-place and
reads are routed by an index map.

Use case: combine SimpleQA's heavily-imbalanced train set with sampled rows from
PopQA so the contrastive method sees ~50/50 halu/non-halu without re-running
~2 TB of inference captures.

Public API mirrors :class:`MemmapActivationParser` so it slots into
``scripts/run_experiment.py`` with minimal plumbing:

    parser = CompositeMemmapActivationParser(
        sources=[
            {"dir": "shared/icr_capture/simpleqa_train_...", "halu": "all",   "nonhalu": "all"},
            {"dir": "shared/icr_capture/popqa_train_...",    "halu": 224,    "nonhalu": 3236},
        ],
        random_seed=42,
        split_strategy="three_way",
        source_sample_seed=0,   # deterministic across folds
    )
    train_ds = parser.get_dataset("train", ...)
    val_ds   = parser.get_dataset("val", ...)

Note: source captures MUST share num_layers / hidden_dim / r_max / dtype /
max_prompt_len / max_response_len / response_logprobs_top_k. The parser raises
if they don't.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from activation_research.memmap_contrastive_dataset import MemmapContrastiveDataset

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

_SHARED_KEYS = (
    "num_layers",
    "hidden_dim",
    "r_max",
    "dtype",
    "max_prompt_len",
    "max_response_len",
    "response_logprobs_top_k",
)


def _parse_count(v: int | str | None) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, str):
        if v.lower() == "all":
            return None
        return int(v)
    return int(v)


def _select_meta_indices(
    meta: list[dict],
    halu: bool,
    n: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    pool = np.array(
        [i for i, r in enumerate(meta) if bool(r["hallucinated"]) == halu],
        dtype=np.int64,
    )
    if n is None:
        return pool
    if n > len(pool):
        raise ValueError(
            f"Requested {n} halu={halu} samples but only {len(pool)} available."
        )
    picked = rng.choice(len(pool), size=n, replace=False)
    picked.sort()
    return pool[picked]


class CompositeMemmapActivationParser:
    """Parser over multiple icr_capture dirs, with per-source halu/nonhalu sampling.

    Stitches sampled rows into one virtual dataset without copying any memmap
    data. Train/val split is a single 90/10 stratified split on the *combined*
    label vector.
    """

    def __init__(
        self,
        sources: Sequence[dict],
        *,
        random_seed: int,
        split_strategy: Literal["none", "three_way"] = "three_way",
        source_sample_seed: int = 0,
        verbose: bool = False,
    ) -> None:
        if not sources:
            raise ValueError("sources must be non-empty.")

        self._random_seed = int(random_seed)
        self._source_sample_seed = int(source_sample_seed)
        self._split_strategy = split_strategy

        # --- Load configs, validate shape compat ---
        self._dirs: list[Path] = []
        self._cfgs: list[dict] = []
        self._metas: list[list[dict]] = []
        ref_cfg: Optional[dict] = None
        for spec in sources:
            d = Path(spec["dir"])
            if not d.exists():
                raise FileNotFoundError(f"source dir not found: {d}")
            with (d / "config.json").open() as fh:
                cfg = json.load(fh)
            if ref_cfg is None:
                ref_cfg = cfg
            else:
                for k in _SHARED_KEYS:
                    if cfg.get(k) != ref_cfg.get(k):
                        raise ValueError(
                            f"Source {d.name} config mismatch on {k!r}: "
                            f"got {cfg.get(k)!r}, expected {ref_cfg.get(k)!r}."
                        )
            meta_rows: list[dict] = []
            with (d / "meta.jsonl").open() as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        meta_rows.append(json.loads(line))
            if not meta_rows:
                raise ValueError(f"empty meta.jsonl in {d}")
            self._dirs.append(d)
            self._cfgs.append(cfg)
            self._metas.append(meta_rows)

        # --- Per-source sampling (deterministic across folds via source_sample_seed) ---
        sample_rng = np.random.default_rng(self._source_sample_seed)
        per_source_meta_indices: list[np.ndarray] = []
        summary_lines: list[str] = []
        for spec, d, meta in zip(sources, self._dirs, self._metas, strict=True):
            n_halu = _parse_count(spec.get("halu", "all"))
            n_nh = _parse_count(spec.get("nonhalu", "all"))
            halu_idx = _select_meta_indices(meta, True, n_halu, sample_rng)
            nh_idx = _select_meta_indices(meta, False, n_nh, sample_rng)
            picked = np.concatenate([halu_idx, nh_idx])
            picked.sort()
            per_source_meta_indices.append(picked)
            summary_lines.append(
                f"{d.name}: halu={len(halu_idx)} non_halu={len(nh_idx)} total={len(picked)}"
            )

        # --- Build composite row_map: (source_idx, meta_idx_in_source) ---
        row_map_parts = []
        for src_idx, picked in enumerate(per_source_meta_indices):
            col_src = np.full((len(picked),), src_idx, dtype=np.int64)
            row_map_parts.append(np.stack([col_src, picked], axis=1))
        self._row_map: np.ndarray = np.concatenate(row_map_parts, axis=0)  # (N, 2)
        N = int(self._row_map.shape[0])
        if N == 0:
            raise ValueError("Composite parser selected zero samples.")

        # Labels for stratified split.
        labels = np.array(
            [
                int(bool(self._metas[int(s)][int(m)]["hallucinated"]))
                for s, m in self._row_map
            ],
            dtype=np.int32,
        )
        self._labels = labels

        # --- 90/10 train/val split ---
        all_idx = np.arange(N, dtype=np.int64)
        self._train_idx: Optional[np.ndarray] = None
        self._val_idx: Optional[np.ndarray] = None
        if split_strategy == "three_way":
            self._train_idx, self._val_idx = train_test_split(
                all_idx, test_size=0.1, stratify=labels, random_state=random_seed,
            )
        elif split_strategy == "none":
            # Composite parsers are intended for train-side merging; the test
            # capture should stay a single dir. Allow "none" anyway in case a
            # future use case needs it.
            pass
        else:
            raise ValueError(f"Unknown split_strategy: {split_strategy!r}")

        # --- Build the per-source MemmapContrastiveDataset placeholders.
        # These are constructed lazily in get_dataset() because dataset kwargs
        # (relevant_layers, num_views, etc.) vary by call.
        # Cache here only the meta + cfg already loaded.

        if verbose:
            logger.info(
                f"CompositeMemmapActivationParser: N={N}  "
                f"halu_rate={labels.mean():.3f}  "
                f"split_strategy={split_strategy}  "
                f"random_seed={random_seed}  source_sample_seed={source_sample_seed}"
            )
            for line in summary_lines:
                logger.info(f"  {line}")
            if split_strategy == "three_way":
                logger.info(
                    f"  train={len(self._train_idx)}  val={len(self._val_idx)}"
                )

        # --- Build a parser-level full DataFrame (same shape as MemmapActivationParser.df) ---
        import pandas as pd
        split_col = np.full(N, "train", dtype=object)
        if self._val_idx is not None:
            split_col[self._val_idx] = "val"
        elif split_strategy == "none":
            split_col[:] = "test"
        rows: list[dict] = []
        for i in range(N):
            s = int(self._row_map[i, 0])
            m = int(self._row_map[i, 1])
            mrow = self._metas[s][m]
            rows.append({
                "prompt_hash": mrow["prompt_hash"],
                "halu": int(bool(mrow["hallucinated"])),
                "split": split_col[i],
                "sample_index": int(mrow["sample_index"]),
                "prompt_len": int(mrow.get("prompt_len", 0)),
                "response_len": int(mrow.get("response_len", 0)),
                "source": self._dirs[s].name,
            })
        self._df = pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    @property
    def df(self) -> "pd.DataFrame":
        return self._df

    @property
    def split_strategy(self) -> str:
        return self._split_strategy

    # ------------------------------------------------------------------ #
    def get_dataset(
        self,
        split: Literal["train", "val", "test"],
        **kwargs: Any,
    ) -> "CompositeMemmapContrastiveDataset":
        if self._split_strategy == "three_way":
            if split == "test":
                raise ValueError(
                    "CompositeMemmapActivationParser was constructed over train "
                    "captures (split_strategy='three_way'). Use a separate "
                    "MemmapActivationParser for the test capture."
                )
            # Derive from the live df (single source of truth) so in-place split
            # edits (e.g. _apply_train_prevalence's P1 subsample, #140) are
            # honored. The df's 0..N-1 RangeIndex aligns with self._row_map
            # positions, so this is drop-in-equivalent for an unmodified parser.
            if split == "train":
                idx = self._df.index[self._df["split"] == "train"].to_numpy()
            elif split == "val":
                idx = self._df.index[self._df["split"] == "val"].to_numpy()
            else:
                raise ValueError(f"Unknown split: {split!r}")
        else:
            # split_strategy == "none" — return everything regardless.
            idx = np.arange(self._row_map.shape[0], dtype=np.int64)

        sub_row_map = self._row_map[idx]
        return CompositeMemmapContrastiveDataset(
            source_dirs=self._dirs,
            sub_row_map=sub_row_map,
            random_seed=self._random_seed,
            split_name=split,
            metas=self._metas,
            **kwargs,
        )


class CompositeMemmapContrastiveDataset:
    """Stitches together rows from multiple MemmapContrastiveDataset instances.

    Each source dataset is constructed with ``split="all"`` so its
    ``__getitem__(i)`` indexes directly into its own meta order. Composite
    ``__getitem__(j)`` looks up (source_idx, meta_idx_in_source) in the row map
    and delegates.
    """

    def __init__(
        self,
        *,
        source_dirs: Sequence[Path],
        sub_row_map: np.ndarray,
        random_seed: int,
        split_name: str,
        metas: Sequence[Sequence[dict]],
        **ds_kwargs: Any,
    ) -> None:
        # Drop parser-only kwargs that MemmapContrastiveDataset doesn't accept.
        for k in ("preload", "check_ram", "min_target_layers", "deterministic"):
            ds_kwargs.pop(k, None)

        self._row_map = np.asarray(sub_row_map, dtype=np.int64)
        if self._row_map.ndim != 2 or self._row_map.shape[1] != 2:
            raise ValueError(f"sub_row_map must be (N, 2), got shape {self._row_map.shape}")
        self._split_name = split_name
        self._metas = metas

        # Retain construction state so slice_layers() can rebuild with a layer subset.
        self._source_dirs = list(source_dirs)
        self._random_seed = int(random_seed)
        self._ds_kwargs = dict(ds_kwargs)

        self._sources: list[MemmapContrastiveDataset] = []
        for d in source_dirs:
            ds = MemmapContrastiveDataset(
                d,
                split="all",
                random_seed=random_seed,
                **ds_kwargs,
            )
            self._sources.append(ds)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return int(self._row_map.shape[0])

    def __getitem__(self, idx: int) -> dict:
        s = int(self._row_map[idx, 0])
        m = int(self._row_map[idx, 1])
        return self._sources[s][m]

    # ------------------------------------------------------------------ #
    @property
    def df(self) -> "pd.DataFrame":
        import pandas as pd
        rows = []
        for s, m in self._row_map:
            mrow = self._metas[int(s)][int(m)]
            rows.append({
                "prompt_hash": mrow["prompt_hash"],
                "halu": int(bool(mrow["hallucinated"])),
                "split": self._split_name,
                "sample_index": int(mrow["sample_index"]),
                "prompt_len": int(mrow.get("prompt_len", 0)),
                "response_len": int(mrow.get("response_len", 0)),
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    def slice_layers(
        self,
        layers: List[int],
        num_views: Optional[int] = None,
    ) -> "CompositeMemmapContrastiveDataset":
        """Return a new composite dataset restricted to a subset of model layers.

        Mirrors :meth:`MemmapContrastiveDataset.slice_layers` — no data is copied,
        each source rebuilds its memmap-backed view with the new layer subset and
        the composite row map is preserved so train/test alignment is intact.
        """
        if num_views is None:
            num_views = len(layers)
        new_kwargs = dict(self._ds_kwargs)
        new_kwargs["relevant_layers"] = list(layers)
        new_kwargs["num_views"] = int(num_views)
        new_kwargs["fixed_layer"] = None
        return CompositeMemmapContrastiveDataset(
            source_dirs=self._source_dirs,
            sub_row_map=self._row_map,
            random_seed=self._random_seed,
            split_name=self._split_name,
            metas=self._metas,
            **new_kwargs,
        )

    @property
    def labels(self) -> np.ndarray:
        out = np.empty((len(self),), dtype=np.int32)
        for i in range(len(self)):
            s = int(self._row_map[i, 0])
            m = int(self._row_map[i, 1])
            out[i] = int(bool(self._metas[s][m]["hallucinated"]))
        return out
