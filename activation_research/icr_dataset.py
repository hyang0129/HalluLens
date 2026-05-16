"""
icr_dataset.py — ICR Probe PyTorch Dataset (Issue #69, Wave 3 Task G / Wave 4c)

Pairs attention.zarr (response-to-response attention maps) with activations.zarr
(cached hidden states) and per-sample hallucination labels from eval_results.json
to yield training-ready samples for the ICR Probe (Issue #70).

Two modes
---------
The ``mode`` constructor parameter selects between two loading strategies:

* ``mode="icr"`` **(default)**: Fast-path. Reads ``icr_scores.npy`` and its
  companion ``icr_scores_meta.jsonl`` sidecar directly.  No attention or
  hidden-state tensors are loaded; the full ``icr_scores.npy`` (~2 MB) is
  loaded into RAM at ``__init__`` time.  ``__getitem__`` returns
  ``{"hashkey", "halu", "icr_score"}`` where ``icr_score`` is a
  ``(num_blocks,)`` float32 Tensor.  This is the normal training path for
  the Issue #70 probe.

* ``mode="raw"``: Ablations path.  Reads attention + activations via
  ``AttentionParser.get_paired()`` exactly as the Wave-3 implementation did.
  ``__getitem__`` returns ``{"hashkey", "halu", "response_attn",
  "h_block_input", "delta_h", "response_len"}``.  Behaviour is
  **bit-for-bit identical** to the original Wave-3 code.

Label loading
-------------
``eval_results_path`` accepts two JSON formats (see module docstring for full
detail):

1. Per-key dict:  ``{sample_key: 0_or_1}``
   OR            ``{sample_key: {"halu": 0_or_1}}``

2. Array format (ActivationParser-compatible):
   ``{"halu_test_res": [bool, ...], "abstantion": [bool, ...], ...}``
   When this format is detected, ``eval_results_path.parent / "generation.jsonl"``
   must exist; each JSONL row's ``"prompt"`` field is SHA-256-hashed to yield
   the sample key (matching the convention in ActivationParser._load_metadata).

Stratified split
----------------
A deterministic three-way split is applied at ``__init__`` time using
``sklearn.model_selection.train_test_split`` with ``stratify=labels``:

  - Stage 1: hold out *test* (``val_fraction`` of total data).
  - Stage 2: from the remaining data hold out *val* (``val_fraction`` of
    remaining, ≈ val_fraction² of total), keeping the rest as *train*.

Both stages use stratify to preserve class balance.  Seeds are
``random_seed`` (stage 1) and ``random_seed + 1`` (stage 2), mirroring the
pattern in ``activation_logging/activation_parser.py:_load_metadata``.

Lazy loading (mode="raw")
--------------------------
In mode="raw" the dataset is lazy: ``__init__`` only loads labels and builds
the split index.  All zarr reads happen in ``__getitem__``, one sample at a
time.

Eager loading (mode="icr")
---------------------------
In mode="icr" ``icr_scores.npy`` is loaded in full at ``__init__`` time
(~2 MB).  ``__getitem__`` is a pure array row-slice + Tensor conversion.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from activation_logging.attention_parser import AttentionParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(text: str) -> str:
    """SHA-256 hex digest — matches ActivationParser prompt-hashing convention."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_labels(eval_results_path: Path) -> dict[str, int]:
    """Load per-sample halu labels from *eval_results_path*.

    Supports two JSON formats (see module docstring).

    Returns
    -------
    dict[str, int]
        Mapping from sample key (str) to label (0 = correct, 1 = hallucination).

    Raises
    ------
    ValueError
        If the file format is unrecognised or a required sibling file is missing.
    """
    raw = json.loads(eval_results_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Format 1: per-key dict  {key: int}  or  {key: {"halu": int}}
    # Heuristic: the JSON root is a dict AND its values are not lists.
    # ------------------------------------------------------------------
    if isinstance(raw, dict):
        first_val = next(iter(raw.values()), None) if raw else None

        # Format 1a: {key: {"halu": int}} or {key: {"halu": bool}}
        if isinstance(first_val, dict):
            labels: dict[str, int] = {}
            for k, v in raw.items():
                halu_val = v.get("halu")
                if halu_val is None:
                    raise ValueError(
                        f"eval_results.json per-key entry for '{k}' is missing 'halu' field."
                    )
                labels[str(k)] = int(bool(halu_val))
            return labels

        # Format 1b: {key: int_or_bool}  — direct label
        if isinstance(first_val, (int, float, bool)):
            return {str(k): int(bool(v)) for k, v in raw.items()}

        # Format 2: ActivationParser-compatible array format
        # Root dict with "halu_test_res" key containing a list.
        if "halu_test_res" in raw and isinstance(raw["halu_test_res"], list):
            halu_arr: list[bool] = [bool(x) for x in raw["halu_test_res"]]
            abstain_arr: list[bool] = [bool(x) for x in raw.get("abstantion", [False] * len(halu_arr))]

            # Locate generation.jsonl in the same directory.
            gen_path = eval_results_path.parent / "generation.jsonl"
            if not gen_path.exists():
                raise ValueError(
                    f"eval_results.json uses the array format but the required sibling "
                    f"'{gen_path}' does not exist.  "
                    "Either provide a per-key eval_results.json or place generation.jsonl "
                    "next to eval_results.json."
                )

            labels = {}
            with gen_path.open("r", encoding="utf-8") as fh:
                for idx, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    if idx >= len(halu_arr):
                        break
                    if abstain_arr[idx]:
                        continue  # skip abstained samples (no activation logged)
                    row = json.loads(line)
                    prompt: str = row.get("prompt", "")
                    key = _sha256(prompt)
                    labels[key] = int(halu_arr[idx])
            return labels

        raise ValueError(
            f"Unrecognised eval_results.json format in '{eval_results_path}'. "
            "Expected: per-key dict, or array dict with 'halu_test_res'."
        )

    raise ValueError(
        f"eval_results.json at '{eval_results_path}' must be a JSON object (dict), "
        f"got {type(raw).__name__}."
    )


def _build_splits(
    keys: list[str],
    labels: dict[str, int],
    *,
    val_fraction: float,
    random_seed: int,
) -> dict[str, list[str]]:
    """Build stratified train / val / test split indices.

    Parameters
    ----------
    keys:
        All sample keys present in *both* the attention store and the label file.
    labels:
        Per-key halu labels (0 or 1).
    val_fraction:
        Fraction of total data to allocate to each of the *test* and *val* sets.
        E.g. ``0.15`` → test=15%, val≈15%, train≈70%.
    random_seed:
        RNG seed.  Stage-1 uses ``random_seed``, stage-2 uses ``random_seed + 1``.

    Returns
    -------
    dict with keys "train", "val", "test", each mapping to a list[str] of keys.
    """
    label_arr = [labels[k] for k in keys]

    # Stage 1: hold out test set.
    train_val_keys, test_keys, tv_labels, _ = train_test_split(
        keys,
        label_arr,
        test_size=val_fraction,
        stratify=label_arr,
        random_state=random_seed,
    )

    # Stage 2: hold out val from the remaining training pool.
    # val_fraction of the *remaining* data ≈ val_fraction^2 / (1 - val_fraction) of total,
    # but we use val_fraction directly here to roughly match the target fraction.
    val_size_of_remaining = val_fraction / (1.0 - val_fraction) if val_fraction < 1.0 else 0.5
    val_size_of_remaining = min(val_size_of_remaining, 0.9)  # guard against degenerate fractions

    train_keys, val_keys = train_test_split(
        train_val_keys,
        test_size=val_size_of_remaining,
        stratify=tv_labels,
        random_state=random_seed + 1,
    )

    return {"train": train_keys, "val": val_keys, "test": test_keys}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ICRDataset(Dataset):
    """PyTorch Dataset for ICR Probe training (Issue #70 consumer).

    Two modes are available, selected by the ``mode`` constructor argument:

    * ``mode="icr"`` (default): fast-path that reads ``icr_scores.npy`` only.
      RAM-trivial; the full array is loaded at ``__init__`` time.
      ``__getitem__`` returns ``{"hashkey", "halu", "icr_score"}``.

    * ``mode="raw"``: ablations path that reads attention + activations via
      :class:`~activation_logging.attention_parser.AttentionParser`.
      ``__getitem__`` returns the full dict with response attention, hidden
      states, and response length.  Behaviour is identical to the Wave-3
      implementation.

    Parameters
    ----------
    attention_zarr_path:
        Path to an ``attention/`` store written by
        :class:`~activation_logging.attention_memmap_writer.AttentionMemmapWriter`.
        Required for ``mode="raw"``.  For ``mode="icr"``, used only to derive
        ``icr_scores_path`` when *icr_scores_path* is ``None``; may be omitted
        (pass ``None``) if *icr_scores_path* is provided explicitly.
    eval_results_path:
        Path to a JSON label file.  See module docstring for supported formats.
    relevant_layers:
        Block indices to include in each sample (e.g. ``list(range(4, 32))``).
        Required for ``mode="raw"``; ignored (with a warning) for
        ``mode="icr"``.
    split:
        One of ``"train"``, ``"val"``, or ``"test"``.
    random_seed:
        RNG seed for the stratified split (default 42).
    val_fraction:
        Fraction of total data allocated to *each* of val and test (default
        0.15).
    activations_parser:
        Optional pre-constructed
        :class:`~activation_logging.zarr_activations_logger.ZarrActivationsLogger`
        opened on the source ``activations.zarr``.  Only used for
        ``mode="raw"``; ignored otherwise.
    mode:
        ``"icr"`` (default) or ``"raw"``.  See class docstring.
    icr_scores_path:
        Explicit path to ``icr_scores.npy``.  Only used for ``mode="icr"``.
        When ``None``, derived from *attention_zarr_path* as
        ``<attention_zarr_path>/../icr_scores.npy``.
    icr_scores_meta_path:
        Explicit path to the ``icr_scores_meta.jsonl`` sidecar.  When
        ``None``, derived as ``<icr_scores_path>.stem + "_meta.jsonl"`` in
        the same parent directory.

    Raises
    ------
    ValueError
        If *split* is not one of the three valid values, *val_fraction* is
        out of range, required paths are missing, or no common keys are found.
    FileNotFoundError
        If ``icr_scores_meta.jsonl`` or ``icr_scores.npy`` cannot be found.
    KeyError
        (deferred to ``__getitem__`` in ``mode="raw"``) If a sample key is
        missing from either the attention or activations store.
    """

    def __init__(
        self,
        attention_zarr_path: str | None,
        eval_results_path: str,
        relevant_layers: list[int] | None = None,
        split: str = "train",
        random_seed: int = 42,
        val_fraction: float = 0.15,
        activations_parser: Optional[object] = None,
        *,
        mode: Literal["icr", "raw"] = "icr",
        icr_scores_path: str | None = None,
        icr_scores_meta_path: str | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"split must be 'train', 'val', or 'test', got '{split}'."
            )
        if not (0.0 < val_fraction < 0.5):
            raise ValueError(
                f"val_fraction must be in (0, 0.5), got {val_fraction}."
            )
        if mode not in ("icr", "raw"):
            raise ValueError(f"mode must be 'icr' or 'raw', got '{mode}'.")

        self.mode: Literal["icr", "raw"] = mode
        self.split = split
        self.random_seed = random_seed
        self.val_fraction = val_fraction
        self.relevant_layers: list[int] | None = list(relevant_layers) if relevant_layers is not None else None

        if mode == "raw":
            self._init_raw(
                attention_zarr_path=attention_zarr_path,
                eval_results_path=eval_results_path,
                relevant_layers=relevant_layers,
                split=split,
                random_seed=random_seed,
                val_fraction=val_fraction,
                activations_parser=activations_parser,
            )
        else:
            self._init_icr(
                attention_zarr_path=attention_zarr_path,
                eval_results_path=eval_results_path,
                split=split,
                random_seed=random_seed,
                val_fraction=val_fraction,
                relevant_layers=relevant_layers,
                icr_scores_path=icr_scores_path,
                icr_scores_meta_path=icr_scores_meta_path,
            )

    # ------------------------------------------------------------------
    # mode="raw" initialisation — unchanged from Wave 3
    # ------------------------------------------------------------------

    def _init_raw(
        self,
        attention_zarr_path: str | None,
        eval_results_path: str,
        relevant_layers: list[int] | None,
        split: str,
        random_seed: int,
        val_fraction: float,
        activations_parser: Optional[object],
    ) -> None:
        if attention_zarr_path is None:
            raise ValueError(
                "mode='raw' requires attention_zarr_path; got None."
            )
        if relevant_layers is None:
            raise ValueError(
                "mode='raw' requires relevant_layers; got None."
            )

        self.relevant_layers = list(relevant_layers)

        # ------------------------------------------------------------------
        # Build AttentionParser (lazy zarr handle — does NOT preload tensors).
        # ------------------------------------------------------------------
        self._parser = AttentionParser(
            attention_zarr_path=attention_zarr_path,
            activations_parser=activations_parser,  # type: ignore[arg-type]
        )

        # ------------------------------------------------------------------
        # Load labels.
        # ------------------------------------------------------------------
        labels_all = _load_labels(Path(eval_results_path))

        # Keep only keys present in BOTH the attention store and the label file.
        attn_keys: set[str] = set(self._parser.list_keys())
        common_keys: list[str] = sorted(k for k in attn_keys if k in labels_all)

        if not common_keys:
            raise ValueError(
                "No keys are shared between the attention store and the label file. "
                "Verify that 'eval_results_path' matches the 'attention_zarr_path'."
            )

        self._labels: dict[str, int] = {k: labels_all[k] for k in common_keys}

        # ------------------------------------------------------------------
        # Stratified split.
        # ------------------------------------------------------------------
        splits = _build_splits(
            common_keys,
            self._labels,
            val_fraction=val_fraction,
            random_seed=random_seed,
        )
        self._keys: list[str] = splits[split]

    # ------------------------------------------------------------------
    # mode="icr" initialisation — fast-path, no zarr
    # ------------------------------------------------------------------

    def _init_icr(
        self,
        attention_zarr_path: str | None,
        eval_results_path: str,
        split: str,
        random_seed: int,
        val_fraction: float,
        relevant_layers: list[int] | None,
        icr_scores_path: str | None,
        icr_scores_meta_path: str | None,
    ) -> None:
        # Warn if caller supplied relevant_layers — it's silently ignored here.
        if relevant_layers is not None:
            warnings.warn(
                "ICRDataset(mode='icr'): relevant_layers is ignored in ICR fast-path mode. "
                "The icr_scores.npy already encodes all blocks; layer subsetting happens "
                "in the probe model, not the dataset.",
                UserWarning,
                stacklevel=3,
            )

        # ------------------------------------------------------------------
        # Derive icr_scores_path if not provided explicitly.
        # ------------------------------------------------------------------
        resolved_scores_path: Path
        if icr_scores_path is not None:
            resolved_scores_path = Path(icr_scores_path)
        elif attention_zarr_path is not None:
            attention_dir = Path(attention_zarr_path)
            resolved_scores_path = attention_dir.parent / "icr_scores.npy"
        else:
            raise ValueError(
                "mode='icr' requires icr_scores_path or attention_zarr_path; "
                "both are None."
            )

        # Derive meta sidecar path.
        resolved_meta_path: Path
        if icr_scores_meta_path is not None:
            resolved_meta_path = Path(icr_scores_meta_path)
        else:
            resolved_meta_path = resolved_scores_path.parent / (
                resolved_scores_path.stem + "_meta.jsonl"
            )

        # ------------------------------------------------------------------
        # Load icr_scores_meta.jsonl → key-to-row mapping.
        # ------------------------------------------------------------------
        if not resolved_meta_path.exists():
            raise FileNotFoundError(
                f"icr_scores_meta.jsonl not found at '{resolved_meta_path}'. "
                "Run recompute_attention.py to generate icr_scores.npy and its "
                "sidecar before constructing ICRDataset(mode='icr')."
            )

        meta_key_to_row: dict[str, int] = {}
        max_sample_index: int = -1
        with resolved_meta_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                key = str(entry["key"])
                row = int(entry["sample_index"])
                meta_key_to_row[key] = row
                if row > max_sample_index:
                    max_sample_index = row

        if not meta_key_to_row:
            raise ValueError(
                f"icr_scores_meta.jsonl at '{resolved_meta_path}' contains zero "
                "samples.  Verify that recompute_attention.py completed without "
                "errors."
            )

        self._meta_key_to_row: dict[str, int] = meta_key_to_row

        # ------------------------------------------------------------------
        # Load labels.
        # ------------------------------------------------------------------
        labels_all = _load_labels(Path(eval_results_path))

        # ------------------------------------------------------------------
        # Common keys: present in meta AND in labels.
        # ------------------------------------------------------------------
        common_keys: list[str] = sorted(
            k for k in self._meta_key_to_row if k in labels_all
        )
        if not common_keys:
            raise ValueError(
                "No keys are shared between icr_scores_meta.jsonl and the label "
                "file.  Verify that 'eval_results_path' matches the experiment "
                "that produced 'icr_scores.npy'."
            )

        self._labels = {k: labels_all[k] for k in common_keys}

        # ------------------------------------------------------------------
        # Stratified split.
        # ------------------------------------------------------------------
        splits = _build_splits(
            common_keys,
            self._labels,
            val_fraction=val_fraction,
            random_seed=random_seed,
        )
        self._keys = splits[split]

        # ------------------------------------------------------------------
        # Load icr_scores.npy into RAM (full load, ~2 MB — not memmap).
        # ------------------------------------------------------------------
        if not resolved_scores_path.exists():
            raise FileNotFoundError(
                f"icr_scores.npy not found at '{resolved_scores_path}'."
            )

        self._icr_scores: np.ndarray = np.load(str(resolved_scores_path))

        # Sanity-check: the file must have at least max_sample_index + 1 rows.
        if self._icr_scores.ndim != 2:
            raise ValueError(
                f"icr_scores.npy at '{resolved_scores_path}' must be 2-D "
                f"(n_samples, num_blocks); got shape {self._icr_scores.shape}."
            )
        required_rows = max_sample_index + 1
        if self._icr_scores.shape[0] < required_rows:
            raise ValueError(
                f"icr_scores.npy has {self._icr_scores.shape[0]} rows but "
                f"icr_scores_meta.jsonl references sample_index {max_sample_index} "
                f"(requires at least {required_rows} rows)."
            )

        self._num_blocks: int = int(self._icr_scores.shape[1])

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, idx: int) -> dict:
        """Return a single sample dict.

        mode="icr"
        ----------
        Returns dict with keys:

        * ``"hashkey"``   — str, sample key
        * ``"halu"``      — int (0 or 1)
        * ``"icr_score"`` — Tensor ``(num_blocks,)`` float32

        mode="raw"
        ----------
        All zarr reads happen here (lazy loading).  Returns dict with keys:

        * ``"hashkey"``       — str, sample key
        * ``"halu"``          — int (0 or 1)
        * ``"response_attn"`` — Tensor ``(len(relevant_layers), R_max, R_max)`` float32
        * ``"h_block_input"`` — Tensor ``(len(relevant_layers), R_max, H)`` float32
        * ``"delta_h"``       — Tensor ``(len(relevant_layers), R_max, H)`` float32
        * ``"response_len"``  — int, actual response length (positions >= this are zero-padded)
        """
        if self.mode == "icr":
            return self._getitem_icr(idx)
        return self._getitem_raw(idx)

    def _getitem_icr(self, idx: int) -> dict:
        key = self._keys[idx]
        halu = self._labels[key]
        row = self._meta_key_to_row[key]
        icr_score = torch.from_numpy(self._icr_scores[row].astype(np.float32))
        return {"hashkey": key, "halu": halu, "icr_score": icr_score}

    def _getitem_raw(self, idx: int) -> dict:
        key = self._keys[idx]
        halu = self._labels[key]

        # Fetch paired hidden states + attention for all relevant layers in one call.
        # get_paired() returns variable-length (R, H) slices; we need to pad to R_max.
        paired = self._parser.get_paired(key, self.relevant_layers)

        response_len: int = paired["response_len"]
        r_max: int = self._parser._r_max

        # Determine hidden dim H from the first available layer.
        h_block_input_raw: dict[int, torch.Tensor] = paired["h_block_input"]
        delta_h_raw: dict[int, torch.Tensor] = paired["delta_h"]
        response_attn_raw: dict[int, torch.Tensor] = paired["response_attn"]

        # Infer H from the first layer tensor (R × H).
        first_layer = self.relevant_layers[0]
        H: int = int(h_block_input_raw[first_layer].shape[-1])

        L = len(self.relevant_layers)

        # Allocate output tensors (zero-initialised → pads beyond response_len are 0).
        out_response_attn = torch.zeros(L, r_max, r_max, dtype=torch.float32)
        out_h_block_input = torch.zeros(L, r_max, H, dtype=torch.float32)
        out_delta_h = torch.zeros(L, r_max, H, dtype=torch.float32)

        R = min(response_len, r_max)  # actual filled rows (capped by R_max)

        for pos, layer_b in enumerate(self.relevant_layers):
            # response_attn[b] shape: (R_actual, R_actual) — already sliced by get_paired
            attn = response_attn_raw[layer_b]          # (R, R)
            h_in = h_block_input_raw[layer_b]          # (R, H)
            dh = delta_h_raw[layer_b]                  # (R, H)

            R_attn = min(int(attn.shape[0]), r_max)
            R_h = min(int(h_in.shape[0]), r_max)

            out_response_attn[pos, :R_attn, :R_attn] = attn[:R_attn, :R_attn]
            out_h_block_input[pos, :R_h, :] = h_in[:R_h, :]
            out_delta_h[pos, :R_h, :] = dh[:R_h, :]

        return {
            "hashkey": key,
            "halu": halu,
            "response_attn": out_response_attn,
            "h_block_input": out_h_block_input,
            "delta_h": out_delta_h,
            "response_len": response_len,
        }
