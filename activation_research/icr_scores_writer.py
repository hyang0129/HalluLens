"""
icr_scores_writer.py

ICR-score sibling of ``AttentionMemmapWriter`` (Task J).  Pre-allocates
``icr_scores.npy`` via ``np.lib.format.open_memmap`` — not bare
``np.memmap`` — so the file has a proper .npy header and can be read back
with ``np.load(path)`` or ``np.load(path, mmap_mode='r')`` by the probe
trainer (Issue #70) without any format negotiation.  A lightweight
``icr_scores_meta.jsonl`` sidecar mirrors the attention writer's
``meta.jsonl`` so the probe trainer has key-to-row mapping without
needing the ``attention/`` directory at all.
"""

import json
import os
from pathlib import Path
from typing import Literal, Optional

import numpy as np


class ICRScoresWriter:
    """Pre-allocated numpy-memmap writer for ICR score vectors.

    Writes ``icr_scores.npy`` (shape ``(n_samples, num_blocks)``, dtype
    ``float32``) and a ``icr_scores_meta.jsonl`` sidecar in the same
    directory.  The sidecar is append-only and authoritative for
    "what has been written"; ``icr_scores.npy`` is the bulk data array.

    Args:
        out_path: Full path to ``icr_scores.npy``.  The sidecar is derived
                  automatically: ``<stem>_meta.jsonl`` in the same parent.
        mode: ``'w'`` creates/overwrites; ``'a'`` resumes an existing file.
        n_samples: Total number of samples (rows) to pre-allocate.
        num_blocks: Number of transformer blocks (columns).  Must be > 0.
    """

    def __init__(
        self,
        out_path: str,
        mode: Literal["w", "a"],
        n_samples: int,
        num_blocks: int,
    ) -> None:
        if mode not in ("w", "a"):
            raise ValueError(f"mode must be 'w' or 'a', got {mode!r}")
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks!r}")

        self._out_path = Path(out_path)
        self._mode = mode
        self._n_samples = n_samples
        self._num_blocks = num_blocks
        self._finalized = False

        # Sidecar lives next to the .npy file, same stem + _meta.jsonl.
        self._meta_path = self._out_path.parent / (
            self._out_path.stem + "_meta.jsonl"
        )

        self._written_keys: set[str] = set()
        self._written_indices: dict[int, str] = {}  # index -> key

        if mode == "w":
            self._init_write()
        else:
            self._init_append()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_write(self) -> None:
        """Create parent directory, pre-allocate .npy, create empty sidecar."""
        self._out_path.parent.mkdir(parents=True, exist_ok=True)

        # open_memmap writes a proper .npy header (mode='w+' creates/overwrites).
        # This makes the file readable by np.load() without format negotiation,
        # which is required by the Issue #70 probe trainer.
        self._mm: np.memmap = np.lib.format.open_memmap(
            str(self._out_path),
            mode="w+",
            dtype="float32",
            shape=(self._n_samples, self._num_blocks),
        )

        # Explicit zero-fill so unwritten rows are exactly zero (not undefined).
        self._mm[:] = 0.0
        self._mm.flush()

        # Truncate / create the sidecar to empty.
        self._meta_path.write_text("", encoding="utf-8")

    def _init_append(self) -> None:
        """Verify existing .npy, read sidecar to rebuild tracking state."""
        if not self._out_path.exists():
            raise FileNotFoundError(
                f"Cannot resume: {self._out_path} does not exist."
            )

        # Load header only to check shape and dtype before opening memmap.
        with open(self._out_path, "rb") as fh:
            version, header_data = np.lib.format.read_magic(fh), None
            try:
                header_data = np.lib.format.read_array_header_1_0(fh)
            except Exception:
                try:
                    header_data = np.lib.format.read_array_header_2_0(fh)
                except Exception:
                    pass

        # Re-open via open_memmap for shape/dtype check (it reads the header).
        mm_check = np.lib.format.open_memmap(str(self._out_path), mode="r")
        actual_shape = mm_check.shape
        actual_dtype = mm_check.dtype
        del mm_check  # close read-only handle before opening r+

        expected_shape = (self._n_samples, self._num_blocks)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Shape mismatch in {self._out_path}: "
                f"expected {expected_shape}, found {actual_shape}."
            )
        if actual_dtype != np.dtype("float32"):
            raise ValueError(
                f"Dtype mismatch in {self._out_path}: "
                f"expected float32, found {actual_dtype}."
            )

        # Re-open in read/write mode.
        self._mm = np.lib.format.open_memmap(str(self._out_path), mode="r+")

        # Rebuild tracking sets from the sidecar (authoritative record).
        if self._meta_path.exists():
            with open(self._meta_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        key = str(entry["key"])
                        idx = int(entry["sample_index"])
                        self._written_keys.add(key)
                        self._written_indices[idx] = key
                    except (json.JSONDecodeError, KeyError):
                        continue

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_written(self, sample_key: str) -> bool:
        """O(1) check whether this sample key has already been written."""
        return sample_key in self._written_keys

    def write(
        self,
        sample_index: int,
        sample_key: str,
        icr_vector: np.ndarray,
    ) -> None:
        """Write one ICR score row.

        Args:
            sample_index: Row index in ``icr_scores.npy``.  Must match the
                          corresponding row index used by ``AttentionMemmapWriter``
                          so the two files stay aligned.
            sample_key: Unique string identifier for this sample.
            icr_vector: Shape ``(num_blocks,)``, castable to float32.
        """
        if not (0 <= sample_index < self._n_samples):
            raise ValueError(
                f"sample_index {sample_index} out of range "
                f"[0, {self._n_samples})."
            )

        vec = np.asarray(icr_vector)
        if vec.shape != (self._num_blocks,):
            raise ValueError(
                f"icr_vector shape {vec.shape} != ({self._num_blocks},)."
            )

        # Idempotent: skip silently if this key is already on disk.
        if sample_key in self._written_keys:
            return

        # Guard against clobbering a different key that already occupies this slot.
        if sample_index in self._written_indices:
            existing_key = self._written_indices[sample_index]
            if existing_key != sample_key:
                raise ValueError(
                    f"sample_index {sample_index} already holds key "
                    f"{existing_key!r}; refusing to overwrite with {sample_key!r}."
                )

        # Write the row into the pre-allocated memmap.
        self._mm[sample_index] = vec.astype(np.float32)

        # Crash-safe append: write to sidecar then fsync before updating
        # in-memory state.  If the process dies mid-write the sidecar line
        # is absent and the slot will be overwritten on resume — no corruption.
        with open(self._meta_path, "a", encoding="utf-8") as fh:
            fh.write(
                json.dumps({"key": sample_key, "sample_index": sample_index})
                + "\n"
            )
            fh.flush()
            # fsync ensures the sidecar line hits disk before we update RAM
            # state; on resume the sidecar is the authoritative record.
            os.fsync(fh.fileno())

        self._written_keys.add(sample_key)
        self._written_indices[sample_index] = sample_key

    def finalize(self) -> None:
        """Flush the memmap to disk.  Idempotent; safe to call multiple times."""
        if self._finalized:
            return
        self._mm.flush()
        self._finalized = True

    def __len__(self) -> int:
        """Number of samples written so far."""
        return len(self._written_keys)

    def __enter__(self) -> "ICRScoresWriter":
        return self

    def __exit__(self, *args: object) -> None:
        self.finalize()
