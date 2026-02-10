"""
WebDataset Option A utilities.

Option A stores all layers per sample in a single WebDataset record and streams
with a shuffle buffer. This favors fewer file opens at the cost of extra bytes.

Expected layout (co-located with Zarr):
<zarr_parent>/
    activations.zarr/
    webdataset/
        wds-%06d.tar
"""
from __future__ import annotations

import io
import json
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from loguru import logger
from torch.utils.data import IterableDataset

try:
    import webdataset as wds
except Exception as exc:  # pragma: no cover - optional dependency
    wds = None
    _WDS_IMPORT_ERROR = exc
else:
    _WDS_IMPORT_ERROR = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


@dataclass(frozen=True)
class WDSOptionAConfig:
    """Configuration for Option A WebDataset streaming."""

    shards: Union[str, Sequence[str]]
    split: str
    shuffle_buffer: int = 10_000
    pad_length: int = 63
    min_target_layers: int = 2
    relevant_layers: Optional[List[int]] = None
    fixed_layer: Optional[int] = None


def _ensure_wds_available() -> None:
    if wds is None:
        detail = "unknown import error"
        if _WDS_IMPORT_ERROR is not None:
            detail = f"{type(_WDS_IMPORT_ERROR).__name__}: {_WDS_IMPORT_ERROR}"
        raise RuntimeError(
            "webdataset is not available in the current Python environment. "
            f"Import error: {detail}. "
            "Ensure the runtime uses the same environment where webdataset is installed."
        ) from _WDS_IMPORT_ERROR


def _coerce_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (bytes, bytearray)):
        with io.BytesIO(value) as bio:
            return np.load(bio, allow_pickle=False)
    raise TypeError(f"Unsupported array payload type: {type(value)}")


def _coerce_json(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        return json.loads(value.decode("utf-8"))
    if isinstance(value, str):
        return json.loads(value)
    raise TypeError(f"Unsupported json payload type: {type(value)}")


def _get_sample_field(sample: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in sample:
            return sample[key]
    return None


def _build_stream(shards: Union[str, Sequence[str]], shuffle_buffer: int) -> Iterable[Dict[str, Any]]:
    _ensure_wds_available()

    # webdataset supports brace expansion patterns, but support for shell-style
    # globs like "*.tar" can vary by version/environment. Expand local globs
    # ourselves when we can.
    if isinstance(shards, str) and any(ch in shards for ch in ["*", "?", "["]):
        expanded = sorted(glob.glob(shards))
        if expanded:
            shards = expanded
    # NOTE: Do not call `.decode("numpy")`.
    # Newer `webdataset` versions treat decode() as image decoding and will
    # raise `ValueError: Unknown imagespec: numpy`.
    # We keep payloads as bytes and decode `.npy` / `.json` ourselves.
    # NOTE: When using many PyTorch DataLoader workers, WebDataset may assign
    # zero shards to some workers (e.g., num_workers > num_shards). Newer
    # webdataset versions raise `ValueError: No samples found in dataset`.
    # `empty_check=False` keeps those workers from erroring.
    dataset = wds.WebDataset(
        shards,
        shardshuffle=100,
        handler=wds.handlers.warn_and_continue,
        empty_check=False,
    )
    if shuffle_buffer and shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer)
    return dataset


class WDSOptionAIterableDataset(IterableDataset):
    """Iterable dataset for Option A WebDataset streaming."""

    def __init__(
        self,
        df,
        config: WDSOptionAConfig,
    ) -> None:
        super().__init__()
        self.df = df
        self.config = config
        self.relevant_layers = (
            config.relevant_layers if config.relevant_layers is not None else list(range(16, 30))
        )
        self.fixed_layer = config.fixed_layer
        self.pad_length = config.pad_length
        self.min_target_layers = config.min_target_layers
        self.shuffle_buffer = config.shuffle_buffer
        self.split = config.split

        self._label_by_hash = dict(zip(self.df["prompt_hash"].tolist(), self.df["halu"].tolist()))
        self._split_by_hash = dict(zip(self.df["prompt_hash"].tolist(), self.df["split"].tolist()))

    def __len__(self) -> int:  # best-effort length for logging
        return int((self.df["split"] == self.split).sum())

    def _select_layers(self, available_layers: List[int]) -> Tuple[int, int]:
        if len(available_layers) < self.min_target_layers:
            raise ValueError(
                f"Not enough targeted layers available (found {len(available_layers)} layers; "
                f"need at least {self.min_target_layers})."
            )

        if self.fixed_layer is not None:
            if self.fixed_layer not in available_layers:
                raise ValueError(f"Fixed layer {self.fixed_layer} is not available in the relevant layers")
            layer1_idx = self.fixed_layer
            other_layers = [i for i in available_layers if i != self.fixed_layer]
            if not other_layers:
                raise ValueError(f"No other layers available besides fixed layer {self.fixed_layer}")
            layer2_idx = int(np.random.choice(other_layers))
            return layer1_idx, layer2_idx

        if len(available_layers) == 1:
            return available_layers[0], available_layers[0]

        layer1_idx, layer2_idx = np.random.choice(available_layers, size=2, replace=False)
        return int(layer1_idx), int(layer2_idx)

    def _prepare_layer(self, response_acts: np.ndarray, layer_pos: int, response_len: int) -> torch.Tensor:
        actual_layer = self.relevant_layers[layer_pos]
        if actual_layer >= response_acts.shape[0]:
            raise ValueError(f"Layer index {actual_layer} out of bounds for response_acts")

        seq_len = min(int(response_len), response_acts.shape[1])
        act = response_acts[actual_layer, :seq_len, :]
        act_t = torch.from_numpy(act).unsqueeze(0)

        if seq_len < self.pad_length:
            noise = torch.randn(
                act_t.shape[0],
                self.pad_length - seq_len,
                act_t.shape[2],
                dtype=act_t.dtype,
            )
            act_t = torch.cat([act_t, noise], dim=1)
        elif seq_len > self.pad_length:
            act_t = act_t[:, : self.pad_length, :]

        return act_t

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for sample in _build_stream(self.config.shards, self.shuffle_buffer):
            meta_raw = _get_sample_field(sample, "meta.json", "meta")
            if meta_raw is None:
                logger.warning("Skipping sample without meta.json")
                continue

            meta = _coerce_json(meta_raw)
            sample_key = meta.get("sample_key") or meta.get("prompt_hash")
            if sample_key is None:
                logger.warning("Skipping sample without sample_key/prompt_hash")
                continue

            split = self._split_by_hash.get(sample_key)
            if split != self.split:
                continue

            response_len = int(meta.get("response_len", 0))
            prompt_len = int(meta.get("prompt_len", 0))

            response_raw = _get_sample_field(sample, "response_acts.npy", "response_acts")
            if response_raw is None:
                logger.warning("Skipping sample without response_acts.npy")
                continue

            response_acts = _coerce_numpy(response_raw)

            padded_activations: List[Optional[torch.Tensor]] = [None] * len(self.relevant_layers)
            for layer_pos in range(len(self.relevant_layers)):
                padded_activations[layer_pos] = self._prepare_layer(
                    response_acts,
                    layer_pos,
                    response_len,
                )

            available_layers = list(range(len(self.relevant_layers)))
            layer1_idx, layer2_idx = self._select_layers(available_layers)
            layer1_activations = padded_activations[layer1_idx]
            layer2_activations = padded_activations[layer2_idx]

            halu_label = self._label_by_hash.get(sample_key, 0)

            yield {
                "hashkey": sample_key,
                "halu": torch.tensor(halu_label, dtype=torch.float32),
                "all_activations": padded_activations,
                "layer1_activations": layer1_activations,
                "layer2_activations": layer2_activations,
                "layer1_idx": layer1_idx,
                "layer2_idx": layer2_idx,
                "input_length": prompt_len,
            }


def infer_activation_dim_from_wds(shards: Union[str, Sequence[str]]) -> int:
    """Infer activation hidden dim H from the first available WDS sample."""
    _ensure_wds_available()
    dataset = wds.WebDataset(shards, shardshuffle=False, handler=wds.handlers.warn_and_continue)
    for sample in dataset:
        response_raw = _get_sample_field(sample, "response_acts.npy", "response_acts")
        if response_raw is None:
            continue
        response_acts = _coerce_numpy(response_raw)
        if response_acts.ndim != 3:
            raise ValueError(f"Unexpected response_acts shape: {response_acts.shape}")
        return int(response_acts.shape[-1])
    raise RuntimeError("Unable to infer activation dimension from WDS samples")


def convert_zarr_to_wds_option_a(
    zarr_path: str,
    output_pattern: str,
    *,
    shard_size_mb: int = 512,
    samples_jsonl_path: Optional[str] = None,
    include_prompt: bool = True,
) -> None:
    """Convert a Zarr activation store to Option A WebDataset shards.

    Args:
        zarr_path: Path to the Zarr store.
        output_pattern: WebDataset shard pattern (e.g., /path/wds-%06d.tar). If None,
            defaults to a webdataset/ folder colocated with the Zarr store.
        shard_size_mb: Target shard size in MB.
        samples_jsonl_path: Optional JSONL to write prompt/response text.
        include_prompt: Whether to include prompt activations in WDS.
    """
    _ensure_wds_available()
    import zarr
    from zarr.errors import GroupNotFoundError

    zarr_dir = Path(zarr_path)
    if zarr_dir.is_dir() and (zarr_dir / "activations.zarr").exists():
        zarr_path = str(zarr_dir / "activations.zarr")

    output_pattern = resolve_wds_output_pattern(zarr_path, output_pattern)

    try:
        root = zarr.open_group(zarr_path, mode="r")
        arrays = root.get("arrays") or root
    except GroupNotFoundError:
        try:
            root = zarr.open_group(zarr_path, mode="r", path="arrays")
            arrays = root
        except GroupNotFoundError as exc:
            raise GroupNotFoundError(
                f"No Zarr group found at {zarr_path}. "
                "Expected a Zarr v2/v3 group or an arrays/ subgroup."
            ) from exc

    prompt_acts = arrays.get("prompt_activations")
    response_acts = arrays.get("response_activations")
    prompt_len = arrays.get("prompt_len")
    response_len = arrays.get("response_len")
    sample_key = arrays.get("sample_key")

    if response_acts is None:
        raise ValueError("Zarr store is missing response_activations")

    num_samples = int(response_acts.shape[0])

    shard_bytes = int(shard_size_mb) * 1024 * 1024
    logger.info(f"Writing {num_samples} samples to WDS: {output_pattern}")

    jsonl_handle = None
    if samples_jsonl_path is not None:
        jsonl_handle = open(samples_jsonl_path, "w", encoding="utf-8")

    iterator = range(num_samples)
    if tqdm is not None:
        iterator = tqdm(iterator, desc="Converting Zarr -> WDS", unit="sample")

    with wds.ShardWriter(output_pattern, maxsize=shard_bytes) as sink:
        for idx in iterator:
            key = None
            if sample_key is not None:
                key = str(sample_key[idx])
            if not key or key == "None":
                key = f"{idx}"

            meta = {
                "sample_index": idx,
                "prompt_len": int(prompt_len[idx]) if prompt_len is not None else 0,
                "response_len": int(response_len[idx]) if response_len is not None else 0,
                "sample_key": key,
            }

            record: Dict[str, Any] = {"__key__": key, "meta.json": json.dumps(meta)}

            if include_prompt and prompt_acts is not None:
                record["prompt_acts.npy"] = np.asarray(prompt_acts[idx])

            record["response_acts.npy"] = np.asarray(response_acts[idx])

            sink.write(record)

            if jsonl_handle is not None:
                jsonl_handle.write(json.dumps({"sample_key": key}) + "\n")

    if jsonl_handle is not None:
        jsonl_handle.close()


def resolve_wds_output_pattern(zarr_path: str, output_pattern: Optional[str] = None) -> str:
    """Resolve a default WDS output pattern colocated with a Zarr store.

    Expected layout:
    <zarr_parent>/
        activations.zarr/
        webdataset/wds-%06d.tar
    """
    if output_pattern:
        return output_pattern
    zarr_dir = Path(zarr_path).resolve()
    if zarr_dir.is_dir() and (zarr_dir / "activations.zarr").exists():
        base_dir = zarr_dir
    else:
        base_dir = zarr_dir.parent
    wds_dir = base_dir / "webdataset"
    wds_dir.mkdir(parents=True, exist_ok=True)
    return str(wds_dir / "wds-%06d.tar")
