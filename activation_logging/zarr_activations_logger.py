"""
zarr_activations_logger.py
Handles Zarr-based logging for LLM activations, prompts, responses, and evaluation results.
API is similar to ActivationsLogger, but uses Zarr for scalable, chunked storage.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, List, Callable, Tuple, Union

import numpy as np
import torch
import zarr
from loguru import logger

from .compression import BaseCompressor, NoCompression, ZstdCompression, ZSTD_AVAILABLE

class ZarrActivationsLogger:
    def __init__(
        self,
        zarr_path: str = "zarr_data/activations.zarr",
        mode: Optional[str] = None,
        chunk_size: int = 1000,
        activation_chunk_shape: Optional[Tuple[int, int, int, int]] = None,
        compression: Union[str, BaseCompressor, None] = None,
        target_layers: str = "all",
        sequence_mode: str = "all",
        read_only: bool = False,
        prompt_max_tokens: Optional[int] = None,
        response_max_tokens: Optional[int] = None,
        prompt_chunk_tokens: Optional[int] = None,
        response_chunk_tokens: Optional[int] = None,
        dtype: str = "float16",
        verbose: bool = True,
    ):
        """
        Initialize the Zarr-based activations logger.
        Args:
            zarr_path: Path to the Zarr group to store activations
            mode: Zarr mode ('a' for append/update, 'w' for overwrite, etc.)
            chunk_size: Number of samples per chunk
            activation_chunk_shape: Optional activation chunk shape (S, L, T, H).
                Use -1 for H to auto-match hidden size.
            compression: Compression method ('zstd', None) or a BaseCompressor instance
            target_layers: Which layers to extract activations from ('all', 'first_half', or 'second_half')
            sequence_mode: Which tokens to extract activations for ('all', 'prompt', 'response')
            read_only: If True, open in read-only mode
            prompt_max_tokens: Fixed maximum prompt tokens (P_max)
            response_max_tokens: Fixed maximum response tokens (R_max)
            prompt_chunk_tokens: Prompt chunk length (tokens)
            response_chunk_tokens: Response chunk length (tokens)
            dtype: Numpy dtype for stored activations
            verbose: Whether to log detailed initialization
        """
        if target_layers not in ['all', 'first_half', 'second_half']:
            raise ValueError("target_layers must be one of: 'all', 'first_half', 'second_half'")
        if sequence_mode not in ['all', 'prompt', 'response']:
            raise ValueError("sequence_mode must be one of: 'all', 'prompt', 'response'")
        self.verbose = verbose
        self.target_layers = target_layers
        self.sequence_mode = sequence_mode
        self.zarr_path = zarr_path
        self.activation_chunk_shape = activation_chunk_shape
        prompt_max_tokens_provided = prompt_max_tokens is not None
        response_max_tokens_provided = response_max_tokens is not None
        if self.activation_chunk_shape is not None:
            self.chunk_size = int(self.activation_chunk_shape[0])
        else:
            self.chunk_size = chunk_size
        self.read_only = read_only
        self.dtype = np.dtype(dtype)
        self.compressor = self._get_compressor(compression)

        if prompt_max_tokens is None:
            prompt_max_tokens = int(os.environ.get("ACTIVATION_PROMPT_MAX_TOKENS", 512))
        if response_max_tokens is None:
            response_max_tokens = int(os.environ.get("ACTIVATION_RESPONSE_MAX_TOKENS", 64))
        if prompt_chunk_tokens is None:
            prompt_chunk_tokens = int(os.environ.get("ACTIVATION_PROMPT_CHUNK_TOKENS", 128))
        if response_chunk_tokens is None:
            response_chunk_tokens = int(os.environ.get("ACTIVATION_RESPONSE_CHUNK_TOKENS", response_max_tokens))

        if self.activation_chunk_shape is not None:
            token_chunk = int(self.activation_chunk_shape[2])
            if not prompt_max_tokens_provided:
                prompt_max_tokens = token_chunk
            if not response_max_tokens_provided:
                response_max_tokens = token_chunk
            if prompt_max_tokens != token_chunk:
                raise ValueError(
                    "prompt_max_tokens must match activation_chunk_shape token dimension "
                    f"({token_chunk}), got {prompt_max_tokens}"
                )
            if response_max_tokens != token_chunk:
                raise ValueError(
                    "response_max_tokens must match activation_chunk_shape token dimension "
                    f"({token_chunk}), got {response_max_tokens}"
                )

        self.prompt_max_tokens = prompt_max_tokens
        self.response_max_tokens = response_max_tokens
        self.prompt_chunk_tokens = prompt_chunk_tokens
        self.response_chunk_tokens = response_chunk_tokens
        if self.activation_chunk_shape is not None:
            self.prompt_chunk_tokens = int(self.activation_chunk_shape[2])
            self.response_chunk_tokens = int(self.activation_chunk_shape[2])

        if read_only:
            mode = "r"
        elif mode is None:
            mode = "a"
        self.mode = mode

        self.root = zarr.open_group(zarr_path, mode=mode)
        if self.read_only:
            if "arrays" not in self.root:
                raise ValueError("Zarr store is missing required 'arrays' group")
            self.arrays_group = self.root["arrays"]
        else:
            self.arrays_group = self.root.require_group("arrays")

        self._prompt_activations = None
        self._response_activations = None
        self._prompt_len = None
        self._response_len = None
        self._sample_key = None
        self._layer_count = None
        self._hidden_size = None
        self._target_layer_indices = None

        self._meta_dir = Path(zarr_path) / "meta"
        self._text_dir = Path(zarr_path) / "text"
        self._index_path = self._meta_dir / "index.jsonl"
        self._index = {}
        self._load_index()
        self._load_existing_arrays()

        logger.info(f"ZarrActivationsLogger initialized at {zarr_path}")

    def _get_compressor(self, compression: Union[str, BaseCompressor, None]) -> BaseCompressor:
        if isinstance(compression, BaseCompressor):
            return compression
        elif compression is None:
            return NoCompression()
        elif compression == 'zstd':
            if ZSTD_AVAILABLE:
                return ZstdCompression(level=19)
            else:
                raise ImportError("zstandard module not available, cannot compress")
        else:
            raise ValueError(f"Unknown compression method '{compression}', must be 'zstd' or a BaseCompressor instance")

    def _load_index(self):
        if not self._index_path.exists():
            return
        try:
            with open(self._index_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    key = entry.get("key")
                    if key:
                        self._index[key] = entry
        except Exception as e:
            logger.warning(f"Failed to load Zarr index file: {e}")

    def _load_existing_arrays(self):
        if "prompt_activations" not in self.arrays_group:
            return

        self._prompt_activations = self.arrays_group["prompt_activations"]
        self._response_activations = self.arrays_group["response_activations"] if "response_activations" in self.arrays_group else None
        self._prompt_len = self.arrays_group["prompt_len"] if "prompt_len" in self.arrays_group else None
        self._response_len = self.arrays_group["response_len"] if "response_len" in self.arrays_group else None
        self._sample_key = self.arrays_group["sample_key"] if "sample_key" in self.arrays_group else None

        self._layer_count = int(self._prompt_activations.shape[1])
        self._hidden_size = int(self._prompt_activations.shape[-1])

        self.prompt_max_tokens = int(self._prompt_activations.shape[2])
        if self._response_activations is not None:
            self.response_max_tokens = int(self._response_activations.shape[2])

        if self.target_layers == "first_half":
            self._target_layer_indices = set(range(self._layer_count // 2))
        elif self.target_layers == "second_half":
            start_idx = self._layer_count // 2
            self._target_layer_indices = set(range(start_idx, self._layer_count))
        else:
            self._target_layer_indices = set(range(self._layer_count))

        if not self._index and self._sample_key is not None:
            for idx in range(self._sample_key.shape[0]):
                key = self._sample_key[idx].decode("utf-8")
                if key:
                    self._index[key] = {"key": key, "sample_index": idx}

    def _ensure_dirs(self):
        if self.read_only:
            return
        self._meta_dir.mkdir(parents=True, exist_ok=True)
        self._text_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_arrays(self, num_layers: int, hidden_size: int):
        if self._prompt_activations is not None:
            return

        self._layer_count = num_layers
        self._hidden_size = hidden_size

        if self.target_layers == "first_half":
            self._target_layer_indices = set(range(num_layers // 2))
        elif self.target_layers == "second_half":
            start_idx = num_layers // 2
            self._target_layer_indices = set(range(start_idx, num_layers))
        else:
            self._target_layer_indices = set(range(num_layers))

        activation_chunk_shape = self._resolve_activation_chunk_shape(num_layers, hidden_size)

        self._prompt_activations = self.arrays_group.require_dataset(
            "prompt_activations",
            shape=(0, num_layers, self.prompt_max_tokens, hidden_size),
            chunks=activation_chunk_shape,
            dtype=self.dtype,
            fill_value=0,
            compressor=None,
            overwrite=False,
            maxshape=(None, num_layers, self.prompt_max_tokens, hidden_size),
        )

        self._response_activations = self.arrays_group.require_dataset(
            "response_activations",
            shape=(0, num_layers, self.response_max_tokens, hidden_size),
            chunks=activation_chunk_shape,
            dtype=self.dtype,
            fill_value=0,
            compressor=None,
            overwrite=False,
            maxshape=(None, num_layers, self.response_max_tokens, hidden_size),
        )

        self._prompt_len = self.arrays_group.require_dataset(
            "prompt_len",
            shape=(0,),
            chunks=(max(1, min(self.chunk_size, 4096)),),
            dtype=np.int32,
            fill_value=0,
            compressor=None,
            overwrite=False,
            maxshape=(None,),
        )

        self._response_len = self.arrays_group.require_dataset(
            "response_len",
            shape=(0,),
            chunks=(max(1, min(self.chunk_size, 4096)),),
            dtype=np.int32,
            fill_value=0,
            compressor=None,
            overwrite=False,
            maxshape=(None,),
        )

        self._sample_key = self.arrays_group.require_dataset(
            "sample_key",
            shape=(0,),
            chunks=(max(1, min(self.chunk_size, 4096)),),
            dtype="S64",
            fill_value=b"",
            compressor=None,
            overwrite=False,
            maxshape=(None,),
        )

        self.root.attrs.update(
            {
                "schema_version": "zarr-v1",
                "num_layers": num_layers,
                "hidden_size": hidden_size,
                "prompt_max_tokens": self.prompt_max_tokens,
                "response_max_tokens": self.response_max_tokens,
                "prompt_chunk_tokens": self.prompt_chunk_tokens,
                "response_chunk_tokens": self.response_chunk_tokens,
                "activation_chunk_shape": activation_chunk_shape,
                "dtype": str(self.dtype),
            }
        )

    def _resolve_activation_chunk_shape(self, num_layers: int, hidden_size: int) -> Tuple[int, int, int, int]:
        if self.activation_chunk_shape is None:
            return (1, 1, self.prompt_chunk_tokens, hidden_size)

        sample_chunk, layer_chunk, token_chunk, hidden_chunk = self.activation_chunk_shape
        if hidden_chunk in (-1, 0):
            hidden_chunk = hidden_size
        if hidden_chunk != hidden_size:
            raise ValueError(
                "activation_chunk_shape hidden dimension must match hidden size "
                f"({hidden_size}), got {hidden_chunk}"
            )

        if layer_chunk > num_layers:
            raise ValueError(
                "activation_chunk_shape layer dimension cannot exceed num_layers "
                f"({num_layers}), got {layer_chunk}"
            )

        return (int(sample_chunk), int(layer_chunk), int(token_chunk), int(hidden_chunk))

    def extract_activations(self, model_outputs, input_length):
        """
        Extract activations from model outputs.
        Returns a list of per-layer tensors according to sequence_mode.
        """
        prompt_acts, response_acts, prompt_len, response_len = self._extract_prompt_response(
            model_outputs, input_length
        )

        if prompt_acts is None:
            return None

        full_hidden_states = []
        for layer_idx in range(len(prompt_acts)):
            if layer_idx not in self._target_layer_indices:
                full_hidden_states.append(None)
                continue
            if self.sequence_mode == "prompt":
                full_hidden_states.append(prompt_acts[layer_idx])
            elif self.sequence_mode == "response":
                full_hidden_states.append(response_acts[layer_idx])
            else:
                if response_acts[layer_idx] is None:
                    full_hidden_states.append(prompt_acts[layer_idx])
                else:
                    full_hidden_states.append(
                        torch.cat([prompt_acts[layer_idx], response_acts[layer_idx]], dim=1)
                    )

        return full_hidden_states

    def _extract_prompt_response(self, model_outputs, input_length: int):
        if model_outputs is None or not hasattr(model_outputs, "hidden_states"):
            if self.verbose:
                logger.info("No hidden states found in model outputs")
            return None, None, 0, 0

        all_hidden_states = model_outputs.hidden_states
        prompt_hidden = all_hidden_states[0]
        gen_hiddens = all_hidden_states[1:]

        trim_pos = None
        if hasattr(model_outputs, "trim_position"):
            trim_pos = model_outputs.trim_position
            if trim_pos is not None:
                gen_hiddens = gen_hiddens[:trim_pos]

        num_layers = len(prompt_hidden)

        if self._target_layer_indices is None:
            if self.target_layers == "first_half":
                self._target_layer_indices = set(range(num_layers // 2))
            elif self.target_layers == "second_half":
                start_idx = num_layers // 2
                self._target_layer_indices = set(range(start_idx, num_layers))
            else:
                self._target_layer_indices = set(range(num_layers))

        prompt_len = prompt_hidden[0].shape[1]
        if input_length is not None:
            prompt_len = min(prompt_len, int(input_length))

        response_len = len(gen_hiddens)

        prompt_acts = []
        response_acts = []
        for layer_idx in range(num_layers):
            if layer_idx in self._target_layer_indices:
                prompt_acts.append(prompt_hidden[layer_idx])
                if response_len > 0:
                    layer_resp = torch.cat([step[layer_idx] for step in gen_hiddens], dim=1)
                else:
                    layer_resp = prompt_hidden[layer_idx].new_zeros((1, 0, prompt_hidden[layer_idx].shape[-1]))
                response_acts.append(layer_resp)
            else:
                prompt_acts.append(None)
                response_acts.append(None)

        return prompt_acts, response_acts, prompt_len, response_len

    def log_entry(self, key: str, entry: Dict[str, Any]):
        """
        Log an entry to the Zarr store.
        Args:
            key: Unique identifier for the entry (e.g., prompt hash)
            entry: Dictionary containing activations, prompt, response, etc.
        """
        if self.read_only:
            raise ValueError("Cannot log entries in read-only mode")

        self._ensure_dirs()

        prompt_acts = None
        response_acts = None
        prompt_len = None
        response_len = None

        if "model_outputs" in entry and "input_length" in entry:
            model_outputs = entry["model_outputs"]
            input_length = entry["input_length"]

            if "trim_position" in entry:
                model_outputs.trim_position = entry["trim_position"]

            prompt_acts, response_acts, prompt_len, response_len = self._extract_prompt_response(
                model_outputs, input_length
            )
        elif "all_layers_activations" in entry:
            all_layers = entry.get("all_layers_activations")
            prompt_len = entry.get("input_length") or entry.get("prompt_len")
            if prompt_len is None:
                logger.warning("Missing input_length for splitting activations; storing all tokens as prompt")
                prompt_len = all_layers[0].shape[1]
                response_len = 0
            prompt_acts = []
            response_acts = []
            for layer_act in all_layers:
                if layer_act is None:
                    prompt_acts.append(None)
                    response_acts.append(None)
                    continue
                if isinstance(layer_act, np.ndarray):
                    layer_tensor = torch.from_numpy(layer_act)
                else:
                    layer_tensor = layer_act
                if layer_tensor.ndim == 2:
                    layer_tensor = layer_tensor.unsqueeze(0)
                prompt_tensor = layer_tensor[:, :prompt_len, :]
                response_tensor = layer_tensor[:, prompt_len:, :]
                prompt_acts.append(prompt_tensor)
                response_acts.append(response_tensor)
            response_len = 0
            for layer_resp in response_acts:
                if layer_resp is not None:
                    response_len = layer_resp.shape[1]
                    break
        else:
            raise ValueError("No model_outputs or all_layers_activations found in entry")

        if prompt_acts is None:
            raise ValueError("No activations extracted for entry")

        num_layers = len(prompt_acts)
        hidden_size = None
        for layer_act in prompt_acts:
            if layer_act is not None:
                hidden_size = layer_act.shape[-1]
                break
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from activations")

        self._ensure_arrays(num_layers, hidden_size)

        idx = self._prompt_activations.shape[0]
        self._prompt_activations.resize((idx + 1, num_layers, self.prompt_max_tokens, hidden_size))
        self._response_activations.resize((idx + 1, num_layers, self.response_max_tokens, hidden_size))
        self._prompt_len.resize((idx + 1,))
        self._response_len.resize((idx + 1,))
        self._sample_key.resize((idx + 1,))

        stored_prompt_len = int(min(prompt_len or 0, self.prompt_max_tokens))
        stored_response_len = int(min(response_len or 0, self.response_max_tokens))

        self._prompt_len[idx] = stored_prompt_len
        self._response_len[idx] = stored_response_len
        self._sample_key[idx] = np.bytes_(key)

        for layer_idx in range(num_layers):
            if layer_idx not in self._target_layer_indices:
                continue

            layer_prompt = prompt_acts[layer_idx]
            if layer_prompt is not None and stored_prompt_len > 0:
                prompt_arr = layer_prompt.squeeze(0)[:stored_prompt_len].to(dtype=torch.float16)
                self._prompt_activations[idx, layer_idx, :stored_prompt_len, :] = prompt_arr.cpu().numpy()

            layer_response = response_acts[layer_idx]
            if layer_response is not None and stored_response_len > 0:
                response_arr = layer_response.squeeze(0)[:stored_response_len].to(dtype=torch.float16)
                self._response_activations[idx, layer_idx, :stored_response_len, :] = response_arr.cpu().numpy()

        metadata_entry = {k: v for k, v in entry.items() if k not in {"model_outputs", "all_layers_activations"}}
        metadata_entry["key"] = key
        metadata_entry["sample_index"] = idx
        metadata_entry["prompt_len"] = stored_prompt_len
        metadata_entry["response_len"] = stored_response_len
        metadata_entry["logging_config"] = {
            "target_layers": self.target_layers,
            "sequence_mode": self.sequence_mode,
        }

        with open(self._index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metadata_entry) + "\n")

        self._index[key] = metadata_entry
        logger.debug(f"Logged entry {key} to Zarr store at index {idx}.")

    def get_entry(self, key: str, metadata_only: bool = False) -> Dict[str, Any]:
        """
        Retrieve an entry from the Zarr store.
        Args:
            key: Unique identifier for the entry
        Returns:
            Dictionary with activations and metadata
        """
        if key not in self._index:
            raise KeyError(f"Key {key} not found in Zarr index")

        meta = dict(self._index[key])
        idx = meta.get("sample_index")
        if idx is None:
            raise KeyError(f"Missing sample_index for key {key}")

        if metadata_only:
            return meta

        prompt_len = int(self._prompt_len[idx]) if self._prompt_len is not None else 0
        response_len = int(self._response_len[idx]) if self._response_len is not None else 0
        num_layers = self._layer_count or int(self._prompt_activations.shape[1])

        activations = []
        for layer_idx in range(num_layers):
            if layer_idx not in self._target_layer_indices:
                activations.append(None)
                continue

            if self.sequence_mode == "prompt":
                layer_prompt = self._prompt_activations[idx, layer_idx, :prompt_len, :]
                activations.append(torch.from_numpy(layer_prompt))
            elif self.sequence_mode == "response":
                if self._response_activations is None:
                    activations.append(None)
                else:
                    layer_response = self._response_activations[idx, layer_idx, :response_len, :]
                    activations.append(torch.from_numpy(layer_response))
            else:
                layer_prompt = self._prompt_activations[idx, layer_idx, :prompt_len, :]
                if self._response_activations is None or response_len == 0:
                    combined = layer_prompt
                else:
                    layer_response = self._response_activations[idx, layer_idx, :response_len, :]
                    combined = np.concatenate([layer_prompt, layer_response], axis=0)
                activations.append(torch.from_numpy(combined))

        meta["all_layers_activations"] = activations
        return meta

    def get_layer_activation(self, key: str, layer_idx: int, sequence_mode: str = "response") -> Optional[torch.Tensor]:
        """
        Retrieve a single layer's activations for a given key and sequence mode.

        Args:
            key: Unique identifier for the entry
            layer_idx: Layer index to retrieve
            sequence_mode: "prompt", "response", or "all"

        Returns:
            Tensor of shape (1, seq_len, hidden) or None if unavailable
        """
        if key not in self._index:
            return None
        if self._target_layer_indices is not None and layer_idx not in self._target_layer_indices:
            return None

        meta = self._index[key]
        idx = meta.get("sample_index")
        if idx is None:
            return None

        prompt_len = int(self._prompt_len[idx]) if self._prompt_len is not None else 0
        response_len = int(self._response_len[idx]) if self._response_len is not None else 0

        if sequence_mode == "prompt":
            if prompt_len == 0:
                return None
            layer_arr = self._prompt_activations[idx, layer_idx, :prompt_len, :]
        elif sequence_mode == "response":
            if self._response_activations is None or response_len == 0:
                return None
            layer_arr = self._response_activations[idx, layer_idx, :response_len, :]
        else:
            layer_prompt = self._prompt_activations[idx, layer_idx, :prompt_len, :]
            if self._response_activations is None or response_len == 0:
                layer_arr = layer_prompt
            else:
                layer_response = self._response_activations[idx, layer_idx, :response_len, :]
                layer_arr = np.concatenate([layer_prompt, layer_response], axis=0)

        tensor = torch.from_numpy(layer_arr)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        return tensor

    def get_entry_by_key(self, key: str, metadata_only: bool = False) -> Optional[Dict[str, Any]]:
        """
        Retrieve an entry from the Zarr store by key.
        Args:
            key: Unique identifier for the entry
            metadata_only: If True, only retrieve metadata
        Returns:
            Dictionary with activations and/or metadata
        """
        try:
            return self.get_entry(key, metadata_only=metadata_only)
        except KeyError:
            return None

    def list_entries(self) -> List[str]:
        """
        List all entry keys in the Zarr store.
        Returns:
            List of entry keys as strings
        """
        return list(self._index.keys())

    def search_metadata(self, filter_fn: Callable[[Dict[str, Any]], bool]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Search through metadata entries using a filter function.
        Args:
            filter_fn: Function that takes a metadata entry and returns True if it matches
        Returns:
            List of tuples (key, metadata_entry) for matches
        """
        results = []
        for key, meta in self._index.items():
            if filter_fn(meta):
                results.append((key, meta))
        return results

    def close(self):
        """
        Close the Zarr store (no-op for zarr, but included for API compatibility).
        """
        pass

    def fix_cuda_tensors(self):
        """
        Placeholder for CUDA tensor fix (not directly needed for Zarr, but included for API compatibility).
        """
        logger.info("fix_cuda_tensors is not implemented for ZarrActivationsLogger.")
        pass

# Example usage (not executed):
# logger = ZarrActivationsLogger()
# logger.log_entry('somekey', {'all_layers_activations': np.random.randn(2,3,4), 'prompt': 'foo'})
# entry = logger.get_entry('somekey')
# print(entry) 