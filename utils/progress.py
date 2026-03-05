"""Shared progress-bar adapter for notebook and terminal compatibility.

This module centralizes tqdm imports so HalluLens can use notebook-friendly
progress bars in Jupyter while preserving normal terminal behavior.

Backend selection priority:
1. Explicit function argument (`backend=`)
2. `HALLULENS_TQDM_BACKEND` environment variable
3. `TQDM_BACKEND` environment variable
4. Default: `auto`

Supported backend values: `auto`, `notebook`, `std`.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Literal, Tuple

TqdmBackend = Literal["auto", "notebook", "std"]
_VALID_BACKENDS = {"auto", "notebook", "std"}
_BACKEND_ENV_VARS = ("HALLULENS_TQDM_BACKEND", "TQDM_BACKEND")


def _normalize_backend(backend: str | None) -> TqdmBackend:
    value = (backend or "").strip().lower()
    if value == "":
        return "auto"

    aliases = {
        "terminal": "std",
        "console": "std",
        "classic": "std",
    }
    value = aliases.get(value, value)
    if value not in _VALID_BACKENDS:
        raise ValueError(f"Invalid tqdm backend '{backend}'. Expected one of {_VALID_BACKENDS}.")
    return value  # type: ignore[return-value]


def get_tqdm_backend(default: str = "auto") -> TqdmBackend:
    """Resolve the configured tqdm backend from environment variables."""
    for env_var in _BACKEND_ENV_VARS:
        env_value = os.getenv(env_var)
        if env_value:
            return _normalize_backend(env_value)
    return _normalize_backend(default)


def _resolve_tqdm_pair(backend: TqdmBackend) -> Tuple[Callable[..., Any], Callable[..., Any]]:
    if backend == "notebook":
        from tqdm.notebook import tqdm as _tqdm, trange as _trange
    elif backend == "std":
        from tqdm.std import tqdm as _tqdm, trange as _trange
    else:
        from tqdm.auto import tqdm as _tqdm, trange as _trange
    return _tqdm, _trange


def refresh_tqdm(backend: str | None = None) -> TqdmBackend:
    """Refresh module-level tqdm aliases.

    Args:
        backend: Optional backend override (`auto`, `notebook`, `std`).

    Returns:
        The resolved backend.
    """
    chosen = _normalize_backend(backend) if backend is not None else get_tqdm_backend()
    selected_tqdm, selected_trange = _resolve_tqdm_pair(chosen)

    global tqdm, trange
    tqdm = selected_tqdm
    trange = selected_trange
    return chosen


def set_tqdm_backend(backend: str) -> TqdmBackend:
    """Set backend preference for the current process and refresh aliases."""
    chosen = _normalize_backend(backend)
    os.environ["HALLULENS_TQDM_BACKEND"] = chosen
    refresh_tqdm(chosen)
    return chosen


def install_tqdm_global(backend: str | None = None) -> TqdmBackend:
    """Patch top-level `tqdm.tqdm` and `tqdm.trange` for legacy imports.

    This allows existing code using `from tqdm import tqdm` to inherit the
    configured backend as long as this function is called before those imports.
    """
    chosen = refresh_tqdm(backend)
    import tqdm as tqdm_module

    tqdm_module.tqdm = tqdm
    tqdm_module.trange = trange
    return chosen


tqdm: Callable[..., Any]
trange: Callable[..., Any]
refresh_tqdm()


__all__ = [
    "tqdm",
    "trange",
    "TqdmBackend",
    "get_tqdm_backend",
    "set_tqdm_backend",
    "refresh_tqdm",
    "install_tqdm_global",
]
