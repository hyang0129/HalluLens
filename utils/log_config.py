"""
Centralized logging configuration for HalluLens.

Uses loguru as the logging backend. By default, only WARNING and above are
shown on stderr so that normal runs stay quiet. Users can raise verbosity
via:

    * CLI flag  ``--log-level DEBUG``
    * Environment variable  ``HALLULENS_LOG_LEVEL=DEBUG``
    * Calling ``configure_logging("DEBUG")`` in Python

The environment variable ``HALLULENS_LOG_LEVEL`` is also propagated to
child processes (e.g. the activation-logging server) so that a single
flag controls the whole pipeline.
"""

from __future__ import annotations

import os
import sys

from loguru import logger

# Sentinel so we only configure once per process
_configured = False

# Valid loguru levels (in ascending severity)
VALID_LEVELS = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")

DEFAULT_LEVEL = "WARNING"

ENV_VAR = "HALLULENS_LOG_LEVEL"

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def _resolve_level(level: str | None = None) -> str:
    """Return the effective log level.

    Priority (highest to lowest):
        1. Explicit *level* argument
        2. ``HALLULENS_LOG_LEVEL`` environment variable
        3. ``DEFAULT_LEVEL`` (WARNING)
    """
    if level:
        level = level.upper()
    else:
        level = os.environ.get(ENV_VAR, DEFAULT_LEVEL).upper()

    if level not in VALID_LEVELS:
        raise ValueError(
            f"Invalid log level '{level}'. Choose from: {', '.join(VALID_LEVELS)}"
        )
    return level


def add_file_sink(log_file_path: str, level: str = "DEBUG") -> "int | None":
    """Add a file sink to loguru, appending to the given path.

    Intended to let the client-side (generate/inference) write into the
    same log file as the server so both streams are visible in one place.

    Args:
        log_file_path: Destination file.  Parent directories are created
                       automatically.  Pass ``None`` / ``""`` to no-op.
        level: Minimum log level written to the file (default DEBUG so
               per-pair accept/reject lines are captured).

    Returns:
        The loguru sink ID (pass to ``logger.remove()`` to detach) or
        ``None`` when *log_file_path* is falsy.
    """
    if not log_file_path:
        return None
    import pathlib
    pathlib.Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
    return logger.add(
        log_file_path,
        level=level,
        format="{time:YYYY-MM-DDTHH:mm:ss.SSSSSS} | {name} | {level} - {message}",
        mode="a",
        enqueue=True,   # thread-safe: safe to call from worker threads
        encoding="utf-8",
    )


def configure_logging(level: str | None = None, *, force: bool = False) -> str:
    """Configure loguru for the current process.

    Removes the default stderr sink and installs a new one at the
    requested *level*.  Safe to call multiple times (no-op after the
    first call unless *force* is ``True``).

    Args:
        level: Desired minimum log level (e.g. ``"DEBUG"``).
               Falls back to ``HALLULENS_LOG_LEVEL`` env var, then
               ``WARNING``.
        force: Re-configure even if already configured.

    Returns:
        The effective log level string.
    """
    global _configured
    if _configured and not force:
        return _resolve_level(level)

    effective = _resolve_level(level)

    # Remove all existing handlers (the default stderr sink)
    logger.remove()

    # Add a single stderr sink at the chosen level
    logger.add(
        sys.stderr,
        level=effective,
        format=LOG_FORMAT,
        colorize=True,
    )

    # Propagate to child processes via environment
    os.environ[ENV_VAR] = effective

    _configured = True
    return effective
