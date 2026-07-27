"""labels.py — centralized correctness-label loading with a selectable source.

Every icr_capture loader derives its binary correctness label from meta.jsonl's
`hallucinated` field (substring-match). This module adds an alternate source — an
LLM-judge sidecar (`judge_labels.jsonl`) — selectable per dataset via `label_source`.

The swap happens at the meta-loading layer: ``load_meta(dir, label_source)``
returns the meta rows with each row's ``hallucinated`` field set from the chosen
source, so all downstream ``row["hallucinated"]`` reads, ``outlier_class``, and
flip conventions keep working unchanged.

Contract (judge sidecar — additive, non-destructive; see issue #145):
  ``<capture_dir>/judge_labels.jsonl`` — one JSON/line, keyed by ``sample_index``::

      {"sample_index": 0, "judge_verdict": "CORRECT|INCORRECT|UNKNOWN",
       "hallucinated": false}   # hallucinated == (verdict == "INCORRECT")

  ``<capture_dir>/judge_labels_meta.json`` — provenance (judge_model, prompt, ...).

Alignment is by ``sample_index`` (join), NOT row order — robust to reordering
(cf. the merge-tail-misalignment bug). Rows with no judge entry or verdict
``UNKNOWN`` fall back to the substring label (counted; logged once).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

VALID_SOURCES = ("substring", "llm_judge")
JUDGE_SIDECAR = "judge_labels.jsonl"


def _read_meta_rows(capture_dir: Path) -> list[dict]:
    rows: list[dict] = []
    with (capture_dir / "meta.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_judge_map(capture_dir: Path) -> dict[int, bool]:
    """Return {sample_index: judge hallucinated bool}; UNKNOWN entries omitted."""
    path = capture_dir / JUDGE_SIDECAR
    if not path.exists():
        raise FileNotFoundError(
            f"label_source='llm_judge' but {path} not found. Run "
            f"scripts/label_audit/backfill_judge_labels.py for this capture first."
        )
    m: dict[int, bool] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if str(o.get("judge_verdict", "")).upper() == "UNKNOWN":
                continue
            m[int(o["sample_index"])] = bool(o["hallucinated"])
    return m


def load_meta(capture_dir: Union[str, Path], label_source: str = "substring") -> list[dict]:
    """Load meta.jsonl rows with ``hallucinated`` set from the chosen source.

    Parameters
    ----------
    capture_dir : str | Path
        An icr_capture directory.
    label_source : {"substring", "llm_judge"}
        "substring" (default) — use meta.jsonl's ``hallucinated`` field as-is.
        "llm_judge" — override ``hallucinated`` from ``judge_labels.jsonl`` (joined
        by ``sample_index``); rows missing a judge entry or marked ``UNKNOWN`` keep
        the substring value.

    Returns
    -------
    list[dict]
        Meta rows in file order (unchanged), with ``hallucinated`` possibly overridden.
    """
    capture_dir = Path(capture_dir)
    if label_source not in VALID_SOURCES:
        raise ValueError(f"label_source must be one of {VALID_SOURCES}, got {label_source!r}")

    rows = _read_meta_rows(capture_dir)
    if label_source == "substring":
        return rows

    judge = _load_judge_map(capture_dir)
    n_override = n_fallback = 0
    for r in rows:
        si = r.get("sample_index")
        if si is not None and int(si) in judge:
            r["hallucinated"] = judge[int(si)]
            n_override += 1
        else:
            n_fallback += 1  # keep the substring value already in the row
    logger.info(
        "load_meta(%s, llm_judge): %d judge-labeled, %d substring-fallback (%.1f%% coverage)",
        capture_dir.name, n_override, n_fallback,
        100.0 * n_override / max(1, len(rows)),
    )
    return rows
