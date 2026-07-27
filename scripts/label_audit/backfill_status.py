#!/usr/bin/env python3
"""Report and validate LLM-judge sidecars for staged capture directories."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def inspect_capture(capture_dir: Path) -> dict:
    generation = capture_dir / "generation.jsonl"
    sidecar = capture_dir / "judge_labels.jsonl"
    generation_indexes: set[int] = set()
    malformed_generation = 0
    if generation.exists():
        with generation.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    generation_indexes.add(int(json.loads(line)["sample_index"]))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    malformed_generation += 1

    finalized: dict[int, str] = {}
    unknown_indexes: set[int] = set()
    duplicate_rows = malformed_sidecar = inconsistent_rows = 0
    seen: set[int] = set()
    if sidecar.exists():
        with sidecar.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    sample_index = int(row["sample_index"])
                    verdict = str(row["judge_verdict"]).upper()
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    malformed_sidecar += 1
                    continue
                if sample_index in seen:
                    duplicate_rows += 1
                seen.add(sample_index)
                if verdict == "UNKNOWN":
                    unknown_indexes.add(sample_index)
                    finalized.pop(sample_index, None)
                elif verdict in {"CORRECT", "INCORRECT"}:
                    expected = verdict == "INCORRECT"
                    if row.get("hallucinated") is not expected:
                        inconsistent_rows += 1
                    finalized[sample_index] = verdict
                    unknown_indexes.discard(sample_index)
                else:
                    malformed_sidecar += 1

    outside_generation = (set(finalized) | unknown_indexes) - generation_indexes
    missing = generation_indexes - set(finalized)
    return {
        "capture": capture_dir.name,
        "generation": len(generation_indexes),
        "finalized": len(set(finalized) & generation_indexes),
        "missing": len(missing),
        "unknown": len(unknown_indexes & generation_indexes),
        "duplicate_rows": duplicate_rows,
        "outside_generation": len(outside_generation),
        "malformed_generation": malformed_generation,
        "malformed_sidecar": malformed_sidecar,
        "inconsistent_rows": inconsistent_rows,
        "has_metadata": (capture_dir / "judge_labels_meta.json").exists(),
    }


def has_integrity_error(row: dict) -> bool:
    return any(row[key] for key in (
        "duplicate_rows", "outside_generation", "malformed_generation",
        "malformed_sidecar", "inconsistent_rows",
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="directory containing capture directories")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--strict", action="store_true",
                        help="fail for integrity errors, missing labels, or missing metadata")
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error(f"not a directory: {args.root}")

    rows = [inspect_capture(path) for path in sorted(args.root.iterdir())
            if path.is_dir() and (path / "generation.jsonl").exists()]
    totals = {key: sum(row[key] for row in rows) for key in (
        "generation", "finalized", "missing", "unknown", "duplicate_rows",
        "outside_generation", "malformed_generation", "malformed_sidecar",
        "inconsistent_rows",
    )}
    totals["coverage_percent"] = round(
        100 * totals["finalized"] / max(1, totals["generation"]), 2
    )
    if args.json:
        print(json.dumps({"captures": rows, "totals": totals}, indent=2))
    else:
        for row in rows:
            flags = []
            if row["unknown"]:
                flags.append(f'{row["unknown"]} unknown')
            if has_integrity_error(row):
                flags.append("INTEGRITY ERROR")
            if not row["has_metadata"]:
                flags.append("no metadata")
            suffix = f"  [{', '.join(flags)}]" if flags else ""
            print(f'{row["capture"]:58} {row["finalized"]:6}/{row["generation"]:<6} '
                  f'({100 * row["finalized"] / max(1, row["generation"]):6.2f}%){suffix}')
        print(f'\nTOTAL {totals["finalized"]}/{totals["generation"]} '
              f'({totals["coverage_percent"]:.2f}%), {totals["missing"]} remaining')

    if args.strict and any(
        row["missing"] or not row["has_metadata"] or has_integrity_error(row)
        for row in rows
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
