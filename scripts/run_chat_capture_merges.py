#!/usr/bin/env python3
"""run_chat_capture_merges.py — drive scripts/merge_icr_captures.py over the
chat-template re-capture (shared/icr_capture_chat/), one merge per
(task, logical_split, model_slug) that was sharded.

Dir naming produced by scripts/dispatch/generate_manifest.py --chat-template
--shard-size ...:

    {base_dir}/{task}_{split}_{slug}                     unsuffixed, single dir
    {base_dir}/{task}_{split}_{slug}_{start}-{end}        one shard of many

This driver:
  1. For each of the 5 tasks x {test, train} x 2 model slugs, looks under
     --base-dir for either the single unsuffixed dir or >=2 shard dirs.
  2. Single-dir datasets are left alone (nothing to merge).
  3. Datasets with 0 or 1 shard dirs present are reported as "not ready yet"
     (the cluster capture run may still be in progress) and skipped, not
     failed.
  4. Datasets with >=2 shard dirs are merged, in index_start order, via
     `scripts/merge_icr_captures.py --inputs <dir0> <dir1> ... --out
     {base}_merged`, run as a subprocess.
  5. After each merge, the merged meta.jsonl line count is verified against
     the independently-recomputed sum of each input's own meta.jsonl line
     count (mirrors merge_icr_captures.py's own written-row accounting —
     see its module docstring for why config n_samples is NOT the right
     count to use here). Any mismatch is a hard failure (non-zero exit).
  6. A merge whose `_merged` dir already exists with the expected meta count
     is skipped (idempotent/resumable): re-running this driver after a
     partial or interrupted run only redoes what is missing or stale.

Token-zero data dependency (see docs in scripts/merge_icr_captures.py and
commit 1c3c753): fixed_token=0 / the token-zero surface used by
token_zero_mlp_probe and the tokenwise first-anchored pipeline is read from
response_activations.npy at sequence position 0, via MemmapContrastiveDataset
/ TokenwiseContrastiveDataset / ACTViTDataset. None of those readers touch
prompt_activations.npy. response_activations.npy position 0 is numerically
the final-prompt-token state (verified in
tests/test_generate_capture_batched.py), but it is already fully present in
response_activations.npy — nothing here needs the ~6 TB prompt_activations.npy
array. The default --skip (prompt_activations.npy, icr_scores.npy,
prompt_token_ids.npy) is therefore correct as-is and this driver does not
override it.

Usage:
    python scripts/run_chat_capture_merges.py [--base-dir shared/icr_capture_chat]
        [--dry-run] [--only hotpotqa_train_Llama-3.1-8B-Instruct ...]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_TASKS = ["hotpotqa", "popqa", "natural_questions", "sciq", "searchqa"]
_SPLITS = ["test", "train"]
_SLUGS = ["Llama-3.1-8B-Instruct", "Qwen3-8B"]

_MERGE_SCRIPT = _PROJECT_ROOT / "scripts" / "merge_icr_captures.py"

# Same default as scripts/merge_icr_captures.py's SKIP_DEFAULT. Passed
# explicitly (rather than relying on the subprocess's own default) so the
# invocation is self-documenting and matches
# scripts/dispatch/run_merge_hotpotqa_llama.sh's precedent.
_SKIP_DEFAULT = ["prompt_activations.npy", "icr_scores.npy", "prompt_token_ids.npy"]


def _meta_count(d: Path) -> int:
    """Written-row count for one capture dir: non-empty meta.jsonl lines.

    Mirrors scripts/merge_icr_captures.py::_meta_count exactly — that is the
    definition of "written" used by the merge itself, not config n_samples
    (which is the pre-allocated row count, including any untouched tail).
    """
    meta_path = d / "meta.jsonl"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.jsonl missing in {d}")
    with meta_path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _shard_dirs(base_dir: Path, base: str) -> list[tuple[int, int, Path]]:
    """Return (start, end, path) for every shard dir matching base_<start>-<end>."""
    pattern = re.compile(rf"^{re.escape(base)}_(\d+)-(\d+)$")
    out: list[tuple[int, int, Path]] = []
    if not base_dir.is_dir():
        return out
    for child in base_dir.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if m:
            out.append((int(m.group(1)), int(m.group(2)), child))
    out.sort(key=lambda t: t[0])
    return out


def plan_merges(base_dir: Path) -> list[dict]:
    """Build the list of planned actions, one per (task, split, slug).

    Each entry has: base, task, split, slug, status ("single" | "not_ready" |
    "merge" | "already_merged"), and for "merge"/"already_merged": inputs
    (ordered Paths), out_dir (Path), expected_total (int).
    """
    plan: list[dict] = []
    for task in _TASKS:
        for split in _SPLITS:
            for slug in _SLUGS:
                base = f"{task}_{split}_{slug}"
                unsuffixed = base_dir / base
                shards = _shard_dirs(base_dir, base)
                entry = {"base": base, "task": task, "split": split, "slug": slug}
                if shards and unsuffixed.is_dir():
                    entry["status"] = "conflict"
                    entry["detail"] = (
                        f"both unsuffixed dir and {len(shards)} shard dir(s) "
                        f"exist for {base} — ambiguous, needs manual cleanup"
                    )
                elif len(shards) >= 2:
                    inputs = [p for _, _, p in shards]
                    out_dir = base_dir / f"{base}_merged"
                    entry["status"] = "merge"
                    entry["inputs"] = inputs
                    entry["input_starts"] = [s for s, _, _ in shards]
                    entry["out_dir"] = out_dir
                elif len(shards) == 1:
                    entry["status"] = "not_ready"
                    entry["detail"] = (
                        f"only 1 shard dir present for {base} (expected >=2) — "
                        f"capture likely still in progress"
                    )
                elif unsuffixed.is_dir():
                    entry["status"] = "single"
                    entry["detail"] = f"single unsuffixed dir {unsuffixed} — left alone"
                else:
                    entry["status"] = "not_ready"
                    entry["detail"] = f"no dir found for {base} yet"
                plan.append(entry)
    return plan


def _expected_total(entry: dict) -> int:
    return sum(_meta_count(d) for d in entry["inputs"])


def _merged_is_complete(entry: dict) -> bool:
    out_dir = entry["out_dir"]
    if not (out_dir / "meta.jsonl").is_file():
        return False
    try:
        expected = _expected_total(entry)
        actual = _meta_count(out_dir)
    except FileNotFoundError:
        return False
    return actual == expected


def run_merge(entry: dict) -> None:
    """Invoke scripts/merge_icr_captures.py via subprocess for one entry.

    Raises RuntimeError (non-zero exit intent) on subprocess failure or
    verification mismatch.
    """
    inputs = entry["inputs"]
    out_dir = entry["out_dir"]
    expected_total = _expected_total(entry)

    cmd = [
        sys.executable,
        str(_MERGE_SCRIPT),
        "--inputs",
        *[str(p) for p in inputs],
        "--out",
        str(out_dir),
        "--skip",
        *_SKIP_DEFAULT,
    ]
    print(f"[run_chat_capture_merges] $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"merge_icr_captures.py failed (exit {result.returncode}) for {entry['base']}"
        )

    actual_total = _meta_count(out_dir)
    if actual_total != expected_total:
        raise RuntimeError(
            f"VERIFICATION FAILED for {entry['base']}: merged meta.jsonl has "
            f"{actual_total} rows, expected {expected_total} "
            f"(sum of input written counts). out_dir={out_dir}"
        )
    entry["merged_total"] = actual_total


def _print_table(rows: list[dict]) -> None:
    headers = ["task", "split", "slug", "status", "n_inputs", "total", "detail"]
    widths = {h: len(h) for h in headers}
    table_rows = []
    for r in rows:
        row = {
            "task": r["task"],
            "split": r["split"],
            "slug": r["slug"],
            "status": r["status"],
            "n_inputs": str(len(r.get("inputs", []))) if "inputs" in r else "-",
            "total": str(r.get("merged_total", r.get("expected_total", "-"))),
            "detail": r.get("detail", r.get("out_dir", "")),
        }
        table_rows.append(row)
        for h in headers:
            widths[h] = max(widths[h], len(str(row[h])))

    def fmt(row: dict) -> str:
        return "  ".join(str(row[h]).ljust(widths[h]) for h in headers)

    print(fmt({h: h for h in headers}))
    print("  ".join("-" * widths[h] for h in headers))
    for row in table_rows:
        print(fmt(row))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        default="shared/icr_capture_chat",
        help="Base dir holding chat-template capture shard/unsuffixed dirs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned merges only.")
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="task_split_slug",
        help="Restrict to this base name (e.g. hotpotqa_train_Llama-3.1-8B-Instruct). "
        "Repeatable.",
    )
    args = parser.parse_args(argv)

    base_dir = Path(args.base_dir)
    if not base_dir.is_absolute():
        base_dir = _PROJECT_ROOT / base_dir

    plan = plan_merges(base_dir)
    if args.only:
        only = set(args.only)
        plan = [e for e in plan if e["base"] in only]
        missing = only - {e["base"] for e in plan}
        if missing:
            print(f"[run_chat_capture_merges] WARNING: --only names not recognized: {sorted(missing)}")

    conflicts = [e for e in plan if e["status"] == "conflict"]
    if conflicts:
        for e in conflicts:
            print(f"[run_chat_capture_merges] ERROR: {e['detail']}", file=sys.stderr)
        return 1

    to_merge = [e for e in plan if e["status"] == "merge"]

    if args.dry_run:
        for e in to_merge:
            e["expected_total"] = _expected_total(e)
            already = _merged_is_complete(e)
            e["status"] = "would_skip_already_merged" if already else "would_merge"
            e["detail"] = str(e["out_dir"])
        _print_table(plan)
        return 0

    failures: list[str] = []
    for e in plan:
        if e["status"] != "merge":
            continue
        if _merged_is_complete(e):
            e["status"] = "already_merged"
            e["merged_total"] = _meta_count(e["out_dir"])
            e["detail"] = str(e["out_dir"])
            print(f"[run_chat_capture_merges] SKIP (already merged, verified): {e['base']}")
            continue
        try:
            run_merge(e)
            e["status"] = "merged"
            e["detail"] = str(e["out_dir"])
        except RuntimeError as exc:
            print(f"[run_chat_capture_merges] FAILED: {e['base']}: {exc}", file=sys.stderr)
            e["status"] = "failed"
            e["detail"] = str(exc)
            failures.append(e["base"])

    _print_table(plan)

    if failures:
        print(
            f"\n[run_chat_capture_merges] {len(failures)} merge(s) failed: {failures}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
