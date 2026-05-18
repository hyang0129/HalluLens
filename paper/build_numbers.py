"""paper/build_numbers.py

Reads every CSV in paper/data/ and every *.numbers.csv sidecar in
paper/generated/figures/, then emits:

  paper/generated/values.tex   -- one csname definition per (stem, key) pair
  paper/generated/provenance.txt -- call sites in prose .tex -> CSV cell -> value

Run:
    python paper/build_numbers.py [--paper-dir <path>]

The script is deliberately free of GPU/cluster dependencies: pandas + stdlib only.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
CellMap = dict[tuple[str, str], str]  # (stem, key) -> raw value string

# ---------------------------------------------------------------------------
# §3 header parsing for paper/data/ CSVs
# ---------------------------------------------------------------------------
REQUIRED_HEADER_FIELDS = {"source_commit", "generated", "generator", "key_schema"}
OPTIONAL_HEADER_FIELDS = {"default_precision"}


class DataCSVHeader(NamedTuple):
    source_commit: str
    generated: str
    generator: str
    key_schema: list[str]  # ordered list of column names forming the row key
    default_precision: int


def parse_data_csv_header(path: Path) -> DataCSVHeader:
    """Parse the §3 comment block from a paper/data/ CSV.

    Raises ValueError if required fields are missing.
    """
    fields: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = re.match(r"#\s*(\w+):\s*(.+)", line)
            if m:
                fields[m.group(1).strip()] = m.group(2).strip()

    missing = REQUIRED_HEADER_FIELDS - set(fields)
    if missing:
        raise ValueError(
            f"{path}: missing required header fields: {', '.join(sorted(missing))}. "
            "Every CSV in paper/data/ must start with the §3 header block."
        )

    key_schema = [col.strip() for col in fields["key_schema"].split(":")]
    default_precision = int(fields.get("default_precision", "3"))

    return DataCSVHeader(
        source_commit=fields["source_commit"],
        generated=fields["generated"],
        generator=fields["generator"],
        key_schema=key_schema,
        default_precision=default_precision,
    )


# ---------------------------------------------------------------------------
# Sidecar header parsing for paper/generated/figures/*.numbers.csv
# ---------------------------------------------------------------------------
class SidecarCSVHeader(NamedTuple):
    generator: str
    source_data: str


def parse_sidecar_header(path: Path) -> SidecarCSVHeader:
    """Parse the simpler header from a *.numbers.csv sidecar."""
    fields: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = re.match(r"#\s*([\w_]+):\s*(.+)", line)
            if m:
                fields[m.group(1).strip()] = m.group(2).strip()
    return SidecarCSVHeader(
        generator=fields.get("generator", "unknown"),
        source_data=fields.get("source_data", "unknown"),
    )


# ---------------------------------------------------------------------------
# Git ancestry check
# ---------------------------------------------------------------------------
def is_ancestor_of_head(commit: str, repo_root: Path) -> bool:
    """Return True if *commit* is an ancestor of HEAD (or is HEAD)."""
    if commit in ("placeholder", "unknown", ""):
        return True  # synthetic / hand-built CSVs skip the check
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            capture_output=True,
            cwd=repo_root,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return True  # git not available — skip silently


# ---------------------------------------------------------------------------
# Flatten paper/data/ CSV into (stem, key) -> value
# ---------------------------------------------------------------------------
def load_data_csv(path: Path) -> tuple[CellMap, DataCSVHeader]:
    """Load one paper/data/ CSV and return its cell map + header."""
    header = parse_data_csv_header(path)
    stem = path.stem  # e.g. "baseline_comparison"

    # Skip comment lines when reading with pandas
    df = pd.read_csv(path, comment="#")

    # Verify key_schema columns exist
    for col in header.key_schema:
        if col not in df.columns:
            raise ValueError(
                f"{path}: key_schema column '{col}' not found in data. "
                f"Available columns: {list(df.columns)}"
            )

    # Identify value columns (everything not in key_schema)
    value_cols = [c for c in df.columns if c not in header.key_schema]

    cells: CellMap = {}

    # Store default_precision as a special key
    cells[(stem, "__default_precision__")] = str(header.default_precision)

    for _, row in df.iterrows():
        # Build the row key from key_schema columns
        row_key_parts = [str(row[col]) for col in header.key_schema]
        row_key_prefix = ":".join(row_key_parts)

        for vcol in value_cols:
            val = row[vcol]
            if pd.isna(val):
                continue
            # Full key: row_key_prefix:column_name
            full_key = f"{row_key_prefix}:{vcol}"
            cells[(stem, full_key)] = str(val)

    return cells, header


# ---------------------------------------------------------------------------
# Flatten sidecar CSV into (fig.<stem>, key) -> value
# ---------------------------------------------------------------------------
def load_sidecar_csv(path: Path) -> tuple[CellMap, SidecarCSVHeader]:
    """Load one *.numbers.csv sidecar; stem is prefixed with 'fig.'."""
    header = parse_sidecar_header(path)
    # Strip .numbers from stem: "transfer_llama_linear.numbers" -> "fig.transfer_llama_linear"
    raw_stem = path.stem  # e.g. "transfer_llama_linear.numbers"
    if raw_stem.endswith(".numbers"):
        base = raw_stem[: -len(".numbers")]
    else:
        base = raw_stem
    stem = f"fig.{base}"

    df = pd.read_csv(path, comment="#")

    required_cols = {"label", "value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{path}: sidecar CSV missing required columns: {missing}. "
            "Expected columns: label, value, role"
        )

    cells: CellMap = {}
    cells[(stem, "__default_precision__")] = "3"

    for _, row in df.iterrows():
        label = str(row["label"])
        value = str(row["value"])
        cells[(stem, label)] = value

    return cells, header


# ---------------------------------------------------------------------------
# Tex-identifier escaping
# ---------------------------------------------------------------------------
def tex_escape_csname(s: str) -> str:
    """Escape characters that are illegal in \\csname...\\endcsname.

    \\csname allows almost any character except \\endcsname itself. The key
    string is used verbatim — colons, dots, underscores are all legal.
    """
    # The only truly forbidden sequence is \endcsname; we don't expect that.
    return s


# ---------------------------------------------------------------------------
# Emit values.tex
# ---------------------------------------------------------------------------
def emit_values_tex(cells: CellMap, out_path: Path) -> None:
    """Write paper/generated/values.tex from the combined cell map."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with open(out_path, "w") as fh:
        fh.write(f"% paper/generated/values.tex\n")
        fh.write(f"% Auto-generated by build_numbers.py on {now}.\n")
        fh.write(f"% DO NOT EDIT. Regenerate with: make values\n")
        fh.write(f"% Total cells: {len(cells)}\n\n")

        # Group by stem for readability
        by_stem: dict[str, list[tuple[str, str]]] = {}
        for (stem, key), value in sorted(cells.items()):
            by_stem.setdefault(stem, []).append((key, value))

        for stem, pairs in sorted(by_stem.items()):
            fh.write(f"% --- {stem} ---\n")
            for key, value in sorted(pairs):
                csname = f"hl@{stem}@{key}"
                safe_key = tex_escape_csname(csname)
                fh.write(
                    f"\\expandafter\\def\\csname {safe_key}\\endcsname{{{value}}}\n"
                )
            fh.write("\n")


# ---------------------------------------------------------------------------
# Regex for citation macros in prose .tex files
# ---------------------------------------------------------------------------
# Matches: \result{csv}{key}[p] or \delta{...}{...}{...}[p] etc.
_MACRO_NAMES = r"(?:result|resdelta|resratio|resultCI|resultPM)"
_RESULT_RE = re.compile(
    r"\\(?P<macro>" + _MACRO_NAMES + r")"
    r"\{(?P<csv>[^}]+)\}"
    r"\{(?P<args>[^}]+)\}"
    r"(?:\{(?P<args2>[^}]+)\})?"  # optional third brace (for \resdelta/\resratio)
    r"(?:\[(?P<prec>[0-9]+)\])?"
)


def extract_citations(tex_path: Path) -> list[dict]:
    """Extract all citation macro calls from a .tex file.

    Returns a list of dicts with keys: file, line, macro, csv, args, prec.
    """
    citations = []
    text = tex_path.read_text(errors="replace")
    for lineno, line in enumerate(text.splitlines(), 1):
        for m in _RESULT_RE.finditer(line):
            citations.append(
                {
                    "file": str(tex_path),
                    "line": lineno,
                    "macro": m.group("macro"),
                    "csv": m.group("csv"),
                    "args": m.group("args"),
                    "args2": m.group("args2"),
                    "prec": m.group("prec"),
                    "raw": m.group(0),
                }
            )
    return citations


def resolve_citation(cite: dict, cells: CellMap) -> str:
    """Attempt to resolve a citation to its value(s), returning a string."""
    csv_stem = cite["csv"]
    args = cite["args"]
    macro = cite["macro"]

    if macro == "result":
        key = args
        return cells.get((csv_stem, key), "<UNRESOLVED>")
    elif macro in ("resdelta", "resratio"):
        key_a = args
        key_b = cite.get("args2") or ""
        va = cells.get((csv_stem, key_a), "<UNRESOLVED>")
        vb = cells.get((csv_stem, key_b), "<UNRESOLVED>")
        op = "-" if macro == "resdelta" else "/"
        if "<UNRESOLVED>" not in (va, vb):
            try:
                result_val = (float(va) - float(vb)) if op == "-" else (float(va) / float(vb))
                return f"{result_val:.6g}"
            except (ValueError, ZeroDivisionError):
                return f"{va} {op} {vb} = <ERROR>"
        return f"{va} {op} {vb}"
    elif macro in ("resultCI", "resultPM"):
        keys = [k.strip() for k in args.split(",")]
        values = [cells.get((csv_stem, k), "<UNRESOLVED>") for k in keys]
        sep = ", " if macro == "resultCI" else " +/- "
        return sep.join(values)
    return "<UNRESOLVED>"


# ---------------------------------------------------------------------------
# Emit provenance.txt
# ---------------------------------------------------------------------------
def emit_provenance(
    cells: CellMap,
    prose_paths: list[Path],
    out_path: Path,
) -> None:
    """Scan all prose .tex files for citation macros and write provenance."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    all_citations = []
    for p in prose_paths:
        all_citations.extend(extract_citations(p))

    with open(out_path, "w") as fh:
        fh.write(f"# paper/generated/provenance.txt\n")
        fh.write(f"# Generated by build_numbers.py on {now}\n")
        fh.write(f"# Total citations: {len(all_citations)}\n\n")
        fh.write(f"{'FILE':40s} {'LINE':5s} {'MACRO':12s} {'CSV':20s} {'KEY':40s} {'VALUE'}\n")
        fh.write("-" * 140 + "\n")

        unresolved = []
        for cite in all_citations:
            value = resolve_citation(cite, cells)
            csv_stem = cite["csv"]
            key = cite["args"]
            fh.write(
                f"{cite['file'][:40]:40s} {cite['line']:5d} "
                f"{cite['macro']:12s} {csv_stem:20s} {key:40s} {value}\n"
            )
            if "<UNRESOLVED>" in value:
                unresolved.append(cite)

        if unresolved:
            fh.write(f"\n# UNRESOLVED ({len(unresolved)}):\n")
            for cite in unresolved:
                fh.write(f"#   {cite['file']}:{cite['line']} — {cite['raw']}\n")

    return unresolved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build paper/generated/values.tex from CSVs")
    parser.add_argument(
        "--paper-dir",
        default="paper",
        help="Path to the paper/ directory (default: paper/)",
    )
    args = parser.parse_args(argv)

    paper_dir = Path(args.paper_dir)
    repo_root = paper_dir.parent

    data_dir = paper_dir / "data"
    sidecar_dir = paper_dir / "generated" / "figures"
    out_values = paper_dir / "generated" / "values.tex"
    out_provenance = paper_dir / "generated" / "provenance.txt"

    all_cells: CellMap = {}
    warnings_list: list[str] = []

    # --- Load paper/data/ CSVs ---
    data_csvs = sorted(data_dir.glob("*.csv"))
    if not data_csvs:
        print(f"WARNING: No CSVs found in {data_dir}", file=sys.stderr)

    for csv_path in data_csvs:
        try:
            cells, header = load_data_csv(csv_path)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

        # Git ancestry check
        if not is_ancestor_of_head(header.source_commit, repo_root):
            warnings_list.append(
                f"WARNING: {csv_path.name}: source_commit {header.source_commit!r} "
                "is not an ancestor of HEAD. CSV may be stale."
            )

        all_cells.update(cells)
        print(f"  Loaded data CSV: {csv_path.name} ({len(cells)} cells)")

    # --- Load sidecar CSVs from generated/figures/ ---
    if sidecar_dir.exists():
        for sidecar_path in sorted(sidecar_dir.glob("*.numbers.csv")):
            try:
                cells, sidecar_header = load_sidecar_csv(sidecar_path)
            except ValueError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                return 1
            all_cells.update(cells)
            print(f"  Loaded sidecar: {sidecar_path.name} ({len(cells)} cells)")

    print(f"  Total cells: {len(all_cells)}")

    # --- Emit values.tex ---
    emit_values_tex(all_cells, out_values)
    print(f"  Written: {out_values}")

    # --- Scan prose .tex files and emit provenance.txt ---
    prose_paths: list[Path] = []
    main_tex = paper_dir / "main.tex"
    if main_tex.exists():
        prose_paths.append(main_tex)
    sections_dir = paper_dir / "sections"
    if sections_dir.exists():
        prose_paths.extend(sorted(sections_dir.glob("*.tex")))

    unresolved = emit_provenance(all_cells, prose_paths, out_provenance)
    print(f"  Written: {out_provenance}")

    # --- Print warnings ---
    for w in warnings_list:
        print(w, file=sys.stderr)

    if unresolved:
        print(
            f"\nWARNING: {len(unresolved)} unresolved citation(s). "
            "Check paper/generated/provenance.txt.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
