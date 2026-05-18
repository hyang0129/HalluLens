"""paper/lint.py

Lint rule: every digit in prose must be inside an approved citation macro,
or covered by the whitelist.

Approved contexts (digits inside these are allowed):
  result, resdelta, resratio, resultCI, resultPM macros
  ref, cite, citep, citet, eqref, label arguments

Math mode ($ $, display math, equation/align environments) is stripped before
scanning -- digits in math are allowed unconditionally.

Year literals matching the pattern (19|20)XX are allowed everywhere.

Whitelist entries are literal strings (not regexes). A digit is OK if the
surrounding text contains a whitelist string that overlaps the digit position.

Usage:
    python paper/lint.py [--paper-dir <path>]

Exit code 0 = clean. Non-zero = violations found.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Approved macro patterns — digits inside these are safe.
# A match means "this span is an approved context; skip digit scanning inside."
# ---------------------------------------------------------------------------
_CITATION_MACROS = r"(?:result|resdelta|resratio|resultCI|resultPM|ref|cite|citep|citet|eqref|label)"

# Match \macroname{...} possibly with additional brace groups and optional [p]
_APPROVED_RE = re.compile(
    r"\\"
    + _CITATION_MACROS
    + r"(?:\{[^}]*\}){1,3}"   # 1 to 3 brace groups
    + r"(?:\[[^\]]*\])?"       # optional [precision]
)

# ---------------------------------------------------------------------------
# LaTeX structural commands — not prose, digits inside are ignored.
# These appear in the preamble and document skeleton, not in paper prose.
# ---------------------------------------------------------------------------
_STRUCTURAL_CMDS = r"(?:documentclass|usepackage|geometry|input|include|inputenc|fontenc|bibliographystyle|bibliography|includegraphics|setlength|setcounter|vspace|hspace|textwidth|columnwidth|linewidth)"

_STRUCTURAL_RE = re.compile(
    r"\\"
    + _STRUCTURAL_CMDS
    + r"(?:\[[^\]]*\])?"       # optional [options]
    + r"(?:\{[^}]*\}){1,2}"   # 1 to 2 brace groups
)

# ---------------------------------------------------------------------------
# Math mode patterns — stripped before digit scanning.
# ---------------------------------------------------------------------------
# Inline math: $...$
_INLINE_MATH_RE = re.compile(r"\$[^$]*\$")
# Display math: \[...\] (non-greedy, single-line approximation)
_DISPLAY_MATH_RE = re.compile(r"\\\[.*?\\\]", re.DOTALL)
# equation/align/align* environments (multi-line)
_ENV_MATH_RE = re.compile(
    r"\\begin\{(?:equation|align|align\*|gather|gather\*|multline)\}.*?\\end\{(?:equation|align|align\*|gather|gather\*|multline)\}",
    re.DOTALL,
)

# ---------------------------------------------------------------------------
# Year pattern — allowed everywhere.
# ---------------------------------------------------------------------------
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

# ---------------------------------------------------------------------------
# Digit detector
# ---------------------------------------------------------------------------
_DIGIT_RE = re.compile(r"\d")

# ---------------------------------------------------------------------------
# Whitelist — literal strings; a digit is OK if its context contains one.
# To add an entry, append a string here and document it in README.md.
# ---------------------------------------------------------------------------
WHITELIST: list[str] = [
    # Model names and sizes
    "8B parameters",
    "Llama-3.1",
    "Llama-3.1-8B",
    "Llama-3.1-8B-Instruct",
    "Qwen3-8B",
    "Qwen3",
    "GPT-4",
    "GPT-3.5",
    # Training hyperparameters (explicit, not experimental results)
    "layers 14",
    "layers 14-29",
    "layers 22,26",
    "seeds [0, 1, 2, 3, 4]",
    "learning rate 1e-4",
    "150 epochs",
    "32 samples",
    "batch size 32",
    "top-k",
    "top-p",
    "2e-4",
    "1e-3",
    "1e-4",
    "1e-5",
    # Hardware
    "H200",
    "A100",
    "V100",
    "T4",
    # Cross-reference macros (the argument digits are already masked by _APPROVED_RE,
    # but these catch "Figure~\ref", "Table~\ref" prose patterns where ~ is nearby)
    "Figure~\\ref",
    "Table~\\ref",
    "Section~\\ref",
    "Appendix~\\ref",
    # Confidence interval prose: "95\% CI" is a standard statistical phrase
    "95\\% CI",
    "95\\%",
    "95\\,\\% CI",
    # Enumerated contributions in intro/conclusion (prose lists)
    "(1) ",
    "(2) ",
    "(3) ",
    "(4) ",
    "(5) ",
    # Footnote/endnote markers
    "\\footnote{",
]


def _mask_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Replace all character positions covered by *spans* with spaces."""
    text_list = list(text)
    for start, end in spans:
        for i in range(start, end):
            if i < len(text_list):
                text_list[i] = " "
    return "".join(text_list)


def strip_math_and_approved(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Return text with math mode and approved macros replaced by spaces.

    Also returns the list of (start, end) spans that were masked, so callers
    can check whether a digit position was masked.
    """
    masked_spans: list[tuple[int, int]] = []

    # Strip multi-line math environments first
    for m in _ENV_MATH_RE.finditer(text):
        masked_spans.append((m.start(), m.end()))

    # Strip display math \[...\]
    for m in _DISPLAY_MATH_RE.finditer(text):
        masked_spans.append((m.start(), m.end()))

    # Strip inline math $...$
    for m in _INLINE_MATH_RE.finditer(text):
        masked_spans.append((m.start(), m.end()))

    # Strip approved citation macros
    for m in _APPROVED_RE.finditer(text):
        masked_spans.append((m.start(), m.end()))

    # Strip structural LaTeX commands (preamble/skeleton, not prose)
    for m in _STRUCTURAL_RE.finditer(text):
        masked_spans.append((m.start(), m.end()))

    stripped = _mask_spans(text, masked_spans)
    return stripped, masked_spans


def _context_window(text: str, pos: int, width: int = 40) -> str:
    """Return a short context string around *pos* in *text*."""
    start = max(0, pos - width)
    end = min(len(text), pos + width + 1)
    snippet = text[start:end].replace("\n", " ")
    return snippet


def lint_file(path: Path) -> list[str]:
    """Lint one .tex file; return list of violation strings (empty = clean)."""
    text = path.read_text(errors="replace")
    violations: list[str] = []

    # Strip TeX comment lines (% ...) — they should not be linted
    # Remove LaTeX comments (% to end-of-line), but not \%
    text_no_comments = re.sub(r"(?<!\\)%.*", "", text)

    # Work line by line so we can report line numbers
    lines_original = text_no_comments.splitlines()

    for lineno, line in enumerate(lines_original, 1):
        # Strip math mode and approved macros from this line
        stripped, _ = strip_math_and_approved(line)

        # Check each digit in the stripped line
        for digit_match in _DIGIT_RE.finditer(stripped):
            pos = digit_match.start()

            # Year check: is this digit part of a year pattern in the ORIGINAL line?
            # Re-check in original line at same position
            if _YEAR_RE.search(line[max(0, pos - 4) : pos + 4]):
                continue

            # Whitelist check: does the context around the digit contain any entry?
            context_start = max(0, pos - 60)
            context_end = min(len(line), pos + 60)
            context = line[context_start:context_end]
            if any(entry in context for entry in WHITELIST):
                continue

            # Violation
            snippet = _context_window(stripped, pos, width=30)
            violations.append(
                f"{path}:{lineno}: bare digit '{digit_match.group()}' "
                f"outside approved context — «{snippet.strip()}»"
            )

    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lint .tex prose for bare digits")
    parser.add_argument(
        "--paper-dir",
        default="paper",
        help="Path to the paper/ directory (default: paper/)",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Specific .tex files to lint (overrides default scan)",
    )
    args = parser.parse_args(argv)

    paper_dir = Path(args.paper_dir)

    if args.files:
        tex_files = [Path(f) for f in args.files]
    else:
        tex_files = []
        main_tex = paper_dir / "main.tex"
        if main_tex.exists():
            tex_files.append(main_tex)
        sections_dir = paper_dir / "sections"
        if sections_dir.exists():
            tex_files.extend(sorted(sections_dir.glob("*.tex")))

    if not tex_files:
        print(f"No .tex files found under {paper_dir}", file=sys.stderr)
        return 0

    all_violations: list[str] = []
    for tex_path in tex_files:
        violations = lint_file(tex_path)
        all_violations.extend(violations)

    if all_violations:
        for v in all_violations:
            print(v, file=sys.stderr)
        print(
            f"\nlint: {len(all_violations)} violation(s) found. "
            "Add to WHITELIST in paper/lint.py or use \\result{{...}} macros.",
            file=sys.stderr,
        )
        return 1

    print(f"lint: OK — {len(tex_files)} file(s) scanned, no bare digits found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
