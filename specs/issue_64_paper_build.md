# Implementation Spec: Paper Build Infrastructure (Issue #64)

**Goal:** Make every number and every figure in the paper trace back to a single CSV
cell. When an experiment is rerun and its CSV regenerates, the paper PDF picks up the
new value automatically — no hand-editing prose, no hand-editing figure captions, no
hand-editing in-figure labels. A stale value should be a build failure, not a
proofreading task.

This revises the original GitHub issue #64. The headline change is **late-binding
citations** instead of pre-generated `\newcommand` macros, plus an explicit
**figure→sidecar-CSV** contract so figures and prose share one provenance pipeline.

---

## 1. The citation contract

Prose never writes a numeric literal that came from an experiment. It writes a
citation that resolves at build time:

```latex
Llama-3.1 reaches \result{baseline}{llama:mmlu:acc}[1]\%{} on MMLU,
a \delta{baseline}{llama:mmlu:acc}{qwen3:mmlu:acc}[1]\,pp{} gain over Qwen3
(95\% CI \resultCI{baseline}{llama:mmlu:acc,acc_lo,acc_hi}[1]).
```

Where:

| Macro | Args | Output |
|-------|------|--------|
| `\result{csv}{key}[p]` | CSV stem, row:col key, precision digits | formatted value |
| `\delta{csv}{key_a}{key_b}[p]` | two keys in same CSV | `key_a - key_b` with sign |
| `\ratio{csv}{key_a}{key_b}[p]` | two keys | `key_a / key_b` |
| `\resultCI{csv}{mean,lo,hi}[p]` | comma-separated triple | `82.3 [80.1, 84.5]` |
| `\resultPM{csv}{mean,err}[p]` | comma-separated pair | `82.3 ± 1.2` |

**Keys** are colon-joined column-name sequences identifying one CSV row, plus a final
column name. The exact key schema is per-CSV and documented in the CSV's header
comment (see §3). The build script raises a hard error if a key doesn't resolve to
exactly one cell.

**Precision** is a per-call argument. The same cell can render as `82` in the abstract
and `82.3` in §4 without a second macro. Default precision (when `[p]` is omitted) is
declared per-CSV in its header.

**Units** are written in prose, not in the macro. `\result{...}\%` not
`\result{...}{percent}`. This keeps the macro purely about the value lookup; units stay
where they read naturally.

### How the macros work

`paper/build_numbers.py` reads every CSV in `paper/data/`, flattens each to a
`(csv_stem, key) → value` mapping, and emits `paper/generated/values.tex`:

```latex
\expandafter\def\csname hl@baseline@llama:mmlu:acc\endcsname{0.823}
\expandafter\def\csname hl@baseline@qwen3:mmlu:acc\endcsname{0.781}
% ...
```

The `\result`/`\delta`/etc. macros are defined once in `paper/macros.tex` and use
`\csname` to look up the key, then `siunitx`'s `\num{...}` to format. Missing keys
produce a `\PackageError` at LaTeX compile time, naming the offending CSV and key. No
silent fallback.

`values.tex` is gitignored; `macros.tex` is committed.

### Why not pre-declared `\newcommand`s

The original spec proposed one `\newcommand` per macroized cell, named by hand in
`macro_map.py`. Late-binding wins because:

1. **No naming problem.** Macro names can't contain digits or hyphens in LaTeX; "Llama-3.1" can't be a macro stem. With late binding the "name" is the CSV coordinate string, free of TeX's identifier rules.
2. **No 900-macro explosion.** The transfer matrix has 900 cells; cite the few you need by coordinate, don't pre-declare them all.
3. **Precision is per-citation, not per-macro.** No `\fooOne`/`\fooTwo` proliferation.
4. **Derived quantities have a real home.** `\delta` does the subtraction at build time; the math doesn't have to live in the aggregator schema or in LaTeX `\pgfmathparse`.
5. **Provenance is the call itself.** `\result{transfer}{llama:hotpot->mmlu:auprc}` already documents where the number came from. The original `*Src` sibling macro pattern is redundant.

Cost: every citation is more verbose than `\llamaMmluAcc`. Accepted.

---

## 2. Figure pipeline (parallel to prose)

Figures split into two categories:

- **Hand-made figures** — schematics, architecture diagrams, screenshots. Live in `paper/figures/`, **committed**. No source-of-truth issue; treated like any other paper asset.
- **Data-derived figures** — anything whose contents come from a CSV. Live in `paper/generated/figures/`, **gitignored**, regenerated on every `make paper` from `paper/data/*.csv`.

This mirrors the prose pipeline: committing a CSV-derived PDF reintroduces the drift
the build is meant to prevent (CSV updates, PDF stays stale). The CSV is the
committed artifact; the figure is a view of the CSV, rebuilt on demand.

### Split of responsibilities

| Script | Reads | Writes | When it runs | Portable? |
|--------|-------|--------|--------------|-----------|
| Aggregator (`scripts/aggregate_transfer.py`, etc.) | `runs/`, zarr stores | `paper/data/*.csv` | Manually, after experiments | No — needs cluster filesystem |
| Figure renderer (`paper/figures_src/render_transfer.py`, etc.) | `paper/data/*.csv` | `paper/generated/figures/*.pdf` + `*.numbers.csv` | Every `make paper` | Yes — matplotlib + a CSV |

Figure renderers live in `paper/figures_src/`, are pure Python with only
matplotlib/pandas as deps, and read exclusively from `paper/data/`. They never touch
`runs/` — that boundary is what keeps the build portable to any contributor's
machine.

### Sidecars

Every data-derived figure writes **two** outputs:

```
paper/generated/figures/transfer_llama_linear.pdf
paper/generated/figures/transfer_llama_linear.numbers.csv
```

The sidecar CSV lists every number that appears on the figure — axis tick labels,
in-plot annotations, callouts, percentages on bars:

```
label,value,role
"diagonal_mean",0.847,"annotation"
"off_diag_max",0.811,"annotation"
"y_axis_max",1.0,"axis"
```

Figure captions in `.tex` cite the sidecar:

```latex
\caption{Transfer matrix for Llama-3.1 (linear probe). Diagonal mean
\result{transfer_llama_linear.numbers}{diagonal_mean}[2], off-diagonal max
\result{transfer_llama_linear.numbers}{off_diag_max}[2].}
```

A CSV regeneration that changes the diagonal mean updates both the figure (renderer
re-runs in `make paper`) and the caption (`\result` resolves to the new sidecar
value). They share the sidecar as their single source.

**Rule for figure renderers:** every number on the PDF goes in the sidecar with a
stable label. Dropping a label from the PDF means dropping it from the sidecar. The
build warns on sidecar keys that no caption cites — catches dropped annotations.

In-figure numbers that aren't cited in any caption (e.g. axis ticks) still go in the
sidecar tagged `role=axis`. Reviewers asking "where did the y-axis max come from"
can still get an answer.

---

## 3. CSV conventions

Every CSV in `paper/data/` and every `*.numbers.csv` sidecar starts with header
comments:

```
# source_commit: 658842c
# generated: 2026-05-15T14:22:31Z
# generator: scripts/aggregate_transfer.py
# key_schema: model:source_dataset:target_dataset
# default_precision: 3
model,source_dataset,target_dataset,auprc,auprc_lo,auprc_hi,ece
llama,hotpotqa,mmlu,0.823,0.801,0.845,0.041
...
```

- `source_commit` is the git SHA the generator ran against. `build_numbers.py` checks every CSV's commit is an ancestor of `HEAD` and prints a warning summary at the end of `make paper`. Not a hard failure — the user often runs experiments on a feature branch and the paper builds on `main`.
- `key_schema` declares which column ordering forms the row key for `\result` lookups.
- `default_precision` is the rounding used when `[p]` is omitted.
- Aggregator scripts are responsible for writing these headers. Hand-edited CSVs are forbidden (lint catches CSVs without the header block).

---

## 4. The lint rule (now actually enforceable)

`paper/lint.py` runs over `main.tex` and `sections/*.tex`. The rule is:

> Any digit appearing in prose must be inside an approved citation macro, or covered by the whitelist.

Approved macros: `\result`, `\delta`, `\ratio`, `\resultCI`, `\resultPM`, plus
`\ref`/`\cite`/`\eqref` argument contents.

Whitelist (small, explicit, in `lint.py`):
- Hyperparameter prose: a curated list of phrases like `8B parameters`, `Llama-3.1`, `Qwen3-8B`, `layers 14–29`, `seeds [0, 1, 2, 3, 4]`, `learning rate 1e-4`, `150 epochs`. Each whitelist entry is a literal string match, not a regex. Forces an explicit decision per hyperparameter.
- Year literals matching `\b(19|20)\d{2}\b` (citations and "since 2020" prose).
- Equation-internal constants inside `$...$` and `\[...\]` blocks (lint ignores math mode entirely).

A lint hit is a build failure with file:line and the offending substring. Adding to
the whitelist is a deliberate edit, reviewed in PRs.

This is stricter than the original `\d+\s*(%|pp|×)` regex — it flags bare numbers
without unit tokens too — and the whitelist is the explicit escape hatch.

---

## 5. Build system

```
paper/
  Makefile
  build_numbers.py        # CSVs + sidecars → generated/values.tex
  lint.py                 # digit-not-in-citation check on prose
  macros.tex              # \result, \delta, \resultCI definitions (uses siunitx)
  main.tex
  sections/
    00_abstract.tex
    01_intro.tex
    ...
  data/
    baseline_comparison.csv
    transfer_matrix.csv
    bootstrap_metrics.csv
  figures/                # hand-made figures only (schematics, diagrams) — committed
    architecture.pdf
    ...
  figures_src/            # figure-rendering scripts (CSV → PDF + sidecar)
    render_transfer.py
    ...
  generated/              # gitignored
    values.tex            # \csname definitions, one per (csv, key) pair
    tab_transfer.tex      # generated tables, if any
    provenance.txt        # call-site → CSV cell → value, written by build_numbers.py
    figures/              # script-produced figures + sidecars — gitignored
      transfer_llama_linear.pdf
      transfer_llama_linear.numbers.csv
      ...
  .gitignore
  README.md
```

`paper/Makefile`:

```
make figures  # run every paper/figures_src/*.py against paper/data/ →
              #   paper/generated/figures/*.pdf + *.numbers.csv
make values   # build_numbers.py: scan data/ + generated/figures/*.numbers.csv → values.tex
make lint     # lint.py: digit-not-in-citation check on prose .tex files
make paper    # figures + values + lint + latexmk -pdf main.tex
make clean
```

`values` depends on `figures` because sidecar CSVs are an input to `build_numbers.py`.
`figures` is fast (matplotlib over already-aggregated CSVs); re-running on every build
is fine.

`make paper` is the only command anyone runs.

`provenance.txt` is the reviewer-letter dump: every `\result` / `\delta` / etc. call
site in the prose, its CSV cell, and the value that was substituted. Generated as a
side effect of the preprocessor pass. Not committed; written fresh every build.

---

## 6. Tables

Generated by `build_numbers.py` when a `.tex.template` exists in `paper/templates/`.
Template syntax uses the same citation macros:

```latex
% paper/templates/tab_transfer.tex.template
\begin{tabular}{lcc}
\toprule
Source & Llama AUPRC & Qwen3 AUPRC \\
\midrule
HotpotQA → MMLU & \result{transfer}{llama:hotpotqa:mmlu:auprc}[2] & \result{transfer}{qwen3:hotpotqa:mmlu:auprc}[2] \\
% ...
\bottomrule
\end{tabular}
```

The preprocessor substitutes `\result{...}` calls in templates exactly as it does in
prose, emitting `generated/tab_transfer.tex`. `main.tex` does
`\input{generated/tab_transfer}`.

This keeps tables editable as LaTeX (column alignment, multicolumn, midrules) while
the cells stay CSV-driven. The earlier proposal of fully-generated `tabular`
environments worked but lost layout flexibility; templates restore it without
giving up the data-binding.

For appendix-vs-main-body variants of the same table, write two templates citing
overlapping cells. Don't try to parameterize one template.

---

## 7. Acceptance criteria

- [ ] `paper/` directory with the layout in §5.
- [ ] `paper/macros.tex` defines `\result`, `\delta`, `\ratio`, `\resultCI`, `\resultPM` using `\csname` lookup + `siunitx` formatting. Missing keys raise `\PackageError`.
- [ ] `paper/build_numbers.py` reads `paper/data/*.csv` plus `paper/generated/figures/*.numbers.csv`, emits `paper/generated/values.tex`. Header parsing (§3) is required for `data/` CSVs; sidecar CSVs use a simpler header (generator + source data CSV) since their schema is fixed (`label,value,role`).
- [ ] At least one aggregator script (start with whatever produces `baseline_comparison.csv`) is updated to write the §3 header block.
- [ ] `paper/lint.py` implements the §4 rule. Math mode is ignored. Whitelist is a literal-string list, not regex.
- [ ] `paper/Makefile` provides `figures`, `values`, `lint`, `paper`, `clean` targets. `make paper` builds end-to-end from a clean `generated/`.
- [ ] One figure renderer in `paper/figures_src/` reads a CSV from `paper/data/` and writes both a PDF and a `*.numbers.csv` sidecar to `paper/generated/figures/`. Pure matplotlib/pandas; no dependency on `runs/` or zarr stores.
- [ ] `sections/01_intro.tex` demonstrates one `\result`, one `\delta`, and one `\resultCI` citation against real CSV data. Compiles cleanly through `make paper`.
- [ ] `paper/generated/provenance.txt` is written on every build, listing call sites → cells → values.
- [ ] `paper/.gitignore` excludes `generated/` (covers values.tex, generated tables, provenance.txt, generated/figures/), `*.aux`, `*.log`, root-level `*.pdf`. `data/` CSVs, `figures/` (hand-made), and `figures_src/` (renderers) are tracked.
- [ ] `paper/README.md` documents `make paper`, the citation macros, the figure-sidecar contract, and the whitelist-edit procedure.

---

## 8. Out of scope

- Bibliography management — assume BibTeX, address separately.
- Overleaf integration — canonical source is git; mirroring is the collaborator's problem.
- CI YAML — `make paper` should run on PRs that touch `paper/`, but the workflow file is a follow-up.
- arXiv vs camera-ready conditional compilation.
- Migrating an existing draft. This is scaffolding; existing prose gets ported when the scaffolding is in place.
- Auto-regenerating CSVs from `runs/` at build time. CSVs are generated by analysis scripts (`scripts/aggregate_*.py`) on demand, checked in, and the paper consumes them. Build doesn't re-run experiments.

---

## 9. Open questions

Scope: this section is build-system-only. Decisions about CSV key schemas, CI column
layout, and per-method aggregator splits affect call-site readability but are
aggregator-side concerns and belong in the issue that owns the aggregator (#62 for the
transfer matrix, #59 for bootstrap CIs).

1. **Namespace collision between `data/` and `generated/figures/` sidecars.** Both produce CSVs that `build_numbers.py` ingests. If a sidecar stem matches a data CSV stem, `\result{stem}{...}` is ambiguous. Resolution: prefix sidecar stems with `fig.` in the `\result` lookup, or namespace via subdirectory in `values.tex` keys. Decide before the first figure renderer lands.
2. **Caching the preprocessor.** Re-scanning every CSV on every `make paper` is fine for the current data volume. Revisit if build time exceeds ~5s.
3. **Whitelist drift.** As prose grows, the hyperparameter whitelist will grow with it. Periodically audit that entries still appear in prose — stale whitelist entries are harmless but accumulate noise.
4. **Dead-CSV warning.** Should `build_numbers.py` warn when a CSV in `data/` has no cell cited by any `\result` call? Useful for catching CSVs that have fallen out of the paper. Probably yes, low priority.
