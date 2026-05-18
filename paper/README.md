# HalluLens Paper Build System

Every number and every figure in the paper traces back to a single CSV cell.
When an experiment re-runs and its CSV regenerates, `make paper` picks up new
values automatically. A stale value is a build failure, not a proofreading task.

## Quick start

```bash
cd paper/
make paper        # renders figures, builds values.tex, lints, compiles PDF
make figures      # render figures_src/*.py only
make values       # build generated/values.tex from data/ + sidecar CSVs
make lint         # check prose for bare digits outside citation macros
make clean        # remove generated/ and LaTeX byproducts
```

LaTeX (`latexmk`) is required for `make paper`. All other targets need only
Python 3.12 + pandas + matplotlib.

---

## Citation macros

All macros are defined in `paper/macros.tex`. Values resolve at LaTeX build
time from `paper/generated/values.tex` (produced by `build_numbers.py`).

### Note on macro names

The issue spec lists `\delta` and `\ratio`; we ship `\resdelta` and `\resratio`.
`\delta` already denotes the Greek letter delta in math mode, so it cannot
safely be redefined; we keep the `\res*` prefix on `\resratio` for naming
consistency.

### `\result{csv}{key}[p]`

Look up one cell and render it with `p` decimal places.

```latex
The probe achieves \result{baseline_comparison}{hotpotqa:icr_probe:auroc:mean}[3] AUROC.
```

- `csv` is the stem of a file in `paper/data/` (e.g. `baseline_comparison`
  for `baseline_comparison.csv`) or `fig.<name>` for a figure sidecar.
- `key` is a colon-joined coordinate per the CSV's `key_schema` plus the
  column name (e.g. `hotpotqa:icr_probe:auroc:mean`).
- `[p]` is the number of decimal places (optional; default from CSV header).

### `\resdelta{csv}{key_a}{key_b}[p]`

Compute `key_a - key_b` at build time via `\fpeval` (from `xfp`). Renders
with explicit sign.

```latex
a gain of \resdelta{baseline_comparison}{hotpotqa:icr_probe:auroc:mean}{hotpotqa:selfcheck:auroc:mean}[3] pp
```

### `\resratio{csv}{key_a}{key_b}[p]`

Compute `key_a / key_b` at build time.

```latex
a speedup of \resratio{baseline_comparison}{hotpotqa:icr_probe:auroc:mean}{hotpotqa:selfcheck:auroc:mean}[2]x
```

### `\resultCI{csv}{mean,lo,hi}[p]`

Render a confidence interval: `mean [lo, hi]`.

```latex
(95\% CI \resultCI{baseline_comparison}{hotpotqa:icr_probe:auroc:mean,hotpotqa:icr_probe:auroc_lo:mean,hotpotqa:icr_probe:auroc_hi:mean}[3])
```

### `\resultPM{csv}{mean,err}[p]`

Render mean with symmetric error: `mean +/- err`.

```latex
\resultPM{baseline_comparison}{hotpotqa:icr_probe:auroc:mean,hotpotqa:icr_probe:auroc_ci95:mean}[3]
```

---

## Figure-sidecar contract

Data-derived figures live in `paper/figures_src/` (source) and
`paper/generated/figures/` (outputs, gitignored). Every figure renderer
writes two outputs:

```
paper/generated/figures/<name>.pdf
paper/generated/figures/<name>.numbers.csv
```

The sidecar lists every number appearing on the figure:

```
# generator: paper/figures_src/render_transfer.py
# source_data: paper/data/baseline_comparison.csv
label,value,role
diagonal_mean,0.847,annotation
off_diag_max,0.811,annotation
y_axis_max,1.0,axis
```

Columns: `label` (stable identifier), `value` (raw float), `role` (`annotation`,
`axis`, or `data`).

### Citing from a sidecar

The sidecar stem is prefixed with `fig.`:

```latex
\caption{Transfer matrix. Diagonal mean
\result{fig.transfer_llama_linear}{diagonal_mean}[2],
off-diagonal max \result{fig.transfer_llama_linear}{off_diag_max}[2].}
```

This namespace (`fig.<stem>`) is how `build_numbers.py` avoids collisions
between `data/` CSVs and sidecar CSVs with the same base name.

---

## How to add a CSV

1. Drop `your_experiment.csv` into `paper/data/` with the §3 header block:

   ```
   # source_commit: <git SHA the aggregator ran at>
   # generated: 2026-05-18T14:22:31Z
   # generator: scripts/aggregate_results.py
   # key_schema: dataset:method:metric
   # default_precision: 3
   dataset,method,metric,mean,ci_95,...
   ```

2. Run `make values` to regenerate `generated/values.tex`.

3. Cite cells with `\result{your_experiment}{...}`.

---

## The lint rule and whitelist

`paper/lint.py` scans `main.tex` and `sections/*.tex`. Any digit appearing
in prose outside an approved context is a build failure.

**Approved contexts:** inside `\result`, `\resdelta`, `\resratio`, `\resultCI`,
`\resultPM`, `\ref`, `\cite`, `\citep`, `\citet`, `\eqref`, `\label`.

**Always allowed:** year literals (`\b(19|20)\d{2}\b`), math mode
(`$...$`, `\[...\]`, equation/align environments).

**Whitelist:** a literal-string list in `paper/lint.py`. Each entry is an
explicit decision. To add one, append to `WHITELIST` in `lint.py` and document
why in your PR. Example entries: `"Llama-3.1"`, `"8B parameters"`,
`"learning rate 1e-4"`.

Do not use regex in the whitelist — keep it to literal strings so the set
of allowed bare digits is small and auditable.

---

## File layout

```
paper/
  Makefile                  # build targets
  build_numbers.py          # CSVs → generated/values.tex
  lint.py                   # bare-digit checker
  macros.tex                # \result, \resdelta, etc. (committed)
  main.tex                  # minimal article wrapper
  sections/                 # prose (committed)
    00_abstract.tex
    01_intro.tex
    ...
  data/                     # committed CSVs (from aggregate scripts)
    baseline_comparison.csv
    ...
  figures/                  # hand-made figures (committed)
  figures_src/              # figure renderers (committed)
    render_transfer.py
    ...
  templates/                # .tex.template files for generated tables
  generated/                # GITIGNORED — rebuilt every make paper
    values.tex
    provenance.txt
    figures/
      *.pdf
      *.numbers.csv
  README.md
  .gitignore
```

---

## Provenance

After every `make values` (or `make paper`), `paper/generated/provenance.txt`
lists every `\result`/`\resdelta` call site in prose with its CSV cell and
resolved value. Use it as a reviewer-letter dump or sanity check.

---

## Aggregator scripts

Aggregator scripts in `scripts/` write CSVs to `paper/data/` with the §3
header block. They never run automatically at build time — the build consumes
committed CSVs. To update a CSV after re-running experiments:

```bash
python scripts/aggregate_results.py --runs-dir runs/baseline_comparison_hotpotqa \
    --paper-output paper/data/baseline_comparison.csv
```
