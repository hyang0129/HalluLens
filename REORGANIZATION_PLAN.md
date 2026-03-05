# Plan: HalluLens Project Reorganization

## Context

The project has grown organically and has three main problems:
1. **Root clutter**: ~15 loose scripts/test files, 20+ markdown docs, 3 notebooks, and data files all sitting in the root
2. **External repo mess**: 8 git repos in `external/` — 7 of which are completely unused in the main codebase (no imports, no references), yet taking up ~229 MB
3. **Large files tracked in git**: `metadata.json` (256 MB), `test_llama.json` (28 MB), `generation.jsonl`, `test_eval_big.json` are all tracked in git and should not be

The goal is a clean, navigable structure that separates core source from tools, docs, experiments, and references.

---

## Target Structure

```
HalluLens/
├── README.md                  # keep
├── CLAUDE.md                  # keep (update paths)
├── LICENSE                    # keep
├── CODE_OF_CONDUCT            # keep
├── requirements.txt           # keep
├── .gitignore                 # update
│
├── activation_logging/        # minor internal cleanup (move test files out)
├── activation_research/       # unchanged
├── tasks/                     # unchanged
├── utils/                     # unchanged
├── data/                      # unchanged
├── assets/                    # unchanged
├── tests/                     # add activation_logging test files here
│   └── activation_logging/    # new subfolder for moved tests
│
├── notebooks/                 # NEW — move 3 notebooks from root
│
├── scripts/                   # consolidate all runnable scripts here
│   └── tools/                 # NEW — one-off utilities/checks from root
│
├── docs/                      # all docs except root README
│   ├── (existing docs/)       # keep as-is
│   ├── planning/              # NEW subfolder — roadmap/planning docs
│   ├── setup/                 # NEW subfolder — remote dev guides
│   └── reference/             # NEW subfolder — changelogs/specs from root
│
└── external/
    ├── LLMsKnow/              # keep (used for NQ data files)
    └── README.md              # NEW — document what LLMsKnow is and why it's here
```

---

## Actions

### Phase 1: Remove unused external repos (7 dirs)
These are all unused — no imports, no references anywhere in the main codebase:
```
external/haloscope          (official submodule — also remove from .gitmodules)
external/Gnosis             (manually cloned, ~large, contains nested trl/transformers)
external/SemanticEnergy     (manually cloned)
external/Sirraya_LSD_Code   (manually cloned)
external/hallucination_probes (manually cloned)
external/selfcheckgpt       (manually cloned)
external/semantic-entropy-probes (manually cloned)
```
Commands:
```bash
git rm -r --cached external/haloscope external/Gnosis external/SemanticEnergy \
  external/Sirraya_LSD_Code external/hallucination_probes external/selfcheckgpt \
  external/semantic-entropy-probes
# Also edit .gitmodules to remove haloscope entry
rm -rf external/haloscope external/Gnosis external/SemanticEnergy \
  external/Sirraya_LSD_Code external/hallucination_probes external/selfcheckgpt \
  external/semantic-entropy-probes
```

### Phase 2: Untrack large data files
```bash
git rm --cached metadata.json generation.jsonl test_llama.json test_eval_big.json
```

### Phase 3: Delete stale migration scripts from root
```bash
rm migrate_phase.sh migrate_to_shared_storage.sh run_migration_in_screen.sh \
   cleanup_originals.sh create_symlinks.sh verify_migration.sh
```
Also delete empty/useless directories:
```bash
rm -rf checkpoints/ goodwiki_json_2/ custom_output/ test_precisewiki_logging/ test_triviaqa_format/
```
Delete duplicate/empty docs:
```bash
rm "HalluLens README.MD"   # duplicate of README.md
rm CHANGELOG.md            # empty file
```

### Phase 4: Create `notebooks/` and move notebooks (SKIP) 
```bash
mkdir notebooks
git mv b_contrastive_training_with_new_trainer.ipynb notebooks/
git mv c_layeraware_training_with_new_trainer.ipynb notebooks/
git mv k_view_loader_profile.ipynb notebooks/
```

### Phase 5: Move root-level scripts into `scripts/`
```bash
git mv connect_gpu.sh scripts/
git mv check_remote_files.sh scripts/
git mv download_q6k_model.sh scripts/
git mv test_gpu_connection.sh scripts/
```

Move one-off utilities to `scripts/tools/`:
```bash
mkdir scripts/tools
git mv check_llamacpp_gpu.py scripts/tools/
git mv check_npy_implementation.py scripts/tools/
git mv generate_inference_from_lmdb.py scripts/tools/
git mv jupyter_api_executor.py scripts/tools/
git mv test_fixed_layer_dataset.py scripts/tools/
git mv test_gguf_loading.py scripts/tools/
```

### Phase 6: Consolidate documentation into `docs/`
Create subfolders and move:
```bash
mkdir docs/planning docs/setup docs/reference

# Planning docs
git mv PAPER_ROADMAP.md docs/planning/
git mv DATASET_ROADMAP.md docs/planning/
git mv DATASET_IMPLEMENTATION_STATUS.md docs/planning/
git mv PROBABILITY_BASED_DETECTION_PLAN.md docs/planning/
git mv SOTA_TRACKER.md docs/planning/

# Setup guides
git mv REMOTE_DEV_SETUP.md docs/setup/
git mv REMOTE_TESTING_GUIDE.md docs/setup/

# Reference/changelog
git mv JSON_ACTIVATION_LOGGING.md docs/reference/
git mv RESULTS_SCHEMA.md docs/reference/
git mv CHANGELOG_SERVER_RESTART.md docs/reference/
git mv ACTIVATION_PARSER_UPDATE_SUMMARY.md docs/reference/
git mv REQUIREMENTS_UPDATE_SUMMARY.md docs/reference/
git mv README_MIGRATION.md docs/reference/
git mv README_file_generation.md docs/reference/
git mv NATURAL_QUESTIONS_SUMMARY.md docs/reference/
git mv relevantpapers.md docs/reference/
```

### Phase 7: Move `activation_logging/` internal tests to `tests/`
```bash
mkdir tests/activation_logging
git mv activation_logging/test_check_lmdb.py tests/activation_logging/
git mv activation_logging/test_gguf_inference.py tests/activation_logging/
git mv activation_logging/test_lmdb_logging.py tests/activation_logging/
```

Also move migration utility:
```bash
git mv activation_logging/migrate_json_to_zarr.py scripts/
```

### Phase 8: Update `.gitignore`
Add the following entries:
```
# Large data files
metadata.json
generation.jsonl
test_llama.json
test_eval_big.json
goodwiki_json*/

# Editor artifacts
*.swp
*.swo
```

### Phase 9: Add `external/README.md`
Document LLMsKnow: what it is, where it came from, that it's used for NQ data files only.

### Phase 10: Update `CLAUDE.md`
Update directory structure section to reflect new paths (notebooks/, docs/ subfolders, scripts/tools/).

---

## Files Kept As-Is
- `activation_logging/activations_logger.py` — deprecated but kept for backward compatibility
- `activation_research/training.py` — legacy functions kept for backward compatibility
- `activation_logging/webdataset_option_a.py` — experimental, not worth disrupting
- `scripts/migrate_lmdb_to_zarr.py` — still potentially useful

---

## Critical Files Modified
- `.gitignore` — add large data files + editor artifacts
- `.gitmodules` — remove haloscope entry
- `CLAUDE.md` — update structure documentation

---

## Verification
1. `git status` — confirm all moves staged correctly, no unintended changes
2. `python scripts/check_setup.py` — environment still valid
3. `pytest tests/` — all tests pass after moving test files
4. Check that `tasks/llmsknow/natural_questions.py` still resolves `external/LLMsKnow/data/` (path unchanged)
5. Check notebooks open correctly from new `notebooks/` location
