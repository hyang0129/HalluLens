# Notebook Workflow

## Master vs Working Copies

- **Master notebooks** live in `notebooks/`. These are the canonical, git-tracked versions.
- **Working notebooks** are copies placed in the repo root for active runs. Root-level `*.ipynb` files are gitignored and will not be committed.

## Rules for Claude

1. **Never run master notebooks** in `notebooks/`. They are reference copies only.
2. **All execution happens from the repo root.** When the user asks to run a notebook, work with the root-dir copy.
3. **Every notebook must start with this cell** to force the correct working directory:
   ```python
   import os
   repo_root = "/mnt/home/hyang1/LLM_research/HalluLens"
   os.chdir(repo_root)
   print(f"cwd: {os.getcwd()}")
   ```
   If a notebook is missing this cell, add it as the first code cell.
4. **After editing a root-dir working copy**, ask the user whether the master version in `notebooks/` should also be updated — unless the working copy and master have diverged significantly, in which case do not ask.
5. **When creating a new notebook**, create it in the repo root for immediate use. Ask the user if a master copy should be saved to `notebooks/`.
