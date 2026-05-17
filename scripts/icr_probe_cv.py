"""Stratified k-fold CV for ICRProbe on a single capture cell.

Trains activation_research.icr_probe.ICRProbe via a minimal training loop
that mirrors ICRProbeTrainer's optimizer / scheduler / early-stop logic.
CPU-only; the probe is ~50k params and the input is (N, L) fp32 scores so
each fold runs in minutes.

Reports per-fold AUROC + mean ± std. Also reports:
  - per-layer single-feature AUROC (baseline: how good is just one layer)
  - best-layer AUROC (baseline: best linear method without combination)
  - probe vs best-layer gap (does the MLP help?)

Usage:
    python scripts/icr_probe_cv.py shared/icr_capture/<cell> [--folds 5]
        [--epochs 50] [--batch-size 256] [--lr 1e-3] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from activation_research.icr_probe import ICRProbe  # noqa: E402


def _load_cell(cell: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores_aligned, labels) for samples present in BOTH meta.jsonl
    and icr_scores.npy.

    icr_scores.npy stacks rows in sample_index order at finalize() time. If
    meta.jsonl has more rows than the scores array (stale finalize), align to
    the first N sample_indices.
    """
    scores = np.load(cell / "icr_scores.npy")
    meta = [json.loads(l) for l in (cell / "meta.jsonl").read_text().splitlines() if l.strip()]
    meta.sort(key=lambda r: r["sample_index"])
    n = scores.shape[0]
    aligned = meta[:n]
    labels = np.array([int(bool(r["hallucinated"])) for r in aligned], dtype=np.int64)
    return scores.astype(np.float32), labels


def _stratified_kfold(labels: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Return list of test-fold index arrays. Stratified on labels.

    Doing this by hand avoids a hard sklearn dep at module import time (we
    still import roc_auc_score below, but sklearn is in requirements).
    """
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        # round-robin assign to folds
        for i, sample_idx in enumerate(idx):
            folds[i % n_folds].append(int(sample_idx))
    return [np.array(sorted(f), dtype=np.int64) for f in folds]


def _train_one_fold(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    plateau_patience: int,
    plateau_factor: float,
    early_stop_patience: int,
    seed: int,
) -> tuple[float, float, np.ndarray]:
    """Train one fold, return (best_val_auroc, n_epochs_run, val_predictions)."""
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    np.random.seed(seed)

    L = x_tr.shape[1]
    model = ICRProbe(input_dim=L)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="max", factor=plateau_factor, patience=plateau_patience,
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr).float())
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    x_va_t = torch.from_numpy(x_va)
    y_va_t = torch.from_numpy(y_va).float()

    best_auroc = 0.0
    best_preds: np.ndarray = np.zeros(len(y_va), dtype=np.float32)
    epochs_since_best = 0
    n_epochs_run = 0

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_va_t).numpy()
        try:
            auroc = float(roc_auc_score(y_va, val_logits))
        except ValueError:
            auroc = float("nan")

        if not np.isnan(auroc):
            scheduler.step(auroc)
            if auroc > best_auroc:
                best_auroc = auroc
                best_preds = val_logits.copy()
                epochs_since_best = 0
            else:
                epochs_since_best += 1

        n_epochs_run = ep + 1
        if epochs_since_best >= early_stop_patience:
            break

    return best_auroc, n_epochs_run, best_preds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cell", type=Path, help="capture directory")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--plateau-patience", type=int, default=5)
    ap.add_argument("--plateau-factor", type=float, default=0.5)
    ap.add_argument("--early-stop-patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cell = args.cell
    if not cell.is_dir():
        print(f"ERROR: not a directory: {cell}", file=sys.stderr)
        return 1

    print(f"=== {cell.name} ===")
    x, y = _load_cell(cell)
    print(f"shape:           x={x.shape}  y={y.shape}")
    print(f"label balance:   pos={y.mean():.3f}  (n_pos={int(y.sum())}/{len(y)})")

    # --- baseline: per-layer single-feature AUROC ---
    from sklearn.metrics import roc_auc_score
    layer_aurocs = np.array([roc_auc_score(y, x[:, l]) for l in range(x.shape[1])])
    df_layer = np.maximum(layer_aurocs, 1.0 - layer_aurocs)
    best_layer = int(np.argmax(df_layer))
    print(
        f"baseline:        best single-layer AUROC = {df_layer.max():.4f} (layer {best_layer})  "
        f"mean = {df_layer.mean():.4f}  n>0.55 = {int((df_layer > 0.55).sum())}/{len(df_layer)}"
    )

    # --- k-fold CV ---
    print()
    print(f"running {args.folds}-fold stratified CV (epochs={args.epochs}, lr={args.lr})")
    fold_indices = _stratified_kfold(y, args.folds, args.seed)
    all_idx = np.arange(len(y))

    fold_aurocs: list[float] = []
    fold_epochs: list[int] = []
    for i, va_idx in enumerate(fold_indices):
        tr_idx = np.setdiff1d(all_idx, va_idx, assume_unique=True)
        x_tr, y_tr = x[tr_idx], y[tr_idx]
        x_va, y_va = x[va_idx], y[va_idx]

        t0 = time.perf_counter()
        auroc, n_ep, _ = _train_one_fold(
            x_tr, y_tr, x_va, y_va,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            plateau_patience=args.plateau_patience,
            plateau_factor=args.plateau_factor,
            early_stop_patience=args.early_stop_patience,
            seed=args.seed + i,  # vary per-fold init
        )
        elapsed = time.perf_counter() - t0
        fold_aurocs.append(auroc)
        fold_epochs.append(n_ep)
        print(
            f"  fold {i + 1}/{args.folds}: "
            f"n_tr={len(tr_idx)}  n_va={len(va_idx)}  "
            f"auroc={auroc:.4f}  epochs={n_ep:2d}  ({elapsed:.1f}s)"
        )

    aurocs = np.array(fold_aurocs)
    print()
    print("=== Results ===")
    print(f"per-fold AUROC:        {[f'{a:.4f}' for a in aurocs.tolist()]}")
    print(f"probe AUROC mean ± std: {aurocs.mean():.4f} ± {aurocs.std():.4f}")
    print(f"best single-layer:      {df_layer.max():.4f}  (layer {best_layer})")
    print(f"probe gap vs best layer: {aurocs.mean() - df_layer.max():+.4f}")
    print(f"mean epochs run:        {np.mean(fold_epochs):.1f} / {args.epochs}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
