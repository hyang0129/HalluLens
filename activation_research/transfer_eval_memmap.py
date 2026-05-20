"""Transfer evaluation: apply source-trained memmap checkpoints to target datasets.

Port of activation_research/transfer_eval.py (feat/issue-62-transfer-matrix) to the
icr_capture memmap backend produced by issue #79.  The logic — load source checkpoint
→ forward target test → AUROC — is unchanged; only the data layer differs.

Key differences from the zarr version:
- Uses MemmapActivationParser instead of ActivationParser.
- Source-train data path comes from dataset_cfg["icr_capture"]["train_dir"].
- llmsknow_probe has no persisted probe; refit at transfer time (see spec §Methods).
- discover_runs() scans runs/baseline_comparison_*_memmap/<ds>_memmap/<method>/seed_*/.
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from activation_research.memmap_activation_parser import MemmapActivationParser
from activation_research.metric_evaluator import MultiMetricHallucinationEvaluator
from activation_research.model import LogprobReconProgressiveCompressor, SimpleHaluClassifier


# Preferred and fallback checkpoint filenames per method.
_CHECKPOINT_PREFERRED = {
    "contrastive_logprob_recon": "contrastive_last.pt",
    "saplma": "linear_probe_last.pt",
    "act_vit": "best_checkpoint.pt",
}
_CHECKPOINT_FALLBACK = "final_weights.pt"

# llmsknow_probe refits at transfer time; its artifact is the sweep summary.
_LLMSKNOW_ARTIFACT = "sweep_summary.json"


def _parse_layer_spec(spec) -> list:
    """Accepts '14-29', '22,26', a list of ints, or None (→ default 14-29)."""
    if spec is None:
        return list(range(14, 30))
    if isinstance(spec, list):
        return [int(x) for x in spec]
    s = str(spec)
    if "-" in s and "," not in s:
        start, end = s.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in s.split(",")]


def _build_memmap_parser(
    capture_dir: str,
    random_seed: int = 0,
    split_strategy: str = "none",
) -> MemmapActivationParser:
    """Construct a MemmapActivationParser over an icr_capture directory.

    Source-train references use split_strategy="three_way" with random_seed equal
    to the source run's split_seed, so the diagonal sanity cell trains on the
    exact same 90% subset the in-dist run did. Target-test references use
    split_strategy="none" to expose all rows of the test capture, matching the
    in-dist eval path (run_experiment builds the test parser the same way).
    """
    return MemmapActivationParser(
        capture_dir=capture_dir,
        random_seed=random_seed,
        split_strategy=split_strategy,
        verbose=False,
    )


def load_checkpoint_model(
    method: str,
    checkpoint_path: str,
    dataset_cfg: dict,
    run_config: Optional[dict] = None,
) -> torch.nn.Module:
    """Load a model from a checkpoint file. Returns the model in eval mode on CPU.

    run_config is the deserialized config.json from the run directory; it supplies
    model_params overrides.  If absent, sensible defaults matching run_experiment.py
    are used.
    """
    input_dim = dataset_cfg.get("input_dim", 4096)

    model_params: dict = {}
    if run_config is not None:
        method_cfg = run_config.get("method", {})
        model_params = method_cfg.get("model_params", {})

    if method == "contrastive_logprob_recon":
        params = dict(
            input_dim=input_dim,
            final_dim=512,
            input_dropout=0.3,
            recon_seq_len=64,
            recon_hidden_dim=256,
            recon_lambda=1.0,
            logprob_var_threshold=1e-4,
        )
        params.update(model_params)
        model = LogprobReconProgressiveCompressor(**params)

    elif method == "saplma":
        params = dict(input_dim=input_dim, hidden_dims=[2048, 1024, 512], dropout=0.1)
        params.update(model_params)
        model = SimpleHaluClassifier(**params)

    elif method == "act_vit":
        from activation_research.act_vit import ACTViT, ACTViTConfig

        params = dict(input_dim=input_dim)
        params.update(model_params)
        cfg = ACTViTConfig(**params)
        model = ACTViT(cfg)

    else:
        raise ValueError(f"load_checkpoint_model: unsupported method '{method}'")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def get_embeddings_contrastive(
    model: torch.nn.Module,
    capture_dir: str,
    relevant_layers: list,
    *,
    device: str = "cpu",
    batch_size: int = 128,
    num_workers: int = 4,
    split_strategy: str = "none",
    split: str = "test",
    random_seed: int = 0,
) -> list:
    """Embed all samples in a capture dir using the contrastive model.

    Returns a list of dicts: [{"hashkey": str, "z_views": Tensor(1, D), "halu": int}, ...]
    The z_views shape of (1, D) matches what metrics.py's _record_embedding expects
    (it takes the mean over the K dimension, giving (D,) per record).

    We request include_response_logprobs=True here to match the cache fingerprint
    built during training — the contrastive model doesn't use logprobs at inference,
    but the MemmapContrastiveDataset caches dataset-level metadata keyed on the full
    kwarg set; mismatching would open a second cache file unnecessarily.
    """
    ap = _build_memmap_parser(
        capture_dir, random_seed=random_seed, split_strategy=split_strategy,
    )
    ds = ap.get_dataset(
        split,
        relevant_layers=relevant_layers,
        num_views=1,
        pad_length=63,
        include_response_logprobs=True,
        response_logprobs_top_k=20,
        preload=False,
        check_ram=False,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
    )

    records = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in dl:
            # views_activations: (B, num_views=1, T, H) → squeeze to (B, T, H)
            x = batch["views_activations"][:, 0, :, :]
            z = model(x.to(device)).cpu()  # (B, D)
            for i in range(z.shape[0]):
                records.append({
                    "hashkey": batch["hashkey"][i],
                    "z_views": z[i].unsqueeze(0),  # (1, D)
                    "halu": int(batch["halu"][i].item()),
                })
    return records


def get_scores_probe(
    model: torch.nn.Module,
    capture_dir: str,
    probe_layer: int,
    *,
    device: str = "cpu",
    batch_size: int = 256,
    num_workers: int = 4,
) -> tuple:
    """Forward-pass a SimpleHaluClassifier over a capture dir at a single probe layer.

    Returns (scores, labels): float32 arrays of shape (N,).
    scores are sigmoid probabilities (higher = more likely hallucinated).

    get_single_layer_dataset() returns activations at the fixed probe_layer only,
    which matches how run_saplma evaluates: it builds SingleLayerDataset from the
    contrastive dataset and uses views_activations directly.
    """
    ap = _build_memmap_parser(capture_dir)
    ds_full = ap.get_dataset(
        "test",
        relevant_layers=[probe_layer],
        num_views=1,
        pad_length=63,
        include_response_logprobs=False,
        preload=False,
        check_ram=False,
    )
    # get_single_layer_dataset returns a SingleLayerDataset with the fixed layer.
    ds = ds_full.get_single_layer_dataset(probe_layer)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
    )

    all_scores, all_labels = [], []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in dl:
            x = batch["views_activations"].to(device)
            # SingleLayerDataset may emit (B, 1, T, H) or (B, T, H) — squeeze dim 1 if present
            if x.dim() == 4:
                x = x.squeeze(1)
            out = model(x).squeeze(-1).cpu()
            all_scores.append(out.numpy())
            all_labels.append(batch["halu"].numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


def _score_act_vit(
    model: torch.nn.Module,
    *,
    tgt_test_dir: str,
    split_seed: int,
    device: str = "cpu",
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple:
    """Forward an ACT-ViT model over a target test capture, return (scores, labels).

    Mirrors run_experiment.run_act_vit's test-eval block: builds ACTViTDataset over
    the test capture's full row set (split_strategy="none") and computes sigmoid
    scores per sample.
    """
    from activation_research.act_vit_dataset import ACTViTDataset

    tgt_parser = _build_memmap_parser(
        tgt_test_dir, random_seed=split_seed, split_strategy="none",
    )
    test_idx = tgt_parser.df["sample_index"].values
    test_ds = ACTViTDataset(tgt_test_dir, test_idx)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=False,
    )
    model = model.to(device)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["activations"].to(device)
            logits = model(x).squeeze(1)
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())
    scores = np.concatenate(all_preds).astype(np.float32)
    labels = np.concatenate(all_labels).astype(np.int32)
    return scores, labels


def score_llmsknow_transfer(
    source_run_dir: str,
    src_train_dir: str,
    tgt_test_dir: str,
    training_seed: int,
    split_seed: int,
    run_config: Optional[dict] = None,
) -> tuple:
    """Refit the LLMsKnow LogReg on source-train and score target-test.

    The sklearn probe is not persisted by run_llmsknow_probe, so we refit here.
    This is fast (~seconds for 50k × H) and deterministic when training_seed matches
    the original run, which is the requirement for the diagonal sanity check.

    Returns (auroc, n_src_train, n_tgt_test).
    """
    from activation_research.llmsknow_probe import _split_view, eval_probe, train_final_probe

    artifacts_dir = os.path.join(source_run_dir, "artifacts")
    with open(os.path.join(artifacts_dir, "sweep_summary.json")) as f:
        sweep_summary = json.load(f)

    # MemmapActivationParser.get_dataset returns the full-L cache regardless of
    # `relevant_layers`. The in-dist sweep_locations writes best_layer_idx as the
    # absolute index into that cache's L axis (see llmsknow_probe.sweep_locations
    # and run_experiment.run_llmsknow_probe — both pass best_layer_idx, not the
    # model-layer name, into train_final_probe). We must do the same here.
    best_layer_idx: int = int(sweep_summary["best_layer_idx"])
    best_token_pos: int = int(sweep_summary["best_token_pos"])

    # Regularisation settings — from source run config if available, else defaults.
    sweep_cfg: dict = {}
    if run_config is not None:
        sweep_cfg = run_config.get("method", {}).get("sweep", {})
    C: float = float(sweep_cfg.get("C", 1.0))
    max_iter: int = int(sweep_cfg.get("max_iter", 100))

    # Source-train: all 50k rows (split_strategy="none" → _split_view gives all).
    # We request include_response_logprobs=True to match the cache fingerprint from
    # the original run_llmsknow_probe call (see run_experiment.py:1608-1618).
    # Source-train: same 90% subset the in-dist run used.
    # split_strategy="three_way" + random_seed=split_seed reproduces the
    # stratified split from training time (see run_experiment.py — the train
    # parser is built with these args and get_dataset("train") returns the
    # in-dist training rows).
    src_ap = _build_memmap_parser(
        src_train_dir, random_seed=split_seed, split_strategy="three_way",
    )
    src_ds = src_ap.get_dataset(
        "train",
        num_views=1,
        pad_length=63,
        include_response_logprobs=True,
        response_logprobs_top_k=20,
        preload=False,
        check_ram=False,
    )
    src_full, src_rows, src_labels = _split_view(src_ds)

    # Target-test: all rows in the test capture dir.
    tgt_ap = _build_memmap_parser(tgt_test_dir, split_strategy="none")
    tgt_ds = tgt_ap.get_dataset(
        "test",
        num_views=1,
        pad_length=63,
        include_response_logprobs=True,
        response_logprobs_top_k=20,
        preload=False,
        check_ram=False,
    )
    tgt_full, tgt_rows, tgt_labels = _split_view(tgt_ds)

    probe = train_final_probe(
        src_full, src_rows, src_labels,
        layer_idx=best_layer_idx, token_pos=best_token_pos,
        seed=training_seed, C=C, max_iter=max_iter,
    )

    outlier_class = 1  # consistent with all memmap dataset configs
    auroc, scores = eval_probe(
        probe, tgt_full, tgt_rows, tgt_labels,
        layer_idx=best_layer_idx, token_pos=best_token_pos,
        outlier_class=outlier_class,
    )

    return (
        float(auroc),
        int(src_rows.shape[0]),
        int(tgt_rows.shape[0]),
        np.asarray(scores, dtype=np.float32),
        np.asarray(tgt_labels, dtype=np.int32),
    )


def _resolve_probe_layer(run_dir: str, run_config: dict) -> int:
    """Determine the SAPLMA probe layer for this run.

    Priority: eval_metrics.json["selected_layer"] → config.json["method"]["data"]["probe_layer"] → 22.

    In memmap-trained SAPLMA, selected_layer is rarely written (the model uses a fixed
    probe_layer, not a sweep).  Falling back to config.json["method"]["data"]["probe_layer"]
    is the common case.  Default 22 matches the saplma.json canonical config.
    """
    eval_metrics_path = os.path.join(run_dir, "eval_metrics.json")
    if os.path.exists(eval_metrics_path):
        with open(eval_metrics_path) as f:
            em = json.load(f)
        if em.get("selected_layer") is not None:
            return int(em["selected_layer"])
    method_data = run_config.get("method", {}).get("data", {})
    if "probe_layer" in method_data:
        return int(method_data["probe_layer"])
    return 22


def _resolve_checkpoint(artifacts_dir: str, method: str) -> Optional[str]:
    """Return the path to the checkpoint file, or None if not found."""
    preferred = _CHECKPOINT_PREFERRED.get(method)
    if preferred:
        p = os.path.join(artifacts_dir, preferred)
        if os.path.exists(p):
            return p
    fallback = os.path.join(artifacts_dir, _CHECKPOINT_FALLBACK)
    if os.path.exists(fallback):
        return fallback
    return None


def evaluate_transfer_cell(
    source_run_dir: str,
    source_dataset_cfg: dict,
    target_dataset_cfg: dict,
    method: str,
    relevant_layers: list,
    probe_layer: int,
    device: str,
    training_seed: int,
) -> dict:
    """Evaluate one transfer matrix cell.

    Loads the source checkpoint, embeds or scores the target test split, and
    returns a dict with AUROC metrics.  Unused fields are set to None so the
    schema stays consistent across methods.
    """
    # Resolve paths relative to repo root so this works regardless of cwd.
    repo_root = Path(__file__).parent.parent

    src_train_dir = str(
        repo_root / source_dataset_cfg["icr_capture"]["train_dir"]
    )
    tgt_test_dir = str(
        repo_root / target_dataset_cfg["icr_capture"]["test_dir"]
    )

    # Load per-run config (needed for model_params and sweep settings).
    run_config: dict = {}
    config_path = os.path.join(source_run_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            run_config = json.load(f)

    outlier_class: int = int(source_dataset_cfg.get("outlier_class", 1))

    # split_seed governs the stratified 90/10 split of the source-train capture
    # at training time. Mirror it here so the source-train reference matches
    # exactly what the persisted probe/encoder was trained against.
    split_seed: int = int(run_config.get("split_seed", 42))

    if method == "llmsknow_probe":
        artifacts_dir = os.path.join(source_run_dir, "artifacts")
        sweep_path = os.path.join(artifacts_dir, _LLMSKNOW_ARTIFACT)
        if not os.path.exists(sweep_path):
            return {"status": "missing_artifact"}

        auroc, n_src, n_tgt, scores, labels = score_llmsknow_transfer(
            source_run_dir=source_run_dir,
            src_train_dir=src_train_dir,
            tgt_test_dir=tgt_test_dir,
            training_seed=training_seed,
            split_seed=split_seed,
            run_config=run_config,
        )
        if np.isnan(auroc):
            return {"status": "single_class", "auroc": float("nan"),
                    "n_src_train": n_src, "n_tgt_test": n_tgt,
                    "mahalanobis_auroc": None, "knn_auroc": None}
        return {
            "status": "ok",
            "auroc": auroc,
            "mahalanobis_auroc": None,
            "knn_auroc": None,
            "n_src_train": n_src,
            "n_tgt_test": n_tgt,
            "_scores": [float(s) for s in scores],
            "_labels": [int(l) for l in labels],
        }

    # contrastive_logprob_recon or saplma — need a checkpoint.
    artifacts_dir = os.path.join(source_run_dir, "artifacts")
    checkpoint_path = _resolve_checkpoint(artifacts_dir, method)
    if checkpoint_path is None:
        return {"status": "missing_checkpoint"}

    model = load_checkpoint_model(method, checkpoint_path, source_dataset_cfg, run_config)

    if method == "contrastive_logprob_recon":
        # Mirror run_experiment.run_contrastive_logprob_recon's eval block 1:1 so
        # the diagonal cell reproduces eval_metrics.json bit-for-bit. Inputs come
        # from the source run's config.json (method.data / method.evaluation).
        method_cfg = run_config.get("method", {})
        data_cfg = method_cfg.get("data", {})
        eval_cfg = method_cfg.get("evaluation", {})

        src_relevant_layers = _parse_layer_spec(data_cfg.get("relevant_layers"))
        target_layers = list(data_cfg.get("target_layers", [22, 26]))
        ds_kwargs = dict(
            relevant_layers=src_relevant_layers,
            num_views=int(data_cfg.get("num_views", 2)),
            pad_length=int(data_cfg.get("pad_length", 63)),
            include_response_logprobs=True,
            response_logprobs_top_k=int(data_cfg.get("response_logprobs_top_k", 20)),
            preload=False,
            check_ram=False,
        )

        src_ap = _build_memmap_parser(
            src_train_dir, random_seed=split_seed, split_strategy="three_way",
        )
        tgt_ap = _build_memmap_parser(tgt_test_dir, split_strategy="none")

        train_ds = src_ap.get_dataset("train", **ds_kwargs)
        test_ds = tgt_ap.get_dataset("test", **ds_kwargs)

        train_ds_target = train_ds.slice_layers(target_layers)
        test_ds_target = test_ds.slice_layers(target_layers)

        train_loader = DataLoader(train_ds_target, batch_size=64, shuffle=False)
        eval_loader = DataLoader(test_ds_target, batch_size=64, shuffle=False)

        metrics_list: list = []
        for m in eval_cfg.get("metrics", ["cosine", "mds", "knn"]):
            if m == "knn":
                knn_params = dict(eval_cfg.get("knn_params", {}))
                knn_params["sample_seed"] = training_seed
                # KNN is the headline scoring for contrastive — request
                # per-sample distances so AUPR / calibration / reweighting can
                # be recomputed downstream without re-running the cell.
                knn_params["include_per_sample"] = True
                metrics_list.append(
                    {"metric": "knn", "kwargs": knn_params, "train_selection": "all"}
                )
            else:
                metrics_list.append(m)

        evaluator = MultiMetricHallucinationEvaluator(
            activation_parser_df=tgt_ap.df,
            train_data_loader=train_loader,
            metrics=metrics_list,
            batch_size=int(eval_cfg.get("eval_batch_size", 256)),
            sub_batch_size=int(eval_cfg.get("sub_batch_size", 64)),
            device=device,
            num_workers=0,
            persistent_workers=False,
            outlier_class=outlier_class,
        )
        model = model.to(device)
        model.eval()
        ood_stats = evaluator.compute(eval_loader, model)

        cell = {
            "status": "ok",
            "auroc": ood_stats.get("knn_auroc"),  # KNN is the headline for contrastive
            "knn_auroc": ood_stats.get("knn_auroc"),
            "mahalanobis_auroc": ood_stats.get("mahalanobis_auroc"),
            "cosine_auroc": ood_stats.get("cosine_auroc"),
            "n_src_train": len(train_ds_target),
            "n_tgt_test": len(test_ds_target),
        }
        # Per-sample KNN distances (the headline score for contrastive).
        k_scores = ood_stats.get("knn_scores")
        k_labels = ood_stats.get("knn_labels")
        if k_scores is not None and k_labels is not None:
            cell["_scores"] = [float(s) for s in np.asarray(k_scores).ravel()]
            cell["_labels"] = [int(l) for l in np.asarray(k_labels).ravel()]
        return cell

    if method == "act_vit":
        scores, labels = _score_act_vit(
            model, tgt_test_dir=tgt_test_dir, split_seed=split_seed, device=device,
        )
    else:
        # saplma
        scores, labels = get_scores_probe(
            model,
            capture_dir=tgt_test_dir,
            probe_layer=probe_layer,
            device=device,
        )

    if len(np.unique(labels)) < 2:
        auroc = float("nan")
        status = "single_class"
    else:
        auroc = float(roc_auc_score(labels, scores))
        status = "ok"
    return {
        "status": status,
        "auroc": auroc,
        "mahalanobis_auroc": None,
        "knn_auroc": None,
        "n_src_train": None,
        "n_tgt_test": int(len(labels)),
        "_scores": [float(s) for s in scores],
        "_labels": [int(l) for l in labels],
    }


def discover_runs(runs_root: str, method: str) -> list:
    """Scan runs_root for completed runs of the given method under memmap experiments.

    Walks: runs_root/baseline_comparison_*_memmap/<ds>_memmap/<method>/seed_*/
    Only returns seeds where eval_metrics.json exists AND the required artifact
    is present:
      - contrastive_logprob_recon: contrastive_last.pt or final_weights.pt
      - saplma: linear_probe_last.pt or final_weights.pt
      - llmsknow_probe: sweep_summary.json

    The experiment dirs are expected to follow the baseline_comparison_*_memmap
    naming convention from issue #79; non-matching dirs are silently skipped.
    """
    results = []
    runs_root_path = Path(runs_root)
    if not runs_root_path.exists():
        return results

    for exp_dir in sorted(runs_root_path.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_name = exp_dir.name
        # Only process memmap experiment dirs.
        if not exp_name.endswith("_memmap"):
            continue

        for dataset_dir in sorted(exp_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            # Inner dataset dirs follow the pattern <dataset>_memmap.
            if not dataset_dir.name.endswith("_memmap"):
                continue

            method_dir = dataset_dir / method
            if not method_dir.is_dir():
                continue

            for seed_dir in sorted(method_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                try:
                    seed = int(seed_dir.name.split("_", 1)[1])
                except (ValueError, IndexError):
                    continue

                eval_metrics_path = seed_dir / "eval_metrics.json"
                if not eval_metrics_path.exists():
                    continue

                artifacts_dir = seed_dir / "artifacts"
                if method == "llmsknow_probe":
                    if not (artifacts_dir / _LLMSKNOW_ARTIFACT).exists():
                        continue
                else:
                    if _resolve_checkpoint(str(artifacts_dir), method) is None:
                        continue

                config_path = seed_dir / "config.json"
                run_config: dict = {}
                if config_path.exists():
                    with open(config_path) as f:
                        run_config = json.load(f)

                results.append({
                    "experiment_name": exp_name,
                    "dataset": dataset_dir.name,   # e.g. "hotpotqa_memmap"
                    "method": method,
                    "seed": seed,
                    "run_dir": str(seed_dir),
                    "config": run_config,
                })

    return results
