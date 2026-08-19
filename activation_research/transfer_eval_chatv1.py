"""Cross-dataset transfer evaluation for the chatv1 (chat-template re-capture) runs.

Extends the issue #89 memmap transfer machinery (activation_research/transfer_eval_memmap.py)
to the three chatv1 method families introduced by issue #151 / #156:

  - tokenwise_contrastive_first_anchored(_qwen3)
  - tokenwise_arch_v1_input_norm_only(_qwen3)
  - token_zero_mlp_probe(_qwen3)

Protocol (identical shape to the issue #89 machinery): load a source-dataset
seed-0 checkpoint, score the TARGET dataset's test split, report AUROC.  This
module is eval-only — it trains nothing.

Run-directory layout (see scripts/run_experiment.py:5775-5780, the
``run_dir = os.path.join(output_base, exp_name, dataset_name, method_name,
f"seed_{seed}")`` construction, and its config.json writer at
scripts/run_experiment.py:5884-5899):

    runs/chatv1_{task}/{task}_chat_memmap/{method}/seed_{seed}/
        config.json                          — {"dataset":..., "method":..., "experiment":...,
                                                 "training_seed":..., "split_seed":...}
        artifacts/final_weights.pt           — {"model_state_dict":..., "training_summary":...}
    runs/chatv1_qwen3_{task}/{task}_qwen3_chat_memmap/{method}_qwen3/seed_{seed}/
        (same shape, Qwen3 method/dataset config variants)

The checkpoint filename ``final_weights.pt`` is the ONLY artifact these three
method families ever write (never a "_last.pt" / "best_checkpoint.pt" style
name):
  - tokenwise contrastive routine: scripts/run_experiment.py:1022-1028
    (``torch.save({"model_state_dict": ..., "training_summary": ...},
    os.path.join(output_dir, "artifacts", "final_weights.pt"))``, reached via
    run_contrastive_logprob_recon with _tokenwise=True — see
    run_tokenwise_contrastive_logprob_recon at scripts/run_experiment.py:1549).
  - token_zero_mlp_probe: scripts/run_experiment.py:2945-2951 (same key shape).

kNN-bank protocol decision (Deliverable #1)
--------------------------------------------
For the tokenwise contrastive family, the in-domain metric this module must
match is ``t0_cosine_knn_auroc`` — see scripts/run_experiment.py:1220-1225
(the ``token_curve["t0_" + metric_name] = value`` loop) sourced from the
"cosine_knn" MultiMetricHallucinationEvaluator spec built at
scripts/run_experiment.py:1070-1090:

    {"name": "cosine_knn", "metric": "knn", "prefix": "cosine",
     "kwargs": cosine_knn_params, "train_selection": "all"}

with ``train_data_loader`` bound to ``train_eval_ds`` — the SOURCE run's own
in-domain train split, fixed_token=0 (scripts/run_experiment.py:670-681,
1037-1038).  In other words: even the in-domain metric's kNN reference bank
is built from the *training* dataset, never the evaluation (test) dataset.

We checked the legacy issue-89 precedent before choosing a transfer-time bank
source (activation_research/transfer_eval_memmap.py, method
"contrastive_logprob_recon"):

  - build_source_scorer (transfer_eval_memmap.py:530-533) builds
    ``train_loader`` from ``src_train_dir`` (source_dataset_cfg's
    ``icr_capture.train_dir``) via ``split_strategy="three_way"``.
  - score_on_target (transfer_eval_memmap.py:613-617) forwards the TARGET
    test capture through the model and scores it against that same
    source-train loader.

This is the same rule the in-domain metric already follows, so there is no
disagreement to resolve: **the kNN bank is always built from the SOURCE
dataset's train split, evaluated against the TARGET dataset's test split.
No target labels are used to build or select the bank.**  This module
reproduces that rule exactly, substituting the token-zero
(``TokenwiseContrastiveDataset(..., fixed_token=0, num_views=1)``) view for
the plain ``slice_layers(target_layers)`` view the legacy method used, since
issue #151 methods score token 0 only.

One deliberate deviation from the legacy contrastive_logprob_recon
score_on_target: that function passes only ``activation_parser_df=tgt_ap.df``
to MultiMetricHallucinationEvaluator, with no ``train_activation_parser_df``.
Since the evaluator's baseline-embedding label lookup
(activation_research/metric_evaluator.py:419-426) defaults
``train_activation_parser_df`` to ``activation_parser_df`` when omitted, the
legacy call resolves SOURCE-train hashkeys against the TARGET-test dataframe
— which will not find them (different question sets), so baseline (bank)
records end up unlabeled.  For a plain-distance KNN metric with
``train_selection="all"`` this is harmless (activation_research/metrics.py:
knn_ood_stats's OOD score is unsupervised — mean neighbor distance, not a
neighbor vote), BUT it silently disables ``calibrate_k`` (metrics.py:496-518
requires labeled train records to run leave-one-out k-calibration), and the
tokenwise methods' ``cosine_knn_params`` sets ``calibrate_k: true``
(configs/methods/tokenwise_contrastive_first_anchored.json).  Skipping
calibration here would make the transfer AUROC not comparable to the
in-domain ``t0_cosine_knn_auroc`` number, which DOES calibrate.  So this
module explicitly passes ``train_activation_parser_df=src_ap.df`` (the
SOURCE train parser's own dataframe) — this uses only source labels, never
target labels, so it does not violate the "no target labels for the bank"
rule; it only fixes the source-bank's own self-labeling so the same
calibration behaviour as the in-domain metric applies at transfer time too.

The token_zero_mlp_probe family is a plain supervised classifier (no bank at
all): forward the target test token-zero features through the loaded MLP and
score AUROC directly, exactly mirroring run_experiment.py's
run_token_zero_mlp_probe eval block (scripts/run_experiment.py:2953-2982).
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from activation_research.metric_evaluator import MultiMetricHallucinationEvaluator
from activation_research.model import LogprobReconProgressiveCompressor, TokenZeroMLPProbe
from activation_research.tokenwise_contrastive_dataset import TokenwiseContrastiveDataset
from activation_research.transfer_eval_memmap import _build_memmap_parser, _parse_layer_spec

# The bare dataset names covered by the chatv1 re-capture (configs/experiments/chatv1_*.json).
CHATV1_DATASETS = ["hotpotqa", "nq", "popqa", "sciq", "searchqa"]

TOKENWISE_METHODS = {
    "tokenwise_contrastive_first_anchored",
    "tokenwise_arch_v1_input_norm_only",
    "tokenwise_contrastive_first_anchored_qwen3",
    "tokenwise_arch_v1_input_norm_only_qwen3",
}
MLP_METHODS = {
    "token_zero_mlp_probe",
    "token_zero_mlp_probe_qwen3",
}
CHATV1_METHODS = TOKENWISE_METHODS | MLP_METHODS

# See module docstring: both chatv1 method families that train a checkpoint
# (tokenwise contrastive and token_zero_mlp_probe) always save this filename,
# and only this filename — scripts/run_experiment.py:1027 and :2950.
_CHECKPOINT_FILENAME = "final_weights.pt"


def _method_family(method: str) -> str:
    """Classify a chatv1 method name into 'tokenwise_contrastive' | 'token_zero_mlp_probe'."""
    if method in TOKENWISE_METHODS:
        return "tokenwise_contrastive"
    if method in MLP_METHODS:
        return "token_zero_mlp_probe"
    raise ValueError(
        f"transfer_eval_chatv1: unsupported method {method!r}; expected one of "
        f"{sorted(CHATV1_METHODS)}"
    )


def _load_run_config(source_run_dir: str) -> Optional[dict]:
    path = os.path.join(source_run_dir, "config.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _resolve_checkpoint(source_run_dir: str) -> Optional[str]:
    path = os.path.join(source_run_dir, "artifacts", _CHECKPOINT_FILENAME)
    return path if os.path.exists(path) else None


def _build_tokenwise_model(method_cfg: dict, input_dim: int) -> torch.nn.Module:
    """Build the tokenwise contrastive encoder from its persisted method config.

    Mirrors the default (model_class unset or "logprob_recon_progressive_compressor")
    branch of scripts/run_experiment.py:840-855 — the only model_class every
    current chatv1 tokenwise method config uses (see configs/methods/
    tokenwise_contrastive_first_anchored.json and tokenwise_arch_v1_input_norm_only.json,
    neither of which sets "model_class").
    """
    model_params = dict(method_cfg.get("model_params", {}))
    model_class = str(
        method_cfg.get("model_class", "logprob_recon_progressive_compressor")
    ).strip().lower()
    if model_class != "logprob_recon_progressive_compressor":
        raise NotImplementedError(
            "transfer_eval_chatv1._build_tokenwise_model only supports "
            f"model_class='logprob_recon_progressive_compressor'; got {model_class!r}. "
            "No chatv1 method config currently sets a different model_class — if one "
            "starts to, this loader needs a matching branch (see "
            "scripts/run_experiment.py:760-855 for the full model_class dispatch)."
        )
    params = dict(
        input_dim=input_dim,
        final_dim=model_params.get("final_dim", 512),
        dropout=model_params.get("dropout", 0.1),
        input_dropout=model_params.get("input_dropout", 0.3),
        normalize_input=model_params.get("normalize_input", False),
        recon_seq_len=model_params.get("recon_seq_len", 64),
        recon_hidden_dim=model_params.get("recon_hidden_dim", 256),
        recon_lambda=model_params.get("recon_lambda", 1.0),
        logprob_var_threshold=model_params.get("logprob_var_threshold", 1e-4),
        block_dims=model_params.get("block_dims"),
        pre_norm=model_params.get("pre_norm", False),
    )
    return LogprobReconProgressiveCompressor(**params)


def _build_mlp_model(method_cfg: dict, input_dim: int, num_layers: int) -> torch.nn.Module:
    """Build the token-zero MLP probe from its persisted method config.

    Mirrors scripts/run_experiment.py:2883-2890.
    """
    model_params = method_cfg.get("model_params", {})
    return TokenZeroMLPProbe(
        input_dim=input_dim,
        num_layers=num_layers,
        hidden_dim=int(model_params.get("hidden_dim", 2048)),
        output_dim=int(model_params.get("output_dim", 1024)),
        dropout=float(model_params.get("dropout", 0.1)),
        normalize_input=bool(model_params.get("normalize_input", True)),
    )


def build_source_scorer_chatv1(
    method: str,
    source_run_dir: str,
    source_dataset_cfg: dict,
    training_seed: int,
    device: str = "cpu",
) -> dict:
    """Pre-compute all source-side state for one (method, source_dataset, seed).

    Returns a scorer dict consumed by score_on_target_chatv1() for each target
    dataset.  On failure: status = "missing_artifact" (no config.json — run
    hasn't started or hasn't reached the point of writing it) |
    "missing_checkpoint" (run is still training — final_weights.pt not yet
    written).  Neither is an error; callers should skip and log a warning so
    the matrix can be re-run later to fill in stragglers.
    """
    family = _method_family(method)
    scorer: dict = {
        "method": method,
        "family": family,
        "training_seed": training_seed,
        "device": device,
    }

    run_config = _load_run_config(source_run_dir)
    if run_config is None:
        scorer["status"] = "missing_artifact"
        return scorer

    checkpoint_path = _resolve_checkpoint(source_run_dir)
    if checkpoint_path is None:
        scorer["status"] = "missing_checkpoint"
        return scorer

    method_cfg = run_config.get("method", {})
    data_cfg = method_cfg.get("data", {})
    split_seed = int(run_config.get("split_seed", 42))
    outlier_class = int(source_dataset_cfg.get("outlier_class", 1))
    input_dim = int(source_dataset_cfg.get("input_dim", 4096))
    relevant_layers = _parse_layer_spec(data_cfg.get("relevant_layers"))
    pad_length = int(data_cfg.get("pad_length", 64))

    scorer.update({
        "split_seed": split_seed,
        "outlier_class": outlier_class,
        "relevant_layers": relevant_layers,
        "pad_length": pad_length,
    })

    repo_root = Path(__file__).parent.parent
    src_train_dir = str(repo_root / source_dataset_cfg["icr_capture"]["train_dir"])

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if family == "tokenwise_contrastive":
        model = _build_tokenwise_model(method_cfg, input_dim)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        pair_mode = data_cfg.get("token_pair_mode", "first_anchored")

        # Source-train bank, split_strategy="three_way" — the exact 90% subset
        # the in-dist run trained on (scripts/run_experiment.py: `ap` is built
        # with split_strategy="three_way", random_seed=split_seed).
        src_ap = _build_memmap_parser(
            src_train_dir, random_seed=split_seed, split_strategy="three_way",
        )
        train_base = src_ap.get_dataset(
            "train",
            relevant_layers=relevant_layers,
            num_views=1,
            pad_length=pad_length,
            include_response_logprobs=False,
            preload=False,
            check_ram=False,
        )
        layer_positions = (
            list(relevant_layers)
            if hasattr(train_base, "_relevant_layers")
            else list(range(len(relevant_layers)))
        )
        train_eval_ds = TokenwiseContrastiveDataset(
            train_base,
            layer_positions=layer_positions,
            num_views=1,
            token_pair_mode=pair_mode,
            fixed_token=0,
            min_response_tokens=1,
            emit_view_logprob_targets=False,
        )
        train_loader = DataLoader(train_eval_ds, batch_size=64, shuffle=False)

        eval_cfg = method_cfg.get("evaluation", {})
        knn_defaults = eval_cfg.get("knn_params", {})
        cosine_knn_params = dict(eval_cfg.get("cosine_knn_params", knn_defaults))
        cosine_knn_params.update({
            "metric": "cosine",
            "l2_normalize": True,
            "sample_seed": training_seed,
            "include_per_sample": True,
        })

        scorer.update({
            "model": model,
            "layer_positions": layer_positions,
            "pair_mode": pair_mode,
            "train_loader": train_loader,
            # Source-only labels — see module docstring for why this is passed
            # explicitly (fixes calibrate_k, not a target-label leak).
            "train_activation_parser_df": src_ap.df,
            "cosine_knn_params": cosine_knn_params,
            "n_src_train": len(train_eval_ds),
        })
        return scorer

    # token_zero_mlp_probe family: plain supervised forward, no bank.
    model = _build_mlp_model(method_cfg, input_dim, num_layers=len(relevant_layers))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    scorer.update({
        "model": model,
        "n_src_train": None,
    })
    return scorer


def score_on_target_chatv1(scorer: dict, target_dataset_cfg: dict) -> dict:
    """Score one target dataset using pre-computed source scorer state.

    Returns {"status", "auroc", "n_test", "n_src_train"}.  Propagates any
    error status from the scorer immediately (no target data is touched).
    """
    if "status" in scorer:
        return {
            "status": scorer["status"], "auroc": None,
            "n_test": None, "n_src_train": scorer.get("n_src_train"),
        }

    repo_root = Path(__file__).parent.parent
    tgt_test_dir = str(repo_root / target_dataset_cfg["icr_capture"]["test_dir"])
    tgt_ap = _build_memmap_parser(tgt_test_dir, split_strategy="none")

    relevant_layers = scorer["relevant_layers"]
    test_base = tgt_ap.get_dataset(
        "test",
        relevant_layers=relevant_layers,
        num_views=1,
        pad_length=scorer["pad_length"],
        include_response_logprobs=False,
        preload=False,
        check_ram=False,
    )
    layer_positions = (
        list(relevant_layers)
        if hasattr(test_base, "_relevant_layers")
        else list(range(len(relevant_layers)))
    )

    if scorer["family"] == "tokenwise_contrastive":
        test_ds = TokenwiseContrastiveDataset(
            test_base,
            layer_positions=layer_positions,
            num_views=1,
            token_pair_mode=scorer["pair_mode"],
            fixed_token=0,
            min_response_tokens=1,
            emit_view_logprob_targets=False,
        )
        eval_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        metrics_list = [{
            "name": "cosine_knn",
            "metric": "knn",
            "prefix": "cosine",
            "kwargs": scorer["cosine_knn_params"],
            "train_selection": "all",
        }]
        evaluator = MultiMetricHallucinationEvaluator(
            activation_parser_df=tgt_ap.df,
            train_activation_parser_df=scorer["train_activation_parser_df"],
            train_data_loader=scorer["train_loader"],
            metrics=metrics_list,
            batch_size=256,
            sub_batch_size=64,
            device=scorer["device"],
            num_workers=0,
            persistent_workers=False,
            outlier_class=scorer["outlier_class"],
        )
        model = scorer["model"].to(scorer["device"])
        ood_stats = evaluator.compute(eval_loader, model)
        auroc = ood_stats.get("cosine_knn_auroc")
        n_test = len(test_ds)

        if auroc is None or (isinstance(auroc, float) and np.isnan(auroc)):
            return {
                "status": "single_class", "auroc": float("nan"),
                "n_test": n_test, "n_src_train": scorer["n_src_train"],
            }
        return {
            "status": "ok", "auroc": float(auroc),
            "n_test": n_test, "n_src_train": scorer["n_src_train"],
        }

    # token_zero_mlp_probe: forward test_ds directly (scripts/run_experiment.py:2953-2982).
    test_ds = TokenwiseContrastiveDataset(
        test_base,
        layer_positions=layer_positions,
        num_views=1,
        token_pair_mode="first_anchored",
        fixed_token=0,
        min_response_tokens=1,
        emit_view_logprob_targets=False,
    )
    eval_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    model = scorer["model"].to(scorer["device"])
    model.eval()

    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in eval_loader:
            x = batch["views_activations"].to(scorer["device"])
            if x.dim() == 4:
                x = x.squeeze(1)
            probs = model(x).view(-1).cpu().numpy()
            all_scores.append(probs)
            all_labels.append(batch["halu"].view(-1).numpy())

    scores = np.concatenate(all_scores) if all_scores else np.array([], dtype=np.float32)
    labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int32)
    n_test = int(len(labels))

    if len(np.unique(labels)) < 2:
        return {
            "status": "single_class", "auroc": float("nan"),
            "n_test": n_test, "n_src_train": None,
        }
    return {
        "status": "ok", "auroc": float(roc_auc_score(labels, scores)),
        "n_test": n_test, "n_src_train": None,
    }


def evaluate_transfer_cell_chatv1(
    method: str,
    source_run_dir: str,
    source_dataset_cfg: dict,
    target_dataset_cfg: dict,
    training_seed: int,
    device: str = "cpu",
) -> dict:
    """Thin wrapper: build_source_scorer_chatv1 + score_on_target_chatv1."""
    scorer = build_source_scorer_chatv1(
        method=method,
        source_run_dir=source_run_dir,
        source_dataset_cfg=source_dataset_cfg,
        training_seed=training_seed,
        device=device,
    )
    return score_on_target_chatv1(scorer, target_dataset_cfg)


def discover_runs_chatv1(runs_root: str) -> list:
    """Scan runs_root for chatv1 runs across both model families.

    Layout (scripts/run_experiment.py:5775-5780 + configs/experiments/chatv1_*.json):
      runs/chatv1_{task}/{task}_chat_memmap/{method}/seed_{seed}/                (llama)
      runs/chatv1_qwen3_{task}/{task}_qwen3_chat_memmap/{method}/seed_{seed}/    (qwen3)

    Only experiment dirs named "chatv1_{task}" or "chatv1_qwen3_{task}" for
    task in CHATV1_DATASETS are scanned; only method dirs in CHATV1_METHODS
    are returned.

    Each result has "ready" = True iff both config.json and
    artifacts/final_weights.pt exist.  Runs with ready=False (still training,
    or not yet started beyond directory creation) are still returned — the
    caller is expected to skip them with a logged warning rather than treat
    them as an error, since the chatv1 matrix is still filling in.
    """
    results: list = []
    root = Path(runs_root)
    if not root.exists():
        return results

    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_name = exp_dir.name
        if exp_name.startswith("chatv1_qwen3_"):
            model_slug = "qwen3"
            bare_task = exp_name[len("chatv1_qwen3_"):]
        elif exp_name.startswith("chatv1_"):
            model_slug = "llama"
            bare_task = exp_name[len("chatv1_"):]
        else:
            continue
        if bare_task not in CHATV1_DATASETS:
            continue

        for dataset_dir in sorted(exp_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue

            for method_dir in sorted(dataset_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                method = method_dir.name
                if method not in CHATV1_METHODS:
                    continue

                for seed_dir in sorted(method_dir.iterdir()):
                    if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                        continue
                    try:
                        seed = int(seed_dir.name.split("_", 1)[1])
                    except (ValueError, IndexError):
                        continue

                    config_path = seed_dir / "config.json"
                    checkpoint_path = seed_dir / "artifacts" / _CHECKPOINT_FILENAME
                    ready = config_path.exists() and checkpoint_path.exists()
                    if not config_path.exists():
                        # Directory exists (e.g. dispatch pre-created it) but the
                        # run hasn't reached the config-write point yet.
                        continue

                    results.append({
                        "experiment_name": exp_name,
                        "dataset": bare_task,
                        "model_slug": model_slug,
                        "method": method,
                        "seed": seed,
                        "run_dir": str(seed_dir),
                        "ready": ready,
                    })

    return results
