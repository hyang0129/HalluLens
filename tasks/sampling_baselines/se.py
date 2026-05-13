"""Semantic Entropy — paper-faithful port of jlko/semantic_uncertainty.

Reference: https://github.com/jlko/semantic_uncertainty/blob/master/semantic_uncertainty/
           uncertainty/uncertainty_measures/semantic_entropy.py

Procedure (per Kuhn/Gal/Farquhar 2023 + Farquhar Nature 2024):

  1. NLI matrix M[i,j,:] = P(contradict | neutral | entail) for premise=text_i,
     hypothesis=text_j (produced upstream by nli_scorer).
  2. Bidirectional entailment clustering via get_semantic_ids — greedy assignment
     with reference's loose-vs-strict rule on argmax labels.
  3. Per-cluster log-likelihood via logsumexp_by_id (sum-normalized).
  4. Headline entropy: predictive_entropy_rao over those cluster log-likelihoods.
     Auxiliary: cluster_assignment_entropy (discrete) and length-normalized variant
     kept for back-compat with assemble_baseline_table.py and compute_sep.py.
"""
import json
import math
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

# Index convention in the (K+1, K+1, 3) NLI matrix (verified at load time by NLIScorer)
CONTRADICTION = 0
NEUTRAL = 1
ENTAILMENT = 2


# ---------------------------------------------------------------------------
# Clustering (port of get_semantic_ids from reference)
# ---------------------------------------------------------------------------

def _are_equivalent(
    nli_matrix: np.ndarray,
    i: int,
    j: int,
    strict_entailment: bool,
) -> bool:
    """Bidirectional equivalence using the reference's argmax-based rule."""
    impl_ij = int(np.argmax(nli_matrix[i, j]))
    impl_ji = int(np.argmax(nli_matrix[j, i]))

    if strict_entailment:
        return impl_ij == ENTAILMENT and impl_ji == ENTAILMENT

    impls = (impl_ij, impl_ji)
    # Loose default: no direction is contradiction, and not both neutral.
    return (CONTRADICTION not in impls) and (impls != (NEUTRAL, NEUTRAL))


def get_semantic_ids(
    nli_matrix: np.ndarray,
    start_idx: int,
    end_idx: int,
    strict_entailment: bool = False,
) -> List[int]:
    """Greedy semantic-id assignment over matrix rows [start_idx, end_idx).

    Reference algorithm: assign next id to the first unlabeled element, then
    forward-scan and propagate that id to every unlabeled element it equivales.

    Returns a list of length (end_idx - start_idx) with ids starting from 0.
    """
    n = end_idx - start_idx
    semantic_ids = [-1] * n
    next_id = 0
    for i in range(n):
        if semantic_ids[i] != -1:
            continue
        semantic_ids[i] = next_id
        m_i = start_idx + i
        for j in range(i + 1, n):
            if semantic_ids[j] != -1:
                continue
            m_j = start_idx + j
            if _are_equivalent(nli_matrix, m_i, m_j, strict_entailment):
                semantic_ids[j] = next_id
        next_id += 1
    assert -1 not in semantic_ids
    return semantic_ids


# ---------------------------------------------------------------------------
# Per-cluster log-likelihood aggregation (port of logsumexp_by_id)
# ---------------------------------------------------------------------------

def logsumexp_by_id(
    semantic_ids: Sequence[int],
    log_likelihoods: Sequence[float],
) -> List[float]:
    """Aggregate sample log-likelihoods into per-cluster log-likelihoods.

    Uses the reference's `sum_normalized` aggregation: normalize each sample
    log-prob by the total log-prob across all samples, then log-sum-exp within
    each cluster.
    """
    unique_ids = sorted(set(semantic_ids))
    assert unique_ids == list(range(len(unique_ids))), f"non-contiguous ids: {unique_ids}"

    lp_arr = np.asarray(log_likelihoods, dtype=np.float64)
    total_log = float(np.log(np.sum(np.exp(lp_arr))))  # log-Z over all samples

    cluster_log_likelihoods: List[float] = []
    for uid in unique_ids:
        idx = [k for k, s in enumerate(semantic_ids) if s == uid]
        normed = lp_arr[idx] - total_log
        cluster_log_likelihoods.append(float(np.log(np.sum(np.exp(normed)))))
    return cluster_log_likelihoods


# ---------------------------------------------------------------------------
# Entropy estimators (port from reference)
# ---------------------------------------------------------------------------

def predictive_entropy_rao(log_probs: Sequence[float]) -> float:
    """Reference's headline estimator: -sum(p * log p) where p = exp(log_p).

    Matches predictive_entropy_rao in the reference. Inputs are *cluster*
    log-likelihoods (already normalized to sum-to-one via logsumexp_by_id).
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    return float(-np.sum(np.exp(lp) * lp))


def cluster_assignment_entropy(semantic_ids: Sequence[int]) -> float:
    """Discrete categorical entropy over empirical cluster-assignment counts."""
    if len(semantic_ids) == 0:
        return 0.0
    counts = np.bincount(semantic_ids)
    probs = counts / counts.sum()
    nz = probs[probs > 0]
    return float(-np.sum(nz * np.log(nz)))


# ---------------------------------------------------------------------------
# Per-record dispatch (used by file-level scorer + tests)
# ---------------------------------------------------------------------------

def compute_se_for_record(
    record: dict,
    nli_matrix: np.ndarray,
    strict_entailment: bool = False,
) -> dict:
    """Compute paper-faithful SE plus back-compat fields for one question.

    Output keys:
        semantic_entropy            — headline (rao over logsumexp_by_id on raw sequence_logprob)
        semantic_entropy_lennorm    — same recipe using length_normalized_logprob
        discrete_se                 — cluster_assignment_entropy (= back-compat key)
        length_normalized_se        — same as semantic_entropy_lennorm, kept for back-compat
        n_clusters, semantic_ids, strict_entailment
    """
    K = record["K"]
    samples = record["samples"]
    seq_lp = [s["sequence_logprob"] for s in samples]
    len_lp = [s["length_normalized_logprob"] for s in samples]

    # Cluster matrix indices 1..K+1 (index 0 is the greedy, excluded from SE).
    semantic_ids = get_semantic_ids(nli_matrix, start_idx=1, end_idx=K + 1,
                                    strict_entailment=strict_entailment)

    cluster_lp_raw = logsumexp_by_id(semantic_ids, seq_lp)
    cluster_lp_len = logsumexp_by_id(semantic_ids, len_lp)

    return {
        "row_idx": record["row_idx"],
        "semantic_entropy": predictive_entropy_rao(cluster_lp_raw),
        "semantic_entropy_lennorm": predictive_entropy_rao(cluster_lp_len),
        "length_normalized_se": predictive_entropy_rao(cluster_lp_len),
        "discrete_se": cluster_assignment_entropy(semantic_ids),
        "n_clusters": len(set(semantic_ids)),
        "semantic_ids": semantic_ids,
        "strict_entailment": strict_entailment,
    }


# ---------------------------------------------------------------------------
# File-level scorer
# ---------------------------------------------------------------------------

def score_files(
    samples_path: str,
    nli_matrix_path: str,
    output_path: str,
    strict_entailment: bool = False,
) -> None:
    """Process selfcheck_samples.jsonl + nli_matrix.jsonl → se_labels.jsonl."""
    done_rows = _load_done_rows(output_path)

    nli_by_row = {}
    with open(nli_matrix_path) as f:
        for line in f:
            rec = json.loads(line)
            nli_by_row[rec["row_idx"]] = np.array(rec["nli_matrix"], dtype=np.float32)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(samples_path) as fin, open(output_path, "a", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            row_idx = rec["row_idx"]
            if row_idx in done_rows:
                continue
            if row_idx not in nli_by_row:
                continue
            result = compute_se_for_record(rec, nli_by_row[row_idx], strict_entailment)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1

    print(f"SE done: {written} rows written → {output_path}")


def _load_done_rows(path: str) -> set:
    done = set()
    p = Path(path)
    if not p.exists():
        return done
    with open(p) as f:
        for line in f:
            try:
                done.add(json.loads(line)["row_idx"])
            except Exception:
                pass
    return done
