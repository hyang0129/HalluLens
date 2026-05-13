"""Semantic Entropy computation from precomputed NLI matrices."""
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def cluster_answers(
    nli_matrix: np.ndarray,
    start_idx: int = 1,
    end_idx: Optional[int] = None,
    threshold: float = 0.5,
) -> List[List[int]]:
    """Greedy bidirectional entailment clustering.

    Operates on matrix indices [start_idx, end_idx).
    Index 0 = greedy answer (excluded from clustering by default).
    nli_matrix[i, j] = [p_contradict, p_neutral, p_entail] (i=premise, j=hypothesis).
    Two texts cluster together iff both directions have p_entail > threshold.
    """
    if end_idx is None:
        end_idx = nli_matrix.shape[0]

    clusters: List[List[int]] = []
    for i in range(start_idx, end_idx):
        placed = False
        for cluster in clusters:
            rep = cluster[0]
            p_entail_ri = nli_matrix[rep, i, 2]
            p_entail_ir = nli_matrix[i, rep, 2]
            if p_entail_ri > threshold and p_entail_ir > threshold:
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])

    return clusters


def discrete_se(clusters: List[List[int]], K: int) -> float:
    """Discrete semantic entropy: -sum(|c|/K * log(|c|/K)) over clusters."""
    if K == 0:
        return 0.0
    entropy = 0.0
    for cluster in clusters:
        p = len(cluster) / K
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def length_normalized_se(
    clusters: List[List[int]],
    samples: List[dict],
    start_idx: int = 1,
) -> float:
    """Length-normalized semantic entropy (Farquhar et al. headline variant).

    Uses length_normalized_logprob from the generating model.
    Log-sum-exp within each cluster, normalize, then entropy.
    samples is 0-indexed relative to the sample list (not matrix index).
    """
    if not clusters:
        return 0.0

    cluster_lp = []
    for cluster in clusters:
        # cluster contains matrix indices; samples are at matrix_idx - start_idx
        sample_lps = [samples[idx - start_idx]["length_normalized_logprob"] for idx in cluster]
        cluster_lp.append(_logsumexp(sample_lps))

    total = _logsumexp(cluster_lp)
    probs = [math.exp(lp - total) for lp in cluster_lp]

    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def compute_se_for_record(record: dict, nli_matrix: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute both SE variants for one question.

    Args:
        record: one row from selfcheck_samples.jsonl
        nli_matrix: (K+1, K+1, 3) array from nli_matrix.jsonl
        threshold: bidirectional entailment threshold

    Returns dict with discrete_se, length_normalized_se, n_clusters, cluster_assignments.
    """
    K = record["K"]
    samples = record["samples"]

    clusters = cluster_answers(nli_matrix, start_idx=1, end_idx=K + 1, threshold=threshold)
    d_se = discrete_se(clusters, K)
    ln_se = length_normalized_se(clusters, samples, start_idx=1)

    return {
        "row_idx": record["row_idx"],
        "discrete_se": d_se,
        "length_normalized_se": ln_se,
        "n_clusters": len(clusters),
        "cluster_assignments": clusters,
    }


def score_files(
    samples_path: str,
    nli_matrix_path: str,
    output_path: str,
    threshold: float = 0.5,
) -> None:
    """Process selfcheck_samples.jsonl + nli_matrix.jsonl → se_labels.jsonl."""
    done_rows = _load_done_rows(output_path)

    # Load NLI matrices indexed by row_idx
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
            result = compute_se_for_record(rec, nli_by_row[row_idx], threshold)
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1

    print(f"SE done: {written} rows written → {output_path}")


def _logsumexp(values: List[float]) -> float:
    if not values:
        return float("-inf")
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


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
