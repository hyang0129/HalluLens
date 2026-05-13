"""SelfCheckGPT variants: NLI, BERTScore, n-gram."""
import json
import math
from collections import Counter
from pathlib import Path
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# NLI variant (uses precomputed matrix — no GPU)
# ---------------------------------------------------------------------------

def selfcheck_nli_score(nli_matrix: np.ndarray, K: int) -> float:
    """Mean P(contradict) with sample_i as premise, greedy as hypothesis.

    nli_matrix[i, j, 0] = P(contradict) for premise=texts[i], hypothesis=texts[j].
    Greedy is at index 0. Samples are at indices 1..K.
    High score = predicted hallucination.
    """
    scores = [nli_matrix[i, 0, 0] for i in range(1, K + 1) if not np.isnan(nli_matrix[i, 0, 0])]
    return float(np.mean(scores)) if scores else float("nan")


# ---------------------------------------------------------------------------
# BERTScore variant (CPU, uses bert-score library)
# ---------------------------------------------------------------------------

def selfcheck_bertscore(greedy: str, sample_texts: List[str]) -> float:
    """1 - mean BERTScore F1(greedy, sample_i) over K samples.

    Requires: pip install bert-score==0.3.13
    High score = predicted hallucination.
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        raise ImportError("Install bert-score==0.3.13 for BERTScore SelfCheckGPT.")

    K = len(sample_texts)
    if K == 0:
        return float("nan")

    _, _, F1 = bert_score_fn(
        cands=[greedy] * K,
        refs=sample_texts,
        lang="en",
        model_type="roberta-large",
        verbose=False,
    )
    return float(1.0 - F1.mean().item())


# ---------------------------------------------------------------------------
# n-gram variant (unigram LM, CPU)
# ---------------------------------------------------------------------------

def selfcheck_ngram(greedy: str, sample_texts: List[str]) -> float:
    """Unigram LM trained on K samples; score = -log P_unigram(greedy) (length-normalized).

    Laplace smoothing. High score = predicted hallucination.
    """
    if not sample_texts:
        return float("nan")

    # Build unigram LM from all sample tokens
    tokens = [t for s in sample_texts for t in s.lower().split()]
    counter = Counter(tokens)
    vocab_size = len(counter)
    total = sum(counter.values())

    greedy_tokens = greedy.lower().split()
    if not greedy_tokens:
        return float("nan")

    log_prob = sum(
        math.log((counter.get(t, 0) + 1) / (total + vocab_size))
        for t in greedy_tokens
    ) / len(greedy_tokens)

    return -log_prob


# ---------------------------------------------------------------------------
# File-level scorer
# ---------------------------------------------------------------------------

def score_files(
    samples_path: str,
    nli_matrix_path: str,
    output_path: str,
    run_bertscore: bool = True,
    run_ngram: bool = True,
    bertscore_batch_size: int = 64,
) -> None:
    """Process selfcheck_samples.jsonl + nli_matrix.jsonl → selfcheck_scores.jsonl.

    Args:
        run_bertscore: Compute BERTScore variant (slow, CPU).
        run_ngram: Compute n-gram variant (fast, CPU).
    """
    done_rows = _load_done_rows(output_path)

    # Index NLI matrices
    nli_by_row = {}
    with open(nli_matrix_path) as f:
        for line in f:
            rec = json.loads(line)
            nli_by_row[rec["row_idx"]] = np.array(rec["nli_matrix"], dtype=np.float32)

    # Collect pending records
    pending = []
    with open(samples_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["row_idx"] not in done_rows and rec["row_idx"] in nli_by_row:
                pending.append(rec)

    if not pending:
        print("SelfCheck: all rows already done — skipping.")
        return

    print(f"SelfCheck scoring {len(pending)} questions...")

    # BERTScore: batch all greedy+sample pairs at once for efficiency
    bertscore_results: Optional[dict] = None
    if run_bertscore:
        bertscore_results = _compute_bertscore_batched(pending, bertscore_batch_size)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as fout:
        for rec in pending:
            row_idx = rec["row_idx"]
            K = rec["K"]
            greedy = rec["greedy_answer"]
            sample_texts = [s["text"] for s in rec["samples"]]
            matrix = nli_by_row[row_idx]

            result = {
                "row_idx": row_idx,
                "nli": selfcheck_nli_score(matrix, K),
            }

            if run_bertscore and bertscore_results is not None:
                result["bertscore"] = bertscore_results.get(row_idx, float("nan"))

            if run_ngram:
                result["ngram"] = selfcheck_ngram(greedy, sample_texts)

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"SelfCheck done → {output_path}")


def _compute_bertscore_batched(records: list, batch_size: int) -> dict:
    """BERTScore all (greedy, sample_i) pairs at once. Returns {row_idx: score}."""
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        raise ImportError("Install bert-score==0.3.13 for BERTScore SelfCheckGPT.")

    all_cands = []
    all_refs = []
    row_idx_per_pair = []

    for rec in records:
        greedy = rec["greedy_answer"]
        for s in rec["samples"]:
            all_cands.append(greedy)
            all_refs.append(s["text"])
            row_idx_per_pair.append(rec["row_idx"])

    if not all_cands:
        return {}

    print(f"  BERTScore: {len(all_cands)} pairs...")
    _, _, F1 = bert_score_fn(
        cands=all_cands,
        refs=all_refs,
        lang="en",
        model_type="roberta-large",
        batch_size=batch_size,
        verbose=False,
    )
    f1_list = F1.tolist()

    # Average F1 per row_idx
    from collections import defaultdict
    sums: dict = defaultdict(float)
    counts: dict = defaultdict(int)
    for row_idx, f1 in zip(row_idx_per_pair, f1_list):
        sums[row_idx] += f1
        counts[row_idx] += 1

    return {r: 1.0 - sums[r] / counts[r] for r in sums}


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
