"""Batched NLI matrix computation matching jlko/semantic_uncertainty reference."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Matches the reference impl (jlko/semantic_uncertainty EntailmentDeberta).
NLI_MODEL_ID = "microsoft/deberta-v2-xlarge-mnli"


class NLIScorer:
    """Compute (K+1)x(K+1) directed NLI matrices over {greedy}∪{K samples} per question.

    Label order for deberta-v2-xlarge-mnli:
        0 = contradiction, 1 = neutral, 2 = entailment
    Verified dynamically via model.config.id2label at load time.
    """

    def __init__(
        self,
        model_id: str = NLI_MODEL_ID,
        batch_size: int = 512,
        device: Optional[str] = None,
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        self._device = device
        self._model = None
        self._tokenizer = None
        self._label2idx: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_matrix(self, texts: List[str]) -> np.ndarray:
        """Compute (N x N x 3) directed NLI matrix for a list of texts.

        texts[0] = greedy answer, texts[1:] = K stochastic samples.
        Returns array of shape (N, N, 3) where axis-2 = [p_contradict, p_neutral, p_entail].
        Diagonal is undefined (set to NaN).
        """
        self._ensure_loaded()
        N = len(texts)

        # Build all (premise, hypothesis) pairs in row-major order, skip diagonal
        pairs: List[Tuple[str, str]] = []
        pair_indices: List[Tuple[int, int]] = []
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                pairs.append((texts[i], texts[j]))
                pair_indices.append((i, j))

        probs = self._run_nli_batched(pairs)  # (M, 3)

        matrix = np.full((N, N, 3), np.nan, dtype=np.float32)
        for k, (i, j) in enumerate(pair_indices):
            matrix[i, j] = probs[k]

        return matrix

    def score_file(
        self,
        samples_path: str,
        output_path: str,
        done_rows: Optional[set] = None,
    ) -> None:
        """Process selfcheck_samples.jsonl → nli_matrix.jsonl."""
        if done_rows is None:
            done_rows = _load_done_rows(output_path)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(samples_path) as fin, open(output_path, "a", encoding="utf-8") as fout:
            lines = [l for l in fin if l.strip()]

        pending = []
        for line in lines:
            rec = json.loads(line)
            if rec["row_idx"] in done_rows:
                continue
            pending.append(rec)

        if not pending:
            print(f"NLI: all rows already done — skipping.")
            return

        print(f"NLI scoring {len(pending)} questions...")
        with open(output_path, "a", encoding="utf-8") as fout:
            for rec in tqdm(pending):
                greedy = rec["greedy_answer"]
                sample_texts = [s["text"] for s in rec["samples"]]
                all_texts = [greedy] + sample_texts

                matrix = self.compute_matrix(all_texts)

                out = {
                    "row_idx": rec["row_idx"],
                    "nli_matrix": matrix.tolist(),
                    "texts": all_texts,
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")

        print(f"NLI done → {output_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id).to(device)
        self._model.eval()

        # Resolve label order dynamically
        id2label = {int(k): v.lower() for k, v in self._model.config.id2label.items()}
        for idx, name in id2label.items():
            if "contradict" in name:
                self._label2idx["contradiction"] = idx
            elif "neutral" in name:
                self._label2idx["neutral"] = idx
            elif "entail" in name:
                self._label2idx["entailment"] = idx
        assert len(self._label2idx) == 3, f"Unexpected NLI labels: {id2label}"

    def _run_nli_batched(self, pairs: List[Tuple[str, str]]) -> np.ndarray:
        """Run NLI on all (premise, hypothesis) pairs in batches. Returns (M, 3) softmax probs."""
        self._ensure_loaded()
        device = next(self._model.parameters()).device
        all_probs = []

        for b_start in range(0, len(pairs), self.batch_size):
            batch = pairs[b_start : b_start + self.batch_size]
            premises = [p for p, _ in batch]
            hypotheses = [h for _, h in batch]

            enc = self._tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                logits = self._model(**enc).logits  # (B, 3)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()  # (B, 3)

            # Reorder to [contradiction, neutral, entailment]
            ordered = np.stack(
                [
                    probs[:, self._label2idx["contradiction"]],
                    probs[:, self._label2idx["neutral"]],
                    probs[:, self._label2idx["entailment"]],
                ],
                axis=1,
            )
            all_probs.append(ordered)

        return np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 3), dtype=np.float32)


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
