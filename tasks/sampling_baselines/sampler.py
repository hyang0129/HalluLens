"""K-sample stochastic generation pass (text + logprobs only, no activation logging)."""
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from tqdm import tqdm


class SamplingPass:
    """Generate K stochastic samples per question at temperature T.

    Writes selfcheck_samples.jsonl incrementally with resume support.
    Does not capture hidden-state activations — logprobs only.
    """

    def __init__(
        self,
        model_name: str,
        K: int = 10,
        temperature: float = 1.0,
        max_tokens: int = 64,
        seed: int = 42,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.K = K
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        generation_jsonl: str,
        output_path: str,
        row_indices: Optional[List[int]] = None,
    ) -> None:
        """Generate K stochastic samples for each question.

        Args:
            generation_jsonl: Path to existing generation.jsonl.
            output_path: Destination selfcheck_samples.jsonl.
            row_indices: 0-based row indices to process (None = all rows).
        """
        self._ensure_loaded()

        gendf = pd.read_json(generation_jsonl, lines=True)
        gendf["row_idx"] = gendf.index

        if row_indices is not None:
            gendf = gendf[gendf["row_idx"].isin(set(row_indices))].reset_index(drop=True)

        done_rows = self._load_done_rows(output_path)
        remaining = gendf[~gendf["row_idx"].isin(done_rows)].reset_index(drop=True)

        if len(remaining) == 0:
            print(f"All {len(gendf)} rows already done — skipping.")
            return

        print(
            f"Sampling {len(remaining)} questions "
            f"({len(done_rows)} already done) | K={self.K} | T={self.temperature}"
        )

        prompts = remaining["prompt"].tolist()
        row_idxs = remaining["row_idx"].tolist()
        greedy_answers = (
            remaining["generation"].tolist()
            if "generation" in remaining.columns
            else [""] * len(remaining)
        )
        question_ids = (
            remaining["id"].tolist()
            if "id" in remaining.columns
            else [str(r) for r in row_idxs]
        )

        n = len(prompts)
        all_samples: List[List[dict]] = [[] for _ in range(n)]

        for k in range(self.K):
            print(f"  Sample slot {k + 1}/{self.K}")
            for b_start in tqdm(range(0, n, self.batch_size), desc=f"k={k}", leave=False):
                b_end = min(b_start + self.batch_size, n)
                batch_results = self._infer_batch(prompts[b_start:b_end], sample_idx=k)
                for i, res in enumerate(batch_results):
                    all_samples[b_start + i].append(res)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as f:
            for i in range(n):
                record = {
                    "row_idx": row_idxs[i],
                    "question_id": question_ids[i],
                    "prompt": prompts[i],
                    "greedy_answer": greedy_answers[i],
                    "samples": all_samples[i],
                    "K": self.K,
                    "temperature": self.temperature,
                    "sampling_seed": self.seed,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Written {n} records → {output_path}")

    # ------------------------------------------------------------------
    # Smoke test
    # ------------------------------------------------------------------

    def validate_alignment(self, generation_jsonl: str, output_path: str, n: int = 50) -> None:
        """Assert greedy_answer in samples matches generation.jsonl. Raises on mismatch."""
        gendf = pd.read_json(generation_jsonl, lines=True)
        gendf["row_idx"] = gendf.index

        samples_df = pd.read_json(output_path, lines=True).head(n)
        mismatches = 0
        for _, row in samples_df.iterrows():
            idx = row["row_idx"]
            expected = gendf.iloc[idx]["generation"]
            if row["greedy_answer"] != expected:
                mismatches += 1
                print(f"  MISMATCH at row_idx={idx}")
        if mismatches:
            raise ValueError(f"Alignment check failed: {mismatches}/{n} rows mismatched.")
        print(f"Alignment check passed ({n} rows).")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from activation_logging.server import get_model_and_tokenizer
        self._model, self._tokenizer = get_model_and_tokenizer(self.model_name, None)

    def _infer_batch(self, prompts: List[str], sample_idx: int) -> List[dict]:
        """One stochastic sample for a batch of prompts."""
        model, tokenizer = self._model, self._tokenizer
        device = next(model.parameters()).device

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        torch.manual_seed(self.seed + sample_idx)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                output_hidden_states=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_start = inputs.input_ids.shape[1]
        results = []
        for i in range(len(prompts)):
            gen_ids = outputs.sequences[i, gen_start:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            token_logprobs = []
            for t, score in enumerate(outputs.scores):
                if t >= len(gen_ids):
                    break
                lp = torch.log_softmax(score[i], dim=-1)
                token_logprobs.append(lp[gen_ids[t]].item())

            seq_lp = sum(token_logprobs) if token_logprobs else 0.0
            len_norm_lp = seq_lp / max(len(token_logprobs), 1)

            results.append({
                "text": text,
                "sequence_logprob": seq_lp,
                "length_normalized_logprob": len_norm_lp,
            })

        del outputs
        return results

    @staticmethod
    def _load_done_rows(output_path: str) -> set:
        done = set()
        p = Path(output_path)
        if not p.exists():
            return done
        with open(p) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["row_idx"])
                except Exception:
                    pass
        return done
