"""P(true) self-evaluation detector (Kadavath et al. 2022).

One forward pass per example: prefix the question + greedy answer into the
Kadavath prompt, read logits at the first generated token for tokens " A" and
" B", compute softmax probability. No chat template (raw string), matching the
original paper and LLMsKnow exactly.

Score direction: p_true is a *correctness* probability.  For AUROC vs
halu_label (1=hallucinated), use ``1 - p_true`` as the anomaly score
(high p_true ⇒ model claims correct ⇒ low hallucination score).

Output per row (ptrue.jsonl):
    {"row_idx": int, "p_true": float, "p_true_reversed": float,
     "p_a": float, "p_b": float, "halu_label": 0|1|null}

``p_true_reversed`` is the P(true) score from the variant where (A)=False,
(B)=True, providing a token-position-bias sanity check.
"""
import json
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Possible answer: {answer}\n"
    "Is the possible answer:\n"
    "(A) True\n"
    "(B) False\n"
    "The possible answer is:"
)

_PROMPT_TEMPLATE_REVERSED = (
    "Question: {question}\n"
    "Possible answer: {answer}\n"
    "Is the possible answer:\n"
    "(A) False\n"
    "(B) True\n"
    "The possible answer is:"
)


def _build_prompt(question: str, answer: str, reversed_: bool = False) -> str:
    tmpl = _PROMPT_TEMPLATE_REVERSED if reversed_ else _PROMPT_TEMPLATE
    return tmpl.format(question=question.strip(), answer=answer.strip())


def _extract_question(row: dict) -> str:
    if "question" in row:
        return row["question"]
    # Fall back: prompt usually starts with "Question: ..."
    prompt = row.get("prompt", "")
    for line in prompt.splitlines():
        line = line.strip()
        if line.lower().startswith("question:"):
            return line[len("question:"):].strip()
    return prompt


class PTrueScorer:
    """Batched P(true) scorer — loads the model once, runs forward + reversed pass.

    Tokenization invariant: " A" and " B" must each be a single token on the
    model's tokenizer. Verified in __init__; fails fast with a clear message if not.
    """

    def __init__(self, model_name: str, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._tok_a: Optional[int] = None
        self._tok_b: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        generation_jsonl: str,
        output_path: str,
        row_indices: Optional[List[int]] = None,
        labels: Optional[List[int]] = None,
    ) -> None:
        """Score every row in generation_jsonl with P(true).

        Args:
            generation_jsonl: Path to existing generation.jsonl.
            output_path: Destination ptrue.jsonl.
            row_indices: 0-based row indices to process (None = all rows).
            labels: Full halu_test_res list so labels[row_idx] gives the
                    label.  Pass None to omit halu_label from output.
        """
        self._ensure_loaded()

        import pandas as pd
        gendf = pd.read_json(generation_jsonl, lines=True)
        gendf["row_idx"] = gendf.index

        if row_indices is not None:
            gendf = gendf[gendf["row_idx"].isin(set(row_indices))].reset_index(drop=True)

        done_rows = _load_done_rows(output_path)
        remaining = gendf[~gendf["row_idx"].isin(done_rows)].reset_index(drop=True)

        if len(remaining) == 0:
            print(f"All {len(gendf)} rows already done — skipping.")
            return

        print(
            f"P(true) scoring {len(remaining)} rows "
            f"({len(done_rows)} already done) | batch_size={self.batch_size}"
        )

        row_idxs = remaining["row_idx"].tolist()
        questions = [_extract_question(r) for r in remaining.to_dict("records")]
        answers = remaining["generation"].tolist()

        fwd_prompts = [_build_prompt(q, a, reversed_=False) for q, a in zip(questions, answers)]
        rev_prompts = [_build_prompt(q, a, reversed_=True) for q, a in zip(questions, answers)]

        print("  Forward pass...")
        fwd_scores = self._score_batches(fwd_prompts)
        print("  Reversed pass...")
        rev_scores = self._score_batches(rev_prompts)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "a", encoding="utf-8") as f:
            for i, row_idx in enumerate(row_idxs):
                p_a_fwd, p_b_fwd = fwd_scores[i]
                p_a_rev, p_b_rev = rev_scores[i]
                # forward: (A)=True → p_true = p_a_fwd
                # reversed: (B)=True → p_true_reversed = p_b_rev
                halu_label = labels[row_idx] if labels is not None and row_idx < len(labels) else None
                record = {
                    "row_idx": row_idx,
                    "p_true": p_a_fwd,
                    "p_true_reversed": p_b_rev,
                    "p_a": p_a_fwd,
                    "p_b": p_b_fwd,
                    "halu_label": halu_label,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Written {len(row_idxs)} records → {output_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from activation_logging.server import get_model_and_tokenizer

        print(f"Loading model: {self.model_name}")
        self._model, self._tokenizer = get_model_and_tokenizer(self.model_name, None)

        tok = self._tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        # Verify " A" and " B" are single tokens (fail fast)
        for token_str in [" A", " B"]:
            ids = tok.encode(token_str, add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(
                    f'Token "{token_str}" encodes to {len(ids)} tokens on '
                    f"{self.model_name}: {ids}. "
                    "P(true) requires single-token decode for ' A' and ' B'."
                )
        self._tok_a = tok.encode(" A", add_special_tokens=False)[0]
        self._tok_b = tok.encode(" B", add_special_tokens=False)[0]
        print(f'Tokenization check passed: " A"={self._tok_a}, " B"={self._tok_b}')

    def _score_batches(self, prompts: List[str]) -> List[tuple]:
        """Return list of (p_a, p_b) tuples, one per prompt."""
        results = []
        for b_start in tqdm(range(0, len(prompts), self.batch_size), leave=False):
            batch = prompts[b_start : b_start + self.batch_size]
            results.extend(self._score_batch(batch))
        return results

    def _score_batch(self, prompts: List[str]) -> List[tuple]:
        model, tokenizer = self._model, self._tokenizer
        device = next(model.parameters()).device

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # scores[0]: logits at the first generated token, shape (batch, vocab)
        logits = outputs.scores[0]
        ab_logits = logits[:, [self._tok_a, self._tok_b]]
        probs = F.softmax(ab_logits.float(), dim=-1).cpu()

        del outputs
        return [(float(probs[i, 0]), float(probs[i, 1])) for i in range(len(prompts))]


# ------------------------------------------------------------------
# Module-level helpers (shared with scripts)
# ------------------------------------------------------------------

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
