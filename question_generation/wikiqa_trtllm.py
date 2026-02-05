"""Generate WikiQA-style question/answer pairs using TensorRT-LLM.

This is a drop-in replacement for the *question generation* stage used by:
- `tasks/shortform/precise_wikiqa.py` (PreciseWikiQA)
- `tasks/longwiki/longwiki_main.py` (LongWiki)

We intentionally avoid modifying the original Meta-authored task code by
producing the same JSONL outputs those tasks expect.

Usage (example)
--------------
python -m question_generation.wikiqa_trtllm \
  --task precise \
  --wiki_input_path data/wiki_data/doc_goodwiki_h_score.jsonl \
  --output_path data/precise_qa/save/qa_goodwiki_Llama-3.1-8B-Instruct_dynamic.jsonl \
  --N 100 \
  --q_model nvidia/Llama-3.3-70B-Instruct-FP8 \
  --max_workers 4

Notes
-----
TensorRT-LLM is typically Linux-only. On unsupported machines this script will
exit with a clear error.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import jsonlines
import pandas as pd
from loguru import logger

from question_generation.trtllm_client import TRTLLMClient, TRTLLMGenerationConfig, TRTLLMUnavailableError


PRECISE_Q_GENERATION_PROMPT = (
    'I would like you to act as a question generator. I will provide reference and you will generate a factual '
    'knowledge based question about "{wiki_title}" based on the reference. The specific requirements are as follows:\n\n'
    '1. The question can be fully answered based only on the reference material.\n'
    '2. The question should be objective and not open-ended.\n'
    '3. The question should be concise.\n'
    '4. The question should not require additional information to answer.\n'
    "5. the question's answer should be a word or a phrase.\n"
    '6. the question should have only one answer.\n\n'
    'Reference:\n{wiki_document}\n\n'
    'Please reply with the question only without any explanation or additional information:\n'
)

PRECISE_ANSWERABILITY_PROMPT = (
    'I would like you to judge question\'s answerability and answer the question.\n'
    'I will provide a question and reference document, and you will judge whether the question is fully answerable '
    'based only on the reference document, i.e., whether the answer is included in the reference.\n'
    'If yes, please reply with the answer only without any explanation or additional information.\n'
    'If no, please reply with "unanswerable" only.\n\n'
    'Reference document: {ref_document}\n\n'
    'Question: {question}'
)

LONGFORM_Q_GENERATION_PROMPT = (
    'I would like you to act as an essay question generator. I will provide a reference and you will generate a '
    'factual knowledge based question about "{wiki_title}" based on the reference. The specific requirements are as follows:\n'
    '1. The question can be fully answered based only on the reference.\n'
    '2. The question should be objective and not open-ended.\n'
    '3. The question should be concise.\n'
    '4. The question\'s answer should be longer than three sentences.\n'
    '5. The question should provide enough context to be answered without ambiguity.\n\n'
    'Reference:\n{wiki_document}\n\n'
    'Please reply with the question only without any explanation or additional information.\n'
    'Remember requirements. Ask only one question. Keep it concise.\n'
    'If you cannot generate an essay question, please reply with "[NO QUESTION]".\n'
    'Question: \n'
)

LONGFORM_ANSWERABILITY_PROMPT = (
    'I would like you to judge question\'s answerability based on the reference document.\n'
    'I will provide a question and reference document, and you will judge whether the question is fully answerable '
    'based only on the reference document, i.e., whether the answer is included in the reference.\n'
    'If yes, please reply with the answer only without any explanation or additional information.\n'
    'If no, please reply with "unanswerable" only.\n\n'
    'Reference document: {ref_document}\n\n'
    'Question: {question}'
)


@dataclass
class WikiQATaskSpec:
    name: str
    q_prompt: str
    a_prompt: str
    min_answer_sentences: Optional[int] = None
    max_answer_words: Optional[int] = None


TASK_SPECS: Dict[str, WikiQATaskSpec] = {
    "precise": WikiQATaskSpec(
        name="precise",
        q_prompt=PRECISE_Q_GENERATION_PROMPT,
        a_prompt=PRECISE_ANSWERABILITY_PROMPT,
        max_answer_words=10,
    ),
    "longform": WikiQATaskSpec(
        name="longform",
        q_prompt=LONGFORM_Q_GENERATION_PROMPT,
        a_prompt=LONGFORM_ANSWERABILITY_PROMPT,
        min_answer_sentences=4,
    ),
}


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _count_existing_jsonl(path: str) -> int:
    if not os.path.exists(path):
        return 0
    count = 0
    with jsonlines.open(path, "r") as reader:
        for _ in reader:
            count += 1
    return count


def _simple_sentence_count(text: str) -> int:
    # Heuristic: count '.', '?', '!' occurrences.
    if not text:
        return 0
    return sum(text.count(ch) for ch in [".", "?", "!"])


def _is_unanswerable(answer: str) -> bool:
    a = (answer or "").strip().lower()
    return (a == "unanswerable") or ("unanswerable" in a) or a.startswith("unfortunately")


def _answer_passes_filters(answer: str, spec: WikiQATaskSpec) -> bool:
    if _is_unanswerable(answer):
        return False

    if spec.min_answer_sentences is not None:
        if _simple_sentence_count(answer) < spec.min_answer_sentences:
            return False

    if spec.max_answer_words is not None:
        if len(answer.split()) > spec.max_answer_words:
            return False

    return True


def _pick_reference_section(document: str, min_chars: int, max_chars: int) -> str:
    """Pick a reference slice of the document.

    We intentionally keep this lightweight (character-based) so question generation
    doesn't require additional heavy tokenizers.
    """
    doc = (document or "").strip()
    if not doc:
        return ""

    # Break into rough paragraphs.
    paras = [p.strip() for p in doc.split("\n") if p.strip()]
    if not paras:
        return doc[:max_chars]

    random.shuffle(paras)
    for p in paras:
        if min_chars <= len(p) <= max_chars:
            return p
    # Fallback: join a few paragraphs.
    joined = "\n".join(paras[:3])
    return joined[:max_chars]


def generate_wikiqa_jsonl(
    *,
    task: str,
    wiki_input_path: str,
    output_path: str,
    n_total: int,
    q_model: str,
    low_level: int,
    high_level: int,
    min_ref_chars: int,
    max_ref_chars: int,
    chunk_size: int,
    seed: int,
) -> None:
    """Generate QA JSONL file in the format expected by HalluLens tasks."""

    if task not in TASK_SPECS:
        raise ValueError(f"Unsupported task: {task}")
    spec = TASK_SPECS[task]

    random.seed(seed)

    _ensure_parent_dir(output_path)
    already = _count_existing_jsonl(output_path)
    if already >= n_total:
        logger.info(f"QA file already complete: {output_path} ({already} >= {n_total})")
        return

    logger.info(f"Loading wiki input: {wiki_input_path}")
    wiki_df = pd.read_json(wiki_input_path, orient="records", lines=True)

    per_level = max(1, n_total // max(1, (high_level - low_level)))
    target_bins = list(range(low_level, high_level))

    # Load model once.
    try:
        client = TRTLLMClient.get(q_model)
    except TRTLLMUnavailableError:
        raise

    q_cfg = TRTLLMGenerationConfig(temperature=0.7, top_p=0.9, max_tokens=96 if task == "precise" else 160)
    a_cfg = TRTLLMGenerationConfig(temperature=0.3, top_p=0.95, max_tokens=96)

    written = already
    filter_count = 0

    logger.info(
        f"Generating QA pairs: task={task} q_model={q_model} output={output_path} "
        f"already={already} target={n_total}"
    )

    with jsonlines.open(output_path, "a") as writer:
        for bin_level in target_bins:
            if written >= n_total:
                break

            level_df = wiki_df[wiki_df.get("h_score_cat") == bin_level]
            if level_df.empty:
                logger.warning(f"No rows for h_score_cat={bin_level}; skipping")
                continue

            # Over-sample a little to account for filtering.
            sample_n = min(len(level_df), per_level + 100)
            sampled = level_df.sample(n=sample_n, replace=(len(level_df) < sample_n), random_state=seed)
            rows = sampled.to_dict(orient="records")
            random.shuffle(rows)

            # Process in chunks for frequent flushing to disk.
            for chunk_start in range(0, len(rows), max(1, chunk_size)):
                if written >= n_total:
                    break

                chunk = rows[chunk_start : chunk_start + chunk_size]
                prompt_objs: List[Dict] = []
                q_prompts: List[str] = []

                for row in chunk:
                    title = row.get("title", "")
                    document = row.get("document", "")
                    reference = _pick_reference_section(document, min_chars=min_ref_chars, max_chars=max_ref_chars)
                    if not reference:
                        continue

                    meta = {
                        "title": title,
                        "h_score_cat": row.get("h_score_cat"),
                        "pageid": row.get("pageid"),
                        "revid": row.get("revid"),
                        "description": row.get("description"),
                        "categories": row.get("categories"),
                        "reference": reference,
                    }

                    q_prompt = spec.q_prompt.format(wiki_title=title, wiki_document=reference.strip())
                    prompt_objs.append(meta)
                    q_prompts.append(q_prompt)

                if not q_prompts:
                    continue

                questions = client.generate_many(q_prompts, config=q_cfg)

                # Answerability check.
                a_prompts = [spec.a_prompt.format(ref_document=o["reference"], question=q.strip()) for o, q in zip(prompt_objs, questions)]
                answers = client.generate_many(a_prompts, config=a_cfg)

                for obj, q, a in zip(prompt_objs, questions, answers):
                    if written >= n_total:
                        break

                    q_clean = (q or "").strip()
                    a_clean = (a or "").strip()

                    if not _answer_passes_filters(a_clean, spec):
                        filter_count += 1
                        continue

                    out = dict(obj)
                    out["prompt"] = q_clean
                    out["answer"] = a_clean

                    writer.write(out)
                    written += 1

                logger.info(f"Progress: written={written}/{n_total} filtered={filter_count}")

    logger.success(f"Done: wrote {written} QA pairs to {output_path} (filtered {filter_count})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate WikiQA prompts using TensorRT-LLM")
    parser.add_argument("--task", choices=["precise", "longform"], required=True)
    parser.add_argument("--wiki_input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--q_model", default="nvidia/Llama-3.3-70B-Instruct-FP8")

    parser.add_argument("--low_level", type=int, default=8)
    parser.add_argument("--high_level", type=int, default=10)

    parser.add_argument("--min_ref_chars", type=int, default=200)
    parser.add_argument("--max_ref_chars", type=int, default=500)
    parser.add_argument("--chunk_size", type=int, default=int(os.environ.get("QA_GENERATION_CHUNK_SIZE", "5")))
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    try:
        generate_wikiqa_jsonl(
            task=args.task,
            wiki_input_path=args.wiki_input_path,
            output_path=args.output_path,
            n_total=args.N,
            q_model=args.q_model,
            low_level=args.low_level,
            high_level=args.high_level,
            min_ref_chars=args.min_ref_chars,
            max_ref_chars=args.max_ref_chars,
            chunk_size=args.chunk_size,
            seed=args.seed,
        )
    except TRTLLMUnavailableError as exc:
        logger.error(str(exc))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
