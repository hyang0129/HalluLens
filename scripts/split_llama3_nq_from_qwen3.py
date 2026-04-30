#!/usr/bin/env python3
"""
Split the legacy Llama-3.1-8B-Instruct Natural Questions dump into canonical
train/test splits that mirror the Qwen3 NQ layout.

Source (non-canonical, single 20,772-sample blob):
    shared/natural_questions_logprob/activations.zarr/
    shared/natural_questions_logprob/natural_questions/Llama-3.1-8B-Instruct/
        generation.jsonl
        generation.sanitized_for_eval.jsonl
        eval_results.json
        eval_results_for_training.json

Targets (canonical):
    shared/natural_questions/activations.zarr/                       (test  ~4155)
    shared/natural_questions_train/activations.zarr/                 (train ~16617)
    output/natural_questions/Llama-3.1-8B-Instruct/{...eval files}
    output/natural_questions_train/Llama-3.1-8B-Instruct/{...eval files}
    configs/datasets/nq.json

Split assignment:
    Question text from each Llama3 row is matched against the question text in
    Qwen3's existing test/train generation files. The two prompt sets are
    identical (verified: 100% overlap, 0 leakage), so this produces splits
    that are directly comparable across models.

Run on the GPU node (zarr installed). Idempotent: skip existing targets unless
--force is passed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import zarr

REPO = Path(__file__).resolve().parents[1]

SRC_ZARR = REPO / "shared/natural_questions_logprob/activations.zarr"
SRC_OUT = REPO / "shared/natural_questions_logprob/natural_questions/Llama-3.1-8B-Instruct"

QWEN_TEST_GEN = REPO / "output/natural_questions/Qwen3-8B/generation.jsonl"
QWEN_TRAIN_GEN = REPO / "output/natural_questions_train/Qwen3-8B/generation.jsonl"

DST_TEST_ZARR = REPO / "shared/natural_questions/activations.zarr"
DST_TRAIN_ZARR = REPO / "shared/natural_questions_train/activations.zarr"
DST_TEST_OUT = REPO / "output/natural_questions/Llama-3.1-8B-Instruct"
DST_TRAIN_OUT = REPO / "output/natural_questions_train/Llama-3.1-8B-Instruct"

CONFIG_PATH = REPO / "configs/datasets/nq.json"

MODEL_NAME = "Llama-3.1-8B-Instruct"


def load_questions(path: Path) -> set:
    qs = set()
    with path.open() as f:
        for line in f:
            qs.add(json.loads(line)["question"])
    return qs


def assign_splits(src_gen: Path, qwen_test_qs: set, qwen_train_qs: set) -> List[str]:
    """Walk source generation.jsonl in order; return list[split] aligned with row index."""
    splits: List[str] = []
    with src_gen.open() as f:
        for line in f:
            q = json.loads(line)["question"]
            if q in qwen_test_qs:
                splits.append("test")
            elif q in qwen_train_qs:
                splits.append("train")
            else:
                raise RuntimeError(f"Question not in either Qwen3 split: {q!r}")
    return splits


def filter_jsonl_by_indices(src: Path, dst: Path, indices: List[int]) -> int:
    """Copy lines at `indices` (sorted ascending) from src to dst, preserving order."""
    idx_set = set(indices)
    written = 0
    with src.open() as fr, dst.open("w") as fw:
        for i, line in enumerate(fr):
            if i in idx_set:
                fw.write(line)
                written += 1
    return written


def filter_jsonl_by_question(src: Path, dst: Path, question_set: set) -> int:
    """Copy lines whose 'question' is in `question_set`. Handles files with row-count drift."""
    written = 0
    with src.open() as fr, dst.open("w") as fw:
        for line in fr:
            if json.loads(line)["question"] in question_set:
                fw.write(line)
                written += 1
    return written


def build_eval_for_training(src_eval: dict, indices: List[int]) -> dict:
    """Slice abstantion + halu_test_res by `indices`, recompute summary fields."""
    abst = [src_eval["abstantion"][i] for i in indices]
    halu = [src_eval["halu_test_res"][i] for i in indices]
    total = len(indices)
    accurate = sum(1 for h in halu if not h)
    halu_ct = sum(1 for h in halu if h)
    refusal = sum(1 for a in abst if a)
    return {
        "evaluator_abstantion": src_eval["evaluator_abstantion"],
        "evaluator_hallucination": src_eval["evaluator_hallucination"],
        "abstantion": abst,
        "halu_test_res": halu,
        "total_count": total,
        "accurate_count": accurate,
        "hallu_count": halu_ct,
        "refusal_count": refusal,
        "correct_rate": accurate / total if total else 0.0,
        "halu_rate_not_abstain": halu_ct / total if total else 0.0,
        "refusal_rate": refusal / total if total else 0.0,
    }


def build_eval_summary(eft: dict) -> dict:
    """Derive eval_results.json (the short summary) from eval_results_for_training."""
    total = eft["total_count"]
    return {
        "model": MODEL_NAME,
        "halu_Rate": eft["halu_rate_not_abstain"],
        "refusal_Rate": eft["refusal_rate"],
        "correct_rate": eft["correct_rate"],
        "accurate_count": eft["accurate_count"],
        "hallu_count": eft["hallu_count"],
        "total_count": total,
        "refusal_count": eft["refusal_count"],
        "hallucination_evaluation": "string_matching",
    }


def split_zarr(
    src_root: zarr.Group,
    dst_path: Path,
    indices: List[int],
    expected_n: int,
) -> None:
    """Copy rows at `indices` from src zarr into a new zarr at dst_path.

    Preserves: schema version, attrs, all per-sample arrays, meta/index.jsonl,
    and zarr.json metadata.
    """
    if dst_path.exists():
        shutil.rmtree(dst_path)
    dst_path.mkdir(parents=True)

    dst_root = zarr.open_group(str(dst_path), mode="w")
    dst_root.attrs.update(dict(src_root.attrs))

    src_arrays = src_root["arrays"]
    dst_arrays = dst_root.create_group("arrays")

    n = len(indices)
    assert n == expected_n, f"expected {expected_n} rows, got {n}"

    idx_arr = np.asarray(indices, dtype=np.int64)

    array_names = [
        "prompt_activations",
        "response_activations",
        "prompt_len",
        "response_len",
        "sample_key",
        "response_token_ids",
        "response_token_logprobs",
        "response_topk_token_ids",
        "response_topk_logprobs",
    ]

    for name in array_names:
        if name not in src_arrays:
            continue
        src_arr = src_arrays[name]
        new_shape = (n,) + tuple(src_arr.shape[1:])
        # Source was written by zarr_activations_logger.py with compressor=None;
        # we mirror that. (zarr v3 deprecated `compressor` singular.)
        dst_arr = dst_arrays.create_dataset(
            name,
            shape=new_shape,
            chunks=src_arr.chunks,
            dtype=src_arr.dtype,
            fill_value=src_arr.fill_value,
            overwrite=True,
        )

        # Copy in batches to bound memory. Activation arrays are O(layers * tokens
        # * hidden) per row — keep the batch small.
        is_heavy = name in ("prompt_activations", "response_activations")
        batch = 32 if is_heavy else 4096
        for start in range(0, n, batch):
            end = min(start + batch, n)
            src_idx = idx_arr[start:end]
            # zarr supports fancy indexing on sorted unique indices via get_orthogonal_selection;
            # for safety we materialize via numpy.
            block = src_arr.get_orthogonal_selection((src_idx,) + (slice(None),) * (src_arr.ndim - 1))
            dst_arr[start:end] = block
            if is_heavy:
                pct = end / n * 100
                print(f"    {name}: {end}/{n} ({pct:.1f}%)", flush=True)

    # Meta/index.jsonl: filter source by row index in-order.
    src_meta = src_root.store.path if hasattr(src_root.store, "path") else None
    src_index_path = Path(SRC_ZARR) / "meta" / "index.jsonl"
    dst_meta_dir = dst_path / "meta"
    dst_meta_dir.mkdir(parents=True, exist_ok=True)
    idx_set = set(indices)
    new_idx_to_old: Dict[int, int] = {old: new for new, old in enumerate(indices)}
    with src_index_path.open() as fr, (dst_meta_dir / "index.jsonl").open("w") as fw:
        for i, line in enumerate(fr):
            if i not in idx_set:
                continue
            entry = json.loads(line)
            entry["sample_index"] = new_idx_to_old[i]
            fw.write(json.dumps(entry) + "\n")

    # text/ dir is optional; copy if present so any auxiliary text files survive.
    src_text = SRC_ZARR / "text"
    if src_text.exists():
        dst_text = dst_path / "text"
        if not dst_text.exists():
            shutil.copytree(src_text, dst_text)


def write_split_outputs(
    name: str,
    indices: List[int],
    question_set: set,
    src_eval_for_training: dict,
    dst_zarr: Path,
    dst_out: Path,
    src_root: zarr.Group,
    force: bool,
) -> None:
    print(f"\n=== {name}: {len(indices)} samples ===", flush=True)

    # Targets
    if dst_zarr.exists() and not force:
        print(f"  [skip zarr] {dst_zarr} exists (use --force to overwrite)")
    else:
        print(f"  splitting zarr → {dst_zarr}")
        split_zarr(src_root, dst_zarr, indices, expected_n=len(indices))

    dst_out.mkdir(parents=True, exist_ok=True)

    # generation.jsonl: row-aligned slice
    gen_dst = dst_out / "generation.jsonl"
    if gen_dst.exists() and not force:
        print(f"  [skip] {gen_dst} exists")
    else:
        n = filter_jsonl_by_indices(SRC_OUT / "generation.jsonl", gen_dst, indices)
        print(f"  wrote {gen_dst.name}: {n} lines")

    # generation.sanitized_for_eval.jsonl: question-keyed (file may be missing 2 rows)
    san_dst = dst_out / "generation.sanitized_for_eval.jsonl"
    if san_dst.exists() and not force:
        print(f"  [skip] {san_dst} exists")
    else:
        n = filter_jsonl_by_question(SRC_OUT / "generation.sanitized_for_eval.jsonl", san_dst, question_set)
        print(f"  wrote {san_dst.name}: {n} lines")

    # eval_results_for_training.json: slice arrays + recompute summary fields
    eft = build_eval_for_training(src_eval_for_training, indices)
    eft_dst = dst_out / "eval_results_for_training.json"
    if eft_dst.exists() and not force:
        print(f"  [skip] {eft_dst} exists")
    else:
        eft_dst.write_text(json.dumps(eft, indent=2))
        print(f"  wrote {eft_dst.name}: total={eft['total_count']}, halu_rate={eft['halu_rate_not_abstain']:.4f}")

    # eval_results.json: short summary
    summary = build_eval_summary(eft)
    sum_dst = dst_out / "eval_results.json"
    if sum_dst.exists() and not force:
        print(f"  [skip] {sum_dst} exists")
    else:
        sum_dst.write_text(json.dumps(summary, indent=4))
        print(f"  wrote {sum_dst.name}")


def write_dataset_config(force: bool) -> None:
    cfg = {
        "name": "nq",
        "model_name": MODEL_NAME,
        "input_dim": 4096,
        "backend": "zarr",
        "label_source": "eval_json",
        "outlier_class": 1,
        "train": {
            "inference_json": f"output/natural_questions_train/{MODEL_NAME}/generation.jsonl",
            "activations_path": "shared/natural_questions_train/activations.zarr",
            "eval_json": f"output/natural_questions_train/{MODEL_NAME}/eval_results_for_training.json",
            "raw_eval_jsonl": f"output/natural_questions_train/{MODEL_NAME}/raw_eval_res.jsonl",
        },
        "test": {
            "inference_json": f"output/natural_questions/{MODEL_NAME}/generation.jsonl",
            "activations_path": "shared/natural_questions/activations.zarr",
            "eval_json": f"output/natural_questions/{MODEL_NAME}/eval_results_for_training.json",
            "raw_eval_jsonl": f"output/natural_questions/{MODEL_NAME}/raw_eval_res.jsonl",
        },
    }
    if CONFIG_PATH.exists() and not force:
        print(f"\n[skip config] {CONFIG_PATH} exists (use --force to overwrite)")
        return
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    print(f"\nwrote config: {CONFIG_PATH}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="overwrite existing targets")
    p.add_argument("--dry-run", action="store_true", help="report split sizes and exit")
    args = p.parse_args()

    if not SRC_ZARR.exists():
        sys.exit(f"missing source zarr: {SRC_ZARR}")
    for required in (SRC_OUT / "generation.jsonl", QWEN_TEST_GEN, QWEN_TRAIN_GEN):
        if not required.exists():
            sys.exit(f"missing required file: {required}")

    print("Loading Qwen3 question sets …", flush=True)
    qwen_test_qs = load_questions(QWEN_TEST_GEN)
    qwen_train_qs = load_questions(QWEN_TRAIN_GEN)
    print(f"  qwen test:  {len(qwen_test_qs)} unique questions")
    print(f"  qwen train: {len(qwen_train_qs)} unique questions")
    overlap = qwen_test_qs & qwen_train_qs
    if overlap:
        sys.exit(f"refusing to proceed: Qwen3 test/train have {len(overlap)} overlapping questions")

    print(f"\nAssigning Llama3 rows to splits …", flush=True)
    splits = assign_splits(SRC_OUT / "generation.jsonl", qwen_test_qs, qwen_train_qs)
    test_idx = [i for i, s in enumerate(splits) if s == "test"]
    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    print(f"  total rows: {len(splits)}")
    print(f"  → test:  {len(test_idx)}")
    print(f"  → train: {len(train_idx)}")

    if len(test_idx) != len(qwen_test_qs):
        sys.exit(f"test row count {len(test_idx)} != qwen test question count {len(qwen_test_qs)}")
    if len(train_idx) != len(qwen_train_qs):
        sys.exit(f"train row count {len(train_idx)} != qwen train question count {len(qwen_train_qs)}")

    if args.dry_run:
        print("\n[dry-run] no files written")
        return

    print("\nLoading source eval_results_for_training.json …", flush=True)
    src_eval_eft = json.loads((SRC_OUT / "eval_results_for_training.json").read_text())
    if len(src_eval_eft["abstantion"]) != len(splits):
        sys.exit(
            f"eval row count {len(src_eval_eft['abstantion'])} != generation row count {len(splits)} "
            "— file misalignment, refusing to proceed"
        )

    print(f"\nOpening source zarr (read-only) …", flush=True)
    src_root = zarr.open_group(str(SRC_ZARR), mode="r")

    write_split_outputs(
        name="TEST",
        indices=test_idx,
        question_set=qwen_test_qs,
        src_eval_for_training=src_eval_eft,
        dst_zarr=DST_TEST_ZARR,
        dst_out=DST_TEST_OUT,
        src_root=src_root,
        force=args.force,
    )
    write_split_outputs(
        name="TRAIN",
        indices=train_idx,
        question_set=qwen_train_qs,
        src_eval_for_training=src_eval_eft,
        dst_zarr=DST_TRAIN_ZARR,
        dst_out=DST_TRAIN_OUT,
        src_root=src_root,
        force=args.force,
    )

    write_dataset_config(force=args.force)

    print("\nDone. Note: raw_eval_res.jsonl is referenced by the new config but not")
    print("present in the source dump. Run `--step eval` later to materialize it,")
    print("or rely on eval_results_for_training.json (which has all the labels).")


if __name__ == "__main__":
    main()
