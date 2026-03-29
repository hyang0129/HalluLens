"""Build ActivationParser-compatible eval JSON from inference and evaluation artifacts.

Extracted from notebooks/d_baseline_comparison.ipynb (cell 4).
"""

import json
from pathlib import Path

import pandas as pd


def _load_jsonl(path: Path):
    """Load a JSONL file into a list of dicts."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_eval_for_activation_parser(
    inference_json_path,
    eval_json_path,
    raw_eval_jsonl_path,
    output_path,
):
    """Build an ActivationParser-compatible eval JSON file.

    Tries three strategies in order:
    1. Load eval_results.json (halu_test_res + abstantion arrays)
    2. Fall back to raw_eval_res.jsonl
    3. Fall back to substring matching

    Parameters
    ----------
    inference_json_path : str or Path
        Path to generation.jsonl.
    eval_json_path : str or Path
        Path to eval_results.json (may not exist).
    raw_eval_jsonl_path : str or Path
        Path to raw_eval_res.jsonl (may not exist).
    output_path : str or Path
        Where to write the compat JSON.

    Returns
    -------
    str
        The output path written to.
    """
    inference_path = Path(inference_json_path)
    if not inference_path.exists():
        raise FileNotFoundError(f"Missing inference file: {inference_path}")

    gendf = pd.read_json(inference_path, lines=True)
    n = len(gendf)
    if n == 0:
        raise RuntimeError("generation.jsonl is empty")

    halu, abstain, source_used = None, None, None

    # Strategy 1: eval_results.json
    eval_path = Path(eval_json_path)
    if eval_path.exists():
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        if (
            isinstance(payload, dict)
            and "halu_test_res" in payload
            and "abstantion" in payload
            and len(payload["halu_test_res"]) == n
            and len(payload["abstantion"]) == n
        ):
            halu = [bool(x) for x in payload["halu_test_res"]]
            abstain = [bool(x) for x in payload["abstantion"]]
            source_used = "eval_results.json"

    # Strategy 2: raw_eval_res.jsonl
    if halu is None:
        raw_path = Path(raw_eval_jsonl_path)
        if raw_path.exists():
            raw_rows = _load_jsonl(raw_path)
            if len(raw_rows) == n:
                halu = [
                    bool(
                        r.get(
                            "is_hallucination",
                            not bool(r.get("is_correct", False)),
                        )
                    )
                    for r in raw_rows
                ]
                abstain = [False] * n
                source_used = "raw_eval_res.jsonl"

    # Strategy 3: substring matching fallback
    if halu is None:
        halu = []
        for _, row in gendf.iterrows():
            answer = str(row.get("answer", "")).strip().lower()
            generation = str(row.get("generation", "")).strip().lower()
            halu.append(not (bool(answer) and answer in generation))
        abstain = [False] * n
        source_used = "substring fallback"

    hallu_count = int(sum(halu))
    compat = {
        "evaluator_abstantion": "natural_questions",
        "evaluator_hallucination": "natural_questions",
        "abstantion": abstain,
        "halu_test_res": halu,
        "total_count": n,
        "accurate_count": n - hallu_count,
        "hallu_count": hallu_count,
        "refusal_count": int(sum(abstain)),
        "correct_rate": float((n - hallu_count) / max(1, n)),
        "halu_rate_not_abstain": float(hallu_count / max(1, n - sum(abstain))),
        "refusal_rate": float(sum(abstain) / max(1, n)),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(compat, indent=2), encoding="utf-8")
    print(
        f"Eval JSON written: {out}  (source: {source_used}, n={n}, halu={hallu_count})"
    )
    return str(out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build ActivationParser-compatible eval JSON from inference artifacts."
    )
    parser.add_argument(
        "--inference-json",
        required=True,
        help="Path to generation.jsonl",
    )
    parser.add_argument(
        "--eval-json",
        required=True,
        help="Path to eval_results.json (may not exist yet)",
    )
    parser.add_argument(
        "--raw-eval-jsonl",
        required=True,
        help="Path to raw_eval_res.jsonl (may not exist yet)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the compat eval JSON",
    )
    args = parser.parse_args()

    build_eval_for_activation_parser(
        inference_json_path=args.inference_json,
        eval_json_path=args.eval_json,
        raw_eval_jsonl_path=args.raw_eval_jsonl,
        output_path=args.output,
    )
