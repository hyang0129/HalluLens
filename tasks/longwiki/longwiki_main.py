# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import pandas as pd
import os
import json
from pathlib import Path

from tasks.longwiki.facthalu import FactHalu
from utils import exp
from utils import generate_question as qa

TASKNAME = "longwiki"

def run_eval(args):
    model_name = args.model.split("/")[-1]
    output_folder = Path(f'output/{TASKNAME}-{args.exp_mode}/{model_name}')
    output_csv = output_folder / "output.csv"
    generations_file_path = output_folder / "generation.jsonl"
    base_path = os.path.dirname(os.path.abspath(__name__))
    eval_cache_path = f"{base_path}/data/longwiki/.cache" if args.eval_cache_path is None else args.eval_cache_path

    facthalu = FactHalu(generations_file_path,
        output_csv,
        abstain_evaluator=args.abstain_evaluator,
        claim_extractor=args.claim_extractor,
        verifier=args.verifier,
        k=args.k,
        eval_cache_path=eval_cache_path,
        db_path = args.db_path,
        args=args
        )

    # save all evalaution details
    eval_details = {
        "output_csv": str(output_csv),
        "abstain_evaluator": args.abstain_evaluator,
        "claim_extractor": args.claim_extractor,
        "verifier": args.verifier,
        "k": args.k,
        "evalauted_model": model_name,
        "exp_mode" : args.exp_mode,
        "eval_time" : str(pd.Timestamp.now())
    }

    with open (output_folder / "eval_details.json", 'w') as f:
        json.dump(eval_details, f)

    facthalu.run()

def run_step(step, model, exp_mode="longwiki", N=5,
             db_path="data/wiki_data/.cache/enwiki-20230401.db",
             q_generator="meta-llama/Meta-Llama-3.1-70B-Instruct",
             claim_extractor="meta-llama/Llama-3.1-405B-Instruct-FP8",
             abstain_evaluator="meta-llama/Llama-3.1-70B-Instruct",
             verifier="meta-llama/Llama-3.1-405B-Instruct-FP8",
             k=32, max_tokens=1024, max_workers=64, max_workers_qgen=1,
             inference_method="vllm", eval_cache_path=None,
             logger_type="lmdb", activations_path=None, log_file=None,
             resume=True, resume_eval=True):
    """Run a single step of the LongWiki task. Callable from Python directly."""
    import types
    base_path = os.getcwd()
    model_name = model.split("/")[-1]
    QA_OUTPUT_PATH = f"data/longwiki/save/longwiki_{model_name}.jsonl"

    if step == "generate":
        if os.path.exists(QA_OUTPUT_PATH):
            print("using existing qa file")
            all_prompts = pd.read_json(QA_OUTPUT_PATH, lines=True)
            assert len(all_prompts) == N
        else:
            if "longwiki" == exp_mode:
                wiki_input_path = f"{base_path}/data/wiki_data/doc_goodwiki_h_score.jsonl"
                print(wiki_input_path)
                QAs = qa.longform_QA_generation_run_batch(
                    wiki_input_path=wiki_input_path,
                    N=N,
                    q_generator=q_generator,
                    output_path=QA_OUTPUT_PATH,
                    from_scratch=False,
                    max_workers=max_workers_qgen)
                all_prompts = pd.DataFrame(QAs)
            else:
                raise NotImplementedError(f"Mode {exp_mode} not implemented")

    elif step == "inference":
        all_prompts = pd.read_json(QA_OUTPUT_PATH, lines=True)
        assert len(all_prompts) == N
        print(f"Start Inference for {model} ", exp_mode, N)
        exp.run_exp(
            task=f"{TASKNAME}-{exp_mode}",
            model_path=model,
            all_prompts=all_prompts,
            inference_method=inference_method,
            max_tokens=max_tokens,
            logger_type=logger_type,
            activations_path=activations_path,
            log_file_path=log_file)
        print('\n***Inference completed')

    elif step == "eval":
        args_ns = types.SimpleNamespace(
            model=model, exp_mode=exp_mode, db_path=db_path,
            abstain_evaluator=abstain_evaluator, claim_extractor=claim_extractor,
            verifier=verifier, k=k, eval_cache_path=eval_cache_path,
            no_resume=not resume, no_resume_eval=not resume_eval,
            logger_type=logger_type, activations_path=activations_path, log_file=log_file,
        )
        print("============= [[ {} ]] =================".format(exp_mode))
        print(f"Running evaluation for {model_name};")
        run_eval(args_ns)
        print('\n***Evaluation completed')

    else:
        raise ValueError(f"Unknown step: {step}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_mode', type=str, default='', help='longwiki')

    parser.add_argument('--do_generate_prompt', default=False, action='store_true')
    parser.add_argument('--do_inference', default=False, action='store_true')
    parser.add_argument('--do_eval', default=False, action='store_true')
    parser.add_argument('--do_extract_only', default=False, action='store_true')

    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.1-405B-Instruct-FP8', help='model that is being "TESTED"')
    parser.add_argument('--q_generator', type=str, default='meta-llama/Meta-Llama-3.1-70B-Instruct', help='model that is used for question generation')

    parser.add_argument('--claim_extractor', type=str, default='meta-llama/Llama-3.1-405B-Instruct-FP8', help='model that is used for claim extraction')
    parser.add_argument('--abstain_evaluator', type=str, default="meta-llama/Llama-3.1-70B-Instruct", help='model that is used for abstantion evaluation')
    parser.add_argument('--verifier', type=str, default='meta-llama/Llama-3.1-405B-Instruct-FP8', help='model that is used for final verification')

    parser.add_argument('--inference_method', type=str, default='smallmodel', help='meta server (metagen/openai) or caire (smallmodel)')
    parser.add_argument('--eval_cache_path', type=str, default=None)
    parser.add_argument('--db_path', type=str, default="data/wiki_data/.cache/enwiki-20230401.db")
    parser.add_argument('--N', type=int, default=250)

    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--max_workers', type=int, default=64)
    parser.add_argument('--max_workers_qgen', type=int, default=1, help='maximum concurrent requests for question generation (default: 1)')

    # Resume control
    parser.add_argument('--no-resume', action='store_true', help='Disable automatic resume from existing generations file (inference)')
    parser.add_argument('--no-resume-eval', action='store_true', help='Disable automatic resume for evaluation step')

    # Activation logging parameters
    parser.add_argument('--logger_type', type=str, default='lmdb', choices=['lmdb', 'json'], help='Activation logger type')
    parser.add_argument('--activations_path', type=str, default=None, help='Path for storing activations')
    parser.add_argument('--log_file', type=str, default=None, help='Path for server behavior logs')

    args = parser.parse_args()

    if args.do_generate_prompt:
        run_step("generate", args.model, exp_mode=args.exp_mode, N=args.N,
                 q_generator=args.q_generator, max_workers_qgen=args.max_workers_qgen)
    if args.do_inference:
        run_step("inference", args.model, exp_mode=args.exp_mode, N=args.N,
                 inference_method=args.inference_method, max_tokens=args.max_tokens,
                 logger_type=args.logger_type, activations_path=args.activations_path,
                 log_file=args.log_file)
    if args.do_eval:
        run_step("eval", args.model, exp_mode=args.exp_mode, db_path=args.db_path,
                 abstain_evaluator=args.abstain_evaluator, claim_extractor=args.claim_extractor,
                 verifier=args.verifier, k=args.k, eval_cache_path=args.eval_cache_path,
                 resume=not args.no_resume, resume_eval=not args.no_resume_eval,
                 logger_type=args.logger_type, activations_path=args.activations_path,
                 log_file=args.log_file)
            


