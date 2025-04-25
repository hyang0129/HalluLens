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
    args = parser.parse_args()

    # save all args details in  
    base_path = os.path.dirname(os.path.abspath(__name__))
    model_name = args.model.split("/")[-1]
    QA_OUTPUT_PATH = f"data/longwiki/save/longwiki_{model_name}.jsonl"

    if args.do_generate_prompt:
        if os.path.exists(QA_OUTPUT_PATH):
            print("using existing qa file")
            all_prompts = pd.read_json(QA_OUTPUT_PATH, lines=True)
            assert len(all_prompts) == args.N
        else:
            if "longwiki" == args.exp_mode:
                wiki_input_path = f"{base_path}/data/wiki_data/doc_goodwiki_h_score.jsonl"
                print(wiki_input_path)
                QAs = qa.longform_QA_generation_run_batch(
                        wiki_input_path=f"{base_path}/data/wiki_data/doc_goodwiki_h_score.jsonl",
                        N=args.N,
                        q_generator=args.q_generator, # "meta-llama/Meta-Llama-3.1-405B-Instruct", 
                        output_path=QA_OUTPUT_PATH,
                        from_scratch=False
                    )
                all_prompts = pd.DataFrame(QAs)
            else:
                raise NotImplementedError(f"Mode {args.exp_mode} not implemented")

    # RUN INFERENCE
    if args.do_inference:
        all_prompts = pd.read_json(QA_OUTPUT_PATH, lines=True)
        assert len(all_prompts) == args.N

        print(f"Start Inference for {args.model} ", args.exp_mode, args.N)

        exp.run_exp(task=f"{TASKNAME}-{args.exp_mode}", 
                    model_path=args.model,
                    all_prompts=all_prompts,
                    inference_method=args.inference_method,
                    max_tokens=args.max_tokens)

        print('\n***Inference completed')

    # RUN EVALUATION:
    if args.do_eval:
        print("============= [[ {} ]] =================".format(args.exp_mode))
        print(f"Running evaluation for {model_name};")
        print(f"** Refusal Evaluator: {args.abstain_evaluator}")
        print(f"** Claim Extractor: {args.claim_extractor}")
        print(f"** Verifier: {args.verifier}") 
        print("=========================================")
        run_eval(args)
        
        print('\n***Evaluation completed')
            


