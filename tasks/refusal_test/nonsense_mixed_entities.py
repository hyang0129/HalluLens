# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse

from tqdm.contrib.concurrent import thread_map
from utils import lm, eval_utils

from tasks.refusal_test.nonsense_name import NonsenseNameInference, NonsenseNameEval
from tasks.refusal_test.entities_generation import NonsenseMixedGeneration
import tasks.refusal_test.prompt as prompt_templates


class NonsenseMixedInference(NonsenseNameInference):
    def __init__(self, taskname, output_base_dir, generate_model, prompt_path, seed, method='vllm'):
        self.output_base_dir = output_base_dir
        self.generate_model = generate_model
        self.inference_method = method
        self.prompt_path = prompt_path
        self.seed = seed
        self.TASKNAME = taskname #prompt_path.split('/')[-1].replace('.csv', '') #  f"{seed}_{N}.csv"
        print('INFER TASKNAME', self.TASKNAME)

class NonsenseMixedEval(NonsenseNameEval):
    def __init__(self, taskname, output_base_dir, model_path, prompt_path, med_safety_filtered_model=False):

        self.prompt_path = prompt_path
        self.TASKNAME = taskname #prompt_path.split('/')[-1].replace('.csv', '') #  f"{seed}_{N}.csv"
        self.model_name = model_path.split("/")[-1]

        self.task_output_dir = f"{output_base_dir}/{self.TASKNAME}/{self.model_name}"
        self.generations_file_path = f'{self.task_output_dir}/generation.jsonl'
        self.res_path = f'{self.task_output_dir}/eval_results.json'
        self.eval_raw_path = f'{self.task_output_dir}/raw_eval_res.jsonl'

        self.med_safety_filtered_model = med_safety_filtered_model
        self.evaluator = "meta-llama/Llama-3.1-8B-Instruct"

        print('EVAL TASKNAME', self.TASKNAME)

    def automatic_abstention(self, generations, evaluator_model="meta-llama/Llama-3.1-8B-Instruct"):
        JSON_KEY = "does_believe"

        eval_prompts = {
            'medicine' : prompt_templates.ABSTAIN_PROMPT_NONSENSE_MEDICINE,
            'animal' : prompt_templates.ABSTAIN_PROMPT_ANIMAL_NONSENSE,
            'plant' : prompt_templates.ABSTAIN_PROMPT_ANIMAL_NONSENSE,
            'bacteria' : prompt_templates.ABSTAIN_PROMPT_ANIMAL_NONSENSE,
        }

        abstain_prompts = [
                eval_prompts.get(gen_obj['type']).format(
                    prompt=gen_obj['prompt'],
                    name=gen_obj['name'],
                    generation=gen_obj['generation'],
                )
                for gen_obj in generations
            ]
        
        abstains_eval_raw = thread_map(
            lambda p: lm.generate(p, self.evaluator),
                    abstain_prompts,
                    max_workers=50,
                    desc=f"using {self.evaluator}")
        
        if self.med_safety_filtered_model:
            for i, gen_obj in enumerate(generations):
                if gen_obj['type'] == 'medicine':
                    abstains_eval_raw[i] = "{\"does_believe\": false}"

        eval_utils.save_eval_raw(abstains_eval_raw, self.eval_raw_path)

        abstains_eval = eval_utils.jsonify_ans(raw_responses=abstains_eval_raw, \
                                                eval_prompts=abstain_prompts, \
                                                evaluator_model=evaluator_model,\
                                                key=JSON_KEY)
        abstains_eval_res = []
        for o in abstains_eval:
            abstains_eval_res.append(not o[JSON_KEY])
        
        return abstains_eval_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='nonsense_all')

    parser.add_argument('--do_generate_prompt', default=False, action='store_true')
    parser.add_argument('--do_inference', default=False, action='store_true')
    parser.add_argument('--do_eval', default=False, action='store_true')
    
    parser.add_argument('--name_overwrite', default=False, action='store_true')
    parser.add_argument('--infer_overwrite', default=False, action='store_true')
    parser.add_argument('--eval_overwrite', default=False, action='store_true')

    parser.add_argument('--output_base_dir', type=str, default="output") # inference and eval output
    parser.add_argument('--prompt_output_path', type=str, default="") # name output
    parser.add_argument('--tested_model', type=str, default='meta-llama/Llama-3.1-405B-Instruct-FP8')
    
    parser.add_argument('--N', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--inference_method', type=str, default='vllm')
    args = parser.parse_args()

    # set variables
    N = args.N
    EXP = args.exp #nonsense_medicine
    seed = args.seed
    tested_model = args.tested_model
    tested_model_name = tested_model.split("/")[-1]
    output_base_dir = args.output_base_dir
    inference_method = args.inference_method

    if not args.prompt_output_path:
        current_path = os.getcwd()
        args.prompt_output_path = '/'.join(current_path.split('/')[:5]) + f"/data/{EXP}/"
    PROMPT_OUTPUT_DIR = args.prompt_output_path
    prompt_path = f"{PROMPT_OUTPUT_DIR}/save/{tested_model_name}/{EXP}_{seed}_{N}.csv"
    TASKNAME = f"{EXP}_{seed}_{N}"

    # generate prompts
    if args.do_generate_prompt:
        generator = NonsenseMixedGeneration(seed, N, EXP)
        prompt_objs = generator.generate_prompts()
        generator.save_prompt_csv(prompt_objs, prompt_path)

    # run inference
    if args.do_inference:
        inference = NonsenseMixedInference(TASKNAME, output_base_dir, tested_model, prompt_path, seed, inference_method)
        if args.infer_overwrite:
            inference.remove_existing_files()
        inference.run_inference()
            
    # run evaluation
    if args.do_eval:
        if 'gemma' in tested_model:
            med_safety_filtered_model = True
            eval = NonsenseMixedEval(TASKNAME, output_base_dir, tested_model, prompt_path, med_safety_filtered_model)
        else:
            eval = NonsenseMixedEval(TASKNAME, output_base_dir, tested_model, prompt_path)
        res = eval.run_eval(args.eval_overwrite)