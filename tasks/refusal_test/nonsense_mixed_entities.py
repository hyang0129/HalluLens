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
    def __init__(self, taskname, output_base_dir, generate_model, prompt_path, seed, method='vllm', logger_type='lmdb', activations_path=None, log_file_path=None):
        super().__init__(output_base_dir, generate_model, prompt_path, seed, method, logger_type, activations_path, log_file_path)
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

    def automatic_abstention(self, generations, evaluator_model="meta-llama/Llama-3.1-8B-Instruct", resume=True):
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
        
        # Check for existing results and resume if requested
        existing_results = []
        prompts_to_process = abstain_prompts

        if resume and os.path.exists(self.eval_raw_path):
            print(f"📂 Found existing evaluation file: {self.eval_raw_path}")
            try:
                with open(self.eval_raw_path, 'r') as f:
                    for line in f:
                        existing_results.append(json.loads(line)['eval_res'])

                if len(existing_results) > 0:
                    print(f"✅ Loaded {len(existing_results)} existing evaluations")
                    if len(existing_results) >= len(abstain_prompts):
                        print(f"All {len(abstain_prompts)} evaluations already complete!")
                        abstains_eval_raw = existing_results
                    else:
                        prompts_to_process = abstain_prompts[len(existing_results):]
                        print(f"📊 Resuming from {len(existing_results)}...")
            except Exception as e:
                print(f"⚠️ Warning: Could not resume: {e}")
                existing_results = []
                prompts_to_process = abstain_prompts

        if not 'abstains_eval_raw' in locals() and len(prompts_to_process) > 0:
            print(f"🔄 Processing {len(prompts_to_process)} evaluation requests...")
            from tqdm import tqdm
            
            # Start server for evaluator model if needed
            server_was_running = lm.check_server_health("http://0.0.0.0:8000")
            server_manager = None

            if not server_was_running:
                print(f"🚀 Starting evaluation server for {self.evaluator}...")
                server_manager = lm.ServerManager(
                    model=self.evaluator,
                    host="0.0.0.0",
                    port=8000,
                    logger_type="lmdb",
                    activations_path=None
                )
                server_manager.start_server()
                lm.set_server_manager(server_manager)

            # Initialize progress tracking for client logging
            lm.initialize_progress_tracking(len(abstain_prompts), already_completed=len(existing_results))

            try:
                file_mode = 'a' if existing_results else 'w'
                new_results = []
                with open(self.eval_raw_path, file_mode, encoding='utf-8') as f:
                    for prompt in tqdm(prompts_to_process, desc=f"using {self.evaluator}"):
                        result = lm.generate(prompt, self.evaluator)
                        new_results.append(result)
                        # Save immediately
                        f.write(json.dumps({"eval_res": result}, ensure_ascii=False) + '\n')
                        f.flush()
                abstains_eval_raw = existing_results + new_results
            finally:
                if server_manager:
                    server_manager.stop_server()
                    lm.set_server_manager(None)
        elif not 'abstains_eval_raw' in locals():
            abstains_eval_raw = existing_results
        
        if self.med_safety_filtered_model:
            for i, gen_obj in enumerate(generations):
                if gen_obj['type'] == 'medicine':
                    abstains_eval_raw[i] = "{\"does_believe\": false}"

        # Note: raw responses are already saved incrementally
        # eval_utils.save_eval_raw(abstains_eval_raw, self.eval_raw_path)

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

    # Activation logging parameters
    parser.add_argument('--logger_type', type=str, default='lmdb', choices=['lmdb', 'json'], help='Activation logger type')
    parser.add_argument('--activations_path', type=str, default=None, help='Path for storing activations')
    parser.add_argument('--log_file', type=str, default=None, help='Path for server behavior logs')

    # Resume control
    parser.add_argument('--no-resume', action='store_true', help='Disable automatic resume from existing generations file (inference)')
    parser.add_argument('--no-resume-eval', action='store_true', help='Disable automatic resume for evaluation step')

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
        inference = NonsenseMixedInference(TASKNAME, output_base_dir, tested_model, prompt_path, seed, inference_method, args.logger_type, args.activations_path, args.log_file)
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
        res = eval.run_eval(args.eval_overwrite, resume=not args.no_resume_eval)