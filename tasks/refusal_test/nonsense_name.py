# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from tqdm.contrib.concurrent import thread_map
from utils import exp, lm, eval_utils
import tasks.refusal_test.prompt as prompt_templates
import os
import json
import pandas as pd

def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


class NonsenseNameInference:
    def __init__(self, output_base_dir, generate_model, prompt_path, seed, method='vllm', logger_type='lmdb', activations_path=None, log_file_path=None):
        self.output_base_dir = output_base_dir
        self.generate_model = generate_model
        self.inference_method = method
        self.prompt_path = prompt_path
        self.seed = seed
        self.logger_type = logger_type
        self.activations_path = activations_path
        self.log_file_path = log_file_path
        self.TASKNAME = prompt_path.split('/')[-1].replace('_all_not_exist.csv', '') #  f"{seed}_{BUSINESS_N}_{EVENT_N}_{PRODUCT_N}"
        print('INFER TASKNAME', self.TASKNAME)
    
    def run_inference(self):
        generate_model = self.generate_model
        print('generate_model', generate_model)
        TASKNAME = self.TASKNAME
        # prompt_path = f"{self.root_path}/save/{self.seed}_{self.BUSINESS_N}_{self.EVENT_N}_{self.PRODUCT_N}_all_not_exist.csv"
        all_prompts = pd.read_csv(self.prompt_path)
        exp.run_exp(task=TASKNAME,
                    model_path=generate_model,
                    all_prompts=all_prompts,
                    inference_method=self.inference_method,
                    max_tokens=256,
                    base_path=self.output_base_dir,
                    logger_type=self.logger_type,
                    activations_path=self.activations_path,
                    log_file_path=self.log_file_path)
        print(TASKNAME, 'Inference completed')

    def remove_existing_files(self):
        model_name = self.generate_model.split("/")[-1]
        generations_file_path = f"{self.output_base_dir}/{self.TASKNAME}/{model_name}/generation.jsonl"
        results_file_path = f"{self.output_base_dir}/{self.TASKNAME}/{model_name}/eval_results.json"
        remove_file(generations_file_path)
        remove_file(results_file_path)

class NonsenseNameEval:
    def __init__(self, output_base_dir, model_path, prompt_path):
        self.prompt_path = prompt_path
        self.TASKNAME = prompt_path.split('/')[-1].replace('_all_not_exist.csv', '') #  f"{seed}_{BUSINESS_N}_{EVENT_N}_{PRODUCT_N}"
        print('EVAL TASKNAME', self.TASKNAME)
        self.model_name = model_path.split("/")[-1]
        self.task_output_dir = f"{output_base_dir}/{self.TASKNAME}/{self.model_name}"
        self.generations_file_path = f'{self.task_output_dir}/generation.jsonl'
        self.res_path = f'{self.task_output_dir}/eval_results.json'
        self.eval_raw_path = f'{self.task_output_dir}/raw_eval_res.jsonl'
        self.evaluator = "meta-llama/Llama-3.1-70B-Instruct"

    def automatic_abstention(self, generations, evaluator_model="meta-llama/Meta-Llama-3.1-70B-Instruct", resume=True):
        abstain_prompts = [
                prompt_templates.ABSTAIN_PROMPT_PLACE_NONSENSE.format(
                    name=generation['name'], 
                    TYPE=generation['type_'],
                    PLACE=" in " + generation['place'] if generation['place'] else "",
                    generation=generation['generation'],
                )
                for generation in generations
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
                        print(f"✅ All {len(abstain_prompts)} evaluations already complete!")
                        abstains_eval_raw = existing_results
                    else:
                        prompts_to_process = abstain_prompts[len(existing_results):]
                        print(f"📊 Resume statistics:")
                        print(f"   - Total prompts: {len(abstain_prompts)}")
                        print(f"   - Already completed: {len(existing_results)}")
                        print(f"   - Remaining to process: {len(prompts_to_process)}")
            except Exception as e:
                print(f"⚠️  Warning: Could not load existing results: {e}")
                print(f"   Starting from scratch...")
                existing_results = []
                prompts_to_process = abstain_prompts

        if len(prompts_to_process) > 0:
            print(f"🔄 Processing {len(prompts_to_process)} evaluation requests...")
            from tqdm import tqdm
            
            # Start server for evaluator model if needed (similar to PreciseWikiQA)
            # This ensures we have a server running for the evaluation
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
        else:
            abstains_eval_raw = existing_results
                        
        abstains_eval = eval_utils.jsonify_ans(raw_responses=abstains_eval_raw, \
                                    eval_prompts=abstain_prompts, \
                                        evaluator_model=evaluator_model,\
                                            key="does_believe")
        abstains_eval_res = []
        for o in abstains_eval:
            try:
                abstains_eval_res.append(not o['does_believe'])
            except:
                print(f"Error in eval_answer: {o}")
                # Fallback or exit
                abstains_eval_res.append(False) 

        return abstains_eval_res

    def run_eval(self, overwrite=False, resume=True):
        if os.path.exists(self.res_path) and not overwrite and not resume:
            print(f'{self.TASKNAME} Evaluation already completed')
            res = json.load(open(self.res_path, "r"))
            return res
        
        generations = [json.loads(line) for line in open(self.generations_file_path, "r")]
        eval_results = self.automatic_abstention(generations, resume=resume)
        refusal_rate = sum(eval_results) / len(eval_results)

        res = {
            'model': self.model_name,
            'false_acceptance_rate': 1 - refusal_rate,
            'refusal_rate': refusal_rate,
            'refusal_eval_raw': eval_results,
        }
        # save the results
        with open(self.res_path, 'w') as f:
            json.dump(res, f, indent=4)

        print()
        print(f'*** {self.TASKNAME} Evaluation completed')
        # Print the results 
        print("=" * 80)
        print(f" Evaluation Results for: <<{self.model_name}>>")
        print("=" * 80)
        print(f"  >> Results saved to: {self.res_path}")
        print("-" * 80)
        print(f"  Evaluator for Abstention: {self.evaluator}")
        print("-" * 80)
        print(f"  Total Number of Samples: {len(generations)}")
        print(f"  False Acceptance Rate: {1 - refusal_rate:.3f} %")
        print("-" * 80)

        return res
