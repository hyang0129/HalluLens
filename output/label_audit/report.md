# Label-quality audit: LLM-judge (Sonnet) vs substring-match

Disagreement = judge and substring-match disagree on correctness. `false_hallucinated` = substring marked a correct answer as a hallucination; `false_correct` = substring marked a wrong answer as correct.

## Per dataset (both models pooled), sorted by disagreement

| dataset | n | substring_acc | judge_acc | **disagree** | false_hallu | false_correct |
|---|--:|--:|--:|--:|--:|--:|
| mmlu | 1998 | 0.145 | 0.358 | **0.261** | 0.237 | 0.024 |
| natural_questions | 2000 | 0.234 | 0.365 | **0.181** | 0.156 | 0.025 |
| sciq | 1999 | 0.659 | 0.782 | **0.168** | 0.146 | 0.022 |
| searchqa | 2000 | 0.506 | 0.626 | **0.165** | 0.142 | 0.022 |
| hotpotqa | 2000 | 0.310 | 0.353 | **0.141** | 0.092 | 0.049 |
| triviaqa | 2000 | 0.701 | 0.690 | **0.059** | 0.024 | 0.035 |
| popqa | 2000 | 0.316 | 0.293 | **0.054** | 0.016 | 0.038 |
| simpleqa | 1732 | 0.057 | 0.054 | **0.020** | 0.009 | 0.011 |

## Per (dataset, model)

| dataset | model | n | disagree | false_hallu | false_correct | unknown |
|---|---|--:|--:|--:|--:|--:|
| hotpotqa | Qwen3-8B | 1000 | 0.133 | 0.083 | 0.050 | 0 |
| hotpotqa | Llama-3.1-8B-Instruct | 1000 | 0.149 | 0.101 | 0.048 | 0 |
| mmlu | Qwen3-8B | 999 | 0.286 | 0.266 | 0.020 | 1 |
| mmlu | Llama-3.1-8B-Instruct | 999 | 0.236 | 0.208 | 0.028 | 1 |
| natural_questions | Qwen3-8B | 1000 | 0.135 | 0.110 | 0.025 | 0 |
| natural_questions | Llama-3.1-8B-Instruct | 1000 | 0.227 | 0.202 | 0.025 | 0 |
| popqa | Qwen3-8B | 1000 | 0.050 | 0.014 | 0.036 | 0 |
| popqa | Llama-3.1-8B-Instruct | 1000 | 0.058 | 0.018 | 0.040 | 0 |
| sciq | Qwen3-8B | 1000 | 0.181 | 0.157 | 0.024 | 0 |
| sciq | Llama-3.1-8B-Instruct | 999 | 0.155 | 0.134 | 0.021 | 1 |
| searchqa | Qwen3-8B | 1000 | 0.141 | 0.121 | 0.020 | 0 |
| searchqa | Llama-3.1-8B-Instruct | 1000 | 0.189 | 0.164 | 0.025 | 0 |
| simpleqa | Qwen3-8B | 866 | 0.021 | 0.007 | 0.014 | 0 |
| simpleqa | Llama-3.1-8B-Instruct | 866 | 0.020 | 0.010 | 0.009 | 0 |
| triviaqa | Qwen3-8B | 1000 | 0.059 | 0.021 | 0.038 | 0 |
| triviaqa | Llama-3.1-8B-Instruct | 1000 | 0.058 | 0.027 | 0.031 | 0 |
