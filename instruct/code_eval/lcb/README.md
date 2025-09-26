# Code Generation with Diffusion Models

## Installation

We recommend creating a dedicated virtual environment for evaluations:
```bash
python3 -m venv envs/codegen_eval
source envs/codegen_eval/bin/activate
```

Please make sure you have `torch>=2.4` and `transformers` installed in the virtual environment. To install all dependencies for evaluations, run the following command:
```bash
pip install -r requirements.txt
pip install --no-deps evalplus==0.3.1
```

## Usage on EvalPlus

To run the evaluation on `HumanEval-plus` with `Dream-v0-Base-7B`, use the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python run_evalplus.py \
    --model Dream-org/Dream-v0-Base-7B \
    --dataset humaneval \
    --temperature 0.1 \
    --diffusion_steps 512 \
    --max_new_tokens 512 \
    --diffusion_remask_alg maskgit_plus \
    --diffusion_remask_alg_temp 0.0
```

## Instruct model

Instruct models can be used by setting the `--use_instruct_prompt` flag and usually give better performance. For example, to run the evaluation on `HumanEval-plus` with `Dream-v0-Instruct-7B`, use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python run_evalplus.py \
    --model Dream-org/Dream-v0-Instruct-7B \
    --dataset humaneval \
    --use_instruct_prompt \
    --temperature 0.1 \
    --diffusion_steps 512 \
    --max_new_tokens 512 \
    --diffusion_remask_alg maskgit_plus \
    --diffusion_remask_alg_temp 0.0
```

We've tested the code on base models with `humaneval` datasets. With the hyper-parameters above on an RTX 3090, it takes about 20G memory and 2 minutes to generate 1 solution for 1 problem. To reduce computational cost while minimizing the performance drop, we can decrease both `diffusion_steps` and `max_new_tokens` to 320.

## Usage on LiveCodeBench

LiveCodeBench includes a lot of competition coding problems, which are very challenging for LLMs and suitable for evaluating the performance of recent stronger code generation models. Although it supports both base and instruct models, we found that the performance of base models is usually not good. Thus only the instruct models, especially the DREAM-v0-Instruct-7B, are used for evaluation.

To run the evaluation on LiveCodeBench with `Dream-v0-Instruct-7B`, use the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python run_lcb.py --n 1 --difficulty easy --model Dream-org/Dream-v0-Instruct-7B --use_instruct_prompt --diffusion_steps 512 --max_new_tokens 512 --evaluate --diffusion_remask_alg maskgit_plus --temperature 0.1 --use_cache
```

The prompt length is usually much longer than EvalPlus and thus might consume more memory. Sometimes it might also run out of memory. We can decrease `diffusion_steps` and `max_new_tokens` to 320 to reduce the memory consumption.
