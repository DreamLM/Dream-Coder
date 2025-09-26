# Evaluation Toolkit
This toolkit integrates official benchmarks from [evalplus](https://github.com/evalplus/evalplus), [bigcodebench](https://github.com/bigcode-project/bigcodebench), [livecodebench](https://github.com/LiveCodeBench/LiveCodeBench), and [cruxeval](https://github.com/facebookresearch/cruxeval). To get started, navigate to each respective subfolder and follow the provided installation instructions.

**Note:** For cruxeval, we use a dedicated script for generation due to compatibility issues with their standard generation pipeline.

```bash
# humaneval
PYTHONPATH=. evalplus.evaluate --model $CKPT_DIR --trust_remote_code True --max_new_tokens 512 --diffusion_steps 512 --dataset humaneval --backend dllm --temperature 0.1

# mbpp
PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 evalplus.evaluate --model $CKPT_DIR --trust_remote_code True --max_new_tokens 512 --diffusion_steps 512 --dataset mbpp --backend dllm --temperature 0.1

# bigcodebench
bigcodebench.evaluate   --model $CKPT_DIR   --execution gradio   --split instruct   --subset hard   --backend dllm   --diffusion_steps 1024   --max_new_tokens 1024   --top_p 0.9   --alg entropy   --alg_temp 0.0   --temperature 0.1   --trust_remote_code   --bs 1

# livecodebench
cd code_eval/lcb; python run_lcb.py \
--n 1 \
--model $CKPT_DIR \
--start_date 2024-10-01 \
--use_instruct_prompt \
--diffusion_steps 768 \
--max_new_tokens 768 \
--evaluate \
--diffusion_remask_alg entropy \
--temperature 0.1 \
--use_cache

# cruxeval
bash code_eval/cruxeval/eval_cruxeval.sh $CKPT_DIR input_cot cruxeval_entropy
bash code_eval/cruxeval/eval_cruxeval.sh $CKPT_DIR output_cot cruxeval_entropy
```