# Dream-Coder-Base
## Data
The adaptation data used to train Dream-Coder-Base mostly comprises [OpenCoder](https://huggingface.co/collections/OpenCoder-LLM/opencoder-datasets-672e6db6a0fed24bd69ef1c2), [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu), [Dolmino](https://huggingface.co/datasets/allenai/dolmino-mix-1124), and [DCLM-Baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0). We provide the data filtering script on the DCLM data in `data_filter.py`. We mix the general text, code and math with weights 2:7:1.

## Evaluation
The evaluation code is based on [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness/tree/main), [Qwen-Coder Eval](https://github.com/QwenLM/Qwen2.5-Coder/tree/main/qwencoder-eval) and [Dream](https://github.com/DreamLM/Dream).

To evaluate on bigcodebench, follow the environment installation guideline in [Qwen-Coder Eval](https://github.com/QwenLM/Qwen3-Coder/tree/main/qwencoder-eval/base) and run:
```
bash eval_bcb.sh
```

To evaluate on other tasks, run:
```
cd lm_eval
pip install -e ".[math]"
```
and
```
bash eval_code_base.sh
```