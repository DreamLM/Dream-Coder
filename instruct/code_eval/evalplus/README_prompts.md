# EvalPlus 评估提示准备脚本

本文件夹包含用于准备 EvalPlus 基准测试提示的脚本，支持 HumanEval+、MBPP+ 和 EvalPerf 数据集。

## 脚本说明

### 1. `prepare_eval_prompts.py` - 完整版本

准备包含所有元数据的完整评估提示，适用于深度分析和评估。

**用法示例：**
```bash
# 生成包含所有数据集的完整提示
python3 prepare_eval_prompts.py --output all_eval_prompts.jsonl

# 只生成 HumanEval+ 和 MBPP+
python3 prepare_eval_prompts.py --output humaneval_mbpp.jsonl --no-evalperf

# 使用 mini 版本（更快）
python3 prepare_eval_prompts.py --mini --output eval_prompts_mini.jsonl

# 只查看摘要，不保存文件
python3 prepare_eval_prompts.py --summary-only
```

**输出格式：**
```json
{
  "task_id": "HumanEval/0",
  "dataset": "humaneval_plus",
  "prompt": "def has_close_elements(...)...",
  "entry_point": "has_close_elements",
  "canonical_solution": "...",
  "test_input_count": 1006,
  "contract": "...",
  "metadata": {
    "atol": 0,
    "has_base_input": true,
    "has_plus_input": true
  }
}
```

### 2. `prepare_simple_prompts.py` - 简化版本

只提取代码生成必需的基本字段，更适合 LLM 推理。

**用法示例：**
```bash
# 生成简化的所有数据集提示
python3 prepare_simple_prompts.py --output simple_prompts.jsonl

# 只生成 HumanEval+
python3 prepare_simple_prompts.py --humaneval-only --output humaneval_only.jsonl

# 只生成 MBPP+
python3 prepare_simple_prompts.py --mbpp-only --output mbpp_only.jsonl

# 排除 EvalPerf
python3 prepare_simple_prompts.py --no-evalperf --output no_evalperf.jsonl
```

**输出格式：**
```json
{
  "task_id": "HumanEval/0",
  "dataset": "humaneval_plus",
  "prompt": "def has_close_elements(...)...",
  "entry_point": "has_close_elements"
}
```

### 3. `prepare_instruct_prompts.py` - Instruct模型专用 🆕

专门为instruct模型准备HuggingFace messages格式的评估提示，完全兼容chat template。

**用法示例：**
```bash
# 生成默认模板的instruct格式
python3 prepare_instruct_prompts.py --output instruct_default.jsonl

# 使用性能优化模板
python3 prepare_instruct_prompts.py --template perf-instruct --output instruct_perf.jsonl

# 使用中文模板
python3 prepare_instruct_prompts.py --template chinese --output instruct_chinese.jsonl

# 包含response模板（用于few-shot或continuation）
python3 prepare_instruct_prompts.py --include-response --output instruct_with_response.jsonl

# 使用自定义instruction
python3 prepare_instruct_prompts.py --custom-instruction "Write Python code to solve:" --output custom.jsonl

# 查看所有可用模板
python3 prepare_instruct_prompts.py --show-templates
```

**输出格式：**
```json
{
  "task_id": "HumanEval/0",
  "dataset": "humaneval_plus",
  "messages": [
    {
      "role": "user",
      "content": "Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\ndef has_close_elements(...)...\n```"
    },
    {
      "role": "assistant",
      "content": "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:\n```python\n"
    }
  ],
  "entry_point": "has_close_elements",
  "instruction_template": "default",
  "metadata": {
    "original_prompt": "def has_close_elements(...)...",
    "has_response_template": true
  }
}
```

### 4. `demo_usage.py` - 使用演示

演示如何使用生成的各种格式数据，包括与transformers库的集成。

```bash
python3 demo_usage.py
```

## 可用的Instruction模板

| 模板名称 | 描述 | 适用场景 |
|---------|------|----------|
| `default` | 标准英文模板 | 通用代码生成任务 |
| `perf-instruct` | 性能优化英文模板 | 强调效率的任务 |
| `perf-CoT` | 思维链+性能优化模板 | 复杂算法问题 |
| `chinese` | 中文模板 | 中文模型 |
| `chinese-perf` | 中文性能优化模板 | 中文模型+性能要求 |

## 命令行参数

### 通用参数
- `--output, -o`: 输出文件路径
- `--summary-only`: 只显示摘要，不保存文件
- `--mini`: 使用 mini 版本数据集（如果可用）

### 数据集选择
- `--no-humaneval`: 不包含 HumanEval+ 数据集
- `--no-mbpp`: 不包含 MBPP+ 数据集
- `--no-evalperf`: 不包含 EvalPerf 数据集
- `--humaneval-only`: 只包含 HumanEval+ 数据集
- `--mbpp-only`: 只包含 MBPP+ 数据集
- `--evalperf-only`: 只包含 EvalPerf 数据集

### Instruct模型特有参数
- `--template, -t`: 选择instruction模板
- `--custom-instruction`: 使用自定义instruction prefix
- `--include-response`: 包含assistant response的开头模板
- `--show-templates`: 显示所有可用模板

## 数据集统计

- **HumanEval+**: 164 个编程任务
- **MBPP+**: 378 个编程任务
- **EvalPerf**: 118 个性能测试任务
- **总计**: 660 个任务

## 生成的文件

运行脚本后会生成以下文件：

1. `eval_prompts_all.jsonl` - 包含所有数据集的完整提示（660个任务）
2. `simple_eval_prompts.jsonl` - 简化版本的所有数据集提示
3. `humaneval_prompts.jsonl` - 只包含 HumanEval+ 的提示
4. `instruct_default.jsonl` - Instruct格式的默认模板提示
5. `instruct_perf.jsonl` - Instruct格式的性能优化模板提示
6. `instruct_chinese.jsonl` - Instruct格式的中文模板提示

## 使用建议

### 用于 Base 模型代码生成
推荐使用 `prepare_simple_prompts.py`：

```bash
python3 prepare_simple_prompts.py --humaneval-only --output humaneval_for_base_model.jsonl
```

### 用于 Instruct 模型代码生成 ⭐
推荐使用 `prepare_instruct_prompts.py`：

```bash
# 标准用法
python3 prepare_instruct_prompts.py --template default --output instruct_standard.jsonl

# 性能优化任务
python3 prepare_instruct_prompts.py --template perf-instruct --output instruct_performance.jsonl

# 中文模型
python3 prepare_instruct_prompts.py --template chinese --output instruct_chinese.jsonl
```

### 与HuggingFace Transformers集成

```python
from transformers import AutoTokenizer
import json

# 加载tokenizer和数据
tokenizer = AutoTokenizer.from_pretrained("your-instruct-model")

with open("instruct_default.jsonl") as f:
    task = json.loads(f.readline())

# 使用chat template格式化
formatted_prompt = tokenizer.apply_chat_template(
    task["messages"],
    tokenize=False,
    add_generation_prompt=True
)

# 进行推理...
```

### 用于详细评估分析
推荐使用 `prepare_eval_prompts.py`，包含完整的元数据用于分析：

```bash
python3 prepare_eval_prompts.py --output complete_eval_data.jsonl
```

### 快速测试
使用 mini 版本进行快速原型验证：

```bash
python3 prepare_instruct_prompts.py --mini --humaneval-only --summary-only
```

## 依赖要求

确保已安装 EvalPlus 包：

```bash
pip install evalplus
# 或者
pip install -e .  # 如果在 evalplus 源码目录
```

## 输出文件格式

所有输出文件都是 JSONL（JSON Lines）格式，每行一个 JSON 对象，可以用标准工具处理：

```bash
# 查看文件内容
head -n 5 instruct_default.jsonl

# 统计任务数量
wc -l instruct_default.jsonl

# 美化显示第一个任务
head -n 1 instruct_default.jsonl | python3 -m json.tool
```

## 特别说明

### Instruct模型的优势

- **标准化格式**: 完全兼容HuggingFace的chat template
- **多样化模板**: 支持不同场景和语言的instruction模板
- **易于集成**: 直接配合transformers库使用
- **遵循最佳实践**: 基于EvalPlus官方的instruction格式