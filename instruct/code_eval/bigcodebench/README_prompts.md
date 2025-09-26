# BigCodeBench 提示提取脚本 (HuggingFace格式)

这些脚本用于从BigCodeBench基准测试中提取所有的提示(prompts)并整理到HuggingFace标准格式的jsonl文件中，方便进行模型训练和评估。

## 脚本说明

### 1. `prepare_prompts.py` - 单独提取脚本
用于提取特定子集和类型的提示，输出HuggingFace格式。

**用法示例：**
```bash
# 提取所有instruct提示 (默认)
python prepare_prompts.py

# 提取hard子集的complete提示
python prepare_prompts.py --subset hard --split complete

# 指定输出文件
python prepare_prompts.py --output my_prompts.jsonl

# 提取所有类型的提示
python prepare_prompts.py --all
```

### 2. `prepare_all_prompts.py` - 综合提取脚本
将所有BigCodeBench提示合并到一个统一的HuggingFace格式jsonl文件中。

**用法示例：**
```bash
# 生成包含所有提示的文件 (默认)
python prepare_all_prompts.py

# 指定输出文件
python prepare_all_prompts.py --output all_bigcodebench_prompts_hf.jsonl

# 不包含解答和测试代码
python prepare_all_prompts.py --no-solutions
```

## 数据集说明

### 子集类型
- **full**: 包含完整的1140个任务
- **hard**: 包含148个更具挑战性的任务

### 提示类型
- **complete**: 代码补全提示，包含函数签名和文档字符串
- **instruct**: 指令式提示，只包含自然语言描述

## 生成的文件

### 单独文件 (使用 `prepare_prompts.py --all`)
- `prompts_full_complete.jsonl` - 完整数据集的代码补全提示 (1140个，7.31MB)
- `prompts_full_instruct.jsonl` - 完整数据集的指令提示 (1140个，6.24MB)
- `prompts_hard_complete.jsonl` - 困难数据集的代码补全提示 (148个，1.11MB)
- `prompts_hard_instruct.jsonl` - 困难数据集的指令提示 (148个，0.96MB)

### 综合文件 (使用 `prepare_all_prompts.py`)
- `all_bigcodebench_prompts_hf.jsonl` - 包含所有2576个提示的统一文件 (15.63MB)

## 数据格式 (HuggingFace标准)

每个jsonl文件中的每一行都是一个JSON对象，采用HuggingFace标准格式：

```json
{
    "task_id": "BigCodeBench/0",
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful programming assistant. Please provide a complete Python solution for the given task."
        },
        {
            "role": "user",
            "content": "具体的任务描述和要求..."
        }
    ],
    "meta_info": {
        "split": "instruct",
        "subset": "full",
        "entry_point": "task_func",
        "libs": "['random', 'itertools']",
        "prompt_id": "BigCodeBench/0_full_instruct",
        "original_prompt": "原始提示内容...",
        "test": "测试代码...",
        "canonical_solution": "标准解答...",
        "doc_struct": "文档结构..."
    }
}
```

### 字段说明

#### 主要字段
- `task_id`: BigCodeBench中的任务ID
- `messages`: HuggingFace标准的对话格式
  - `role`: 消息角色 ("system" 或 "user")
  - `content`: 消息内容
- `meta_info`: 包含所有元数据信息的字典

#### meta_info 字段详解
- `split`: 提示类型 ("complete" 或 "instruct")
- `subset`: 数据子集 ("full" 或 "hard")
- `entry_point`: 函数入口点名称
- `libs`: 所需的库列表
- `prompt_id`: 唯一标识符 (格式: `{task_id}_{subset}_{split}`)
- `original_prompt`: 原始BigCodeBench提示内容
- `test`: 单元测试代码 (可选)
- `canonical_solution`: 标准解决方案 (可选)
- `doc_struct`: 文档结构信息 (可选)

### Messages格式差异

#### Instruct类型
```json
"messages": [
    {
        "role": "system",
        "content": "You are a helpful programming assistant. Please provide a complete Python solution for the given task."
    },
    {
        "role": "user",
        "content": "任务的自然语言描述..."
    }
]
```

#### Complete类型
```json
"messages": [
    {
        "role": "system",
        "content": "You are a helpful programming assistant. Please complete the following Python function."
    },
    {
        "role": "user",
        "content": "Please complete this Python function:\n\n函数签名和文档字符串..."
    }
]
```

## 统计信息

### 数据量统计
- **总提示数**: 2576个
  - full-complete: 1140个
  - full-instruct: 1140个
  - hard-complete: 148个
  - hard-instruct: 148个

### 文件大小
- **单独文件总计**: ~15.6MB
- **综合文件**: 15.63MB
- **平均每个提示**: ~6KB

### 提示类型差异
- **Complete提示**: 包含函数签名、参数、文档字符串，适合代码补全任务
- **Instruct提示**: 只包含自然语言描述和要求，适合指令驱动的代码生成任务

## 与HuggingFace生态的兼容性

这种格式完全兼容HuggingFace的训练和推理工具链：

### 训练
```python
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("json", data_files="all_bigcodebench_prompts_hf.jsonl")

# 直接用于训练Chat模型
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    ...
)
```

### 推理
```python
# 直接使用messages格式进行推理
response = model.chat(messages=data["messages"])
```

## 依赖要求

确保已安装BigCodeBench包：
```bash
pip install bigcodebench --upgrade
```

## 注意事项

1. 首次运行时，脚本会自动下载BigCodeBench数据集到缓存目录
2. hard子集是full子集的一个更具挑战性的子集，任务有重叠
3. 生成的文件使用UTF-8编码，确保正确处理多语言字符
4. 所有文件都是jsonl格式，每行一个JSON对象，便于流式处理
5. 新格式文件稍大，因为包含了HuggingFace的messages结构和完整的元数据

## 使用建议

- **模型训练**: 使用综合文件`all_bigcodebench_prompts_hf.jsonl`进行大规模训练
- **特定评估**: 根据需要选择特定的子集和提示类型
- **模型类型选择**:
  - 基础模型：使用complete类型进行代码补全训练
  - 指令微调模型：使用instruct类型进行对话式训练
- **评估优先级**: 建议优先使用hard子集，因为它更能反映真实编程任务的复杂性

## 版本更新

- **v2.0**: 更新为HuggingFace标准格式，添加messages结构和meta_info整合