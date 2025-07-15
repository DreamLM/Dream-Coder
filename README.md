<div align="center">

#  Dream-Coder 7B
</div>
<div>
<br>

<div align="center">

[![Blog](https://img.shields.io/badge/blog-black?style=for-the-badge&logo=googledocs&logoColor=white)]()
[![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)]()
[![Twitter](https://img.shields.io/badge/twitter-white?style=for-the-badge&logo=X&logoColor=white&color=black)]()
[![Hugging Face Model](https://img.shields.io/badge/models-%23000000?style=for-the-badge&logo=huggingface&logoColor=000&logoColor=white)]()

</div>
</div>

## Introduction
Dream-Coder 7B is a **diffusion LLM for code** trained exclusively on open-source data across its development stages—adaptation, supervised fine-tuning, and reinforcement learning. It achieves an impressive 21.4% pass@1 on LiveCodeBench (2410-2505), outperforming other open-source diffusion LLMs by a wide margin.

<p align="center">
    <img src="assets/results.png" width="90%"/>
<p>

## Features
### Flexible Code Generation
We observe Dream-Coder 7B exhibits emergent any-order generation that adaptively determines its decoding style based on the coding task. For example, Dream-Coder 7B Instruct displays patterns such as:

<table>
  <tr>
    <td rowspan="2" align="center">
      <img src="assets/demo1.gif" alt="Figure 1" width="90%">
      <p align="center"><em>Sketch-First Generation</em></p>
    </td>
    <td align="center">
      <img src="assets/demo2.gif" alt="Figure 2" width="90%">
      <p align="center"><em>Left-to-Right Generation</em></p>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/demo3.gif" alt="Figure 3" width="90%">
      <p align="center"><em>Iterative Back-and-Forth Generation</em></p>
    </td>
  </tr>
</table>

These demos were collected using consistent sampling parameters: `temperature=0.1, diffusion_steps=512, max_new_tokens=512, alg="entropy", top_p=1.0, alg_temp=0.0, eos_penalty=3.0`.

### Variable-Length Code Infilling
We also introduce an infilling variant, **DreamOn-7B**, that naturally adjusts the length of masked spans during generation for variable-length code infilling. For more details, please refer to our accompanying [blog post]().

## Quickstart
To get start with,
please install `transformers==4.46.2` and `torch==2.5.1`. Here is an example to use Dream-Coder 7B:

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_path = "Dream-org/Dream-Coder-v0-Instruct-7B"
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = model.to("cuda").eval()

messages = [
    {"role": "user", "content": "Write a quick sort algorithm."}
]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
)
input_ids = inputs.input_ids.to(device="cuda")
attention_mask = inputs.attention_mask.to(device="cuda")

output = model.diffusion_generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=768,
    output_history=True,
    return_dict_in_generate=True,
    steps=768,
    temperature=0.1,
    top_p=0.95,
    alg="entropy",
    alg_temp=0.,
)
generations = [
    tokenizer.decode(g[len(p) :].tolist())
    for p, g in zip(input_ids, output.sequences)
]

print(generations[0].split(tokenizer.eos_token)[0])
```

## Releases
- [x] [Our blog]() and [model weights]()
- [ ] Data processing scripts
- [ ] Training recipes based on [verl](https://github.com/volcengine/verl)

## Citation
```
@misc{dreamcoder2025,
    title = {Dream-Coder 7B},
    url = {https://hkunlp.github.io/blog/2025/dream-coder},
    author = {Xie, Zhihui and Ye, Jiacheng and Zheng, Lin and Gao, Jiahui and Dong, Jingwei and Wu, Zirui and Zhao, Xueliang and Gong, Shansan and Jiang, Xin and Li, Zhenguo and Kong, Lingpeng},
    year = {2025}
}
```
