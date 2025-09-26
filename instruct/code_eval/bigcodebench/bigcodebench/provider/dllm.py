from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

from bigcodebench.provider.base import DecoderBase
from bigcodebench.provider.utility import (
    extra_eos_for_direct_completion,
    make_raw_chat_prompt,
)


class DLLMDecoder(DecoderBase):
    def __init__(
        self,
        name: str,
        dataset: str,
        attn_implementation: str = "eager",
        **kwargs,
    ):
        # extract custom kwargs for dllm
        self.diffusion_steps = kwargs.pop("diffusion_steps", 256)
        self.top_p = kwargs.pop("top_p", 0.9)
        self.alg = kwargs.pop("alg", "entropy")
        self.alg_temp = kwargs.pop("alg_temp", 0.0)

        super().__init__(name=name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        kwargs = {
            "device_map": "auto",
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": getattr(torch, self.dtype),
            "attn_implementation": attn_implementation,  # "eager", "flash_attention_2", "sdpa"
            "revision": self.revision,
        }
        self.skip_special_tokens = True

        print(f"{kwargs = }")

        self.tokenizer = AutoTokenizer.from_pretrained(
            name,
            use_fast=False,
            legacy=self.tokenizer_legacy,
            trust_remote_code=self.trust_remote_code,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # assume the model is decoder-only
        self.tokenizer.padding_side = "left"

        if self.is_direct_completion():  # no chat template
            self.eos += extra_eos_for_direct_completion(dataset)
        else:  # with chat template
            if self.prefill and "```" in self.response_prefix:
                self.eos += ["\n```\n"]

        print(f"{self.eos = }")
        self.model = AutoModel.from_pretrained(name, **kwargs)

    def is_direct_completion(self) -> bool:
        return self.direct_completion or self.tokenizer.chat_template is None

    @torch.inference_mode()
    def codegen(
        self, prompts: List[str], do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        prompts = [
            (
                prompt
                if self.is_direct_completion()
                else make_raw_chat_prompt(
                    prompt,
                    self.subset,
                    self.split,
                    self.instruction_prefix,
                    self.response_prefix,
                    self.tokenizer,
                    self.direct_completion,
                )
            )
            for prompt in prompts
        ]

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        input_tokens = inputs.input_ids
        attn_mask = inputs.attention_mask

        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = self.temperature

        ret = self.model.diffusion_generate(
            input_tokens,
            attention_mask=attn_mask,
            max_new_tokens=self.max_new_tokens,
            output_history=False,
            return_dict_in_generate=True,
            steps=self.diffusion_steps,
            temperature=self.temperature,
            top_p=self.top_p,
            alg=self.alg,
            alg_temp=self.alg_temp,
        ).sequences

        # Reshape ret into a list of lists, each sublist containing num_samples elements
        ret_chunks = [ret[i : i + num_samples] for i in range(0, len(ret), num_samples)]

        all_outputs = []
        # Process each chunk in ret_chunks
        for i, ret_chunk in enumerate(ret_chunks):
            gen_strs = self.tokenizer.batch_decode(
                ret_chunk[:, input_tokens[i].size(-1) :],
                skip_special_tokens=self.skip_special_tokens,
            )
            outputs = []
            for output in gen_strs:
                min_index = 10000
                for eos in self.eos:
                    if eos in output:
                        min_index = min(min_index, output.index(eos))
                outputs.append(output[:min_index].replace("\t", "    "))
            all_outputs.append(outputs)

        print(f"Context:\n{prompts[0]}\n\nGenerated:\n{all_outputs[0][0]}")
        return all_outputs
