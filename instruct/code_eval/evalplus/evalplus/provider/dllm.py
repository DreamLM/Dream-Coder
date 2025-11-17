from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

from evalplus.provider.base import DecoderBase
from evalplus.provider.utility import (
    extra_eos_for_direct_completion,
    make_raw_chat_prompt,
)


class DLLMDecoder(DecoderBase):
    def __init__(
        self,
        name: str,
        dataset: str,
        force_base_prompt: bool = False,
        attn_implementation: str = "eager",
        device_map: str = None,
        gguf_file: str = None,
        **kwargs,
    ):
        # extract custom kwargs for dllm
        self.diffusion_steps = kwargs.pop("diffusion_steps", 256)
        self.top_p = kwargs.pop("top_p", 0.9)
        self.alg = kwargs.pop("alg", "entropy")
        self.alg_temp = kwargs.pop("alg_temp", 0.0)

        self.fast_dllm = kwargs.pop("fast_dllm", False)

        super().__init__(name=name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        kwargs = {
            "device_map": device_map,
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": getattr(torch, self.dtype),
            "attn_implementation": attn_implementation,  # "eager", "flash_attention_2", "sdpa"
            "gguf_file": gguf_file,
        }

        self.skip_special_tokens = True

        print(f"{kwargs = }")

        self.force_base_prompt = force_base_prompt

        # gguf format embeds tokenizer and is not compatible with hf tokenizer `use_fast` param
        tokenizer_kwargs = {"trust_remote_code": self.trust_remote_code}
        if gguf_file is not None:
            tokenizer_kwargs["gguf_file"] = gguf_file
        self.tokenizer = AutoTokenizer.from_pretrained(name, **tokenizer_kwargs)
        if self.is_direct_completion():  # no chat template
            self.eos += extra_eos_for_direct_completion(dataset)
        else:  # with chat template
            self.eos += ["\n```\n"]

        print(f"{self.eos = }")
        if self.fast_dllm:
            from src.inference.fast_dllm.modeling_dream import DreamModel
            from src.inference.fast_dllm.generation_utils_block import (
                DreamGenerationMixin,
            )
            import types

            self.model = DreamModel.from_pretrained(name, **kwargs)
            self.model.diffusion_generate = types.MethodType(
                DreamGenerationMixin.diffusion_generate, self.model
            )
            self.model._sample = types.MethodType(
                DreamGenerationMixin._sample, self.model
            )
        else:
            self.model = AutoModel.from_pretrained(name, **kwargs)
        self.model = self.model.to(self.device)

    def is_direct_completion(self) -> bool:
        return self.force_base_prompt or self.tokenizer.chat_template is None

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        if self.temperature == 0:
            assert not do_sample
            assert num_samples == 1

        prompt = (
            prompt
            if self.is_direct_completion()
            else make_raw_chat_prompt(
                prompt, self.instruction_prefix, self.response_prefix, self.tokenizer
            )
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_tokens = inputs.input_ids
        attn_mask = inputs.attention_mask

        kwargs = {}
        if do_sample:
            kwargs["top_p"] = 0.95
            kwargs["temperature"] = self.temperature

        outputs = self.model.diffusion_generate(
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

        gen_strs = self.tokenizer.batch_decode(
            outputs[:, input_tokens.size(-1) :],
            skip_special_tokens=self.skip_special_tokens,
        )
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index].replace("\t", "    "))

        print(f"Context:\n{prompt}\n\nGenerated:\n{outputs[0]}")

        print(self.max_new_tokens, self.diffusion_steps, outputs)
        return outputs
