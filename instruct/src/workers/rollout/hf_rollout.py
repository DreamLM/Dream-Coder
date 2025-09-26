# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single GPU model.
Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model to perform generation.
"""
import contextlib
import gc
import torch
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from .base import BaseRollout


__all__ = ["HFRollout"]


class HFRollout(BaseRollout):

    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        self.module.eval()

        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(
            batch_size // self.config.get("micro_batch_size", batch_size), 1
        )
        batch_prompts = prompts.chunk(chunks=num_chunks)

        # generate in mini-batches
        param_ctx = contextlib.nullcontext()
        if isinstance(self.module, FSDP):
            param_ctx = FSDP.summon_full_params(
                self.module, writeback=False, recurse=False
            )
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
        with param_ctx, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = [self._generate_minibatch(p) for p in batch_prompts]

        # empty cache before compute old_log_prob
        gc.collect()
        torch.cuda.empty_cache()

        self.module.train()
        return DataProto.concat(output)

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        pad_token_id = prompts.meta_info["pad_token_id"]

        batch_size = idx.size(0)
        prompt_length = idx.size(1)

        # make sampling args can be overriden by inputs
        response_length = prompts.meta_info.get(
            "response_length", self.config.response_length
        )
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 0.95))
        diffusion_steps = prompts.meta_info.get(
            "diffusion_steps", self.config.get("diffusion_steps", 512)
        )
        alg = prompts.meta_info.get("alg", self.config.get("alg", "maskgit_plus"))
        alg_temp = prompts.meta_info.get("alg_temp", self.config.get("alg_temp", 0.0))
        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        num_return_sequences = prompts.meta_info.get("n", self.config.n)
        return_batch_size = self.config.get("return_batch_size", 4)
        assert batch_size == 1, "can only support batch_size=1 for now"

        # generate in single batch
        if batch_size == 1:
            original_prompt_length = prompt_length
            original_idx = idx.clone()
            original_attention_mask = attention_mask.clone()

            pad_mask = idx == pad_token_id
            idx = idx[~pad_mask].unsqueeze(0)
            attention_mask = attention_mask[~pad_mask].unsqueeze(0)
            prompt_length = idx.shape[1]

        seqs = []
        for i in range(num_return_sequences // return_batch_size):
            torch.manual_seed(i)
            output = self.module.diffusion_generate(
                idx,
                attention_mask=attention_mask,
                max_new_tokens=response_length,
                output_history=False,
                return_dict_in_generate=True,
                steps=diffusion_steps,
                temperature=temperature,
                top_p=top_p,
                alg=alg,
                alg_temp=alg_temp,
                num_return_sequences=return_batch_size,
            )
            # TODO: filter out the seq with no answers like ds-chat
            seq = output.sequences
            seqs.append(seq)
        seq = torch.cat(seqs, dim=0)

        if batch_size == 1:
            seq = torch.cat(
                (
                    torch.full(
                        (num_return_sequences, original_prompt_length - prompt_length),
                        pad_token_id,
                        device=seq.device,
                        dtype=seq.dtype,
                    ),
                    seq,
                ),
                dim=1,
            )
            idx = original_idx
            attention_mask = original_attention_mask
            prompt_length = original_prompt_length

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(
                size=(batch_size, delta_length), device=seq.device, dtype=seq.dtype
            )
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)

        assert seq.shape[1] == sequence_length

        prompt = seq[:, :prompt_length]  # (bs, prompt_length)
        response = seq[:, prompt_length:]  # (bs, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(
            1, response_length + 1, device=position_ids.device
        )
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        # NOTE: we use full attention mask on both non-eos and eos tokens
        # response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        response_attention_mask = torch.ones_like(response, dtype=attention_mask.dtype)

        # repeat interleave the attention mask by self.config.n
        attention_mask = torch.repeat_interleave(
            attention_mask, num_return_sequences, dim=0
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # repeat interleave the position_ids by self.config.n
        position_ids = torch.repeat_interleave(
            position_ids, num_return_sequences, dim=0
        )

        # NOTE: just keep it the same as SFT
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size * num_return_sequences,
        )
        meta_info = prompts.meta_info.copy()

        return DataProto(batch=batch, meta_info=meta_info)
