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
Single Process Actor
"""

from typing import Tuple

import numpy as np
import torch
import verl.utils.torch_functional as verl_F
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.workers.actor import BasePPOActor

from src.diffllm.train_utils import context_adaptive_reweight

# from src.trainer.ppo import core_algos

__all__ = ["DataParallelPPOActor"]


def compute_policy_loss(old_log_prob, log_prob, advantages, cliprange, cliprange_high):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        cliprange_high: (float)
            The high clip range used in DAPO. See https://arxiv.org/abs/2503.14476

    Returns:
        token_pg_loss: `(bs, response_length)`
            token-level policy gradient loss computed via PPO
        token_pg_clipfrac: `(bs, response_length)`
            token-level fraction of policy gradient loss being clipped
        token_ppo_kl: `(bs, response_length)`
            token-level KL divergence computed via PPO
    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    token_ppo_kl = -negative_approx_kl

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange_high)

    token_pg_loss = torch.max(pg_losses, pg_losses2)
    token_pg_clipfrac = torch.gt(pg_losses2, pg_losses).float()
    return token_pg_loss, token_pg_clipfrac, token_ppo_kl


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get("use_remove_padding", False)
        print(f"Actor use_remove_padding={self.use_remove_padding}")
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get(
                "use_torch_compile", True
            )  #  use torch compile by default
            else verl_F.entropy_from_logits
        )

    def _forward_micro_batch(
        self, micro_batch, temperature
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            masked_input_ids = micro_batch["masked_input_ids"]
            attention_mask = micro_batch["attention_mask"].bool()
            position_ids = micro_batch["position_ids"]
            responses = micro_batch["responses"]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )

            if self.use_remove_padding:
                raise NotImplementedError("use_remove_padding is not supported")
            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(
                    input_ids=masked_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[
                    :, -response_length - 1 : -1, :
                ]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, responses)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(
                max_norm=self.config.grad_clip
            )
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_module.parameters(), max_norm=self.config.grad_clip
            )
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info[
            "temperature"
        ]  # temperature must be in the data.meta_info to avoid slient error

        select_keys = [
            "responses",
            "attention_mask",
            "position_ids",
            "masked_input_ids",
            "loss_mask",
            "t",
        ]
        batch = data.select(batch_keys=select_keys).batch
        micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(
                    micro_batch, temperature=temperature
                )
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        return log_probs

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info[
            "temperature"
        ]  # temperature must be in the data.meta_info to avoid slient error
        pad_token_id = data.meta_info["pad_token_id"]

        select_keys = [
            "responses",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "masked_input_ids",
            "loss_mask",
            "t",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch

        # NOTE: since ppo_epoches is at the inner loop, we need to transpose the batch
        epoch_batches = (
            batch.reshape(-1, self.config.ppo_epochs, self.config.mask_epochs)
            .transpose(0, 1)
            .transpose(1, 2)
        )  # (ppo_epochs, mask_epochs, batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            dataloader = (
                epoch_batches[epoch].reshape(-1).split(self.config.ppo_mini_batch_size)
            )

            for batch_idx, data in enumerate(dataloader):
                if self.config.perbatch_cutoff:
                    effective_response_lens = (data["responses"] != pad_token_id).sum(
                        -1
                    )
                    kept_response_len = np.random.choice(effective_response_lens.cpu())
                    truncate_len = data["responses"].shape[-1] - kept_response_len

                    if truncate_len > 0:
                        new_batch = {k: v[:, :-truncate_len] for k, v in data.items()}
                        data = TensorDict(new_batch, batch_size=data.batch_size)

                # split batch into micro_batches
                mini_batch = data
                self.gradient_accumulation = (
                    self.config.ppo_mini_batch_size
                    // self.config.ppo_micro_batch_size_per_gpu
                )
                # split batch into micro_batches
                micro_batches = mini_batch.split(
                    self.config.ppo_micro_batch_size_per_gpu
                )

                self.actor_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all hardwares
                    if isinstance(data, DataProto):
                        data = {
                            **data.batch.to(torch.cuda.current_device()),
                            **data.non_tensor_batch,
                        }
                    else:
                        data = data.to(
                            torch.cuda.current_device()
                        )  # actor device is cpu when using offload
                    old_log_prob = data["old_log_probs"]
                    advantages = data["advantages"]
                    loss_mask = data["loss_mask"]
                    response_length = data["responses"].shape[-1]
                    t = data["t"]

                    use_t_in_loss = True
                    if self.config.context_adaptive_reweight_p > 0:
                        seq_len = data["masked_input_ids"].shape[-1]
                        weight_matrix = context_adaptive_reweight(
                            seq_len, cart_p=self.config.context_adaptive_reweight_p
                        )
                        _weight_matrix = weight_matrix[:seq_len, :seq_len].to(
                            loss_mask.device
                        )
                        _weight_matrix = _weight_matrix[
                            -response_length:, -response_length:
                        ]
                        non_mask = ~loss_mask.to(
                            loss_mask.device
                        )  # loss_mask indicates where is mask
                        weight = (
                            non_mask.type_as(_weight_matrix)
                            .matmul(_weight_matrix)
                            .masked_fill(non_mask, 0)
                        )
                        loss_mask = loss_mask.to(weight.dtype) * weight
                        use_t_in_loss = False

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_high = self.config.get("clip_ratio_high", clip_ratio)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch=data, temperature=temperature
                    )

                    if self.config.no_loss_on_pad:
                        loss_mask = loss_mask * (data["responses"] != pad_token_id).to(
                            loss_mask.dtype
                        )

                    # # NOTE: debug
                    # response_mask = data["attention_mask"][:, -response_length:]
                    # old_log_prob *= loss_mask.to(old_log_prob.dtype)
                    # log_prob *= loss_mask.to(log_prob.dtype)

                    token_pg_loss, token_pg_clipfrac, token_ppo_kl = (
                        compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            # eos_mask=loss_mask,
                            cliprange=clip_ratio,
                            cliprange_high=clip_ratio_high,
                        )
                    )
                    if use_t_in_loss:
                        token_pg_loss *= 1 / t

                    if loss_agg_mode == "token-mean":
                        pg_loss = masked_mean(token_pg_loss, loss_mask)
                        pg_clipfrac = masked_mean(token_pg_clipfrac, loss_mask)
                        ppo_kl = masked_mean(token_ppo_kl, loss_mask)
                    else:
                        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

                    # pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = core_algos.compute_policy_loss(
                    #     old_log_prob=old_log_prob,
                    #     log_prob=log_prob,
                    #     advantages=advantages,
                    #     response_mask=loss_mask,
                    #     cliprange=clip_ratio,
                    #     loss_agg_mode=loss_agg_mode,
                    # )

                    entropy_loss = masked_mean(entropy, loss_mask)
                    # entropy_loss = core_algos.agg_loss(loss_mat=entropy, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode)

                    # compute policy loss
                    policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = core_algos.kl_penalty(
                            logprob=log_prob,
                            ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = masked_mean(kld, loss_mask)
                        # kl_loss = core_algos.agg_loss(loss_mat=kld, loss_mask=loss_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        "actor/entropy_loss": entropy_loss.detach().item(),
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        # "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
