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
The main entry point to run the PPO algorithm
"""

import logging
import os
import types
import warnings

import psutil
import torch
import torch.distributed
import verl.utils.torch_functional as verl_F
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from torch.distributed.device_mesh import init_device_mesh
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from src.diffllm.gen_utils import q_sample

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_PPO_LOGGING_LEVEL", "WARN"))


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(
            "cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"]
        )
    else:
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(world_size // fsdp_size, fsdp_size),
            mesh_dim_names=["ddp", "fsdp"],
        )
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(
            f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2"
        )
    return sharding_strategy


class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group()

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(
            world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size
        )

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get(
            "ulysses_sequence_parallel_size", 1
        )
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"],
            )

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(
            self.ulysses_device_mesh
        )

        self.role = role
        assert self.role in [
            "actor",
            "rollout",
            "ref",
            "actor_rollout",
            "actor_rollout_ref",
        ]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in [
            "rollout",
            "actor_rollout",
            "actor_rollout_ref",
        ]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get(
                "param_offload", False
            )
            self._is_offload_optimizer = self.config.actor.fsdp_config.get(
                "optimizer_offload", False
            )
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get(
                "param_offload", False
            )

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= (
                self.config.rollout.n * self.config.actor.mask_epochs
            )
            self.config.actor.ppo_mini_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            assert (
                self.config.actor.ppo_mini_batch_size > 0
            ), f"ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than 0 after normalization"
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (
                    self.device_mesh.size() // self.ulysses_sequence_parallel_size
                )
                self.config.actor.ppo_micro_batch_size_per_gpu = (
                    self.config.actor.ppo_micro_batch_size
                )
                assert (
                    self.config.actor.ppo_mini_batch_size
                    % self.config.actor.ppo_micro_batch_size_per_gpu
                    == 0
                ), f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"
                assert (
                    self.config.actor.ppo_mini_batch_size
                    // self.config.actor.ppo_micro_batch_size_per_gpu
                    > 0
                ), f"normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}"

        # normalize rollout config
        if (
            self._is_rollout
            and self.config.rollout.log_prob_micro_batch_size is not None
        ):
            self.config.rollout.log_prob_micro_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            self.config.rollout.log_prob_micro_batch_size_per_gpu = (
                self.config.rollout.log_prob_micro_batch_size
            )
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= (
                self.device_mesh.size() // self.ulysses_sequence_parallel_size
            )
            self.config.ref.log_prob_micro_batch_size_per_gpu = (
                self.config.ref.log_prob_micro_batch_size
            )

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        role="actor",
    ):
        from verl.utils.model import (
            print_model_size,
            update_model_config,
            get_generation_config,
        )
        from verl.utils.torch_dtypes import PrecisionType
        from transformers import AutoModel, AutoConfig
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            CPUOffload,
        )
        from torch import optim

        assert role in ["actor", "ref"]

        log_gpu_memory_usage("Before init from HF AutoModel", logger=logger)
        local_path = copy_to_local(model_path)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code
        )

        self.generation_config = get_generation_config(
            local_path, trust_remote_code=trust_remote_code
        )

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_penalty": self.config.rollout.get("eos_penalty", 0),
            "eps": self.config.rollout.get("eps", 1e-3),
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(
            actor_model_config, override_config_kwargs=override_config_kwargs
        )
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings,
            mesh=self.device_mesh,
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actor_module = AutoModel.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            if self.config.rollout.accelerator.enable:
                accelerator_type = self.config.rollout.accelerator.type
                accelerator_use_cache = self.config.rollout.accelerator.use_cache

                # TODO: debug
                if accelerator_type == "fast_dllm":
                    assert (
                        self.config.rollout.get("micro_batch_size", 1) == 1
                    ), "Fast DLLM only supports micro_batch_size=1"
                    from src.inference.fast_dllm.modeling_dream import DreamModel

                    actor_module = DreamModel.from_pretrained(
                        pretrained_model_name_or_path=local_path,
                        torch_dtype=torch_dtype,
                        config=actor_model_config,
                        attn_implementation="flash_attention_2",
                        trust_remote_code=trust_remote_code,
                    )

                    if accelerator_use_cache:
                        from src.inference.fast_dllm.generation_utils_block import (
                            DreamGenerationMixin,
                        )

                        actor_module.diffusion_generate = types.MethodType(
                            DreamGenerationMixin.diffusion_generate, actor_module
                        )
                        actor_module._sample = types.MethodType(
                            DreamGenerationMixin._sample, actor_module
                        )
                    else:
                        from src.inference.fast_dllm.generation_utils import (
                            DreamGenerationMixin,
                        )

                        actor_module.diffusion_generate = types.MethodType(
                            DreamGenerationMixin.diffusion_generate, actor_module
                        )
                        actor_module._sample = types.MethodType(
                            DreamGenerationMixin._sample, actor_module
                        )
                else:
                    raise NotImplementedError(
                        f"Accelerator type {accelerator_type} is not supported"
                    )

            if use_remove_padding or self.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(
                    model=actor_module,
                    ulysses_sp_size=self.ulysses_sequence_parallel_size,
                )

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage("After init from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("param_dtype", "bf16")
            )
            reduce_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("reduce_dtype", "fp32")
            )
            buffer_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("buffer_dtype", "fp32")
            )
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module, config=fsdp_config.get("wrap_policy", None)
        )

        # if self._is_rollout and self.config.rollout.name == "hf":
        #     # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
        #     auto_wrap_policy = None

        if self.rank == 0:
            print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == "actor" else CPUOffload(offload_params=True)
        actor_module_fsdp = FSDP(
            actor_module,
            cpu_offload=cpu_offload,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            forward_prefetch=False,
        )

        log_gpu_memory_usage("After Actor FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import get_constant_schedule_with_warmup

            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
            )

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            actor_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
            )
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage("After actor optimizer init", logger=logger)

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler

    def _build_rollout(self):
        rollout_name = self.config.rollout.name

        if rollout_name == "hf":
            from src.workers.rollout import HFRollout

            rollout = HFRollout(
                module=self.actor_module_fsdp, config=self.config.rollout
            )
            # TODO: a sharding manager that do nothing?
        else:
            raise NotImplementedError(f"Rollout name {rollout_name} is not supported")

        return rollout

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from src.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(
            self.config.model.get("override_config", OmegaConf.create())
        )

        use_remove_padding = self.config.model.get("use_remove_padding", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler = (
                self._build_model_optimizer(
                    model_path=self.config.model.path,
                    fsdp_config=fsdp_config,
                    optim_config=optim_config,
                    override_model_config=override_model_config,
                    use_remove_padding=use_remove_padding,
                    enable_gradient_checkpointing=self.config.model.get(
                        "enable_gradient_checkpointing", False
                    ),
                    trust_remote_code=self.config.model.get("trust_remote_code", False),
                    role="actor",
                )
            )

            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage(
                    "After offload actor optimizer during init", logger=logger
                )
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
            self.actor = DataParallelPPOActor(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

        if self._is_rollout:
            self.rollout = self._build_rollout()

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
            self.ref_policy = DataParallelPPOActor(
                config=self.config.ref, actor_module=self.ref_module_fsdp
            )

        if self._is_actor:
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=(
                    self.processor if self.processor is not None else self.tokenizer
                ),
                checkpoint_contents=self.config.actor.checkpoint.contents,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        # Support all hardwares
        data = data.to(torch.cuda.current_device())

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(
                optimizer=self.actor_optimizer, device_id=torch.cuda.current_device()
            )

        log_gpu_memory_usage("Before update policy", logger=logger)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            metrics["perf/max_memory_allocated_gb"] = (
                torch.cuda.max_memory_allocated() / (1024**3)
            )
            metrics["perf/max_memory_reserved_gb"] = (
                torch.cuda.max_memory_reserved() / (1024**3)
            )
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (
                1024**3
            )

            self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr

            log_gpu_memory_usage("After update policy", logger=logger)

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={"metrics": metrics})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        prompts = prompts.to(torch.cuda.current_device())

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        meta_info = {
            "eos_token_id": (
                self.generation_config.eos_token_id
                if self.generation_config is not None
                else self.tokenizer.eos_token_id
            ),
            "pad_token_id": (
                self.generation_config.pad_token_id
                if self.generation_config is not None
                else self.tokenizer.pad_token_id
            ),
            "mask_token_id": self.tokenizer.mask_token_id,
        }
        prompts.meta_info.update(meta_info)

        # Generate
        output = self.rollout.generate_sequences(prompts=prompts)
        log_gpu_memory_usage("After rollout generation", logger=logger)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)

        # After generating sequences, we add noise immediately for further training
        # NOTE: to match ppo epoch, we repeat the same data for ppo_epochs times
        original_bs = len(output)
        output = output.repeat(
            repeat_times=self.config.actor.ppo_epochs * self.config.actor.mask_epochs,
            interleave=True,
        )
        response_length = output.batch["responses"].size(-1)
        maskable_mask = torch.zeros_like(output.batch["attention_mask"].bool())
        maskable_mask[:, -response_length:] = True

        t, t_mask = None, None
        if self.config.actor.t_schedule == "couple":
            assert (
                self.config.actor.mask_epochs == 2
            ), "couple t_schedule only supports mask_epochs == 2"
            t = torch.rand(
                (self.config.actor.ppo_epochs,),
                dtype=torch.float,
                device=output.batch["input_ids"].device,
            )
            epoch_maskable_mask = maskable_mask[
                : self.config.actor.ppo_epochs
            ]  # (epochs, seq_len)
            u = torch.rand_like(epoch_maskable_mask, dtype=torch.float)

            t_mask = (u < t[:, None]) & epoch_maskable_mask
            coupled_t_mask = (u >= t[:, None]) & epoch_maskable_mask
            t_mask = (
                torch.stack([t_mask, coupled_t_mask], dim=1)
                .flatten(0, 1)
                .repeat(original_bs, 1)
            )
            t = torch.stack([t, 1 - t], dim=1).flatten(0, 1).repeat(original_bs)
        else:
            raise ValueError(f"Invalid t_schedule: {self.config.actor.t_schedule}")

        masked_input_ids, t, loss_mask_nonflatten = q_sample(
            input_ids=output.batch["input_ids"],
            maskable_mask=maskable_mask,
            mask_token_id=meta_info["mask_token_id"],
            t=t,
            t_mask=t_mask,
        )

        output.batch["masked_input_ids"] = masked_input_ids
        output.batch["loss_mask"] = loss_mask_nonflatten[:, -response_length:]
        output.batch["t"] = t
        output = output.to("cpu")

        # clear kv cache
        log_gpu_memory_usage("After recompute log prob", logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        data = data.to(torch.cuda.current_device())
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = (
            self.config.rollout.log_prob_micro_batch_size_per_gpu
        )
        data.meta_info["max_token_len"] = (
            self.config.rollout.log_prob_max_token_len_per_gpu
        )
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.actor.compute_log_prob(data=data)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output},
                meta_info={"temperature": self.config.rollout.temperature},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        log_gpu_memory_usage("After compute_log_prob", logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        # Support all hardwares
        data = data.to(torch.cuda.current_device())

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={"ref_log_prob": output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.ref_policy.actor_module._handle.reshard(True)

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(
        self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None
    ):
        # only support save and load ckpt for actor
        assert self._is_actor
        import torch

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            del_local_after_load=del_local_after_load,
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)
