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
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import math
import random
import re
from contextlib import nullcontext
from typing import List

import hydra
import numpy as np
import torch
import torch.distributed
import verl.utils.hdfs_io as hdfs_io
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn, optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, PreTrainedModel
from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import get_cosine_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.workers.sharding_manager import FSDPUlyssesShardingManager

from src.diffllm.gen_utils import q_sample
from src.trainer.sft_dataset import SFTDataset, TokenizedSFTDataset

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import DictConfig, ListConfig

    if isinstance(obj, (ListConfig, DictConfig)):
        return (
            {k: convert_to_regular_types(v) for k, v in obj.items()}
            if isinstance(obj, DictConfig)
            else list(obj)
        )
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj


def context_adaptive_reweight(seq_len, distribution="symmetric-geometric", **kwargs):
    position_ids_l = np.arange(seq_len).reshape(-1, 1)
    position_ids_r = np.arange(seq_len).reshape(1, -1)
    distance = position_ids_l - position_ids_r
    distance = torch.from_numpy(distance)

    def geometric_distribution(k, cart_p=0.8, **kwargs):
        if not 0 < cart_p <= 1:
            raise ValueError("p must be between 0 and 1")

        res = (math.log(cart_p) + (k.abs() - 1) * math.log(1 - cart_p)).exp() * 0.5
        res.masked_fill_(k == 0, 0)  # ignore distance=0
        return res

    if distribution == "symmetric-geometric":
        matrix = geometric_distribution(distance, **kwargs)
    else:
        raise ValueError(f"Unknown distribution {distribution}")

    return matrix


class OptimizedCollateFunction:
    """
    Optimized collate function that completes preprocessing during data loading
    Reduces GPU computation overhead and improves training efficiency
    """

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.pad_eos_token_id = (
            config.data.pad_token_id
            if config.data.pad_token_id is not None
            else tokenizer.pad_token_id
        )

        # Cache configuration items to avoid repeated access
        self.enable_perbatch_cutoff = getattr(config.data, "perbatch_cutoff", False)
        self.perbatch_cutoff_type = getattr(config.data, "perbatch_cutoff_type", None)
        self.resp_cutoff_ratio = getattr(config.data, "resp_cutoff_ratio", 0.0)

        self.random = random.Random(42)
        self.np_random = np.random.RandomState(42)

    def __call__(self, batch_samples):
        """
        Efficient collate function implementation
        Args:
            batch_samples: List of samples from dataset
        Returns:
            Preprocessed batch dict
        """
        # Use default collate to merge samples into batch tensors
        from torch.utils.data.dataloader import default_collate

        batch = default_collate(batch_samples)

        # Extract tensors
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].bool()
        position_ids = batch["position_ids"]
        loss_mask = batch["loss_mask"].bool()

        # 1. Handle perbatch_cutoff related logic
        if self.enable_perbatch_cutoff:
            input_ids, attention_mask, position_ids, loss_mask = (
                self._apply_perbatch_cutoff(
                    input_ids, attention_mask, position_ids, loss_mask
                )
            )

        # 2. Handle response truncation
        if self.resp_cutoff_ratio > 0.0:
            input_ids, attention_mask, position_ids, loss_mask = (
                self._apply_resp_cutoff(
                    input_ids, attention_mask, position_ids, loss_mask
                )
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

    def _apply_perbatch_cutoff(
        self, input_ids, attention_mask, position_ids, loss_mask
    ):
        """Apply perbatch cutoff logic"""
        if self.perbatch_cutoff_type == "random":
            non_pad_lens = (input_ids != self.pad_eos_token_id).sum(-1).cpu()

            # randomly choose a cutoff length from non_pad_lens
            cutoff_seq_len = np.random.choice(non_pad_lens)

            # cutoff
            input_ids = input_ids[:, :cutoff_seq_len]
            attention_mask = attention_mask[:, :cutoff_seq_len]
            position_ids = position_ids[:, :cutoff_seq_len]
            loss_mask = loss_mask[:, :cutoff_seq_len]
        elif self.perbatch_cutoff_type == "random_with_input_pad":
            prompt_mask = loss_mask == 0
            response_mask = (loss_mask == 1) & (input_ids != self.pad_eos_token_id)

            prompt_lens = prompt_mask.sum(-1)
            response_lens = response_mask.sum(-1)
            max_prompt_len = prompt_lens.max()
            pad_lens = max_prompt_len - prompt_lens

            # randomly choose a response length from response_lens
            kept_response_len = np.random.choice(response_lens.cpu())

            # rebuild input_ids, attention_mask, loss_mask
            new_input_ids = (
                torch.ones(
                    input_ids.shape[0],
                    max_prompt_len + kept_response_len,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                * self.pad_eos_token_id
            )
            new_attention_mask = torch.ones_like(
                new_input_ids, dtype=attention_mask.dtype
            )
            new_loss_mask = torch.ones_like(new_input_ids, dtype=loss_mask.dtype)

            for i in range(input_ids.shape[0]):
                kept_response_len_i = min(kept_response_len, response_lens[i])
                new_input_ids[i, pad_lens[i] : pad_lens[i] + prompt_lens[i]] = (
                    input_ids[i][prompt_mask[i]]
                )
                new_input_ids[
                    i,
                    pad_lens[i]
                    + prompt_lens[i] : pad_lens[i]
                    + prompt_lens[i]
                    + kept_response_len_i,
                ] = input_ids[i][response_mask[i]][:kept_response_len_i]

                new_attention_mask[i, : pad_lens[i]] = 0
                new_loss_mask[i, : pad_lens[i] + prompt_lens[i]] = 0

            input_ids = new_input_ids
            attention_mask = new_attention_mask
            position_ids = compute_position_id_with_mask(new_attention_mask)
            loss_mask = new_loss_mask
        else:
            pad_lens = (input_ids == self.pad_eos_token_id).sum(-1)
            # cutoff_len = eos_lens.min() - 1
            # assert cutoff_len > 0, input_ids
            cutoff_len = pad_lens.min()
            assert cutoff_len >= 0

            # cutoff
            seq_len = input_ids.shape[-1]
            input_ids = input_ids[:, : seq_len - cutoff_len]
            attention_mask = attention_mask[:, : seq_len - cutoff_len]
            position_ids = position_ids[:, : seq_len - cutoff_len]
            loss_mask = loss_mask[:, : seq_len - cutoff_len]

        return input_ids, attention_mask, position_ids, loss_mask

    def _apply_resp_cutoff(self, input_ids, attention_mask, position_ids, loss_mask):
        """Apply response truncation logic"""
        import numpy as np

        if self.np_random.rand() < self.resp_cutoff_ratio:
            # Calculate response length for each sample (loss_mask True portion)
            resp_lens = loss_mask.sum(-1)
            min_resp_len = resp_lens.min().item()

            if min_resp_len > 1:
                # Randomly select truncation length
                cutoff_len = self.np_random.randint(1, min_resp_len)

                # Truncate from the end of sequence
                new_seq_len = input_ids.shape[-1] - cutoff_len
                input_ids = input_ids[:, :new_seq_len].contiguous()
                attention_mask = attention_mask[:, :new_seq_len].contiguous()
                position_ids = position_ids[:, :new_seq_len].contiguous()
                loss_mask = loss_mask[:, :new_seq_len].contiguous()

        return input_ids, attention_mask, position_ids, loss_mask


class StreamingCollateFunction(OptimizedCollateFunction):
    """
    Streaming version of collate function
    Suitable for large batches or long sequences, reduces memory peaks
    """

    def __call__(self, batch_samples):
        """
        Streaming version, processes samples one by one then concatenates
        """
        if len(batch_samples) == 0:
            return {}

        # Process samples one by one to reduce memory usage
        processed_samples = []
        for sample in batch_samples:
            processed_sample = self._preprocess_single_sample(sample)
            processed_samples.append(processed_sample)

        # Concatenate all processed samples
        from torch.utils.data.dataloader import default_collate

        batch = default_collate(processed_samples)

        # Apply batch-level preprocessing (inherited from parent class logic)
        return self._apply_batch_preprocessing(batch)

    def _preprocess_single_sample(self, sample):
        """Preprocess single sample"""
        # Sample-level preprocessing can be implemented here
        # e.g., dynamic truncation, special token processing, etc.
        return sample

    def _apply_batch_preprocessing(self, batch):
        """Apply batch-level preprocessing logic"""
        # Extract tensors
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].bool()
        position_ids = batch["position_ids"]
        loss_mask = batch["loss_mask"].bool()

        if self.enable_perbatch_cutoff:
            input_ids, attention_mask, position_ids, loss_mask = (
                self._apply_perbatch_cutoff(
                    input_ids, attention_mask, position_ids, loss_mask
                )
            )

        if self.resp_cutoff_ratio > 0.0:
            input_ids, attention_mask, position_ids, loss_mask = (
                self._apply_resp_cutoff(
                    input_ids, attention_mask, position_ids, loss_mask
                )
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }


class FSDPSFTTrainer(object):

    def __init__(
        self, config, device_mesh: DeviceMesh, ulysses_device_mesh: DeviceMesh
    ):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # Add tracking for current epoch
        self.current_epoch = 0

        # Check if resuming training
        self.resume_training = getattr(self.config.trainer, "resume_training", False)
        self.resume_checkpoint_path = getattr(self.config.trainer, "resume_path", None)

        # build tokenizer first
        if self.resume_training and self.resume_checkpoint_path:
            # If resuming from specific checkpoint, use that path for tokenizer
            local_model_path = copy_local_path_from_hdfs(
                src=self.resume_checkpoint_path, verbose=True
            )
        else:
            local_model_path = copy_local_path_from_hdfs(
                src=self.config.model.partial_pretrain, verbose=True
            )

        from verl.utils import hf_tokenizer

        self.tokenizer = hf_tokenizer(
            local_model_path, trust_remote_code=self.config.model.trust_remote_code
        )
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(
            self.config, "ulysses_sequence_parallel_size", 1
        )
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(
                f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}"
            )
            print(f"Using remove padding: {self.use_remove_padding}")

        self._build_dataloader()
        # build model
        self._build_model_optimizer()

        # TODO: add checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(self.config)

    def _normalize_config_bsz(self):
        dp_size = (
            self.device_mesh.size(0)
            if not self.ulysses_device_mesh
            else self.ulysses_device_mesh.size(0)
        )
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert (
            self.config.data.train_batch_size % dp_size == 0
        ), f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"

        self.config.data.train_batch_size //= dp_size

        assert (
            self.config.data.train_batch_size
            % self.config.data.micro_batch_size_per_gpu
            == 0
        )

    def _build_dataloader(self):
        config = self.config
        # build dataset
        if config.data.tokenized:
            self.train_dataset = TokenizedSFTDataset(
                parquet_files=config.data.train_files,
            )
            self.val_dataset = TokenizedSFTDataset(
                parquet_files=config.data.val_files,
            )
        else:
            self.train_dataset = SFTDataset(
                parquet_files=config.data.train_files,
                tokenizer=self.tokenizer,
                prompt_key=config.data.prompt_key,
                response_key=config.data.response_key,
                max_length=config.data.max_length,
                truncation=config.data.truncation,
                pad_token_id=config.data.pad_token_id,
            )
            self.val_dataset = SFTDataset(
                parquet_files=config.data.val_files,
                tokenizer=self.tokenizer,
                prompt_key=config.data.prompt_key,
                response_key=config.data.response_key,
                max_length=config.data.max_length,
                truncation=config.data.truncation,
                pad_token_id=config.data.pad_token_id,
            )

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(
                    f"Using SP rank {rank} and size {world_size} for data distribution"
                )
                print(
                    f"Each SP rank gets different data, but the same data WITHIN the same rank"
                )
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        self.train_sampler = DistributedSampler(
            self.train_dataset,
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            drop_last=True,
        )

        # Create optimized collate function
        collate_fn_type = getattr(config.data, "collate_fn_type", "optimized")
        if collate_fn_type == "streaming":
            train_collate_fn = StreamingCollateFunction(config, self.tokenizer)
            val_collate_fn = StreamingCollateFunction(config, self.tokenizer)
        else:
            train_collate_fn = OptimizedCollateFunction(config, self.tokenizer)
            val_collate_fn = OptimizedCollateFunction(config, self.tokenizer)

        if self.device_mesh.get_rank() == 0:
            print(f"Using {collate_fn_type} collate function for data preprocessing")

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4,
            persistent_workers=True,
            collate_fn=train_collate_fn,  # Use optimized collate function
        )

        self.val_sampler = DistributedSampler(
            self.val_dataset,
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            drop_last=True,
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4,
            persistent_workers=True,
            collate_fn=val_collate_fn,  # Use optimized collate function
        )

    def _build_model_optimizer(self, checkpoint_path=None):
        """Build model and optimizer, optionally from a checkpoint."""
        # Determine which path to load from
        if checkpoint_path:
            local_model_path = checkpoint_path
        else:
            local_model_path = copy_local_path_from_hdfs(
                src=self.config.model.partial_pretrain, verbose=True
            )

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        # load config first
        config = AutoConfig.from_pretrained(
            local_model_path,
            trust_remote_code=trust_remote_code,
            attention_dropout=self.config.model.attention_dropout,
        )
        if self.config.ulysses_sequence_parallel_size > 1:
            assert (
                self.use_remove_padding
            ), "Sequence parallel is only supported when remove_padding is enabled"
            from verl.models.registry import check_model_support_rmpad

            check_model_support_rmpad(config.model_type)

        if self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch

            apply_monkey_patch(config, verbose=True)

        # This may be very large
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings
        )

        with init_context():
            self.model: PreTrainedModel = AutoModel.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch.float32,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import (
                    _apply_liger_kernel_to_instance,
                )

                _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get("lora_rank", 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(
                        self.config.model.target_modules
                    ),
                    "bias": "none",
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(
                offload_params=self.config.model.fsdp_config.offload_params
            )

        self.fsdp_model = FSDP(
            module=self.model,
            auto_wrap_policy=auto_wrap_policy,
            param_init_fn=init_fn,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_mesh=self.device_mesh,
            sync_module_states=True,
            device_id=torch.cuda.current_device(),
            cpu_offload=cpu_offload,
            use_orig_params=False,
        )

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(
                f"Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}"
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.total_steps,
        )

    def _load_from_checkpoint(self, checkpoint_path):
        """Initialize training state from checkpoint."""
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        if self.device_mesh.get_rank() == 0:
            print(f"Resuming from checkpoint: {checkpoint_path}")

        # Load training state
        training_state_path = os.path.join(checkpoint_path, "training_state.pt")
        optimizer_state_path = os.path.join(checkpoint_path, "optimizer_state.pt")

        # Only rank 0 loads the full state initially
        if self.device_mesh.get_rank() == 0 and os.path.exists(training_state_path):
            training_state = torch.load(training_state_path)
            epoch = training_state["epoch"]
            global_step = training_state["global_step"]

            # Load scheduler state
            self.lr_scheduler.load_state_dict(training_state["lr_scheduler"])
        else:
            # For other ranks or if file is missing, get step from path
            epoch = 0
            global_step = extract_step(checkpoint_path) or 0

        # Broadcast values to all ranks
        if torch.distributed.get_world_size() > 1:
            tensor = torch.tensor([epoch, global_step], device="cuda")
            torch.distributed.broadcast(tensor, src=0)
            if self.device_mesh.get_rank() != 0:
                epoch, global_step = tensor.tolist()

        # Load optimizer state if exists
        if os.path.exists(optimizer_state_path):
            from torch.distributed.fsdp import FullStateDictConfig
            from torch.distributed.fsdp.api import FullOptimStateDictConfig

            with FSDP.state_dict_type(
                self.fsdp_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                # Load optimizer state - rank 0 loads, others receive broadcast
                if self.device_mesh.get_rank() == 0:
                    optim_state = torch.load(optimizer_state_path)
                else:
                    optim_state = None

                # Use FSDP utility to load optimizer state
                optim_state_dict = FSDP.scatter_full_optim_state_dict(
                    optim_state, self.fsdp_model
                )
                self.optimizer.load_state_dict(optim_state_dict)

        self.current_epoch = epoch
        return global_step

    def _compute_loss_and_backward(self, batch, do_backward=True):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = (
            self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1
        )

        input_ids = batch["input_ids"].cuda(non_blocking=True)
        attention_mask = batch["attention_mask"].cuda(non_blocking=True).bool()
        position_ids = batch["position_ids"].cuda(non_blocking=True)
        loss_mask = batch["loss_mask"].cuda(non_blocking=True).bool()

        loss_fct = nn.CrossEntropyLoss(reduction="none")
        pad_eos_token_id = (
            self.config.data.pad_token_id
            if self.config.data.pad_token_id is not None
            else self.tokenizer.pad_token_id
        )

        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if not use_sp:
                    # Standard forward pass without sequence parallel
                    labels = input_ids.contiguous()

                    # Forward pass
                    # NOTE: loss_mask is of size (batch_size, seq_len - 1)
                    batch_size = input_ids.shape[0]
                    masked_input_ids, t, loss_mask_nonflatten = q_sample(
                        input_ids,
                        maskable_mask=loss_mask,
                        mask_token_id=self.tokenizer.mask_token_id,
                        eos_token_id=(
                            pad_eos_token_id
                            if self.config.data.get("treat_eos_as_one", False)
                            else None
                        ),
                    )
                    loss_mask = loss_mask_nonflatten.reshape(-1)

                    # 2d -> 4d conversion for attention_mask
                    attention_mask = torch.logical_and(
                        attention_mask.unsqueeze(1).unsqueeze(-2),
                        attention_mask.unsqueeze(1).unsqueeze(-1),
                    )

                    output = self.fsdp_model(
                        input_ids=masked_input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                    )
                    logits = output.logits

                    shift_logits = torch.cat(
                        [logits[:, 0:1], logits[:, :-1]], dim=1
                    ).contiguous()
                    shift_labels = labels.contiguous()
                    # Flatten the tokens
                    shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)

                    # We use weighted loss
                    loss_mask = loss_mask.to(loss.device)
                    loss = loss.masked_fill(~loss_mask, 0)
                    if self.config.diffusion.token_reweighting:
                        loss = (
                            self.config.diffusion.alpha
                            * (1 - torch.exp(-loss)) ** self.config.diffusion.gamma
                            * loss
                        )

                    if self.config.diffusion.time_reweighting == "original":
                        weight = 1 / t[:, None].float().expand(labels.size())
                    elif self.config.diffusion.time_reweighting == "linear":
                        weight = 1 - t[:, None].float().expand(labels.size())
                    elif self.config.diffusion.time_reweighting == "cart":
                        # seq_len = self.config.data.max_length
                        seq_len = input_ids.shape[-1]
                        weight_matrix = context_adaptive_reweight(
                            seq_len, cart_p=self.config.diffusion.cart_p
                        )
                        _weight_matrix = weight_matrix[:seq_len, :seq_len].to(
                            loss_mask.device
                        )
                        non_mask = ~loss_mask_nonflatten.to(
                            loss.device
                        )  # loss_mask indicates where is mask
                        weight = (
                            non_mask.type_as(_weight_matrix)
                            .matmul(_weight_matrix)
                            .masked_fill(non_mask, 0)
                        )
                    else:
                        weight = (
                            t.new_ones((batch_size, 1)).float().expand(labels.size())
                        )

                    loss = loss * weight.reshape(-1)
                else:
                    raise NotImplementedError(
                        "Sequence parallel is not implemented yet"
                    )

                valid_token_this_rank = torch.sum(loss_mask)
                loss = torch.sum(loss) / valid_token_this_rank

                if do_backward:
                    loss.backward()
                return loss

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for micro_batch in micro_batches:
            loss = (
                self._compute_loss_and_backward(batch=micro_batch, do_backward=False)
                / n_micro_batches
            )
            loss.backward()
            step_loss += loss.item()

        grad_norm = self.fsdp_model.clip_grad_norm_(
            max_norm=self.config.optim.clip_grad
        )

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        self.optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage("After offload weights", logger=logger)

        step_loss = torch.tensor(step_loss, device="cuda")
        torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)

        # record how many tokens are padded as EOS
        all_num_eos = []
        for micro_batch in micro_batches:
            loss_mask = batch["loss_mask"]
            num_eos = loss_mask.sum(-1)
            num_eos -= num_eos.min() - 1
            all_num_eos.append(num_eos)
        all_num_eos = torch.cat(all_num_eos)

        return {
            "train/loss": step_loss.detach().item(),
            "train/lr(1e-3)": lr * 1e3,
            "train/grad_norm": grad_norm,
            "train/num_eos_mean": all_num_eos.float().mean().item(),
            "train/num_eos_max": all_num_eos.float().max().item(),
        }

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss = self._compute_loss_and_backward(batch, do_backward=False)
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        return loss

    def save_checkpoint(self, step):
        """Save model, optimizer, and training state."""
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp.api import FullOptimStateDictConfig

        # Create checkpoint directory
        path = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{step}"
        )

        # Save model state
        model_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        with FSDP.state_dict_type(
            self.fsdp_model, StateDictType.FULL_STATE_DICT, model_cfg
        ):
            model_state = self.fsdp_model.state_dict()
            optim_state = FSDP.full_optim_state_dict(self.fsdp_model, self.optimizer)

        # Save training state
        training_state = {
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "global_step": step,
            "epoch": self.current_epoch,
        }

        # Save on rank 0 only
        if self.device_mesh.get_rank() == 0:
            os.makedirs(path, exist_ok=True)

            # Save model using HF's save_pretrained
            self.model.save_pretrained(path, state_dict=model_state)
            self.tokenizer.save_pretrained(path)

            # Save optimizer and training state
            torch.save(optim_state, os.path.join(path, "optimizer_state.pt"))
            torch.save(training_state, os.path.join(path, "training_state.pt"))

            # Copy to HDFS if configured
            if self.config.trainer.default_hdfs_dir:
                hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
                hdfs_io.copy(
                    src=path,
                    dst=self.config.trainer.default_hdfs_dir,
                    dirs_exist_ok=True,
                )
        torch.distributed.barrier()

    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in checkpoint directories."""
        latest_checkpoint = None
        latest_step = -1

        # Check local directory first
        local_dir = self.config.trainer.default_local_dir
        if os.path.exists(local_dir):
            checkpoints = [
                d
                for d in os.listdir(local_dir)
                if os.path.isdir(os.path.join(local_dir, d))
                and d.startswith("global_step_")
            ]

            for ckpt in checkpoints:
                step = extract_step(ckpt)
                if step is not None and step > latest_step:
                    latest_step = step
                    latest_checkpoint = os.path.join(local_dir, ckpt)

        # If not found locally and HDFS is configured, check there
        if latest_checkpoint is None and self.config.trainer.default_hdfs_dir:
            try:
                if hdfs_io.exists(self.config.trainer.default_hdfs_dir):
                    checkpoints = [
                        d
                        for d in hdfs_io.listdir(self.config.trainer.default_hdfs_dir)
                        if d.startswith("global_step_")
                    ]
                    for ckpt in checkpoints:
                        step = extract_step(ckpt)
                        if step is not None and step > latest_step:
                            latest_step = step
                            remote_path = os.path.join(
                                self.config.trainer.default_hdfs_dir, ckpt
                            )

                            # Copy from HDFS to local
                            local_path = os.path.join(local_dir, ckpt)
                            os.makedirs(local_dir, exist_ok=True)
                            hdfs_io.copy(
                                src=remote_path, dst=local_path, dirs_exist_ok=True
                            )
                            latest_checkpoint = local_path
            except Exception as e:
                if self.device_mesh.get_rank() == 0:
                    print(f"Error checking HDFS for checkpoints: {e}")

        return latest_checkpoint

    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
            )

        global_step = 0

        # Handle resuming training
        if self.resume_training:
            # Find latest checkpoint if not specified
            if not self.resume_checkpoint_path:
                self.resume_checkpoint_path = self._find_latest_checkpoint()

            if self.resume_checkpoint_path:
                global_step = self._load_from_checkpoint(self.resume_checkpoint_path)
                if rank == 0:
                    print(
                        f"Resumed training from step {global_step}, epoch {self.current_epoch}"
                    )
            elif rank == 0:
                print("No checkpoint found, starting training from scratch")

        # Compute total training steps
        total_training_steps = (
            len(self.train_dataloader) * self.config.trainer.total_epochs
        )

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        if rank == 0:
            print(f"Total training steps: {self.total_training_steps}")

        # Begin training from the current epoch
        for epoch in range(self.current_epoch, self.config.trainer.total_epochs):
            self.current_epoch = epoch
            self.train_sampler.set_epoch(epoch=epoch)

            # Create a data iterator
            dataloader_iter = iter(self.train_dataloader)

            # If resuming mid-epoch, skip to the right position
            if epoch == self.current_epoch and global_step > 0 and self.resume_training:
                steps_in_epoch = global_step % self.steps_per_epoch
                if steps_in_epoch > 0:
                    if rank == 0:
                        print(
                            f"Skipping {steps_in_epoch} steps to resume at the right position"
                        )
                    for _ in range(steps_in_epoch):
                        try:
                            next(dataloader_iter)
                        except StopIteration:
                            dataloader_iter = iter(self.train_dataloader)

            # Calculate remaining steps in this epoch
            remaining_steps = self.steps_per_epoch
            if epoch == self.current_epoch and global_step > 0 and self.resume_training:
                remaining_steps -= global_step % self.steps_per_epoch

            for data in tqdm(
                dataloader_iter,
                initial=self.steps_per_epoch - remaining_steps,
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch+1}/{self.config.trainer.total_epochs}",
            ):
                data = TensorDict(
                    data, batch_size=self.config.data.train_batch_size
                ).cuda()
                metric = self.training_step(data)
                if rank == 0:
                    tracking.log(data=metric, step=global_step)
                global_step += 1

                # for early exit validation
                if global_step >= self.total_training_steps:
                    # Perform final validation
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_data = TensorDict(
                            val_data,
                            batch_size=self.config.data.micro_batch_size_per_gpu,
                        ).cuda()
                        val_loss = self.validation_step(val_data)
                        val_losses.append(val_loss)
                    if rank == 0:
                        avg_val_loss = torch.mean(torch.stack(val_losses))
                        metric = {"val/loss": avg_val_loss.detach().item()}
                        tracking.log(data=metric, step=global_step)
                    torch.distributed.barrier()

                    # Save final checkpoint
                    self.save_checkpoint(step=global_step)
                    return

                if global_step % self.config.trainer.save_checkpoint_steps == 0:
                    # Perform validation
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_data = TensorDict(
                            val_data,
                            batch_size=self.config.data.micro_batch_size_per_gpu,
                        ).cuda()
                        val_loss = self.validation_step(val_data)
                        val_losses.append(val_loss)
                    if rank == 0:
                        avg_val_loss = torch.mean(torch.stack(val_losses))
                        metric = {"val/loss": avg_val_loss.detach().item()}
                        tracking.log(data=metric, step=global_step)
                    torch.distributed.barrier()

                    # Save checkpoint
                    self.save_checkpoint(step=global_step)

            # validation
            val_losses = []
            for data in self.val_dataloader:
                data = TensorDict(
                    data, batch_size=self.config.data.micro_batch_size_per_gpu
                ).cuda()
                val_loss = self.validation_step(data)
                val_losses.append(val_loss)
            if rank == 0:
                val_loss = torch.mean(torch.stack(val_losses))
                metric = {"val/loss": val_loss.detach().item()}
                tracking.log(data=metric, step=global_step)
            torch.distributed.barrier()

            # save checkpoint
            self.save_checkpoint(step=global_step)


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(
        device_type="cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",)
    )
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )
    trainer = FSDPSFTTrainer(
        config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh
    )
    trainer.fit()


if __name__ == "__main__":
    main()
