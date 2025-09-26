"""
Optimized Collate Function for SFT Training
Wrap preprocessing logic into DataLoader's collate_fn to improve GPU utilization
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from torch.nn.utils.rnn import pad_sequence


class OptimizedSFTCollate:
    """
    Optimized SFT training collate function that completes all preprocessing on CPU
    """

    def __init__(self, config, tokenizer, pad_token_id: Optional[int] = None):
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id or tokenizer.pad_token_id

        # Cache configuration parameters to avoid repeated access
        self.enable_simple_preprocessing = getattr(
            config.data, "enable_simple_preprocessing", False
        )
        self.perbatch_cutoff = getattr(config.data, "perbatch_cutoff", False)
        self.perbatch_cutoff_type = getattr(config.data, "perbatch_cutoff_type", None)
        self.resp_cutoff_ratio = getattr(config.data, "resp_cutoff_ratio", 0.0)
        self.shuffle_pad_token_ids = getattr(
            config.data, "shuffle_pad_token_ids", False
        )
        self.unattn_pad_token_ids = getattr(config.data, "unattn_pad_token_ids", False)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate function that performs preprocessing on CPU
        """
        # Step 1: Extract and stack tensors
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch]).bool()
        position_ids = torch.stack([item["position_ids"] for item in batch])
        loss_mask = torch.stack([item["loss_mask"] for item in batch]).bool()

        # Step 2: Apply preprocessing on CPU
        if not self.enable_simple_preprocessing:
            input_ids, attention_mask, position_ids, loss_mask = (
                self._apply_preprocessing(
                    input_ids, attention_mask, position_ids, loss_mask
                )
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }

    def _apply_preprocessing(self, input_ids, attention_mask, position_ids, loss_mask):
        """
        Apply all preprocessing logic on CPU
        """
        # 1. Batch cutoff preprocessing
        if self.perbatch_cutoff:
            input_ids, attention_mask, position_ids, loss_mask = (
                self._apply_perbatch_cutoff(
                    input_ids, attention_mask, position_ids, loss_mask
                )
            )

        # 2. Random cutoff with different strategies
        elif self.perbatch_cutoff_type == "random":
            input_ids, attention_mask, position_ids, loss_mask = (
                self._apply_random_cutoff(
                    input_ids, attention_mask, position_ids, loss_mask
                )
            )

        elif self.perbatch_cutoff_type == "random_with_input_pad":
            input_ids, attention_mask, position_ids, loss_mask = (
                self._apply_random_with_input_pad(
                    input_ids, attention_mask, position_ids, loss_mask
                )
            )

        # 3. Response cutoff
        if np.random.rand() < self.resp_cutoff_ratio:
            input_ids, attention_mask, position_ids, loss_mask = (
                self._apply_response_cutoff(
                    input_ids, attention_mask, position_ids, loss_mask
                )
            )

        # 4. Shuffle pad tokens
        if self.shuffle_pad_token_ids:
            input_ids = self._shuffle_pad_tokens(input_ids, loss_mask)

        return input_ids, attention_mask, position_ids, loss_mask

    def _apply_perbatch_cutoff(
        self, input_ids, attention_mask, position_ids, loss_mask
    ):
        """Apply per-batch cutoff"""
        pad_lens = (input_ids == self.pad_token_id).sum(-1)
        cutoff_len = pad_lens.min()

        if cutoff_len > 0:
            seq_len = input_ids.shape[-1]
            input_ids = input_ids[:, : seq_len - cutoff_len].contiguous()
            attention_mask = attention_mask[:, : seq_len - cutoff_len].contiguous()
            position_ids = position_ids[:, : seq_len - cutoff_len].contiguous()
            loss_mask = loss_mask[:, : seq_len - cutoff_len].contiguous()

        return input_ids, attention_mask, position_ids, loss_mask

    def _apply_random_cutoff(self, input_ids, attention_mask, position_ids, loss_mask):
        """Apply random cutoff"""
        non_pad_lens = (input_ids != self.pad_token_id).sum(-1).cpu()
        cutoff_seq_len = np.random.choice(non_pad_lens)

        input_ids = input_ids[:, :cutoff_seq_len].contiguous()
        attention_mask = attention_mask[:, :cutoff_seq_len].contiguous()
        position_ids = position_ids[:, :cutoff_seq_len].contiguous()
        loss_mask = loss_mask[:, :cutoff_seq_len].contiguous()

        return input_ids, attention_mask, position_ids, loss_mask

    def _apply_random_with_input_pad(
        self, input_ids, attention_mask, position_ids, loss_mask
    ):
        """Apply random cutoff with input padding"""
        prompt_mask = loss_mask == 0
        response_mask = (loss_mask == 1) & (input_ids != self.pad_token_id)

        prompt_lens = prompt_mask.sum(-1)
        response_lens = response_mask.sum(-1)
        max_prompt_len = prompt_lens.max()
        pad_lens = max_prompt_len - prompt_lens

        kept_response_len = np.random.choice(response_lens.cpu())

        # Rebuild tensors
        batch_size = input_ids.shape[0]
        new_seq_len = max_prompt_len + kept_response_len

        new_input_ids = torch.full(
            (batch_size, new_seq_len),
            self.pad_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        new_attention_mask = torch.ones(
            (batch_size, new_seq_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        new_loss_mask = torch.ones(
            (batch_size, new_seq_len), dtype=loss_mask.dtype, device=loss_mask.device
        )

        for i in range(batch_size):
            kept_response_len_i = min(kept_response_len, response_lens[i])

            # Fill prompt
            new_input_ids[i, pad_lens[i] : pad_lens[i] + prompt_lens[i]] = input_ids[i][
                prompt_mask[i]
            ]

            # Fill response
            new_input_ids[
                i,
                pad_lens[i]
                + prompt_lens[i] : pad_lens[i]
                + prompt_lens[i]
                + kept_response_len_i,
            ] = input_ids[i][response_mask[i]][:kept_response_len_i]

            # Set attention and loss mask
            new_attention_mask[i, : pad_lens[i]] = 0
            new_loss_mask[i, : pad_lens[i] + prompt_lens[i]] = 0

        # Recalculate position_ids
        from verl.utils.model import compute_position_id_with_mask

        new_position_ids = compute_position_id_with_mask(new_attention_mask)

        return new_input_ids, new_attention_mask, new_position_ids, new_loss_mask

    def _apply_response_cutoff(
        self, input_ids, attention_mask, position_ids, loss_mask
    ):
        """Apply response cutoff"""
        resp_lens = loss_mask.sum(-1)
        cutoff_len = np.random.randint(1, resp_lens.min().item())

        input_ids = input_ids[:, :-cutoff_len].contiguous()
        attention_mask = attention_mask[:, :-cutoff_len].contiguous()
        position_ids = position_ids[:, :-cutoff_len].contiguous()
        loss_mask = loss_mask[:, :-cutoff_len].contiguous()

        return input_ids, attention_mask, position_ids, loss_mask

    def _shuffle_pad_tokens(self, input_ids, loss_mask):
        """Shuffle the positions of pad tokens"""

        def shuffle_pad_token_ids(
            input_ids_without_pad: List[int], pad_length: int, pad_token_id: int
        ) -> List[int]:
            is_pad_list = [False] * len(input_ids_without_pad) + [True] * pad_length
            np.random.shuffle(is_pad_list)

            new_input_ids = []
            for is_pad in is_pad_list:
                if is_pad:
                    new_input_ids.append(pad_token_id)
                else:
                    new_input_ids.append(input_ids_without_pad.pop(0))
            return new_input_ids

        new_input_ids = []
        for i in range(input_ids.shape[0]):
            loss_mask_i = loss_mask[i]
            prompt_ids_i = input_ids[i][~loss_mask_i]
            response_ids_i = input_ids[i][loss_mask_i]

            response_ids_without_pad_i = response_ids_i[
                response_ids_i != self.pad_token_id
            ].tolist()
            pad_length = (response_ids_i == self.pad_token_id).sum().item()

            new_response_ids_i = torch.tensor(
                shuffle_pad_token_ids(
                    response_ids_without_pad_i, pad_length, self.pad_token_id
                ),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )

            new_input_ids.append(torch.cat([prompt_ids_i, new_response_ids_i]))

        return torch.stack(new_input_ids)


class DynamicBatchSampler:
    """
    Dynamic batch sampler that intelligently groups by sequence length
    Reduce padding and improve GPU utilization
    """

    def __init__(self, dataset, batch_size, max_length, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.drop_last = drop_last

        # Group data by sequence length
        self.length_groups = self._group_by_length()

    def _group_by_length(self):
        """Group data by sequence length"""
        length_to_indices = {}

        for idx in range(len(self.dataset)):
            # Assume dataset has get_length method or direct access to length
            if hasattr(self.dataset, "get_length"):
                length = self.dataset.get_length(idx)
            else:
                # Fall back to direct data access
                sample = self.dataset[idx]
                length = len(sample["input_ids"])

            # Quantize length to buckets to reduce fragmentation
            bucket = ((length - 1) // 32 + 1) * 32  # Multiple of 32
            bucket = min(bucket, self.max_length)

            if bucket not in length_to_indices:
                length_to_indices[bucket] = []
            length_to_indices[bucket].append(idx)

        return length_to_indices

    def __iter__(self):
        """Generate batches"""
        all_batches = []

        for length, indices in self.length_groups.items():
            # Shuffle the order within the same length group
            np.random.shuffle(indices)

            # Create batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    all_batches.append(batch)

        # Shuffle the order of all batches
        np.random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):
        total_batches = 0
        for indices in self.length_groups.values():
            num_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size > 0:
                num_batches += 1
            total_batches += num_batches
        return total_batches


def create_optimized_dataloader(dataset, config, tokenizer, **dataloader_kwargs):
    """
    Create optimized DataLoader using custom collate_fn and sampling strategy
    """
    # Create optimized collate function
    collate_fn = OptimizedSFTCollate(config, tokenizer)

    # Whether to use dynamic batch sampling
    use_dynamic_batching = getattr(config.data, "use_dynamic_batching", False)

    if use_dynamic_batching:
        # Use dynamic batch sampler
        batch_sampler = DynamicBatchSampler(
            dataset=dataset,
            batch_size=config.data.train_batch_size,
            max_length=config.data.max_length,
            drop_last=dataloader_kwargs.get("drop_last", True),
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            num_workers=dataloader_kwargs.get("num_workers", 8),
            pin_memory=dataloader_kwargs.get("pin_memory", True),
            prefetch_factor=dataloader_kwargs.get("prefetch_factor", 4),
            persistent_workers=dataloader_kwargs.get("persistent_workers", True),
        )
    else:
        # Use standard sampler
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.data.train_batch_size,
            collate_fn=collate_fn,
            **dataloader_kwargs
        )

    return dataloader


# Usage example
def integrate_with_trainer(trainer_instance):
    """
    Integrate optimized dataloader in trainer
    """
    # Replace existing dataloader creation logic
    trainer_instance.train_dataloader = create_optimized_dataloader(
        dataset=trainer_instance.train_dataset,
        config=trainer_instance.config,
        tokenizer=trainer_instance.tokenizer,
        sampler=trainer_instance.train_sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    trainer_instance.val_dataloader = create_optimized_dataloader(
        dataset=trainer_instance.val_dataset,
        config=trainer_instance.config,
        tokenizer=trainer_instance.tokenizer,
        sampler=trainer_instance.val_sampler,
        batch_size=trainer_instance.config.data.micro_batch_size_per_gpu,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
