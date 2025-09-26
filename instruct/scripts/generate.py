#!/usr/bin/env python3
"""
Multi-GPU inference script using Hugging Face Accelerate for data parallelism.
Processes JSONL input files and generates model responses.
"""

import argparse
import json
from pathlib import Path
import types
from typing import Any, Dict, List

import torch
import yaml
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


class JSONLDataset(Dataset):
    """Dataset class for loading JSONL files with message support."""

    def __init__(self, jsonl_path: str, messages_field: str = "messages"):
        self.data = []
        self.messages_field = messages_field

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def model_init(model_name: str, accelerator, use_cache: bool = False):
    """
    Initialize the model and tokenizer for instruct/chat models.

    Args:
        model_name: Name or path of the Hugging Face model
        accelerator: Accelerator instance for distributed setup

    Returns:
        tuple: (model, tokenizer, generation_config)
    """
    print(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # For multi-GPU data parallelism, don't use device_map='auto'
    # Let accelerate handle the device placement
    if accelerator.num_processes > 1:
        # Multi-GPU data parallelism - load on CPU first, let accelerate move to GPU
        if use_cache:
            from src.inference.fast_dllm.generation_utils_block import (
                DreamGenerationMixin,
            )
            from src.inference.fast_dllm.modeling_dream import DreamModel

            model = DreamModel.from_pretrained(
                model_name, torch_dtype=torch.float16, trust_remote_code=True
            )
            model = model.to("cuda").eval()

            model.diffusion_generate = types.MethodType(
                DreamGenerationMixin.diffusion_generate, model
            )
            model._sample = types.MethodType(DreamGenerationMixin._sample, model)
        else:
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
    else:
        # Single GPU - can use device_map='auto' for model parallelism if needed
        if use_cache:
            from src.inference.fast_dllm.generation_utils_block import (
                DreamGenerationMixin,
            )
            from src.inference.fast_dllm.modeling_dream import DreamModel

            model = DreamModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            model = model.to("cuda").eval()

            model.diffusion_generate = types.MethodType(
                DreamGenerationMixin.diffusion_generate, model
            )
            model._sample = types.MethodType(DreamGenerationMixin._sample, model)
        else:
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

    return model, tokenizer


def batch_generate(
    model,
    tokenizer,
    generation_config,
    batch_data: List[Dict[str, Any]],
    messages_field: str = "messages",
    verbose: bool = False,
) -> List[str]:
    """
    Generate responses for a batch of conversation inputs using chat templates.

    Args:
        model: The loaded model (potentially wrapped by Accelerate)
        tokenizer: The tokenizer
        generation_config: Generation configuration
        batch_data: List of data items from the dataset
        messages_field: Field name containing the messages list

    Returns:
        List of generated responses
    """
    # Extract messages from batch and format them
    formatted_prompts = []

    for item in batch_data:
        messages = item[messages_field]

        # Use the model's chat template
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except:
            formatted_prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{messages[-1]["content"].strip()}
<|im_end|>
<|im_start|>assistant
"""
        formatted_prompts.append(formatted_prompt)

    # Tokenize inputs
    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        padding=True,
    )

    # Move inputs to model device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Access the underlying model for generation (handle DDP wrapping)
    if hasattr(model, "module"):
        # Model is wrapped by DistributedDataParallel
        generation_model = model.module
    else:
        # Model is not wrapped
        generation_model = model

    # Generate responses
    with torch.no_grad():
        outputs = generation_model.diffusion_generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **generation_config,
        )

    # Decode only the new tokens (exclude input)
    input_length = inputs["input_ids"].shape[1]
    new_tokens = outputs[:, input_length:]

    # Decode generated text
    generated_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    # (Optional) Print the first generated text
    if verbose:
        print(f"Context:\n{formatted_prompts[0]}")
        print(f"Generation:\n{generated_texts[0]}")

    return generated_texts


def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU inference with Accelerate for instruct models"
    )
    parser.add_argument("--input", required=True, help="Input JSONL file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument(
        "--model",
        default="Dream-org/Dream-v0-Instruct-7B",
        help="Model name or path (default: Dream-org/Dream-v0-Instruct-7B)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size per GPU (default: 4)"
    )
    parser.add_argument(
        "--messages_field",
        default="messages",
        help="Field name containing messages list (default: messages)",
    )
    parser.add_argument(
        "--generation_config",
        default="src/inference/generation_config.yaml",
        help="Generation config file path (default: src/inference/generation_config.yaml)",
    )
    parser.add_argument(
        "--config_name",
        default="default",
        help="Config name (default: default)",
    )

    args = parser.parse_args()

    # Initialize accelerator
    accelerator = Accelerator()

    # Load model and tokenizer (only on main process to avoid conflicts)
    if accelerator.is_main_process:
        print(f"Using {accelerator.num_processes} GPU(s)")
        print("Initializing model...")

    # Load generation config
    with open(args.generation_config, "r", encoding="utf-8") as f:
        generation_config = yaml.safe_load(f)
    generation_config = generation_config[args.config_name]
    use_cache = generation_config.pop("use_cache")

    if use_cache and accelerator.is_main_process:
        print("WARNING: Using cache for acceleration.")

    model, tokenizer = model_init(args.model, accelerator, use_cache=use_cache)

    # Create dataset and dataloader
    dataset = JSONLDataset(args.input, args.messages_field)

    # For better control over data distribution, manually split the dataset
    if accelerator.num_processes > 1:
        # Use more even data distribution without dropping any samples
        total_samples = len(dataset)

        # Calculate start and end indices for each process
        samples_per_process = total_samples // accelerator.num_processes
        remainder = total_samples % accelerator.num_processes

        # The first 'remainder' processes handle one more sample
        if accelerator.process_index < remainder:
            start_idx = accelerator.process_index * (samples_per_process + 1)
            end_idx = start_idx + samples_per_process + 1
        else:
            start_idx = (
                remainder * (samples_per_process + 1)
                + (accelerator.process_index - remainder) * samples_per_process
            )
            end_idx = start_idx + samples_per_process

        # Create data subset for current process
        process_dataset = [dataset[i] for i in range(start_idx, end_idx)]

        print(
            f"GPU {accelerator.process_index}: processing samples {start_idx} to {end_idx-1} ({len(process_dataset)} samples)"
        )

        # Create data loader
        dataloader = DataLoader(
            process_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda x: x,
            drop_last=False,
        )

    else:
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda x: x
        )

    # Only prepare model with accelerator, not dataloader (since we manually distributed)
    if accelerator.num_processes > 1:
        model = accelerator.prepare(model)
    else:
        model, dataloader = accelerator.prepare(model, dataloader)

    # Storage for results
    all_results = []

    # Process batches
    if accelerator.is_main_process:
        print(
            f"Total {len(dataset)} samples to process, distributed across {accelerator.num_processes} GPUs"
        )

    # Synchronize all processes to ensure they all start processing
    accelerator.wait_for_everyone()

    with tqdm(
        total=len(dataloader),
        disable=not accelerator.is_main_process,
        desc=f"GPU {accelerator.process_index}",
    ) as pbar:
        for batch_idx, batch in enumerate(dataloader):
            # Generate responses
            batch_generated_texts = batch_generate(
                model,
                tokenizer,
                generation_config,
                batch,
                args.messages_field,
                verbose=(batch_idx == 0 and accelerator.is_main_process),
            )
            # Handle n > 1
            n = len(batch_generated_texts) // len(batch)
            batch_generated_texts = [
                batch_generated_texts[i * n : (i + 1) * n] for i in range(len(batch))
            ]

            # Create result entries
            batch_results = []
            for item, generated_texts in zip(batch, batch_generated_texts):
                result = item.copy()  # Keep original fields
                result["generations"] = generated_texts

                # Add processing metadata for debugging
                result["_gpu_id"] = accelerator.process_index
                result["_batch_id"] = batch_idx

                batch_results.append(result)

            all_results.extend(batch_results)
            pbar.update(1)

    print(f"GPU {accelerator.process_index} completed processing, handled {len(all_results)} samples")

    # Ensure all processes complete processing before gathering
    accelerator.wait_for_everyone()

    # Gather results from all processes
    if accelerator.num_processes > 1:
        # Gather results from all GPUs - use gather_object for variable-length lists
        all_results = accelerator.gather_for_metrics(
            all_results, use_gather_object=True
        )

    # Save results (only on main process)
    if accelerator.is_main_process:
        print(f"Total results collected: {len(all_results)}")
        print(f"Original dataset size: {len(dataset)}")

        # Sort by original index if available, or by GPU and batch info
        if all_results:
            try:
                # Try to sort by original ID/index
                if "id" in all_results[0]:
                    all_results.sort(key=lambda x: x.get("id", 0))
                elif "_gpu_id" in all_results[0] and "_batch_id" in all_results[0]:
                    all_results.sort(
                        key=lambda x: (x.get("_gpu_id", 0), x.get("_batch_id", 0))
                    )
            except:
                pass

        # Remove processing metadata
        clean_results = []
        for result in all_results:
            clean_result = {k: v for k, v in result.items() if not k.startswith("_")}
            clean_results.append(clean_result)

        print(f"Processing complete: {len(clean_results)} responses generated")

        # Write to output file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for result in clean_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"Results saved to: {output_path}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
