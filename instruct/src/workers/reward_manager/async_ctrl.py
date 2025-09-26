"""Adapted from https://github.com/HKUNLP/critic-rl/tree/main"""

import asyncio
import json
import pickle
import os
from datetime import datetime

import torch
from verl import DataProto

from src.eval.score_utils import apply_verifiable_reward


async def run_with_semaphore(semaphore, func, *args, **kwargs):
    async with semaphore:
        return await func(*args, **kwargs)


def check_instruct_following(prompt: str, response: str):
    is_correct = True

    response = response.strip()
    if "You will NOT return anything except for the program." in prompt:
        if not (response.startswith("```") and response.endswith("```")):
            is_correct = False

    return is_correct


class AsyncRewardManager:
    """The reward manager."""

    def __init__(
        self, tokenizer, num_examine, num_concurrent_tasks=128, **kwargs
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.num_concurrent_tasks = num_concurrent_tasks

        self.force_instruct_following = kwargs.get("force_instruct_following", False)
        self.metadata_dir = kwargs.get("metadata_dir", None)

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        tasks = []
        prompt_strs, response_strs = [], []
        semaphore = asyncio.Semaphore(self.num_concurrent_tasks)
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][
                :prompt_length
            ].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(
                valid_prompt_ids, skip_special_tokens=True
            )
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=True
            )
            prompt_strs.append(prompt_str)
            response_strs.append(response_str)

            data_source = data_item.non_tensor_batch["data_source"]
            info = json.loads(data_item.non_tensor_batch["info"])
            tasks.append(
                run_with_semaphore(
                    semaphore,
                    apply_verifiable_reward,
                    response=response_str,
                    dataset=data_source,
                    info=info,
                )
            )

        async def run_tasks():
            results = await asyncio.gather(*tasks)
            scores, metadatas = zip(*results)
            return scores, metadatas

        scores, metadatas = asyncio.run(run_tasks())
        metadatas = [
            metadata | {"prompt": prompt_str, "response": response_str, "score": score}
            for metadata, prompt_str, response_str, score in zip(
                metadatas, prompt_strs, response_strs, scores
            )
        ]

        if self.metadata_dir is not None:
            timestamp = datetime.now().timestamp()
            os.makedirs(os.path.join(self.metadata_dir, "metadata"), exist_ok=True)
            metadata_path = os.path.join(
                self.metadata_dir, "metadata", f"{timestamp}.pickle"
            )
            with open(metadata_path, "wb") as handle:
                pickle.dump(metadatas, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved metadata to {metadata_path}")

        if self.force_instruct_following:
            scores = [
                score if check_instruct_following(prompt_str, response_str) else 0
                for score, prompt_str, response_str in zip(
                    scores, prompt_strs, response_strs
                )
            ]

        for i, score in enumerate(scores):
            reward_tensor[i, valid_response_length - 1] = score

        if self.num_examine:
            print("[prompt]", prompt_str)
            print("[response]", response_str)
            print("[info]", info)
            print("[score]", score)

        return reward_tensor
