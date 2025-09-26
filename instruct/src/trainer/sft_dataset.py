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
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
- **NOTE**: We support multi-turn prompts.
Each parquet file contains
"""

from typing import List, Union

import pandas as pd

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from functools import partial

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset
    """

    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer,
        prompt_key="prompt",
        response_key="response",
        max_length=1024,
        truncation="error",
        pad_token_id=None,
        pad_input=False,
    ):
        assert truncation in ["error", "left", "right"]
        self.truncation = truncation

        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key
        self.response_key = response_key

        self.max_length = max_length
        self.pad_token_id = (
            pad_token_id if pad_token_id is not None else self.tokenizer.pad_token_id
        )
        self.pad_input = pad_input
        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(
                parquet_file, verbose=True
            )

    def _read_files_and_tokenize(self):

        def series_to_item(ls):
            import pandas, numpy

            while (
                isinstance(ls, (pandas.core.series.Series, numpy.ndarray))
                and len(ls) == 1
            ):
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

    @staticmethod
    def _tokenize_static(example, tokenizer, prompt_key, response_key, max_length, truncation, pad_token_id):
        prompt = example[prompt_key]
        response = example[response_key]

        # apply chat template
        if not isinstance(prompt, str):
            prompt_chat = list(prompt)
        else:
            prompt_chat = [{"role": "user", "content": prompt}]

        # string
        prompt_chat_str = tokenizer.apply_chat_template(
            prompt_chat, add_generation_prompt=True, tokenize=False
        )
        response_chat_str = response + tokenizer.eos_token

        # tokenize
        prompt_ids_output = tokenizer(
            prompt_chat_str, return_tensors="pt", add_special_tokens=False
        )
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        response_ids_output = tokenizer(
            response_chat_str, return_tensors="pt", add_special_tokens=False
        )
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat(
            (prompt_attention_mask, response_attention_mask), dim=-1
        )

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < max_length:
            padded_input_ids = (
                torch.ones(
                    size=(max_length - sequence_length,), dtype=input_ids.dtype
                )
                * pad_token_id
            )
            padded_attention_mask = torch.ones(  # NOTE: we use 1 here
                size=(max_length - sequence_length,), dtype=attention_mask.dtype
            )

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > max_length:
            if truncation == "left":
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-max_length :]
                attention_mask = attention_mask[-max_length :]
            elif truncation == "right":
                input_ids = input_ids[: max_length]
                attention_mask = attention_mask[: max_length]
            elif truncation == "error":
                raise NotImplementedError(
                    f"{sequence_length=} is larger than {max_length=}"
                )
            else:
                raise NotImplementedError(
                    f"Unknown truncation method {truncation}"
                )

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[: min(prompt_length, loss_mask.size(0))] = 0

        return {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
            "position_ids": position_ids.numpy(),
            "loss_mask": loss_mask.numpy(),
        }

    def _tokenize(self, example):
        return self._tokenize_static(
            example,
            self.tokenizer,
            self.prompt_key,
            self.response_key,
            self.max_length,
            self.truncation,
            self.pad_token_id
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        example = self.dataframe.iloc[item]
        data = self._tokenize(example)
        return {
            "input_ids": torch.tensor(data["input_ids"]),
            "attention_mask": torch.tensor(data["attention_mask"]),
            "position_ids": torch.tensor(data["position_ids"]),
            "loss_mask": torch.tensor(data["loss_mask"]),
        }

    def save_tokenized(self, path, num_proc=16):
        hf_dataset = HFDataset.from_pandas(self.dataframe)
        tokenize_fn = partial(
            self._tokenize_static,
            tokenizer=self.tokenizer,
            prompt_key=self.prompt_key,
            response_key=self.response_key,
            max_length=self.max_length,
            truncation=self.truncation,
            pad_token_id=self.pad_token_id
        )
        hf_dataset = hf_dataset.map(tokenize_fn, num_proc=num_proc)
        hf_dataset.to_pandas().to_parquet(path)


class TokenizedSFTDataset(Dataset):
    """
    This is an in-memory tokenized SFTDataset
    """

    def __init__(
        self,
        parquet_files: Union[str, List[str]],
    ):
        if not isinstance(parquet_files, List):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self._read_files()

    def _read_files(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        dataframe = pd.concat(dataframes)
        self.hf_dataset = HFDataset.from_pandas(dataframe)
        self.hf_dataset.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "position_ids", "loss_mask"],
        )

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, item):
        return self.hf_dataset[item]
