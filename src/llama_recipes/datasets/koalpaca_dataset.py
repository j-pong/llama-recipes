# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json

import torch
from torch.utils.data import Dataset


PROMPT_DICT = {
    "prompt_input": (
        "아래는 작업을 설명하는 명령어와 추가 컨텍스트를 제공하는 입력이 짝을 이루는 예제입니다. "
        "요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### 명령어:\n{instruction}\n\n### 입력:\n{input}\n\n### 응답:"
    ),
    "prompt_no_input": (
        "아래는 작업을 설명하는 명령어입니다. "
        "요청을 적절히 완료하는 응답을 작성하세요.\n\n"
        "### 명령어:\n{instruction}\n\n### 응답:"
    ),
}

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        with open(dataset_config.data_path) as f:
            self.ann = [json.loads(line) for line in f]
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:200]

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt + ann["output"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }
