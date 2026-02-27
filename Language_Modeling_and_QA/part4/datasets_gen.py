"""
Dataset classes for pre-training and fine-tuning.
Example submission.
"""
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class PretrainingDataset(Dataset):
    def __init__(self, file_path: str | Path, tokenizer, max_length: int = 256, stride: int | None = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride or max_length
        
        file_path = Path(file_path)
        cache_path = file_path.with_suffix(f".{max_length}.pt")

        if cache_path.exists():
            print(f"Loading cached tokenized data from {cache_path}...")
            self.token_ids = torch.load(cache_path)
        else:
            print(f"Tokenizing {file_path}... (This may take a while for large files)")
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            self.token_ids = tokenizer.encode(text)
            
            print(f"Saving tokenized data to {cache_path} for future use...")
            torch.save(self.token_ids, cache_path)

        if len(self.token_ids) <= max_length:
            self.num_sequences = 1
        else:
            self.num_sequences = (len(self.token_ids) - max_length) // self.stride + 1
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx = idx * self.stride
        end_idx = min(start_idx + self.max_length + 1, len(self.token_ids))
        sequence = self.token_ids[start_idx:end_idx]
        if len(sequence) < self.max_length + 1:
            sequence = sequence + [0] * (self.max_length + 1 - len(sequence))
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        labels = torch.tensor(sequence[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


class MultipleChoiceQADataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 256, num_choices: int = 4):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_choices = num_choices
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _format_choice_input(self, context: str, question: str, choice: str) -> str:
        return f"{context}\n\nQuestion: {question}\n\nAnswer: {choice}"
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        context = example["context"]
        question = example["question"]
        choices = example["choices"]
        answer = example.get("answer", -1)
        
        all_input_ids = []
        all_attention_masks = []
        
        for choice in choices:
            text = self._format_choice_input(context, question, choice)
            token_ids = self.tokenizer.encode(text)
            if len(token_ids) > self.max_length:
                token_ids = token_ids[:self.max_length]
            attention_mask = [1] * len(token_ids)
            padding_length = self.max_length - len(token_ids)
            token_ids = token_ids + [0] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            all_input_ids.append(token_ids)
            all_attention_masks.append(attention_mask)
        
        return {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_attention_masks, dtype=torch.long),
            "labels": torch.tensor(answer, dtype=torch.long),
        }
    
    @classmethod
    def from_json(cls, file_path: str | Path, tokenizer, **kwargs) -> "MultipleChoiceQADataset":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data, tokenizer, **kwargs)


def create_pretraining_dataloader(file_path, tokenizer, batch_size=8, max_length=256, stride=None, shuffle=True, num_workers=0):
    if stride is None:
        stride = max_length
    dataset = PretrainingDataset(file_path, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def create_qa_dataloader(data, tokenizer, batch_size=4, max_length=256, num_choices=4, shuffle=True, num_workers=0):
    if isinstance(data, (str, Path)):
        dataset = MultipleChoiceQADataset.from_json(data, tokenizer, max_length=max_length, num_choices=num_choices)
    else:
        dataset = MultipleChoiceQADataset(data, tokenizer, max_length=max_length, num_choices=num_choices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


# =============================================================================
# [CRITICAL FIX] Robust Generative Dataset
# =============================================================================

class GenerativeQADataset(Dataset):
    """
    Dataset for Generative Fine-tuning.
    Fix: Ensures target tokens exist and masking calculation is robust.
    """
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        context = example["context"]
        question = example["question"]
        choices = example["choices"]
        answer_idx = example.get("answer", -1)
        
        # 1. Build Prompt
        choices_str = "\n".join(f"{l}. {c}" for l, c in zip(["A", "B", "C", "D"], choices))
        prompt_text = f"{context}\n{question}\n{choices_str}\nThe answer is"
        
        # 2. Build Target (Robust Method)
        ans_char = self.answer_map.get(answer_idx, "A") # Fallback to A if invalid
        
        # Try adding space first (standard GPT2 format " A")
        target_text = " " + ans_char
        target_ids = self.tokenizer.encode(target_text)
        
        # [Fix] If tokenizer fails to produce tokens with space, try without space
        if len(target_ids) == 0:
            target_ids = self.tokenizer.encode(ans_char)
            
        # If STILL empty (unlikely), force a dummy token ID (e.g. 0) just to prevent crash
        if len(target_ids) == 0:
            target_ids = [0] 

        prompt_ids = self.tokenizer.encode(prompt_text)
        
        # 3. Concatenate
        full_ids = prompt_ids + target_ids
        
        # 4. Truncate (Prioritize keeping the ANSWER at the end)
        # We need length N for input and N for label, so N+1 total tokens before shift
        if len(full_ids) > self.max_length + 1:
            allowed_prompt_len = (self.max_length + 1) - len(target_ids)
            prompt_ids = prompt_ids[-allowed_prompt_len:] # Keep end of prompt
            full_ids = prompt_ids + target_ids
            
        full_tensor = torch.tensor(full_ids, dtype=torch.long)
        
        # 5. Shift (Standard Causal LM Logic)
        input_ids = full_tensor[:-1]
        labels = full_tensor[1:]
        
        # 6. Masking (Robust Method)
        # We want to train on the TARGET tokens.
        # target tokens are at the END of the `labels` array.
        # Length of target part in labels is `len(target_ids)`.
        # So we mask everything BEFORE that.
        
        labels_masked = labels.clone()
        target_len = len(target_ids)
        seq_len = len(labels)
        
        # Mask everything except the last `target_len` tokens
        mask_len = seq_len - target_len
        if mask_len > 0:
            labels_masked[:mask_len] = -100
            
        # 7. Padding
        if len(input_ids) < self.max_length:
            padding_len = self.max_length - len(input_ids)
            padding = torch.zeros(padding_len, dtype=torch.long)
            padding_labels = torch.full((padding_len,), -100, dtype=torch.long)
            
            input_ids = torch.cat([input_ids, padding])
            labels_masked = torch.cat([labels_masked, padding_labels])
            
        return {
            "input_ids": input_ids,
            "labels": labels_masked
        }

def create_generative_qa_dataloader(data, tokenizer, batch_size=8, max_length=512, shuffle=True, num_workers=0):
    if isinstance(data, (str, Path)):
        with open(data, "r", encoding="utf-8") as f:
            data = json.load(f)
    dataset = GenerativeQADataset(data, tokenizer, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)