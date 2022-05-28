# coding=utf-8

import json
import torch
from typing import Dict, List, Tuple, Union
from torch.utils.data import Dataset, DataLoader


def read_clas_samples(
    data_path, 
    label_path: str = None,
    is_pair: bool = False, 
    sep: str = '\t', 
    label2id: Dict[str, int] = None
):
    if label2id is None:
        if label_path is None:
            raise ValueError("label_path and label2id can't be empty at the same time.")
        labels = []
        label2id = {}
        with open(label_path, mode='r', encoding='utf-8') as label_handle:
            for line in label_handle:
                line = line.strip()
                if line:
                    labels.append(line)
        for index, label in enumerate(labels):
            label2id[label] = index
    
    texts, labels = [], []
    with open(data_path, mode='r', encoding='utf-8') as data_handle:
        for line in data_handle:
            data = line.rstrip().split(sep)
            if is_pair:
                if len(data) != 3:
                    continue
                texts.append((data[0], data[1]))
                labels.append(label2id[data[2]])
            else:
                if len(data) != 2:
                    continue
                texts.append(data[0])
                labels.append(label2id[data[1]])

    return (texts, labels, label2id)
                

def read_nli_samples(
    data_path, 
    label_path: str = None,
    sep: str = '\t', 
    label2id: Dict[str, int] = None
):
    if label2id is None:
        labels = []
        label2id = {}
        with open(label_path, mode='r', encoding='utf-8') as label_handle:
            for line in label_handle:
                line = line.strip()
                if line:
                    labels.append(line)
        for index, label in enumerate(labels):
            label2id[label] = index
    
    texts = []
    with open(data_path, mode='r', encoding='utf-8') as data_handle:
        for line in data_handle:
            data = line.rstrip().split(sep)
            if len(data) != 3:
                continue
            texts.append((data[0], data[1], label2id[data[2]]))
    
    return (texts, label2id)
    

def read_semantic_samples(
    data_path, 
    is_pair: bool = False, 
    has_label: bool = False, 
    sep: str = '\t'
):
    texts = []
    with open(data_path, mode='r', encoding='utf-8') as data_handle:
        for line in data_handle:
            if is_pair:
                data = line.rstrip().split(sep)
                if has_label:
                    if len(data) != 3:
                        continue
                    label = float(data[2]) if '.' in data[2] else int(data[2])
                    texts.append((data[0], data[1], label))
                else:
                    if len(data) != 2:
                        continue
                    texts.append((data[0], data[1]))
            else:
                if line:
                    texts.append(line)
    return texts


def read_ner_samples(
    data_path, 
    label_path: str = None, 
    label2id: Dict[str, int] = None
):
    if label2id is None:
        if label_path is None:
            raise ValueError("label_path and label2id can't be empty at the same time.")
        labels, label2id = [], {}
        with open(label_path, mode='r', encoding='utf-8') as label_handle:
            for line in label_handle:
                line = line.strip()
                if line:
                    labels.append(line)
        for index, label in enumerate(labels):
            label2id[label] = index
    
    texts = []
    with open(data_path, mode='r', encoding='utf-8') as data_handle:
        for line in data_handle:
            line = line.rstrip()
            if line:
                data = json.loads(line)
                assert len(data["text"]) == len(data["label"])
                label = [label2id[_] for _ in data["label"]]
                texts.append((data["text"], label))
    return (texts, label2id)
        

class ClasDataset(Dataset):
    def __init__(
        self, texts: Union[List[str], List[Tuple[str]]], labels: List[int], tokenizer, max_seq_length: int = 512
    ):
        self.data = tokenizer(texts, truncation=True, max_length=max_seq_length)
        self.labels = labels
        
    def __len__(self):
        """
        Returns the number of samples in dataset.
        """
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Basic function of `ClasDataset` to get sample from dataset with a given 
        index.
        """
        return [
            self.data['input_ids'][idx],
            self.data['token_type_ids'][idx],
            self.data['attention_mask'][idx],
            self.labels[idx]
        ]
    

class SimcseDataset(Dataset):
    def __init__(
        self, texts: List[str], tokenizer, max_seq_length: int = 512
    ):
        self.data = tokenizer(texts, truncation=True, max_length=max_seq_length)
    
    def __len__(self):
        """
        Returns the number of samples in dataset.
        """
        return len(self.data['input_ids'])
    
    def __getitem__(self, idx):
        """
        Basic function of `SimcseDataset` to get sample from dataset with a given 
        index.
        """
        return [
            self.data['input_ids'][idx],
            self.data['token_type_ids'][idx],
            self.data['attention_mask'][idx]
        ]


class BatchNegDataset(Dataset):
    def __init__(
        self, texts: List[Tuple[str]], tokenizer, max_seq_length: int = 512
    ):
        text_a_list = [_[0] for _ in texts]
        text_b_list = [_[1] for _ in texts]
        self.data = tokenizer(text_a_list, truncation=True, max_length=max_seq_length)
        self.pair_data = tokenizer(text_b_list, truncation=True, max_length=max_seq_length)
    
    def __len__(self):
        """
        Returns the number of samples in dataset.
        """
        return len(self.data['input_ids'])
    
    def __getitem__(self, idx):
        """
        Basic function of `BatchNegDataset` to get sample from dataset with a given 
        index.
        """
        return [
            self.data['input_ids'][idx],
            self.data['token_type_ids'][idx],
            self.data['attention_mask'][idx],
            self.pair_data['input_ids'][idx],
            self.pair_data['token_type_ids'][idx],
            self.pair_data['attention_mask'][idx]
        ]


class PairWithLabelDataset:
    def __init__(
        self, texts, tokenizer, max_seq_length: int = 512
    ):
        text_a_list = [_[0] for _ in texts]
        text_b_list = [_[1] for _ in texts]
        labels = [_[2] for _ in texts]
        self.data = tokenizer(text_a_list, truncation=True, max_length=max_seq_length)
        self.pair_data = tokenizer(text_b_list, truncation=True, max_length=max_seq_length)
        self.labels = labels
    
    def __len__(self):
        """
        Returns the number of samples in dataset.
        """
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Basic function of `PairWithLabelDataset` to get sample from dataset with a given 
        index.
        """
        return [
            self.data['input_ids'][idx],
            self.data['token_type_ids'][idx],
            self.data['attention_mask'][idx],
            self.pair_data['input_ids'][idx],
            self.pair_data['token_type_ids'][idx],
            self.pair_data['attention_mask'][idx],
            self.labels[idx]
        ]


class NerDataset(Dataset):
    def __init__(
        self, texts, tokenizer, o_label_id, max_seq_length: int = 512
    ):
        input_ids_list, token_type_ids_list, attention_mask_list, labels = [], [], [], []
        for text, label in texts:
            input_ids = tokenizer.convert_tokens_to_ids(text, add_special_tokens=True)
            label = [o_label_id] + label + [o_label_id]
            if len(input_ids) > max_seq_length:
                input_ids = input_ids[: max_seq_length]
                label = label[: max_seq_length]
                input_ids[-1] = tokenizer.sep_token_id
                label[-1] = o_label_id
                
            token_type_ids = [0] * len(input_ids)
            attention_mask = [1] * len(input_ids)
            input_ids_list.append(input_ids)
            token_type_ids_list.append(token_type_ids)
            attention_mask_list.append(attention_mask)
            labels.append(label)
        
        self.data = {
            "input_ids": input_ids_list,
            "token_type_ids": token_type_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels    
        }
    
    def __len__(self):
        """
        Returns the number of samples in dataset.
        """
        return len(self.data["labels"])
    
    def __getitem__(self, idx):
        """
        Basic function of `NerDataset` to get sample from dataset with a given 
        index.
        """
        return [
            self.data["input_ids"][idx],
            self.data["token_type_ids"][idx],
            self.data["attention_mask"][idx],
            self.data["labels"][idx]
        ]


class ClasCollate:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        input_ids = [item[0] for item in batch]
        token_type_ids = [item[1] for item in batch]
        attention_mask = [item[2] for item in batch]
        labels = [item[3] for item in batch]
        
        # padding
        max_seq_len = max(len(l) for l in input_ids)
        for i in range(len(input_ids)):
            pad_num = max_seq_len - len(input_ids[i])
            input_ids[i].extend([self.pad_token_id] * pad_num)
            token_type_ids[i].extend([0] * pad_num)
            attention_mask[i].extend([0] * pad_num)
        
        input_ids = torch.as_tensor(input_ids)
        token_type_ids = torch.as_tensor(token_type_ids)
        attention_mask = torch.as_tensor(attention_mask)
        labels = torch.as_tensor(labels)
        return [input_ids, token_type_ids, attention_mask, labels]


class SimcseCollate:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        input_ids = [item[0] for item in batch]
        token_type_ids = [item[1] for item in batch]
        attention_mask = [item[2] for item in batch]
    
        # padding
        max_seq_len = max(len(l) for l in input_ids)
        for i in range(len(input_ids)):
            pad_num = max_seq_len - len(input_ids[i])
            input_ids[i].extend([self.pad_token_id] * pad_num)
            token_type_ids[i].extend([0] * pad_num)
            attention_mask[i].extend([0] * pad_num)

        input_ids = torch.as_tensor(input_ids)
        token_type_ids = torch.as_tensor(token_type_ids)
        attention_mask = torch.as_tensor(attention_mask)
        return [input_ids, token_type_ids, attention_mask]
        

class PairCollate:
    def __init__(self, pad_token_id: int = 0, has_label: bool = False):
        self.pad_token_id = pad_token_id
        self.has_label = has_label
    
    def __call__(self, batch):
        input_ids = [item[0] for item in batch]
        token_type_ids = [item[1] for item in batch]
        attention_mask = [item[2] for item in batch]
        pair_input_ids = [item[3] for item in batch]
        pair_token_type_ids = [item[4] for item in batch]
        pair_attention_mask = [item[5] for item in batch]
        
        if self.has_label:
            labels = [item[6] for item in batch]
        
        # padding
        max_seq_len_a = max(len(l) for l in input_ids)
        for i in range(len(input_ids)):
            pad_num = max_seq_len_a - len(input_ids[i])
            input_ids[i].extend([self.pad_token_id] * pad_num)
            token_type_ids[i].extend([0] * pad_num)
            attention_mask[i].extend([0] * pad_num)
        
        max_seq_len_b = max(len(l) for l in pair_input_ids)
        for i in range(len(pair_input_ids)):
            pad_num = max_seq_len_b - len(pair_input_ids[i])
            pair_input_ids[i].extend([self.pad_token_id] * pad_num)
            pair_token_type_ids[i].extend([0] * pad_num)
            pair_attention_mask[i].extend([0] * pad_num)
        
        input_ids = torch.as_tensor(input_ids)
        token_type_ids = torch.as_tensor(token_type_ids)
        attention_mask = torch.as_tensor(attention_mask)
        pair_input_ids = torch.as_tensor(pair_input_ids)
        pair_token_type_ids = torch.as_tensor(pair_token_type_ids)
        pair_attention_mask = torch.as_tensor(pair_attention_mask)
        
        if self.has_label:
            labels = torch.as_tensor(labels)
            return [
                input_ids, token_type_ids, attention_mask,
                pair_input_ids, pair_token_type_ids, pair_attention_mask,
                labels
            ]
        
        return [
            input_ids, token_type_ids, attention_mask,
            pair_input_ids, pair_token_type_ids, pair_attention_mask
        ]


class NerCollate:
    def __init__(self, pad_label_id, pad_token_id: int = 0):
        self.pad_label_id = pad_label_id
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        input_ids = [item[0] for item in batch]
        token_type_ids = [item[1] for item in batch]
        attention_mask = [item[2] for item in batch]
        labels = [item[3] for item in batch]
        
        # padding
        max_seq_len = max(len(l) for l in input_ids)
        for i in range(len(input_ids)):
            pad_num = max_seq_len - len(input_ids[i])
            input_ids[i].extend([self.pad_token_id] * pad_num)
            token_type_ids[i].extend([0] * pad_num)
            attention_mask[i].extend([0] * pad_num)
            labels[i].extend([self.pad_label_id] * pad_num)
        
        input_ids = torch.as_tensor(input_ids)
        token_type_ids = torch.as_tensor(token_type_ids)
        attention_mask = torch.as_tensor(attention_mask)
        labels = torch.as_tensor(labels)
        return [input_ids, token_type_ids, attention_mask, labels]
