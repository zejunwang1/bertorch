# coding=utf-8

import argparse
import os
import torch

from loguru import logger
from torch.utils.data import DataLoader
from bertorch.dataset import (
    read_semantic_samples,
    BatchNegDataset,
    PairCollate
)
from bertorch.modeling import BertModel
from bertorch.tokenization import BertTokenizer
from bertorch.semantic_model import BaseModel


def parse_args():
    parser = argparse.ArgumentParser(description="Cosine similarity of sentence pairs.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="bert-base-chinese",
        type=str,
        help="The pretrained huggingface model name or path."
    )
    parser.add_argument(
        "--input_file",
        required=True,
        type=str,
        help="The path of sentence pairs file."
    )
    parser.add_argument(
        "--output_file",
        required=True,
        type=str,
        help="The path of predictions."
    )
    parser.add_argument(
        "--init_from_ckpt",
        required=True,
        type=str,
        help="The path of checkpoint to be loaded."
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for prediction."
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
    )
    parser.add_argument(
        "--pooling_mode",
        default="linear",
        choices=["linear", "cls", "mean"],
        help="Pooling method on the token embeddings."
    )
    parser.add_argument(
        "--output_emb_size",
        default=None,
        type=int,
        help="Output sentence vector dimension, None means use hidden_size as output embedding size."
    )
    args = parser.parse_args()
    return args


def predict(args):
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build dataloader
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    texts = read_semantic_samples(args.input_file, is_pair=True)
    dataset = BatchNegDataset(texts, tokenizer, max_seq_length=args.max_seq_length)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        collate_fn=PairCollate(pad_token_id=tokenizer.pad_token_id),
        shuffle=False,
        pin_memory=True
    )

    # load model
    pretrained_model = BertModel.from_pretrained(args.pretrained_model_name_or_path)
    model = BaseModel(pretrained_model, output_emb_size=args.output_emb_size)
    
    if os.path.isfile(args.init_from_ckpt):
        state_dict = torch.load(args.init_from_ckpt)
        keys = state_dict.keys()
        if "classifier.weight" in keys:
            state_dict.pop("classifier.weight")
            state_dict.pop("classifier.bias")
        model.load_state_dict(state_dict)
        logger.info("initializing weights from: {}".format(args.init_from_ckpt))

    model.to(device)
    model.eval()

    similarity = []
    for batch in dataloader:
        input_ids, token_type_ids, attention_mask, pair_input_ids, pair_token_type_ids, pair_attention_mask = batch
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        pair_input_ids = pair_input_ids.to(device)
        pair_token_type_ids = pair_token_type_ids.to(device)
        pair_attention_mask = pair_attention_mask.to(device)

        cosine_sim = model.cosine_similarity(
            input_ids,
            pair_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pair_attention_mask=pair_attention_mask,
            pair_token_type_ids=pair_token_type_ids,
            pooling_mode=args.pooling_mode
        )
        similarity.extend(cosine_sim.tolist())
    
    with open(args.output_file, mode='w', encoding='utf-8') as handle:
        for (text_a, text_b), s in zip(texts, similarity):
            handle.write("{}\t{}\t{}\n".format(text_a, text_b, s))


if __name__ == "__main__":
    args = parse_args()
    predict(args)
