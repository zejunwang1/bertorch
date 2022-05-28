# coding=utf-8

import argparse
import math
import os
import random
import time
import torch
import numpy as np

from loguru import logger
from bertorch.dataset import (
    read_semantic_samples, 
    BatchNegDataset, 
    PairWithLabelDataset,
    PairCollate
)

from scipy import stats
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from bertorch.tokenization import BertTokenizer
from bertorch.modeling import BertConfig, BertModel
from bertorch.optimization import get_scheduler, AdamW
from bertorch.semantic_model import BatchNegModel

def parse_args():
    parser = argparse.ArgumentParser(description="In-Batch-Negatives")
    parser.add_argument(
        "--local_rank",
        default=-1,
        type=int,
        help="Node rank for distributed training."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        default="bert-base-chinese",
        type=str,
        help="The pretrained huggingface model name or path."
    )
    parser.add_argument(
        "--init_from_ckpt",
        default=None,
        type=str,
        help="The path of checkpoint to be loaded."
    )
    parser.add_argument(
        "--train_data_file",
        type=str,
        required=True,
        help="The full path of train_data_file."
    )
    parser.add_argument(
        "--dev_data_file",
        default=None,
        type=str,
        help="The full path of dev_data_file."
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for training."
    )
    parser.add_argument(
        "--scheduler",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        default="linear",
        help="The name of the scheduler to use."
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.0,
        type=float,
        help="Warmup proption over the training process."
    )
    parser.add_argument(
        "--seed",
        default=1000,
        type=int,
        help="Random seed for initialization."
    )
    parser.add_argument(
        "--save_steps",
        default=100,
        type=int,
        help="The interval steps to save checkpoints."
    )
    parser.add_argument(
        "--logging_steps",
        default=20,
        type=int,
        help="The interval steps to logging."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
    )
    parser.add_argument(
        "--saved_dir",
        default="./checkpoint",
        type=str,
        help="The output directory where the model checkpoints will be written."
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Used for gradient normalization."
    )
    parser.add_argument(
        "--save_best_model",
        action="store_true",
        help="Whether to save checkpoint on best validation performance."
    )
    parser.add_argument(
        "--margin",
        default=0.2,
        type=float, 
        help="Margin beteween pos_sample and neg_samples."
    )
    parser.add_argument(
        "--scale", 
        default=20, 
        type=int, 
        help="Scale for pair-wise margin_rank_loss."
    )
    parser.add_argument(
        "--pooling_mode", 
        choices=["linear", "cls", "mean"], 
        default="linear", 
        help="Pooling method on the token embeddings."
    )
    parser.add_argument(
        "--output_emb_size", 
        default=None, 
        type=int, 
        help="Output sentence vector dimension, None means use hidden_size as output embedding size."
    )
    parser.add_argument(
        "--mean_loss",
        action="store_true",
        help="Whether to use mean cross-entropy loss."
    )
    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    # set device
    n_gpu = torch.cuda.device_count()
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    
    # set seed
    set_seed(args.seed)
    
    # build dataloader
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    texts = read_semantic_samples(args.train_data_file, is_pair=True)
    train_dataset = BatchNegDataset(texts, tokenizer, max_seq_length=args.max_seq_length)
    if args.local_rank == -1:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            collate_fn=PairCollate(pad_token_id=tokenizer.pad_token_id),
            shuffle=True,
            pin_memory=True
        )
    else:
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            collate_fn=PairCollate(pad_token_id=tokenizer.pad_token_id),
            sampler=train_sampler,
            pin_memory=True
        )
    
    if args.dev_data_file:
        texts = read_semantic_samples(args.dev_data_file, is_pair=True, has_label=True)
        dev_dataset = PairWithLabelDataset(texts, tokenizer, max_seq_length=args.max_seq_length)
        dev_dataloader = DataLoader(
            dataset=dev_dataset,
            batch_size=args.batch_size,
            collate_fn=PairCollate(pad_token_id=tokenizer.pad_token_id, has_label=True),
            shuffle=False,
            pin_memory=True
        )
    
    # load pretrained model
    pretrained_model = BertModel.from_pretrained(
        args.pretrained_model_name_or_path
    )
    model = BatchNegModel(
        pretrained_model, 
        margin=args.margin, 
        scale=args.scale,
        pooling_mode=args.pooling_mode,
        output_emb_size=args.output_emb_size
    )
   
    if args.init_from_ckpt is not None and os.path.isfile(args.init_from_ckpt):
        state_dict = torch.load(args.init_from_ckpt)
        model.load_state_dict(state_dict)
        logger.info("initializing weights from: {}".format(args.init_from_ckpt))
    
    model.to(device)
    if n_gpu > 1:
        if args.local_rank == -1:
            model = nn.DataParallel(model)
        else:
            model = DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank,
                find_unused_parameters=True
            )
        
    # preparation before training
    num_training_steps = len(train_dataloader) * args.epochs
    warmup_steps = math.ceil(num_training_steps * args.warmup_proportion)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_scheduler(
        name=args.scheduler, optimizer=optimizer,
        num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
    )
    
    global_step = 0
    best_metrics = 0
    os.makedirs(args.saved_dir, exist_ok=True)
    saved_model_file = os.path.join(args.saved_dir, "pytorch_model.bin")
    
    model.train()
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_dataloader, start=1):
            model.zero_grad()
            
            input_ids, token_type_ids, attention_mask, pair_input_ids, pair_token_type_ids, pair_attention_mask = batch
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            pair_input_ids = pair_input_ids.to(device)
            pair_token_type_ids = pair_token_type_ids.to(device)
            pair_attention_mask = pair_attention_mask.to(device)
            
            loss = model(
                input_ids,
                pair_input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                pair_token_type_ids=pair_token_type_ids,
                pair_attention_mask=pair_attention_mask,
                mean_loss=args.mean_loss
            )
            if n_gpu > 1 and args.local_rank == -1:
                loss = torch.mean(loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            global_step += 1
            if global_step % args.logging_steps == 0 and (args.local_rank == -1 or args.local_rank == 0):
                time_diff = time.time() - tic_train
                logger.info("global step: %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                            % (global_step, epoch, step, loss, args.logging_steps / time_diff))
                tic_train = time.time()
            
            if global_step % args.save_steps == 0 and (args.local_rank == -1 or args.local_rank == 0):
                if args.dev_data_file:
                    spearman_corr, pearson_corr = evaluate(model, dev_dataloader, device)
                    logger.info("spearman corr: %.4f, pearson corr: %.4f" % (spearman_corr, pearson_corr))
                    if args.save_best_model:
                        if best_metrics < spearman_corr:
                            best_metrics = spearman_corr
                            tokenizer.save_pretrained(args.saved_dir, save_tokenizer_config=True)
                            if n_gpu > 1:
                                torch.save(model.module.state_dict(), saved_model_file)
                            else:
                                torch.save(model.state_dict(), saved_model_file)
                        tic_train = time.time()
                        continue

                tokenizer.save_pretrained(args.saved_dir, save_tokenizer_config=True)
                if n_gpu > 1:
                    torch.save(model.module.state_dict(), saved_model_file)
                else:
                    torch.save(model.state_dict(), saved_model_file)
                tic_train = time.time()
                            

@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Evaluate model performance on a given dataset.
    Compute spearman correlation coefficient.
    """
    model.eval()
    similarity, label = [], []
    for batch in dataloader:
        input_ids, token_type_ids, attention_mask, pair_input_ids, pair_token_type_ids, pair_attention_mask, batch_label = batch
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        pair_input_ids = pair_input_ids.to(device)
        pair_token_type_ids = pair_token_type_ids.to(device)
        pair_attention_mask = pair_attention_mask.to(device)
        
        if torch.cuda.device_count() > 1:
            cosine_similarity = model.module.cosine_similarity
            pooling_mode = model.module.pooling_mode
        else:
            cosine_similarity = model.cosine_similarity
            pooling_mode = model.pooling_mode
        
        batch_similarity = cosine_similarity(
            input_ids,
            pair_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pair_attention_mask=pair_attention_mask,
            pair_token_type_ids=pair_token_type_ids,
            pooling_mode=pooling_mode
        )
        
        similarity.extend(batch_similarity.tolist())
        label.extend(batch_label.tolist())
        
    spearman_corr = stats.spearmanr(similarity, label).correlation
    pearson_corr = stats.pearsonr(similarity, label)[0]
    model.train()
    return (spearman_corr, pearson_corr)


if __name__ == "__main__":
    args = parse_args()
    train(args)
