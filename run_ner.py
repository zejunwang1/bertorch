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
    read_ner_samples, 
    NerDataset,
    NerCollate 
)

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from bertorch.tokenization import BertTokenizer
from bertorch.modeling import BertModel
from bertorch.optimization import get_scheduler, AdamW
from bertorch.clas_model import NerTagger
from bertorch.ner_utils import EntityScore

def parse_args():
    parser = argparse.ArgumentParser(description="BERT for Sequence Labeling.")
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
        "--label_file",
        type=str,
        required=True,
        help="The full path of label_file."
    )
    parser.add_argument(
        "--tag",
        default="bios",
        type=str,
        choices=["bios", "bio"],
        help="Entity annotation method."
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
        help="The initial learning rate for BERT."
    )
    parser.add_argument(
        "--crf_learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for CRF."
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
        "--use_crf",
        action="store_true",
        help="Whether to add CRF layer."
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
    args.device = device

    # set seed
    set_seed(args.seed)
    
    # build dataloader
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    texts, label2id = read_ner_samples(args.train_data_file, label_path=args.label_file)
    # add pad token label
    pad_label_id = len(label2id)
    label2id[tokenizer.pad_token] = pad_label_id
    id2label = list(label2id.keys())
    args.id2label = id2label
    args.pad_label_id = pad_label_id
    
    o_label_id = label2id['O']
    train_dataset = NerDataset(texts, tokenizer, o_label_id, max_seq_length=args.max_seq_length)
    if args.local_rank == -1:
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            collate_fn=NerCollate(pad_label_id=pad_label_id, pad_token_id=tokenizer.pad_token_id),
            shuffle=True,
            pin_memory=True
        )
    else:
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            collate_fn=NerCollate(pad_label_id=pad_label_id, pad_token_id=tokenizer.pad_token_id),
            sampler=train_sampler,
            pin_memory=True
        )
    
    if args.dev_data_file:
        texts, _ = read_ner_samples(args.dev_data_file, label2id=label2id)
        dev_dataset = NerDataset(texts, tokenizer, o_label_id, max_seq_length=args.max_seq_length)
        dev_dataloader = DataLoader(
            dataset=dev_dataset,
            batch_size=args.batch_size,
            collate_fn=NerCollate(pad_label_id=pad_label_id, pad_token_id=tokenizer.pad_token_id),
            shuffle=False,
            pin_memory=True
        )
    
    # load pretrained model
    pretrained_model = BertModel.from_pretrained(
        args.pretrained_model_name_or_path
    )
    model = NerTagger(
        pretrained_model,
        num_labels=len(label2id),
        use_crf=args.use_crf
    )
    
    if args.init_from_ckpt is not None and os.path.isfile(args.init_from_ckpt):
        state_dict = torch.load(args.init_from_ckpt)
        model.load_state_dict(state_dict)
        logger.info("initializing weights from: {}".format(args.init_from_ckpt))
    
    model.to(device)
    
    # preparation before training
    num_training_steps = len(train_dataloader) * args.epochs
    warmup_steps = math.ceil(num_training_steps * args.warmup_proportion)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if args.use_crf:
        bert_param_optimizer = list(model.encoder.named_parameters())
        crf_param_optimizer = list(model.crf.named_parameters())
        linear_param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
                'lr': args.learning_rate
            },
            {
                'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': args.learning_rate
            },
            {
                'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
                'lr': args.crf_learning_rate
            },
            {
                'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': args.crf_learning_rate
            },
            {
                'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay,
                'lr': args.crf_learning_rate
            },
            {
                'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': args.crf_learning_rate
            }
        ]
    else:
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

    if n_gpu > 1:
        if args.local_rank == -1:
            model = nn.DataParallel(model)
        else:
            model = DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank,
                find_unused_parameters=True
            )
    
    global_step = 0
    best_metrics = 0
    os.makedirs(args.saved_dir, exist_ok=True)
    saved_model_file = os.path.join(args.saved_dir, "pytorch_model.bin")
    
    model.train()
    # metrics = EntityScore(id2label, tag=args.tag)
    tic_train = time.time()
    for epoch in range(1, args.epochs + 1):
        for step, batch in enumerate(train_dataloader, start=1):
            model.zero_grad()
            
            input_ids, token_type_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            loss, logits = model(
                input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                ignore_index=pad_label_id,
                labels=labels
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
                    dev_loss, dev_eval = evaluate(model, dev_dataloader, args)
                    f1 = dev_eval["f1"]
                    logger.info("eval loss: %.5f, F1: %.5f" % (dev_loss, f1))
                    if args.save_best_model:
                        if best_metrics < f1:
                            best_metrics = f1
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
def evaluate(model, dataloader, args):
    """
    Evaluate model performance on a given dataset.
    Compute precision recall and f1.
    """
    model.eval()
    metrics = EntityScore(args.id2label, tag=args.tag)
    total_loss = 0.
    for batch in dataloader:
        input_ids, token_type_ids, attention_mask, labels = batch
        input_ids = input_ids.to(args.device)
        token_type_ids = token_type_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        labels = labels.to(args.device)
        
        loss, logits = model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            ignore_index=args.pad_label_id,
            labels=labels
        )
        if torch.cuda.device_count() > 1 and args.local_rank == -1:
            total_loss += torch.mean(loss)
        else:
            total_loss += loss
        
        # remove [CLS] and [SEP] token label
        labels = labels[:, 1:].tolist()
        active_lens = (torch.sum(attention_mask, dim=-1) - 2).tolist()
        if args.use_crf:
            if torch.cuda.device_count() > 1:
                preds = model.module.crf.decode(logits, attention_mask).squeeze(0)
            else:
                preds = model.crf.decode(logits, attention_mask).squeeze(0)
            preds = preds[:, 1:].tolist()
        else:
            preds = torch.argmax(logits, dim=-1)[:, 1:].tolist()

        true_labels, pred_labels = [], []
        for i in range(len(labels)):
            true_labels.append(labels[i][: active_lens[i]])
            pred_labels.append(preds[i][: active_lens[i]])
        metrics.update(true_labels, pred_labels)
    
    total_loss /= len(dataloader)
    model.train()
    return (total_loss, metrics.result()[0])


if __name__ == "__main__":
    args = parse_args()
    train(args)
