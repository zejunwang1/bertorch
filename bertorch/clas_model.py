# coding=utf-8

import torch
from torch import nn
from .crf import CRF

class Classifier(nn.Module):
    def __init__(self, pretrained_model, num_labels):
        super(Classifier, self).__init__()
        
        self.encoder = pretrained_model
        self.num_labels = num_labels
        self.hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
    
    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        labels=None
    ):
        outputs = self.encoder(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return (loss, logits)

        return logits
        

class NerTagger(nn.Module):
    def __init__(self, pretrained_model, num_labels, use_crf=False):
        super(NerTagger, self).__init__()
        
        self.encoder = pretrained_model
        self.num_labels = num_labels
        self.use_crf = use_crf
        
        self.pad_token_id = self.encoder.config.pad_token_id
        self.hidden_size = self.encoder.config.hidden_size

        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        if self.use_crf:
            self.crf = CRF(self.num_labels, batch_first=True)
    
    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        ignore_index=None,
        labels=None
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            attention_mask[input_ids==self.pad_token_id] = 0

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        if labels is not None:
            if self.use_crf:
                loss = -self.crf(emissions=logits, tags=labels, mask=attention_mask)
            else:
                # Only keep active parts of the loss
                if ignore_index is None:
                    loss_fct = nn.CrossEntropyLoss()
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                else:
                    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return (loss, logits)

        return logits
