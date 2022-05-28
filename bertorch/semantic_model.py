# coding=utf-8

import abc
import torch
import torch.nn.functional as F

from torch import nn

class BaseModel(nn.Module):
    def __init__(self, pretrained_model, output_emb_size=None):
        super(BaseModel, self).__init__()
        
        self.bert = pretrained_model

        self.output_emb_size = output_emb_size if output_emb_size else 0
        self.pad_token_id = self.bert.config.pad_token_id
        self.hidden_size = self.bert.config.hidden_size
        if self.output_emb_size > 0:
            self.emb_reduce_linear = nn.Linear(self.hidden_size, output_emb_size)
    
    def encode(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        pooling_mode="linear",
        normalize_to_unit=False
    ):
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        
        if pooling_mode == "linear":
            emb = pooled_output
            if self.output_emb_size > 0:
                emb = self.emb_reduce_linear(emb)
        elif pooling_mode == "cls":
            emb = sequence_output[:, 0, :]
            if self.output_emb_size > 0:
                emb = self.emb_reduce_linear(emb)
        elif pooling_mode == "mean":
            if self.output_emb_size > 0:
                sequence_output = self.emb_reduce_linear(sequence_output)
            # set token embeddings to 0 for padding tokens
            attention_mask = torch.unsqueeze(attention_mask, dim=2)
            sequence_output = sequence_output * attention_mask
            emb = torch.sum(sequence_output, dim=1)
            seqlen = torch.sum(attention_mask, dim=1)
            emb /= seqlen
        else:
            raise NotImplementedError
        
        if normalize_to_unit:
            emb = F.normalize(emb, p=2, dim=-1)
        
        return emb

    def cosine_similarity(
        self,
        input_ids,
        pair_input_ids,
        attention_mask=None,
        token_type_ids=None,
        pair_attention_mask=None,
        pair_token_type_ids=None,
        pooling_mode="linear",
        return_matrix=False
    ):
        emb = self.encode(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pooling_mode=pooling_mode,
            normalize_to_unit=True
        )
        
        pair_emb = self.encode(
            pair_input_ids,
            attention_mask=pair_attention_mask,
            token_type_ids=pair_token_type_ids,
            pooling_mode=pooling_mode,
            normalize_to_unit=True
        )
        
        if return_matrix:
            similarity = torch.mm(emb, pair_emb.transpose(0, 1))
        else:
            similarity = torch.sum(emb * pair_emb, dim=-1)
        
        return similarity

    @abc.abstractmethod
    def forward(self):
        pass
    

class SimCSE(BaseModel):
    def __init__(
        self,
        pretrained_model,
        margin=0.2,
        scale=20,
        pooling_mode="linear",
        output_emb_size=None
    ):
        super().__init__(pretrained_model, output_emb_size)
        
        self.scale = scale
        self.margin = margin
        self.pooling_mode = pooling_mode
        
    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None
    ):
        similarity = self.cosine_similarity(
            input_ids=input_ids,
            pair_input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pair_attention_mask=attention_mask,
            pair_token_type_ids=token_type_ids,
            pooling_mode=self.pooling_mode,
            return_matrix=True
        )
        
        # substract margin from all positive samples cosine_sim
        margin_diag = torch.tensor([self.margin] * similarity.shape[0],
            device=similarity.device, dtype=similarity.dtype)        
        similarity -= torch.diag(margin_diag)

        # scale cosine similarity
        similarity *= self.scale
        
        labels = torch.arange(0, similarity.shape[0],
            device=similarity.device, dtype=torch.int64)
        
        loss = F.cross_entropy(similarity, labels)
        return loss


class BatchNegModel(BaseModel):
    def __init__(
        self,
        pretrained_model,
        margin=0.2,
        scale=20,
        pooling_mode="linear",
        output_emb_size=None
    ):
        super().__init__(pretrained_model, output_emb_size)
        
        self.scale = scale
        self.margin = margin
        self.pooling_mode = pooling_mode
    
    def forward(
        self,
        input_ids,
        pair_input_ids,
        attention_mask=None,
        token_type_ids=None,
        pair_attention_mask=None,
        pair_token_type_ids=None,
        mean_loss=False
    ):
        similarity = self.cosine_similarity(
            input_ids=input_ids,
            pair_input_ids=pair_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pair_attention_mask=pair_attention_mask,
            pair_token_type_ids=pair_token_type_ids,
            pooling_mode=self.pooling_mode,
            return_matrix=True
        )
        
        # substract margin from all positive samples cosine_sim
        margin_diag = torch.tensor([self.margin] * similarity.shape[0],
            device=similarity.device, dtype=similarity.dtype)
        similarity -= torch.diag(margin_diag)
        
        # scale cosine similarity
        similarity *= self.scale
        
        labels = torch.arange(0, similarity.shape[0],
            device=similarity.device, dtype=torch.int64)
        
        if mean_loss:
            loss = (F.cross_entropy(similarity, labels) + F.cross_entropy(similarity.t(), labels)) / 2
        else:
            loss = F.cross_entropy(similarity, labels)
        return loss


class SentenceBERT(BaseModel):
    def __init__(
        self,
        pretrained_model,
        num_labels,
        pooling_mode='mean',
        concat_rep=True,
        concat_diff=True,
        concat_multiply=False,
        output_emb_size=None,
    ):
        super().__init__(pretrained_model, output_emb_size)
        
        self.concat_rep = concat_rep
        self.concat_diff = concat_diff
        self.concat_multiply = concat_multiply
        self.pooling_mode = pooling_mode
        
        num = 0
        if concat_rep:
            num += 2
        if concat_diff:
            num += 1
        if concat_multiply:
            num += 1
        in_features = num * self.output_emb_size if self.output_emb_size > 0 else num * self.hidden_size
        self.classifier = nn.Linear(in_features, num_labels)        
        
    def forward(
        self,
        input_ids,
        pair_input_ids,
        attention_mask=None,
        token_type_ids=None,
        pair_attention_mask=None,
        pair_token_type_ids=None,
        labels=None
    ):
        emb = self.encode(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pooling_mode=self.pooling_mode,
            normalize_to_unit=False
        )
        
        pair_emb = self.encode(
            pair_input_ids,
            attention_mask=pair_attention_mask,
            token_type_ids=pair_token_type_ids,
            pooling_mode=self.pooling_mode,
            normalize_to_unit=False
        )
        
        concat_vectors = []
        if self.concat_rep:
            concat_vectors.append(emb)
            concat_vectors.append(pair_emb)
        
        if self.concat_diff:
            concat_vectors.append(torch.abs(emb - pair_emb))
        
        if self.concat_multiply:
            concat_vectors.append(emb * pair_emb)
        
        projection = torch.cat(concat_vectors, dim=-1)
        logits = self.classifier(projection)
        if labels is None:
            return logits
        loss = F.cross_entropy(logits, labels)
        return (loss, logits)


class MSECosineSimilarity(BaseModel):
    def __init__(
        self,
        pretrained_model,
        scale=1,
        pooling_mode='mlp',
        output_emb_size=None
    ):
        super().__init__(pretrained_model, output_emb_size)
        
        self.scale = scale
        self.pooling_mode = pooling_mode
    
    def forward(
        self,
        input_ids,
        pair_input_ids,
        labels,
        attention_mask=None,
        token_type_ids=None,
        pair_attention_mask=None,
        pair_token_type_ids=None,
    ):
        similarity = self.cosine_similarity(
            input_ids,
            pair_input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pair_attention_mask=pair_attention_mask,
            pair_token_type_ids=pair_token_type_ids,
            pooling_mode=self.pooling_mode
        )
        
        # scale cosine similarity
        similarity *= self.scale
        loss = F.mse_loss(similarity, labels)
        return loss
