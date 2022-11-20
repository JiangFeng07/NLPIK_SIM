#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/20 16:37
# @Author: lionel
import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer, models


class TextRepr(nn.Module):
    def __init__(self, encoder, mode='mean'):
        """
        :param encoder: 预训练模型
        :param mode: 向量抽取策略：mean，max，cls
        """
        super(TextRepr, self).__init__()
        self.encoder = encoder
        self.mode = mode

    def forward(self, token_ids, attention_mask):
        token_embeddings = self.encoder(input_ids=token_ids, attention_mask=attention_mask)[0]
        if self.mode == 'cls':
            features = token_embeddings[:, 0, :]
        elif self.mode == 'max':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = 1e-9
            features = torch.max(token_embeddings, dim=1)[0]
        elif self.mode == 'mean':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            features = sum_embeddings / sum_mask
        else:
            return None
        return features


if __name__ == '__main__':
    bert_model_path = '/tmp/chinese-roberta-wwm-ext'
    bert_model = BertModel.from_pretrained(bert_model_path)
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    texts = ['我是李四', '我是张三']
    encoder = tokenizer(texts, return_tensors='pt', padding=True)
    token_ids, attention_mask = encoder['input_ids'], encoder['attention_mask']
    pool = TextRepr(encoder=bert_model, mode='mean')
    features = pool(token_ids, attention_mask)
    print(features)

    word_embedding_model = models.Transformer(bert_model_path, max_seq_length=128)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    a = model.encode(texts)
    print(a)
