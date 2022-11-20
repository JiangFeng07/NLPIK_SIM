#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/20 18:57
# @Author: lionel
import torch
from torch import nn
from transformers import BertModel, BertTokenizer

from models.text_repr import TextRepr


class SentenceBert(nn.Module):
    def __init__(self, encoder):
        super(SentenceBert, self).__init__()
        self.textRepr = TextRepr(encoder)
        self.fc = nn.Linear(3 * encoder.config.hidden_size, 2)

    def forward(self, a_token_ids, a_attention_mask, b_token_ids, b_attention_mask):
        a_token_embeddings = self.textRepr(a_token_ids, a_attention_mask)
        b_token_embeddings = self.textRepr(b_token_ids, b_attention_mask)
        features = torch.concat([a_token_embeddings, b_token_embeddings, a_token_embeddings - b_token_embeddings],
                                dim=1)
        logits = self.fc(features)
        return logits


if __name__ == '__main__':
    bert_model_path = '/tmp/chinese-roberta-wwm-ext'
    bert_model = BertModel.from_pretrained(bert_model_path)
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    a_texts = ['我是李四', '我是张三']
    b_texts = ['我是李四', '我是张三']
    encoder = tokenizer(a_texts, return_tensors='pt', padding=True)
    a_token_ids, a_attention_mask = encoder['input_ids'], encoder['attention_mask']

    encoder = tokenizer(b_texts, return_tensors='pt', padding=True)
    b_token_ids, b_attention_mask = encoder['input_ids'], encoder['attention_mask']

    sbert = SentenceBert(encoder=bert_model)
    logits = sbert(a_token_ids, a_attention_mask, b_token_ids, b_attention_mask)
    print(logits)
