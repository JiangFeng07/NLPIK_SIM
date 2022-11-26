#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/24 22:06
# @Author: lionel
import argparse
import os
import pickle

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from datasets.bq_dataset import BQDataset

torch.set_printoptions(8)
from models.sentence_repr import TextRepr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_kernel_bias(vecs):
    """vecs.shape = [num_samples, embbeding_size]"""
    mean = torch.mean(vecs, dim=0, keepdim=True)
    cov = torch.cov(vecs.T)
    U, S, Vh = torch.linalg.svd(cov)
    print(U)
    W = torch.matmul(U, torch.diag(1 / torch.sqrt(S)))
    return W, -mean


def collect_fn(batch):
    a_texts, b_texts, _ = zip(*batch)
    a_outputs = tokenizer(a_texts, return_tensors='pt', padding=True)
    a_token_ids, a_attention_mask = a_outputs['input_ids'], a_outputs['attention_mask']
    b_outputs = tokenizer(b_texts, return_tensors='pt', padding=True)
    b_token_ids, b_attention_mask = b_outputs['input_ids'], b_outputs['attention_mask']
    a_token_ids = a_token_ids.to(device)
    a_attention_mask = a_attention_mask.to(device)
    b_token_ids = b_token_ids.to(device)
    b_attention_mask = b_attention_mask.to(device)
    return a_token_ids, a_attention_mask, b_token_ids, b_attention_mask


def get_global_data_mean_cov():
    sentence_expr = TextRepr(encoder=bert_model)
    dataset = BQDataset(os.path.join(args.file_path, 'dev.tsv'))
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collect_fn)
    sum_embeddings = torch.zeros(size=(1, bert_model.config.hidden_size), dtype=torch.float)
    vecs = []
    with tqdm(total=len(loader), desc='程序执行进度') as pbar:
        for index, batch in enumerate(loader):
            a_token_ids, a_attention_mask, b_token_ids, b_attention_mask = batch
            embeddings = sentence_expr(a_token_ids, a_attention_mask)
            vecs.append(embeddings)
            embeddings = sentence_expr(a_token_ids, a_attention_mask)
            vecs.append(embeddings)
            pbar.update()

    W, mean = compute_kernel_bias(features)
    pickle.dump({'kernels': W, 'bias': mean}, open('bert_whiten.pkl', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', help='训练数据路径', type=str, default='/tmp/bq_corpus/')
    parser.add_argument('--bert_model_path', help='预训练模型路径', type=str, default='/tmp/chinese-roberta-wwm-ext')
    parser.add_argument('--epochs', help='训练轮数', type=int, default=100)
    parser.add_argument('--dropout', help='', type=float, default=0.5)
    parser.add_argument('--warm_up_ratio', help='', type=float, default=0.1)
    parser.add_argument('--embedding_size', help='', type=int, default=100)
    parser.add_argument('--batch_size', help='', type=int, default=64)
    parser.add_argument('--hidden_size', help='', type=int, default=200)
    parser.add_argument('--num_layers', help='', type=int, default=1)
    parser.add_argument('--lr', help='学习率', type=float, default=1e-3)
    parser.add_argument('--mode', help='长文本处理方式', type=str, default='cut')
    parser.add_argument('--model_path', help='模型存储路径', type=str, default='/tmp/bq_corpus/bq_sbert.pt')
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)

    bert_model_path = '/tmp/chinese-roberta-wwm-ext'
    bert_model = BertModel.from_pretrained(bert_model_path)
    texts = ['我是李四', '我是张三']
    encoder = tokenizer(texts, return_tensors='pt', padding=True)
    token_ids, attention_mask = encoder['input_ids'], encoder['attention_mask']
    pool = TextRepr(encoder=bert_model, mode='mean')
    features = pool(token_ids, attention_mask)

    get_global_data_mean_cov()
