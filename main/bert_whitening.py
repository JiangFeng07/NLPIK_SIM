#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/24 22:06
# @Author: lionel
import argparse
import os
import pickle
from scipy import spatial
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
    mean = np.mean(vecs, axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    U, S, Vh = np.linalg.svd(cov)
    W = np.dot(U, np.diag(1 / np.sqrt(S)))
    return W, -mean


def collect_fn(batch):
    a_texts, b_texts, labels = zip(*batch)
    a_outputs = tokenizer(a_texts, return_tensors='pt', padding=True)
    a_token_ids, a_attention_mask = a_outputs['input_ids'], a_outputs['attention_mask']
    b_outputs = tokenizer(b_texts, return_tensors='pt', padding=True)
    b_token_ids, b_attention_mask = b_outputs['input_ids'], b_outputs['attention_mask']
    a_token_ids = a_token_ids.to(device)
    a_attention_mask = a_attention_mask.to(device)
    b_token_ids = b_token_ids.to(device)
    b_attention_mask = b_attention_mask.to(device)

    return a_token_ids, a_attention_mask, b_token_ids, b_attention_mask, labels


def get_global_data_mean_cov():
    sentence_expr = TextRepr(encoder=bert_model)
    dataset = BQDataset(os.path.join(args.file_path, 'train.tsv'))
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collect_fn)
    vecs = []
    with tqdm(total=len(loader), desc='程序执行进度') as pbar:
        for index, batch in enumerate(loader):
            a_token_ids, a_attention_mask, b_token_ids, b_attention_mask, _ = batch
            embeddings = sentence_expr(a_token_ids, a_attention_mask)
            for embedding in embeddings.detach().numpy():
                vecs.append(embedding)
            embeddings = sentence_expr(a_token_ids, a_attention_mask)
            for embedding in embeddings.detach().numpy():
                vecs.append(embedding)
            pbar.update()
    W, mean = compute_kernel_bias(np.array(vecs))
    pickle.dump({'kernels': W, 'bias': mean}, open(args.model_path, 'wb'))


def embedding_wihten(embedding, kernels, bias):
    embedding = np.dot((embedding + bias), kernels)
    return embedding


def evaluate():
    whiten = pickle.load(open(args.model_path, 'rb'))
    kernels = whiten['kernels']
    bias = whiten['bias']
    sentence_expr = TextRepr(encoder=bert_model)
    dataset = BQDataset(os.path.join(args.file_path, 'train.tsv'))
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collect_fn)
    for batch in loader:
        a_token_ids, a_attention_mask, b_token_ids, b_attention_mask, labels = batch
        embeddings_a = sentence_expr(a_token_ids, a_attention_mask)
        embeddings_a = embedding_wihten(embeddings_a.detach().numpy(), kernels, bias)
        embeddings_b = sentence_expr(b_token_ids, b_attention_mask)
        embeddings_b = embedding_wihten(embeddings_b.detach().numpy(), kernels, bias)
        for embedding_a, embedding_b, label in zip(embeddings_a, embeddings_b, labels):
            cos_sim = 1 - spatial.distance.cosine(embedding_a, embedding_b)
            print(cos_sim, label)
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', help='训练数据路径', type=str, default='/tmp/bq_corpus/')
    parser.add_argument('--bert_model_path', help='预训练模型路径', type=str, default='/tmp/chinese-roberta-wwm-ext')
    parser.add_argument('--batch_size', help='', type=int, default=64)
    parser.add_argument('--model_path', help='模型存储路径', type=str, default='/tmp/bert_whiten.pkl')
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    bert_model = BertModel.from_pretrained(args.bert_model_path)
    # get_global_data_mean_cov()
    evaluate()
