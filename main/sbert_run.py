#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/20 22:46
# @Author: lionel
import argparse
import os

import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

from datasets.bq_dataset import BQDataset
from torch.utils import data

from models.sentence_bert import SentenceBert

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    labels = torch.LongTensor(labels).to(device)
    return a_token_ids, a_attention_mask, b_token_ids, b_attention_mask, labels


def metric(valid_loader, model):
    correct_num, predict_num, gold_num = 0, 0, 0
    with tqdm(total=len(valid_loader), desc='模型验证进度条') as pbar:
        for index, batch in enumerate(valid_loader):
            a_token_ids, a_attention_mask, b_token_ids, b_attention_mask, gold_labels = batch
            pred_labels = model(a_token_ids, a_attention_mask, b_token_ids, b_attention_mask)
            pred_labels = torch.argmax(pred_labels, dim=1)
            predict_num += len(pred_labels)
            gold_num += len(gold_labels)
            correct_num += int(torch.sum(pred_labels == gold_labels))
            pbar.update()

    print("correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".format(correct_num, predict_num, gold_num))
    precision = correct_num / (predict_num + 1e-10)
    recall = correct_num / (gold_num + 1e-10)
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    print('f1: {:4.4f}, precision: {:4.4f}, recall: {:4.4f}'.format(f1_score, precision, recall))
    return precision, recall, f1_score


def train():
    train_dataset = BQDataset(os.path.join(args.file_path, 'train.tsv'))
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collect_fn)
    valid_dataset = BQDataset(os.path.join(args.file_path, 'dev.tsv'))
    valid_loader = data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collect_fn)
    bert_model = BertModel.from_pretrained(args.bert_model_path)
    model = SentenceBert(encoder=bert_model).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    best_f1_score = 0.0
    early_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        with tqdm(total=len(train_loader), desc='Epoch：%d，模型训练进度条' % epoch) as pbar:
            for batch_idx, batch in enumerate(train_loader):
                a_token_ids, a_attention_mask, b_token_ids, b_attention_mask, labels = batch
                optimizer.zero_grad()
                logits = model(a_token_ids, a_attention_mask, b_token_ids, b_attention_mask)
                loss = torch.nn.CrossEntropyLoss()(logits, labels)
                pbar.set_postfix({'loss': '{0:1.5f}'.format(float(loss))})
                pbar.update()
                loss.backward()
                optimizer.step()
                scheduler.step()

        model.eval()
        with torch.no_grad():
            precision, recall, f1_score = metric(valid_loader, model)
            if f1_score > best_f1_score:
                torch.save(model.state_dict(), args.model_path)
                best_f1_score = f1_score
                early_epochs = 0
            else:
                early_epochs += 1

            if early_epochs > 7:  # 连续7个epoch，验证集f1_score没有提升，训练结束
                print('验证集f1_score连续7个epoch没有提升，训练结束')
                break
        print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', help='训练数据路径', type=str, default='/tmp/bq_corpus/')
    parser.add_argument('--bert_model_path', help='预训练模型路径', type=str, default='/tmp/chinese-roberta-wwm-ext')
    parser.add_argument('--epochs', help='训练轮数', type=int, default=100)
    parser.add_argument('--dropout', help='', type=float, default=0.5)
    parser.add_argument('--embedding_size', help='', type=int, default=100)
    parser.add_argument('--batch_size', help='', type=int, default=32)
    parser.add_argument('--hidden_size', help='', type=int, default=200)
    parser.add_argument('--num_layers', help='', type=int, default=1)
    parser.add_argument('--lr', help='学习率', type=float, default=1e-3)
    parser.add_argument('--mode', help='长文本处理方式', type=str, default='cut')
    parser.add_argument('--model_path', help='模型存储路径', type=str, default='/tmp/bq_corpus/bq_sbert.pt')
    args = parser.parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)

    train()
