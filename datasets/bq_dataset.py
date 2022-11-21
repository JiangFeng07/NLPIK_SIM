#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/20 22:33
# @Author: lionel

from torch.utils import data


class BQDataset(data.Dataset):
    def __init__(self, file_path, max_seq_len=128):
        self.datas = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) != 3:
                    continue
                self.datas.append(tuple(fields))
        self.max_seq_len = max_seq_len

    def __getitem__(self, item):
        a_text = self.datas[item][0]
        b_text = self.datas[item][1]
        label = self.datas[item][2]

        if len(a_text) > self.max_seq_len:
            a_text = a_text[:self.max_seq_len]
        if len(b_text) > self.max_seq_len:
            b_text = b_text[:self.max_seq_len]
        return a_text, b_text, int(label)

    def __len__(self):
        return len(self.datas)


if __name__ == '__main__':
    bq = BQDataset('/tmp/bq_corpus/train.tsv')
    for ele in bq:
        print(ele)
        break
