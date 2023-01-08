#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2023/1/8 15:19
# @Author: lionel

import logging
import multiprocessing
import os
import re
import sys

from gensim import utils
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.INFO)

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logger.info("running %s" % ' '.join(sys.argv))


def text_process(text):
    """文本处理"""
    text = re.sub('<.*?>', '', text)
    text = re.sub('〇', '0', text)
    return text


def load_stop_words(file_path):
    """加载停用词库"""
    stop_words = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            stop_words.add(line.strip())
    return stop_words


def cut_text(text, tokenizer, stop_words):
    """文本分词"""
    word_list = []
    if not text:
        return word_list
    words = tokenizer.lcut(text, cut_all=False)
    for word in words:
        if not re.search('[\u4e00-\u9fa5A-Za-z]', word) or word in stop_words:
            continue
        word_list.append(word)
    return word_list


class Docs(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        try:
            for line in open(self.filename, 'rb'):
                pieces = utils.to_unicode(line).split('\t')
                words = pieces[1].split(' ')
                tags = list()
                tags.append(pieces[0])
                yield TaggedDocument(words, tags)
        except:
            logging.info('e')


if __name__ == '__main__':
    import argparse

    base_path = ''
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', help='训练语料', type=str, default='')
    parser.add_argument('--vector_size', help='向量维度', type=int, default=200)
    parser.add_argument('--epochs', help='训练轮次', type=int, default=10)
    parser.add_argument('--window', help='窗口大小', type=int, default=5)
    parser.add_argument('--min_count', help='词出现最低次数', type=int, default=10)
    parser.add_argument('--data_process', help='训练语料是否需要预处理, 如分词等', type=bool, default=False)
    args = parser.parse_args()
    file_path = os.path.join(args.base_path, 'train.csv')
    stop_words_path = os.path.join(args.base_path, 'stop_words.csv')
    words_path = os.path.join(args.base_path, 'words.csv')
    model_path = os.path.join(args.base_path, 'doc2vec')
    word2vec_path = os.path.join(args.base_path, 'word2vec')

    in_file = file_path
    if args.data_process:
        import jieba

        stop_words = load_stop_words(stop_words_path)

        if words_path and os.path.exists(words_path):
            jieba.load_userdict(words_path)
        out_path = '/tmp/train2.csv'
        out_file = open(out_path, 'w', encoding='utf-8')
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) != 2:
                    continue
                key_id, text = fields
                text = text_process(text)
                word_list = cut_text(text, tokenizer=jieba, stop_words=stop_words)
                out_file.write('{0}\t{1}\n'.format(key_id, ' '.join(word_list)))
        out_file.close()
        in_file = out_path

    model = Doc2Vec(Docs(in_file), size=args.vector_size, epochs=args.epochs, window=args.window,
                    min_count=args.min_count, workers=multiprocessing.cpu_count())

    # 模型保存
    model.save(model_path)
    # 保存word2vec向量
    model.save_word2vec_format(word2vec_path)
