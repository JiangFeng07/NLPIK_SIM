#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time  : 2022/11/25 10:47
# @Author: lionel
import re


def text_to_chars(text):
    char_list = []
    if not text:
        return char_list
    if len(text) == 1:
        char_list = list(text)
        return char_list
    start = 0
    pattern = re.compile('^[\u4e00-\u9fa5， ,；。;:：]$')
    while start < len(text) - 1:
        char = text[start]
        if pattern.search(char):
            char_list.append(char)
            start += 1
            continue
        end = start + 1
        while end < len(text):
            if not pattern.search(text[end]):
                end += 1
                continue
            break
        char_list.append(text[start:end])
        start = end
    if start == len(text) - 1:
        char_list.append(text[start])
    return char_list


if __name__ == '__main__':
    print(text_to_chars('张刚 认缴出资额：27.000000 实缴出资额：27.000000，2.五金艳 认缴出资额：3.000000 实缴出资额：3.000000江'))
