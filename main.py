#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/6/4 17:59
"""
import argparse
from config import *


def data_loader(train_path, test_path):
    train, test = [], []
    return train, test


def train(self, model):
    """
    在过去13天数据上训练
    """
    pass


def evaluate(self):
    """
    过去13天训练的模型预测第14天
    """
    pass


def predict(self):
    """
    过去14天训练的模型预测第15天
    """
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--model', type=str, default='deepfm')
    parser.add_argument('-b', type=int, help='batch size', default=512)
    parser.add_argument('--epoch', type=int, help='batch size', default=1)
    args = parser.parse_args()






