#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/6/4 17:59
"""
import argparse
from config import *
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__file__)



def train(self, model):
    pass


def evaluate(self):
    pass


def predict(self):
    pass


if __name__ == '__main__':
    logger.info("123")

    # TODO 参数化
    # parser = argparse.ArgumentParser()
    # opt = parser.parse_args()




