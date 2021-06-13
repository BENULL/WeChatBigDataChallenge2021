#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/6/11 15:59
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models.deepfm import *
from deepctr_torch.models.difm import *
from deepctr_torch.models.wdl import *
from config import *
import time
import numpy as np
import sys
from evaluation import uAUC, compute_weighted_score

SEED = 2021

# dense_features = ['videoplayseconds']
#
# sparse_features = [i for i in USE_FEAT if i not in dense_features and i not in target]
#

class NNCtr:

    def __init__(self, model_name='deepfm', stage='off_train',feed_embed=True, user_tags=True):
        self.stage = stage
        self.feed_embed = feed_embed
        self.user_tags = user_tags
        self.df_train, self.df_test, self.df_submit = self.load_dataset()
        self.proc_data()
        self.model = self.make_model(model_name)

    def load_dataset(self):
        """
        load df_train, df_test, df_submit
        """
        df_train, df_test, df_submit = None, None, None
        if self.stage == 'off_train':
            df_train = pd.read_csv(FEATURE_PATH + f'/nosample_train_data.csv')[['date_'] < 14]
            df_test = pd.read_csv(f'{EVALUATE_PATH}/evaluate_14.csv')

        elif self.stage == 'online_train':
            df_train = pd.read_csv(FEATURE_PATH + f'/nosample_train_data.csv')
            df_test = pd.read_csv(f'{FEATURE_PATH}/test_data.csv')
            df_submit = df_test[['userid', 'feedid']]
        elif self.stage == 'predict':
            df_test = pd.read_csv(f'{FEATURE_PATH}/test_data.csv')
            df_submit = df_test[['userid', 'feedid']]

        if df_train:
            df_train = df_train.sample(frac=1, random_state=SEED).reset_index(drop=True)
        df_test[ACTION_LIST] = 0
        return df_train, df_test, df_submit

    def proc_data(self):
        """
        add feed_embed and user_tags
        """
        if self.feed_embed:
            feed_embed = pd.read_csv(f'{FEATURE_PATH}/feed_embed_pca_32.csv')
            self.df_train = self.df_train.merge(feed_embed, on='feedid', how='left')
            self.df_test = self.df_test.merge(feed_embed, on='feedid', how='left')

        if self.user_tags:
            user_tags = pd.read_csv(f'{FEATURE_PATH}/user_tags_encoded_336.csv')
            self.df_train = self.df_train.merge(user_tags, on='userid', how='left')
            self.df_test = self.df_test.merge(user_tags, on='userid', how='left')

    def make_feature(self):
        pass


    def make_model(self, model_name):
        pass


    def train(self):
        pass

    def predict(self):
        pass






