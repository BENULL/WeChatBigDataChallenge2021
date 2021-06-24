#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/6/4 19:31
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models.deepfm import *
from deepctr_torch.models.difm import *
from deepctr_torch.models.wdl import *
from deepctr_torch.models.dcn import *
from config import *
import gc
import time
import numpy as np
from evaluation import uAUC, compute_weighted_score
from sklearn.model_selection import KFold

import os
os.environ['NUMEXPR_MAX_THREADS'] = '40'

ACTION_SAMPLE_RATE = {"read_comment": 0.2, "like": 0.2, "click_avatar": 0.2, "forward": 0.2,
                      "comment": 0.8, "follow": 0.8, "favorite": 0.8}

HIGH_DIMENSION_FEATURE = ['feed_embed', 'user_tags', 'user_kws']

def fit_predict(train, test, submit):
    for action in ACTION_LIST:
        if has_sample:
            df_neg = train[train[action] == 0]
            df_neg = df_neg.sample(frac=ACTION_SAMPLE_RATE[action], random_state=42, replace=False)
            action_train = pd.concat([df_neg, train[train[action] == 1]])
        else:
            action_train = train

        action_train = merge_feature(action_train)

        data = pd.concat((action_train, test)).reset_index(drop=True)

        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])

        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])

        fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                                  for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                                  for feat in dense_features]
        if has_feed_embed:
            fixlen_feature_columns.append(DenseFeat('feed_embed', 168))
            # fixlen_feature_columns.extend([DenseFeat(f'embed{i}', 1) for i in range(168)])
        if has_user_tags:
            fixlen_feature_columns.append(DenseFeat('user_tags', 74))
            # fixlen_feature_columns.extend([DenseFeat(f'tags{i}', 1) for i in range(74)])
        if has_user_keywords:
            fixlen_feature_columns.append(DenseFeat('user_kws', 64))

        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns[:7]

        feature_names = get_feature_names(
            linear_feature_columns + dnn_feature_columns)

        action_train = data.iloc[:action_train.shape[0]].reset_index(drop=True)
        action_test = data.iloc[action_train.shape[0]:].reset_index(drop=True)

        print(feature_names)

        train_model_input = {name: action_train[name] for name in feature_names if name not in HIGH_DIMENSION_FEATURE}
        test_model_input = {name: action_test[name] for name in feature_names if name not in HIGH_DIMENSION_FEATURE}

        for high_dim_fea in HIGH_DIMENSION_FEATURE:
            if high_dim_fea in feature_names:
                train_model_input[high_dim_fea] = np.array(action_train[high_dim_fea].tolist())
                test_model_input[high_dim_fea] = np.array(action_test[high_dim_fea].tolist())

        model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                       l2_reg_embedding=1e-1,
                       task='binary', dnn_hidden_units=[256, 512, 1024, 512, 256, 128], device=device,
                       gpus=[0, 1, 2, 3])

        model.compile("adam", "binary_crossentropy")
        model.fit(train_model_input, action_train[action].values, batch_size=1024, epochs=1, verbose=2)
        # predict
        pred_ans = model.predict(test_model_input, batch_size=512)
        submit[action] = pred_ans
        # torch.save(model, f'{MODEL_PATH}/NEW_DeepFM_model_{action}.h5')
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # config
    off_train = True
    kfold = 0
    has_sample = True
    has_feed_embed = False
    has_user_tags = False
    has_user_keywords = False

    # load dataset
    train = pd.read_csv(USER_ACTION)
    if off_train:
        test = train[train['date_'] == 14].reset_index(drop=True)
        train = train[train['date_'] < 14].reset_index(drop=True)
    else:
        test = pd.read_csv(TEST_FILE)

    submit = test[['userid', 'feedid']]

    # define features
    dense_features = ['videoplayseconds', 'device']

    sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', ]

    # load feature
    feed_feature = pd.read_csv(FEED_INFO)[['feedid', 'videoplayseconds', 'authorid', 'bgm_song_id', 'bgm_singer_id',]]

    if has_feed_embed:
        feed_embed = pd.read_csv(f'{FEATURE_PATH}/feed_embed_pca_168.csv')

    if has_user_tags:
        user_tags = pd.read_csv(f'{FEATURE_PATH}/use_tags_pca_74.csv')

    if has_user_keywords:
        user_kws = pd.read_csv(f'{FEATURE_PATH}/user_kws_w2v_64.csv')

    # merge feature
    def merge_feature(df):

        # merge feedid
        df = df.merge(feed_feature, on='feedid', how='left')

        # merge userid

        if has_feed_embed:
            # high dimension
            feed_embed_col = pd.merge(feed_embed, df['feedid'], on='feedid', how='right')
            feed_embed_col = feed_embed_col.fillna(0)
            df['feed_embed'] = feed_embed_col.iloc[:, 1:].values.tolist()

            # spilt to columns
            # df = df.merge(feed_embed, on='feedid', how='left')

        if has_user_tags:
            user_tags_col = pd.merge(user_tags, df['userid'], on='userid', how='right')
            user_tags_col = user_tags_col.fillna(0)
            df['user_tags'] = user_tags_col.iloc[:, 1:].values.tolist()

            # df = df.merge(user_tags, on='userid', how='left')

        if has_user_keywords:
            user_kws_col = pd.merge(user_kws, df['userid'], on='userid', how='right')
            user_kws_col = user_kws_col.fillna(0)
            df['user_kws'] = user_kws_col.iloc[:, 1:].values.tolist()
        return df

    test = merge_feature(test)

    # check gpu
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    if kfold > 0:
        kf = KFold(n_splits=kfold, shuffle=True, random_state=SEED)
        all_result = []
        for train_index, _ in kf.split(train):
            X_train = train.iloc[train_index]
            fit_predict(X_train, test, submit)
            filepath = f'{SUBMIT_PATH}/fold_result_{int(time.time())}.csv'
            submit.to_csv(filepath, index=False)
            label_data = open(f'{EVALUATE_PATH}/evaluate_all_13.csv', 'r')
            result_data = open(filepath, 'r')
            from evaluation import score
            res = score(result_data, label_data, mode='初赛')
            all_result.append(submit.copy())

        # merge kfold result
        all_result_merge = pd.concat(all_result)
        filepath = f'{SUBMIT_PATH}/DeepFM_{kfold}fold_result_{int(time.time())}.csv'
        merged_result = all_result_merge.groupby(['userid', 'feedid']).agg('mean').reset_index()
        merged_result.to_csv(filepath, index=False)
        print(f'submit result save to {filepath}')

    else:
        fit_predict(train, test, submit)
        filepath = f'{SUBMIT_PATH}/DeepFM_result_{int(time.time())}.csv'
        submit.to_csv(filepath, index=False)
        print(f'submit result save to {filepath}')

    if off_train:
        t = time.time()
        label_data = open(f'{EVALUATE_PATH}/evaluate_all_13.csv', 'r')
        result_data = open(f'{filepath}', 'r')
        from evaluation import score

        res = score(result_data, label_data, mode='初赛')
        print('Time cost: %.2f s' % (time.time() - t))
