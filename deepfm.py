#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/6/4 19:31
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models.deepfm import *
from deepctr_torch.models.difm import *
from deepctr_torch.models.wdl import *
from deepctr_torch.models.dcn import *
from deepctr_torch.models.autoint import *
from config import *
import gc
import time
import numpy as np
from evaluation import uAUC, compute_weighted_score
from sklearn.model_selection import KFold

import os
os.environ['NUMEXPR_MAX_THREADS'] = '40'

ACTION_SAMPLE_RATE = {"read_comment": 0.5, "like": 0.5, "click_avatar": 0.5, "forward": 0.5,
                      "comment": 0.8, "follow": 0.8, "favorite": 0.8}

BEFORE_ACTION_SAMPLE_RATE = {"read_comment": 0.1, "like": 0.1, "click_avatar": 0.2, "forward": 0.2,
                      "comment": 0.8, "follow": 0.8, "favorite": 0.8}

AFTER_ACTION_SAMPLE_RATE = {"read_comment": 1, "like": 1, "click_avatar": 0.5, "forward": 0.5,
                      "comment": 0.8, "follow": 0.8, "favorite": 0.8}

HIGH_DIMENSION_FEATURE = [f'embed{i}' for i in range(168)]+[f'tags{i}' for i in range(74)]


# phase
off_train = False
kfold = 0

# sample
use_last_day = False
has_sample = False
phase_sample = False
sample_day = False

# feature
has_feed_embed = True
has_feed_keywords = True
has_feed_tags = True
has_user_tags = True
has_user_keywords = True

# model args
dnn_hidden_units = [128, 128, 128]


def fit_predict(train, test, submit):
    for action in ACTION_LIST:
        if has_sample:
            if phase_sample:
                df_neg_before = train[(train[action] == 0) & (train['date_'] >= 7)]
                df_neg_before = df_neg_before.sample(frac=AFTER_ACTION_SAMPLE_RATE[action], random_state=SEED, replace=False)
                df_neg_after = train[(train[action] == 0) & (train['date_'] < 7)]
                df_neg_after = df_neg_after.sample(frac=BEFORE_ACTION_SAMPLE_RATE[action], random_state=SEED, replace=False)
                action_train = pd.concat([df_neg_before, df_neg_after, train[train[action] == 1]])
            else:
                df_neg = train[train[action] == 0]
                if sample_day:
                    df_neg = df_neg.groupby(['userid', 'date_']).sample(frac=ACTION_SAMPLE_RATE[action], random_state=SEED, replace=False)
                else:
                    df_neg = df_neg.sample(frac=ACTION_SAMPLE_RATE[action], random_state=SEED, replace=False)
                action_train = pd.concat([df_neg, train[train[action] == 1]])
        else:
            action_train = train

        # action_train = action_train.drop_duplicates(['userid', 'feedid', action], keep='last')

        action_train = action_train.sample(frac=1, random_state=SEED).reset_index(drop=True)


        print(f"{action} posi prop: {sum((action_train[action] == 1) * 1) / action_train.shape[0]}")

        train_model_input = {name: action_train[name] for name in feature_names}
        test_model_input = {name: test[name] for name in feature_names}

        model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                    task='binary', dnn_hidden_units=dnn_hidden_units, device=device, l2_reg_embedding=1e-5,
                    gpus=[0, 1, 2, 3])
        # l2_reg_dnn = 1e-2, dnn_dropout = 0.5,
        #

        model.compile("adam", "binary_crossentropy")
        model.fit(train_model_input, action_train[action].values, batch_size=1024, epochs=3, verbose=2)
        # predict
        pred_ans = model.predict(test_model_input, batch_size=512)
        submit[action] = pred_ans

        if off_train:
            labels = test[action].values
            userid_list = test.userid.astype(str).tolist()
            uauc = uAUC(labels.flatten(), pred_ans.flatten(), userid_list)
            print(f"{action} uAUC: ", uauc)
        # torch.save(model, f'{MODEL_PATH}/NEW_DeepFM_model_{action}.h5')
        del model
        gc.collect()
        torch.cuda.empty_cache()


def merge_feature(df):
    # merge feedid
        # 'videoplayseconds',
    feed_feature = pd.read_csv(FEED_INFO)[['feedid','videoplayseconds', 'authorid', 'bgm_song_id', 'bgm_singer_id',]]
    df = df.merge(feed_feature, on='feedid', how='left')

    # # merge userid_feat
    # userid_feat = pd.read_csv(f'{FEATURE_PATH}/userid_feat.csv')
    # df = df.merge(userid_feat, on=['userid', 'date_'], how='left')
    #
    # # merge feedid_feat
    # feedid_feat = pd.read_csv(f'{FEATURE_PATH}/feedid_feat.csv')
    # df = df.merge(feedid_feat, on=['feedid', 'date_'], how='left')
    #
    # # merge authorid_feat
    # authorid_feat = pd.read_csv(f'{FEATURE_PATH}/authorid_feat.csv')
    # df = df.merge(authorid_feat, on=['authorid', 'date_'], how='left')
    #
    # # merge bgm_song_id_feat
    # bgm_song_id_feat = pd.read_csv(f'{FEATURE_PATH}/bgm_song_id_feat.csv')
    # df = df.merge(bgm_song_id_feat, on=['bgm_song_id', 'date_'], how='left')
    #
    # # merge userid_authorid_feat
    # userid_authorid_feat = pd.read_csv(f'{FEATURE_PATH}/userid_authorid_feat.csv')
    # df = df.merge(userid_authorid_feat, on=['userid', 'authorid', 'date_'], how='left')

    if has_feed_embed:
        # high dimension
        feed_embed = pd.read_csv(f'{FEATURE_PATH}/feed_embed_pca_168.csv')
        # feed_embed_col = pd.merge(feed_embed, df['feedid'], on='feedid', how='right')
        # feed_embed_col = feed_embed_col.fillna(0)
        # df['feed_embed'] = feed_embed_col.iloc[:, 1:].values.tolist()
        # del feed_embed, feed_embed_col

        # spilt to columns
        df = df.merge(feed_embed, on='feedid', how='left')

        feed_embed['feed_embed_avg'] = feed_embed.iloc[:, 1:].mean(axis=1)
        feed_embed['feed_embed_max'] = feed_embed.iloc[:, 1:].max(axis=1)
        feed_embed['feed_embed_sum'] = feed_embed.iloc[:, 1:].sum(axis=1)
        df = df.merge(feed_embed[['feed_embed_avg', 'feed_embed_max', 'feed_embed_sum', 'feedid']], on='feedid', how='left')


    if has_feed_tags:
        feed_tags = pd.read_csv(f'{FEATURE_PATH}/feed_tags_w2v_32.csv')
        feed_tags['feed_tags_avg'] = feed_tags.iloc[:, 1:].mean(axis=1)
        feed_tags['feed_tags_max'] = feed_tags.iloc[:, 1:].max(axis=1)
        feed_tags['feed_tags_sum'] = feed_tags.iloc[:, 1:].sum(axis=1)
        df = df.merge(feed_tags[['feed_tags_avg', 'feed_tags_max', 'feed_tags_sum', 'feedid']], on='feedid', how='left')

    if has_feed_keywords:
        feed_kws = pd.read_csv(f'{FEATURE_PATH}/feed_kws_w2v_64.csv')
        feed_kws['feed_kws_avg'] = feed_kws.iloc[:, 1:].mean(axis=1)
        feed_kws['feed_kws_max'] = feed_kws.iloc[:, 1:].max(axis=1)
        feed_kws['feed_kws_sum'] = feed_kws.iloc[:, 1:].sum(axis=1)
        df = df.merge(feed_kws[['feed_kws_avg', 'feed_kws_max', 'feed_kws_sum', 'feedid']], on='feedid', how='left')

    if has_user_tags:
        user_tags = pd.read_csv(f'{FEATURE_PATH}/user_tags_w2v_32.csv')
        # user_tags = pd.read_csv(f'{FEATURE_PATH}/use_tags_pca_74.csv')  # 336
        # df = df.merge(user_tags, on='userid', how='left')

        user_tags['user_tags_avg'] = user_tags.iloc[:, 1:].mean(axis=1)
        user_tags['user_tags_max'] = user_tags.iloc[:, 1:].max(axis=1)
        user_tags['user_tags_sum'] = user_tags.iloc[:, 1:].sum(axis=1)
        df = df.merge(user_tags[['user_tags_avg', 'user_tags_max', 'user_tags_sum', 'userid']], on='userid', how='left')

        user_tags = pd.read_csv(f'{FEATURE_PATH}/use_tags_pca_74.csv')
        df = df.merge(user_tags, on='userid', how='left')



    if has_user_keywords:
        user_kws = pd.read_csv(f'{FEATURE_PATH}/user_kws_w2v_64.csv')
        user_kws['user_kws_avg'] = user_kws.iloc[:, 1:].mean(axis=1)
        user_kws['user_kws_max'] = user_kws.iloc[:, 1:].max(axis=1)
        user_kws['user_kws_sum'] = user_kws.iloc[:, 1:].sum(axis=1)
        df = df.merge(user_kws[['user_kws_avg', 'user_kws_max', 'user_kws_sum', 'userid']], on='userid', how='left')

    df = df.fillna(0)
    gc.collect()
    return df


if __name__ == "__main__":
    # load dataset
    train = pd.read_csv(USER_ACTION)
    # train = pd.read_csv(FEATURE_PATH + f'/total_feats.csv')
    # del train['Unnamed: 0']
    if off_train:
        test = train[train['date_'] == 14].reset_index(drop=True)
        train = train[train['date_'] < 14].reset_index(drop=True)
    else:
        test = pd.read_csv(SUBMIT_B)

    if use_last_day:
        train = train[train['date_'] >= 7].reset_index(drop=True)

    submit = test[['userid', 'feedid']]



    # build_feature
    data = pd.concat((train, test)).reset_index(drop=True)
    data = merge_feature(data)

    # define features
    sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'device']

    dense_features = [col for col in data.columns
                      if col not in ['date_', 'userid', 'stay', 'play'] +
                      sparse_features + FEA_COLUMN_LIST + HIGH_DIMENSION_FEATURE]


    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])


    # for feat in dense_features:
    #     q = data[feat].quantile(q=0.8)
    #     data.loc[data[feat] > q, feat] = q
    # mms = MinMaxScaler(feature_range=(0, 1))
    # data[dense_features] = mms.fit_transform(data[dense_features])

    rs = RobustScaler()
    data[dense_features] = rs.fit_transform(data[dense_features])

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), )
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    if has_feed_embed:
        # fixlen_feature_columns.append(DenseFeat('feed_embed', 168))
        fixlen_feature_columns.extend([DenseFeat(f'embed{i}', 1) for i in range(168)])
    if has_user_tags:
        fixlen_feature_columns.extend([DenseFeat(f'tags{i}', 1) for i in range(74)])

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    train = data.iloc[:train.shape[0]].reset_index(drop=True)
    test = data.iloc[train.shape[0]:].reset_index(drop=True)

    del data
    gc.collect()

    print(feature_names, len(feature_names))
    # print(dense_features, len(dense_features))

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
            if off_train:
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
