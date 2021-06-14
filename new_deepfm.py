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
from config import *
import gc
import time
import numpy as np
from evaluation import uAUC, compute_weighted_score

if __name__ == "__main__":
    off_train = True

    if off_train:
        test = pd.read_csv(f'{EVALUATE_PATH}/evaluate_all_13.csv')
        submit = test[['userid', 'feedid']]

        train = pd.read_csv(FEATURE_PATH + f'/total_feats.csv')
        train = train[train['date_'] < 14].reset_index(drop=True)
    else:
        test = pd.read_csv(f'{SUBMIT_PATH}/submit_15.csv')
        submit = test[['userid', 'feedid']]

        train = pd.read_csv(FEATURE_PATH + f'/total_feats.csv')

    test[ACTION_LIST] = 0

    # feed_embed
    feed_embed = pd.read_csv(f'{FEATURE_PATH}/feed_embed_pca_168.csv')  # _pca_32

    # user_tags
    # user_tags = pd.read_csv(f'{FEATURE_PATH}/use_tags_pca_74.csv')  # 336

    # 散装
    train = pd.merge(train, feed_embed, on='feedid', how='left')
    test = pd.merge(test, feed_embed, on='feedid', how='left')

    #
    # train = pd.merge(train, user_tags, on='userid', how='left')
    # test = pd.merge(test, user_tags, on='userid', how='left')

    train = train.sample(frac=1, random_state=21).reset_index(drop=True)

    data = pd.concat((train, test)).reset_index(drop=True)

    dense_features = ['videoplayseconds', 'device', 'user_action_cnt', 'user_feed_cnt', 'user_play_avg',
       'user_stay_avg', 'user_read_comment_sum', 'user_comment_sum',
       'user_like_sum', 'user_click_avatar_sum', 'user_forward_sum',
       'user_follow_sum', 'user_favorite_sum', 'user_read_comment_avg',
       'user_comment_avg', 'user_like_avg', 'user_click_avatar_avg',
       'user_forward_avg', 'user_follow_avg', 'user_favorite_avg',
       'feed_expo_cnt', 'feed_user_cnt', 'feed_play_avg', 'feed_stay_avg',
       'feed_read_comment_sum', 'feed_comment_sum', 'feed_like_sum',
       'feed_click_avatar_sum', 'feed_forward_sum', 'feed_follow_sum',
       'feed_favorite_sum', 'feed_read_comment_avg', 'feed_comment_avg',
       'feed_like_avg', 'feed_click_avatar_avg', 'feed_forward_avg',
       'feed_follow_avg', 'feed_favorite_avg', 'feed_play_rate',
       'feed_stay_rate']

    sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id',]

    data[sparse_features] = data[sparse_features].fillna(0)
    data[dense_features] = data[dense_features].fillna(0)

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    # 高维
    # fixlen_feature_columns.append(DenseFeat('feed_embed', 32))

    # 散装
    fixlen_feature_columns.extend([DenseFeat(f'embed{i}', 1) for i in range(168)])
    # fixlen_feature_columns.extend([DenseFeat(f'tags{i}', 1) for i in range(74)])

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    train, test = data.iloc[:train.shape[0]].reset_index(drop=True), data.iloc[train.shape[0]:].reset_index(
        drop=True)

    # print(feature_names, len(feature_names))

    # 散装
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # eval_dict = {}
    for action in ACTION_LIST:
        device = 'cpu'
        use_cuda = True
        if use_cuda and torch.cuda.is_available():
            print('cuda ready...')
            device = 'cuda:0'

        model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                       l2_reg_embedding=1e-1,
                       task='binary', dnn_hidden_units=[256, 128], device=device, gpus=[0, 1, 2, 3])
        # l2_reg_embedding = 1e-1,

        model.compile("adagrad", "binary_crossentropy")
        model.fit(train_model_input, train[action].values, batch_size=1024, epochs=1, verbose=1, validation_split=0.2)

        # eval_dict
        # pred = model.predict(train_model_input, batch_size=512)
        # labels = train[target].values
        # userid_list = train.userid.astype(str).tolist()
        # uauc = uAUC(labels.flatten(), pred.flatten(), userid_list)
        #
        # eval_dict[action] = uauc
        # print(f"{action} uAUC: ", uauc)


        # predict
        pred_ans = model.predict(test_model_input, batch_size=512)
        submit[action] = pred_ans
        # torch.save(model, f'{MODEL_PATH}/NEW_DeepFM_model_{action}.h5')
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # weight_auc = compute_weighted_score(eval_dict)
    # print("Weighted uAUC: ", weight_auc)

    filepath = f'{SUBMIT_PATH}/NEW_DeepFM_{int(time.time())}.csv'
    submit.to_csv(filepath, index=False)
    print(f'submit result save to {filepath}')

    if off_train:
        t = time.time()
        label_data = open(f'{EVALUATE_PATH}/evaluate_all_13.csv', 'r')
        result_data = open(f'{filepath}', 'r')
        from evaluation import score

        res = score(result_data, label_data, mode='初赛')
        print('Time cost: %.2f s' % (time.time() - t))
