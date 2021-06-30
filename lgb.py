#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/6/10 18:40
"""
import pandas as pd
import os
import lightgbm
from config import *
from evaluation import uAUC, compute_weighted_score
from sklearn.preprocessing import LabelEncoder
import sys

ONE_HOT_FEATURE = ['feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'userid', 'device']


class LGB:
    """
    lightgbm lgb
    """

    def __init__(self, stage, action):
        self.params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'boost_from_average': True,
            'train_metric': True,
            'feature_fraction_seed': 1,
            'learning_rate': 0.05,
            'is_unbalance': True,  # 当训练数据是不平衡的，正负样本相差悬殊的时候，可以将这个属性设为true,此时会自动给少的样本赋予更高的权重
            'num_leaves': 128,  # 一般设为少于2^(max_depth) 128
            'max_depth': -1,  # 最大的树深，设为-1时表示不限制树的深度
            'min_child_samples': 15,  # 每个叶子结点最少包含的样本数量，用于正则化，避免过拟合
            'max_bin': 200,  # 设置连续特征或大量类型的离散特征的bins的数量
            'subsample': 0.8,  # Subsample ratio of the training instance.
            'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
            'colsample_bytree': 0.8,  # Subsample ratio of columns when constructing each tree.
            'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
            'subsample_for_bin': 200000,  # Number of samples for constructing bin
            'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
            'reg_alpha': 2.99,  # L1 regularization term on weights
            'reg_lambda': 1.9,  # L2 regularization term on weights
            'nthread': 12,
            'verbose': 0,
            # 'force_row_wise': True
        }
        self.stage = stage
        self.action = action
        self.select_frts = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'videoplayseconds','device']
                            # 'device', 'user_action_cnt', 'user_feed_cnt', 'user_play_avg',
                            # 'user_stay_avg', 'user_read_comment_sum', 'user_comment_sum',
                            # 'user_like_sum', 'user_click_avatar_sum', 'user_forward_sum',
                            # 'user_follow_sum', 'user_favorite_sum', 'user_read_comment_avg',
                            # 'user_comment_avg', 'user_like_avg', 'user_click_avatar_avg',
                            # 'user_forward_avg', 'user_follow_avg', 'user_favorite_avg',
                            # 'feed_expo_cnt', 'feed_user_cnt', 'feed_play_avg', 'feed_stay_avg',
                            # 'feed_read_comment_sum', 'feed_comment_sum', 'feed_like_sum',
                            # 'feed_click_avatar_sum', 'feed_forward_sum', 'feed_follow_sum',
                            # 'feed_favorite_sum', 'feed_read_comment_avg', 'feed_comment_avg',
                            # 'feed_like_avg', 'feed_click_avatar_avg', 'feed_forward_avg',
                            # 'feed_follow_avg', 'feed_favorite_avg', 'feed_play_rate','feed_stay_rate']

        # feed embedding by PCA
        # self.select_frts += [f'embed{i}' for i in range(168)]
        # self.select_frts += [f'tags{i}' for i in range(74)]

    def process_data(self, train_path, test_path):
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        df = pd.concat((df_train, df_test)).reset_index(drop=True)
        for feature in ONE_HOT_FEATURE:
            df[feature] = LabelEncoder().fit_transform(df[feature].apply(str))
        df_feed = pd.read_csv(f'{FEATURE_PATH}/feed_embed_pca_168.csv')
        # df = df.merge(df_feed)

        # df_tags = pd.read_csv(f'{FEATURE_PATH}/use_tags_pca_74.csv')
        # df = df.merge(df_tags)

        train = df.iloc[:df_train.shape[0]].reset_index(drop=True)
        test = df.iloc[df_train.shape[0]:].reset_index(drop=True)
        return train, test

    def train_test(self):
        # 读取训练集数据
        train_path = f'{FEATURE_PATH}/total_feats_{self.action}.csv'
        test_path = f'{FEATURE_PATH}/test_data.csv'

        df_train, df_test = self.process_data(train_path, test_path)

        if self.stage == 'offline_train':
            df_val = df_train[df_train['date_'] == 14].reset_index(drop=True)
            df_train = df_train[df_train['date_'] < 14].reset_index(drop=True)
        else:
            df_val = None

        train_x = df_train[self.select_frts]
        train_y = df_train[self.action]

        train_matrix = Lightgbm.Dataset(train_x, label=train_y)

        self.model = Lightgbm.train(self.params, train_matrix, num_boost_round=200)

        print("\n".join(
            ("%s: %.2f" % x) for x in list(sorted(zip(list(train_x.columns), self.model.feature_importance("gain")),
                                                  key=lambda x: x[1], reverse=True))[:5]))
        if self.stage == 'offline_train':
            return self.evaluate(df_val)
        elif self.stage == 'online_train':
            return self.predict(df_test)

    def evaluate(self, df):
        # 测试集
        test_x = df[self.select_frts].values
        labels = df[self.action].values
        userid_list = df['userid'].astype(str).tolist()
        logits = self.model.predict(test_x)
        uauc = uAUC(labels, logits, userid_list)
        return df[["userid", "feedid"]], logits, uauc

    def predict(self, df):
        # 测试集
        test_x = df[self.select_frts].values
        logits = self.model.predict(test_x)
        return df[["userid", "feedid"]], logits


def main(argv):
    stage = argv[1]
    eval_dict = {}
    predict_dict = {}
    ids = None
    submit = pd.read_csv(SUBMIT_FILE)[['userid', 'feedid']]
    for action in ACTION_LIST:
        print("-------------------Action-----------------:", action)
        model = LGB(stage, action)
        if stage == "offline_train":
            # 离线训练并评估
            ids, logits, action_uauc = model.train_test()
            eval_dict[action] = action_uauc
            predict_dict[action] = logits

        elif stage == "online_train":
            # 评估线下测试集结果，计算单个行为的uAUC值，并保存预测结果
            ids, logits = model.train_test()
            predict_dict[action] = logits

        else:
            print("stage must be in [online_train,offline_train]")

    if stage == "offline_train":
        print(eval_dict)
        weight_dict = {"read_comment": 4, "like": 3, "click_avatar": 2, "favorite": 1, "forward": 1,
                       "comment": 1, "follow": 1}
        weight_auc = compute_weighted_score(eval_dict, weight_dict)
        print("Weighted uAUC: ", weight_auc)

    if stage == "online_train":
        # 计算所有行为的加权uAUC
        actions = pd.DataFrame.from_dict(predict_dict)
        print("Actions:", actions)
        ids[["userid", "feedid"]] = ids[["userid", "feedid"]].astype(int)
        res = pd.concat([ids, actions], sort=False, axis=1)
        # 写文件
        file_name = "submit_lgb.csv"
        submit_file = f'{SUBMIT_PATH}/{file_name}'
        print(f'Save to:{submit_file}')
        res.to_csv(submit_file, index=False)


if __name__ == "__main__":
    main(sys.argv)
