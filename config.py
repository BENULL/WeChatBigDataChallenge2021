#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/6/4 19:57
"""

# 存储数据的根目录
ROOT_PATH = "/root/WeChatBigData/data"

# 比赛数据集路径
DATASET_PATH = ROOT_PATH + "/wechat_algo_data1"
# 训练集
USER_ACTION = DATASET_PATH + "/user_action.csv"
FEED_INFO = DATASET_PATH + "/feed_info.csv"
FEED_EMBEDDINGS = DATASET_PATH + "/feed_embeddings.csv"
# 测试集
SUBMIT_FILE = DATASET_PATH + "/test_a.csv"

FEATURE_PATH = ROOT_PATH + "/feature"
MODEL_PATH = ROOT_PATH + "/model"
SUBMIT_PATH = ROOT_PATH + "/submit"
EVALUATE_PATH = ROOT_PATH + "/evaluate"

END_DAY = 15
SEED = 2021

# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
# 复赛待预测行为列表
# ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]

# 负样本下采样比例((下采样后负样本数/原负样本数)) TODO 下采样参数测试
ACTION_SAMPLE_RATE = {"read_comment": 0.2, "like": 0.2, "click_avatar": 0.2, "forward": 0.2,
                      "comment": 0.2, "follow": 0.2, "favorite": 0.2}

# 各个阶段数据集的设置的最后一天
STAGE_END_DAY = {"train": 13, "evaluate": 14, "submit": 14}

# 构造训练数据的天数
BEFORE_DAY = 5

# 用于构造特征的字段列表
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]

# FEED 特征
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id', 'feed_stay_avg',
                 'feed_play_avg', 'feed_like_sum', 'feed_read_comments_sum',
                 'feed_click_avatar_sum', 'feed_forward_sum', 'feed_comment_sum', 'feed_follow_sum',
                 'feed_favorite_sum', 'feed_tags']

# TODO USER nounique 可以尝试
FEA_USER_LIST = ['userid', 'user_stay_avg', 'user_play_avg', 'user_like_sum', 'user_read_comments_sum',
                 'user_click_avatar_sum', 'user_forward_sum', 'user_comment_sum', 'user_follow_sum',
                 'user_favorite_sum', 'user_tags']
