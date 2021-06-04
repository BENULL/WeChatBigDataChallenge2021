#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/6/4 19:57
"""

# 存储数据的根目录
ROOT_PATH = "/root/WeChatBigData/data"

# 比赛数据集路径
DATASET_PATH = ROOT_PATH + "wechat_algo_data1"
# 训练集
USER_ACTION = DATASET_PATH + "user_action.csv"
FEED_INFO = DATASET_PATH + "feed_info.csv"
FEED_EMBEDDINGS = DATASET_PATH + "feed_embeddings.csv"
# 测试集
TEST_FILE = DATASET_PATH + "test_a.csv"

FEATURE_PATH = ROOT_PATH + "feature"
MODEL_PATH = ROOT_PATH + "model"
SUBMIT_PATH = ROOT_PATH + "submit"

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
# STAGE_END_DAY = {"online_train": 14, "offline_train": 12, "evaluate": 13, "submit": 15}

# 构造训练数据的天数
BEFORE_DAY = 5

# 用于构造特征的字段列表
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]

# FEED 特征
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']

# TODO USER nounique 可以尝试
# FEA_USER_LIST = []
