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
TEST_FILE = DATASET_PATH + "/test_a.csv"
SUBMIT_B = DATASET_PATH + "/test_b.csv"
FEED_EMBEDDINGS = DATASET_PATH + "/feed_embeddings.csv"

# 测试集
SUBMIT_FILE = DATASET_PATH + "/test_a.csv"

# 清理过后的数据
FEATURE_PATH = ROOT_PATH + "/feature"
MODEL_PATH = ROOT_PATH + "/model"
SUBMIT_PATH = ROOT_PATH + "/submit"
EVALUATE_PATH = ROOT_PATH + "/evaluate"

# user action 列名
USER_ACTION_COLUMNS = ["userid","feedid","date_","device","read_comment","comment","like","play","stay","click_avatar","forward","follow","favorite"]
# feed info 列名
FEED_INFO_COLUMNS = ["feedid","authorid","videoplayseconds","description","ocr","asr","bgm_song_id","bgm_singer_id","manual_keyword_list",\
                     "machine_keyword_list","manual_tag_list","machine_tag_list","description_char","ocr_char","asr_char"]
# feed drop columns
FEED_DROP_COLUMNS = ['description', 'ocr', 'asr', 'description_char', 'ocr_char', 'asr_char', 'manual_keyword_list', 'machine_keyword_list',
                     'manual_tag_list', 'machine_tag_list']

# test data 列名
TEST_COLUMNS = ["userid","feedid","device"]

# feed embedding 列名
FEED_EMBEDDINGS_COLUMNS = ["feedid","feed_embedding"]

# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward", ]
# 复赛待预测行为列表
# ACTION_LIST = ["read_comment", "like", "click_avatar",  "forward", "comment", "follow", "favorite"]

# 各个行为构造训练数据的天数
ACTION_DAY_NUM = {"read_comment": 7, "like": 7, "click_avatar": 7, "forward": 7, "comment": 7, "follow": 7, "favorite": 7}

# 负样本下采样比例((下采样后负样本数/原负样本数)) TODO 下采样参数测试
# ACTION_SAMPLE_RATE = {"read_comment": 0.2, "like": 0.2, "click_avatar": 0.2, "forward": 0.2,
#                       "comment": 0.2, "follow": 0.2, "favorite": 0.2}

# 各个阶段数据集的设置的最后一天
STAGE_END_DAY = {"train": 13, "evaluate": 14, "submit": 14}

# 构造训练数据的天数
BEFORE_DAY = 5

END_DAY = 15
SEED = 2021

# 机器预测的标签阈值, 大于此阈值，则认为可信
MACHINE_TAG_THRESHOLD = 0.6

# Tag缺失值填充
FILL_NAN_TAG_STR = "-1 1.0"
FILL_NAN_TAG_NUM = -1

# KeyWord缺失值填充
FILL_NAN_KEYWORD_STR = "-1"
FILL_NAN_KEYWORD_NUM = -1

#通过统计计算得出
NUMBER_TAGS = 354 # 不同Tag的个数
NUMBER_KEYWORD = 27272 # 不同keyword的个数

# 用于构造特征的字段列表
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]

# FEED相关特征
FEA_FEED_LIST = ['feedid', 'authorid',
                 'videoplayseconds',
                 'feed_expo_cnt', # 此feed在过去一段时间内被曝光的总次数
                 'feed_user_cnt', # 此feed在过去一段时间内被不同用户看到的个数
                 'bgm_song_id', 'bgm_singer_id',
                 'feed_stay_avg', 'feed_play_avg', # feed被用户播放和停留的平均时长
                 'feed_play_rate', 'feed_stay_rate', # feed被用户播放和停留的平均时长占总时长的比例
                 'feed_like_sum', 'feed_read_comment_sum', 'feed_click_avatar_sum', 'feed_forward_sum', 'feed_comment_sum', 'feed_follow_sum', 'feed_favorite_sum',
                 'feed_like_avg', 'feed_read_comment_avg', 'feed_click_avatar_avg', 'feed_forward_avg', 'feed_comment_avg', 'feed_follow_avg', 'feed_favorite_avg',
                 'feed_tag_list', # 每个feed的tag列表
                 'feed_keyword_list' # 每个feed的keyword列表
                 ]

# USER相关特征
FEA_USER_LIST = ['userid',
                 'user_action_cnt', # 用户在过去一段时间内的行为总数
                 'user_feed_cnt', # 用户在过去一段时间内的观看的不同的feed的个数
                 'user_stay_avg', 'user_play_avg', # 用户播放和停留时长平均值
                 'user_like_sum', 'user_read_comment_sum', 'user_click_avatar_sum', 'user_forward_sum', 'user_comment_sum', 'user_follow_sum', 'user_favorite_sum',#用户各种行为的总数
                 'user_like_avg', 'user_read_comment_avg', 'user_click_avatar_avg', 'user_forward_avg', 'user_comment_avg', 'user_follow_avg', 'user_favorite_avg',#用户各种行为的平均出现次数
                 'user_feed_sum', #用户过去一共浏览过的feed总数
                 'user_tag_list', # 用户喜爱的feed类别
                 'user_keyword_list' #用户喜爱的feed关键字
                 ]

