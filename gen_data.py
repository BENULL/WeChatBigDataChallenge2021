#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: BENULL
@time: 2021/6/4 13:52
"""
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import *

def create_dir():
    """
    创建所需要的目录
    """
    if not os.path.exists(ROOT_PATH):
        print('Create dir: %s' % ROOT_PATH)
        os.mkdir(ROOT_PATH)
    for need_dir in [FEATURE_PATH,MODEL_PATH,SUBMIT_PATH]:
        if not os.path.exists(need_dir):
            print('Create dir: %s' % need_dir)
            os.mkdir(need_dir)

def check_file():
    """
    检查数据文件是否存在
    """
    paths = [USER_ACTION, FEED_INFO, TEST_FILE]
    flag = True
    not_exist_file = []
    for f in paths:
        if not os.path.exists(f):
            not_exist_file.append(f)
            flag = False
    return flag, not_exist_file

def statis_feature(start_day=1, before_day=5, agg=['mean', 'sum', 'count']):
    """
    统计特征
    """
    # TODO 统计特征
    pass

def make_sample():
    feed_info_df = pd.read_csv(FEED_INFO)
    # 用户行为 除去播放、停留时间
    user_action_df = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid"] + FEA_COLUMN_LIST]
    feed_embed = pd.read_csv(FEED_EMBEDDINGS)
    test = pd.read_csv(TEST_FILE)
    # add feed feature
    train = pd.merge(user_action_df, feed_info_df[FEA_FEED_LIST], on='feedid', how='left')
    test = pd.merge(test, feed_info_df[FEA_FEED_LIST], on='feedid', how='left')
    test["videoplayseconds"] = np.log(test["videoplayseconds"] + 1.0)
    test.to_csv(FEATURE_PATH + f'/test_data.csv', index=False)

    # TODO 对数据添加统计特征 1.生成统计特征文件 2. dataloader 中增强数据
    for action in tqdm(ACTION_LIST):
        print(f"prepare data for {action}")
        tmp = train.drop_duplicates(['userid', 'feedid', action], keep='last')
        # TODO 没有取最后5天？
        df_neg = tmp[tmp[action] == 0]
        df_neg = df_neg.sample(frac=ACTION_SAMPLE_RATE[action], random_state=SEED, replace=False)
        df_all = pd.concat([df_neg, tmp[tmp[action] == 1]])
        df_all["videoplayseconds"] = np.log(df_all["videoplayseconds"] + 1.0)
        df_all.to_csv(FEATURE_PATH + f'/train_data_for_{action}.csv', index=False)

def main():
    t = time.time()
    create_dir()
    flag, not_exists_file = check_file()
    if not flag:
        print("files not exist: ", ",".join(not_exists_file))
        return
    # statis_feature(start_day=1, before_day=BEFOR_DAY, agg=['mean','sum','count'])
    make_sample()
    print('Time cost: %.2f s' % (time.time() - t))


if __name__ == '__main__':
    main()