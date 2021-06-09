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
import os

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

def process_embed(train):
    """
    处理feed_embed
    """
    # TODO 处理feed_embed 怎么使用？
    feed_embed_array = np.zeros((train.shape[0], 512))
    for i in tqdm(range(train.shape[0])):
        x = train.loc[i, 'feed_embedding']
        if x != np.nan and x != '':
            y = [float(i) for i in str(x).strip().split(" ")]
        else:
            y = np.zeros((512,)).tolist()
        feed_embed_array[i] += y
    temp = pd.DataFrame(columns=[f"embed{i}" for i in range(512)], data=feed_embed_array)
    train = pd.concat((train, temp), axis=1)
    return train

def statis_feature(start_day=1, before_day=7, agg='sum'):
    """
    统计用户/feed 过去n天各类行为的次数
    :param start_day: Int. 起始日期
    :param before_day: Int. 时间范围（天数）
    :param agg: String. 统计方法
    """
    history_data = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid"] + FEA_COLUMN_LIST]
    history_data.sort_values(by="date_", inplace=True)
    for dim in ["userid", "feedid"]:
        user_data = history_data[[dim, "date_"] + FEA_COLUMN_LIST]
        res_arr = []
        tmp_name = '_' + dim + '_'
        for start in range(2, before_day + 1):
            temp = user_data[user_data['date_'] <= start]
            temp = temp.drop(columns=['date_'])
            temp = temp.groupby([dim]).agg(agg).reset_index()
            temp.columns = [dim] + list(map(tmp_name.join, temp.columns.values[1:]))
            temp['date_'] = start
            res_arr.append(temp)

        for start in range(start_day, END_DAY-before_day+1):
            temp = user_data[((user_data["date_"]) >= start) & (user_data["date_"] < (start + before_day))]
            temp = temp.drop(columns=['date_'])
            temp = temp.groupby([dim]).agg([agg]).reset_index()
            temp.columns = list(map(''.join, temp.columns.values))
            temp["date_"] = start + before_day
            res_arr.append(temp)

        dim_feature = pd.concat(res_arr)
        feature_path = os.path.join(feature_dir, dim+"_feature.csv")
        dim_feature.to_csv(f'{FEATURE_PATH}/{dim}_feature.csv', index=False)
        print(f'Saved {FEATURE_PATH}/{dim}_feature.csv')

def merge_feature():
    pass

def make_sample():
    feed_info_df = pd.read_csv(FEED_INFO)
    # user action
    user_action_df = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid"] + FEA_COLUMN_LIST]
    feed_embed = pd.read_csv(FEED_EMBEDDINGS)

    test = pd.read_csv(TEST_FILE)

    # add feed feature
    train = pd.merge(user_action_df, feed_info_df[FEA_FEED_LIST], on='feedid', how='left')
    test = pd.merge(test, feed_info_df[FEA_FEED_LIST], on='feedid', how='left')
    test["videoplayseconds"] = np.log(test["videoplayseconds"] + 1.0)
    test.to_csv(f'{FEATURE_PATH}/test_data.csv', index=False)

    # TODO 对数据添加统计特征 1.生成统计特征文件
    for action in tqdm(ACTION_LIST):
        print(f"prepare data for {action}")
        tmp = train.drop_duplicates(['userid', 'feedid', action], keep='last')
        # TODO 没有取最后5天？
        df_neg = tmp[tmp[action] == 0]
        df_neg = df_neg.sample(frac=ACTION_SAMPLE_RATE[action], random_state=SEED, replace=False)
        df_all = pd.concat([df_neg, tmp[tmp[action] == 1]])
        df_all["videoplayseconds"] = np.log(df_all["videoplayseconds"] + 1.0)
        df_all.to_csv(f'{FEATURE_PATH}/train_data_for_{action}.csv', index=False)

def make_evaluate():
    pass

def make_submit():
    pass

def main():
    t = time.time()
    create_dir()
    flag, not_exists_file = check_file()
    if not flag:
        print("files not exist: ", ",".join(not_exists_file))
        return
    # statis_feature(start_day=1, before_day=BEFOR_DAY, agg=['mean','sum','count'])
    make_sample()
    make_evaluate()
    make_submit()
    print('Time cost: %.2f s' % (time.time() - t))


if __name__ == '__main__':
    main()