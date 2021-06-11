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

def loadFile():
    user_action_data = pd.read_csv(USER_ACTION)[USER_ACTION_COLUMNS]
    print("user action data description:\n ", user_action_data.describe())
    print("user action data:\n ", user_action_data)
    print("the number of users: ", len(user_action_data["userid"].unique()))
    print("the number of user interact feeds: ", len(user_action_data["feedid"].unique()))

    feed_data = pd.read_csv(FEED_INFO)[FEED_INFO_COLUMNS]
    print("feed data description:\n ", feed_data.describe())
    print("the number of feeds: ", feed_data["feedid"].count())
    print("the number of authorids: ", feed_data["authorid"].count())

    return user_action_data, feed_data

def user_statistics_feats(userdataDF):
    """
    用户统计特征提取
    """
    user_static = userdataDF
    user_static = user_static.groupby('userid', as_index=False).agg(
        # 用户过去一段时间的行为总数
        user_action_cnt=('feedid', 'count'),
        # 用户观看的不同feed的数量
        user_feed_cnt=("feedid", lambda x: x.nunique()),
        # 用户的观看以及停留平均时长
        user_play_avg=('play', 'mean'),
        user_stay_avg=('stay', 'mean'),
        # 用户过去一段时间的各种行为的总数
        user_read_comment_sum=('read_comment', 'sum'),
        user_comment_sum=('comment', 'sum'),
        user_like_sum=('like', 'sum'),
        user_click_avatar_sum=('click_avatar', 'sum'),
        user_forward_sum=("forward", 'sum'),
        user_follow_sum=("follow", 'sum'),
        user_favorite_sum=("favorite", 'sum'),
        # 用户过去一段时间的各种行为的平均次数
        user_read_comment_avg=('read_comment', 'mean'),
        user_comment_avg=('comment', 'mean'),
        user_like_avg=('like', 'mean'),
        user_click_avatar_avg=('click_avatar', 'mean'),
        user_forward_avg=("forward", 'mean'),
        user_follow_avg=("follow", 'mean'),
        user_favorite_avg=("favorite", 'mean'))
    print(user_static)
    return user_static

def feed_statistics_feat(userDataDF, feedDataDF):
    """
    处理feed统计特征
    """
    feed_static = userDataDF
    print(feed_static.columns.values)

    feed_static = feed_static.groupby('feedid', as_index=False).agg(
        # 每个feed在过去一段时间内被看到的总数
        feed_count=('userid', 'count'),
        # 每个feed在过去一段时间内被多少个不同用户观看到过
        feed_user_cnt=("userid", lambda x: x.nunique()),
        # 每个feed被用户播放以及停留的平均时长
        feed_play_avg=('play', 'mean'),
        feed_stay_avg=('stay', 'mean'),
        # 每个feed在过去一段时间的各种行为的总数
        feed_read_comment_sum=('read_comment', 'sum'),
        feed_comment_sum=('comment', 'sum'),
        feed_like_sum=('like', 'sum'),
        feed_click_avatar_sum=('click_avatar', 'sum'),
        feed_forward_sum=("forward", 'sum'),
        feed_follow_sum=("follow", 'sum'),
        feed_favorite_sum=("favorite", 'sum'),
        # 每个feed在过去一段时间的各种行为的平均次数
        feed_read_comment_avg=('read_comment', 'mean'),
        feed_comment_avg=('comment', 'mean'),
        feed_like_avg=('like', 'mean'),
        feed_click_avatar_avg=('click_avatar', 'mean'),
        feed_forward_avg=("forward", 'mean'),
        feed_follow_avg=("follow", 'mean'),
        feed_favorite_avg=("favorite", 'mean'))
    print(feed_static.columns.values)
    print(feed_static)

    feed_static = feed_static.merge(feedDataDF, how="left", on="feedid")

    # 计算feed平均播放和停留时长占feed总时长的比例
    def calcu(row):
        play, total = row["feed_play_avg"], row["videoplayseconds"]
        print(play, total)
        return play / 1000 / total

    feed_static["feed_play_rate"] = feed_static[["feed_play_avg", "videoplayseconds"]].apply(
        lambda x: x["feed_play_avg"] / 1000 / x["videoplayseconds"], axis=1)
    feed_static["feed_stay_rate"] = feed_static[["feed_stay_avg", "videoplayseconds"]].apply(
        lambda x: x["feed_stay_avg"] / 1000 / x["videoplayseconds"], axis=1)
    print(feed_static.columns.values)
    print(feed_static)

def getLength(row, S):
    for x in row:
        S.add(x)

def feed_keyword_process(feedDF):
    """
    处理feed的Keyword
    """
    def process_keywords(row):
        """
        合并手工标注的keyword和机器标注的keyword列
        """
        manual, machine = row['manual_keyword_list'], row['machine_keyword_list']
        # print("row: ", manual, machine)
        if manual and machine:
            manual_list = list(map(int, manual.split(";")))
            machine_list = list(map(int, machine.split(";")))
            return list(set(manual_list + machine_list))
        else:
            if manual:
                return list(map(int, manual.split(";")))
            elif machine:
                return list(map(int, machine.split(";")))
            else:
                return [FILL_NAN_KEYWORD_NUM]  # 默认关键字类型

    feed_keyword = feedDF.loc[:, ["feedid", "manual_keyword_list", "machine_keyword_list"]]
    feed_keyword['manual_keyword_list'].fillna(FILL_NAN_KEYWORD_STR, inplace=True)
    feed_keyword['machine_keyword_list'].fillna(FILL_NAN_KEYWORD_STR, inplace=True)
    feed_keyword["feed_keyword_list"] = feed_keyword.apply(process_keywords, axis=1) #逐行处理

    #print(feed_keyword['feed_keyword_list'])
    # 对所有keyword都加1
    feed_keyword['feed_keyword_list'] = feed_keyword['feed_keyword_list'].apply(lambda row: list(map(lambda x: x+1, row)))
    print(feed_keyword['feed_keyword_list'])

    S = set()
    feed_keyword['feed_keyword_list'].apply(lambda row: getLength(row, S))
    print("the number of keywords: ", len(S))
    print("the number of keywords embeddings: ", max(S)+1)
    feed_keyword = feed_keyword[['feedid', 'feed_keyword_list']] # 筛选出keyword列表
    print(feed_keyword)
    return feed_keyword

def feed_tag_process(feedDF):
    """
    处理feed的Tag
    """
    def process_tags(row):
        """
        合并手工标注的tag和机器标注的tag列
        """
        manual, machine = row['manual_tag_list'], row['machine_tag_list']
        # print("row: ", manual, machine)
        if manual and machine:
            manual_list = list(map(int, manual.split(";")))
            machine_list = list(filter(lambda x: float(x.split(" ")[1]) > MACHINE_TAG_THRESHOLD, machine.split(";")))
            machine_list = list(map(lambda x: int(x.split()[0]), machine_list))
            return list(set(manual_list + machine_list))
        else:
            if manual:
                return list(map(int, manual.split(";")))
            elif machine:
                machine_list = list(
                    filter(lambda x: float(x.split(" ")[1]) > MACHINE_TAG_THRESHOLD, machine.split(";")))
                return list(map(lambda x: int(x.split()[0]), machine_list))
            else:
                return [FILL_NAN_TAG_NUM]  # 默认Tag类型

    feed_tag = feedDF.loc[:, ["feedid", "manual_tag_list", "machine_tag_list"]]
    feed_tag['manual_tag_list'].fillna(FILL_NAN_KEYWORD_STR, inplace=True)
    feed_tag['machine_tag_list'].fillna(FILL_NAN_TAG_STR, inplace=True)
    feed_tag['feed_tag_list'] = feed_tag.apply(process_tags, axis=1)  # 逐行处理

    # print(feed_tag['feed_tag_list'])
    # 对所有tag都加1
    feed_tag['feed_tag_list'] = feed_tag['feed_tag_list'].apply(lambda row: list(map(lambda x: x + 1, row)))

    S = set()
    feed_tag['feed_tag_list'].apply(lambda row: getLength(row, S))
    print("the number of tags embeddings: ", max(S) + 1)

    feed_tag = feed_tag[['feedid', 'feed_tag_list']] # 筛选出tag列表
    print(feed_tag)
    return feed_tag

def user_keyword_tag_process(userdataDF, feedkeywordDF, feedtagDF):
    """
    统计每个用户的keyword和tag列表
    """
    # 构造用户特征
    userdataDF = userdataDF.loc[:, ["userid", "feedid", "read_comment", "comment", "like", "click_avatar", "forward", "follow", "favorite"]]
    feedkwtagDF = feedkeywordDF.merge(feedtagDF, how="left", on="feedid")
    print(feedkwtagDF)
    feedkwtagDF.to_csv(FEATURE_PATH + "/feed_keyword_tag.csv", index=True)

    userKeywordTagDF = userdataDF.merge(feedkwtagDF, how='left', on='feedid')
    print(userKeywordTagDF)

    # 筛选掉7种动作至少有一种不为0的用户行为
    userKeywordTagDF = userKeywordTagDF[(userKeywordTagDF["read_comment"] != 0) | (userKeywordTagDF["comment"] != 0) | (userKeywordTagDF["like"] != 0) | (userKeywordTagDF["click_avatar"] != 0) |
                                  (userKeywordTagDF["forward"] != 0) | (userKeywordTagDF["follow"] != 0) | (userKeywordTagDF["favorite"] != 0)]
    print(userKeywordTagDF)

    def userTagKeywordProcess(row, mode):
        if mode == 'keyword':
            return row['feed_keyword_list']
        elif mode == 'tag':
            return row['feed_tag_list']

    userKeywordTagDF['user_tag_list'] = userKeywordTagDF.apply(lambda row: userTagKeywordProcess(row, 'tag'), axis=1)
    userKeywordTagDF['user_keyword_list'] = userKeywordTagDF.apply(lambda row: userTagKeywordProcess(row, 'keyword'), axis=1)

    def userTagKeywordCollect(row):
        import operator
        from functools import reduce
        # print("row:\n ", row)
        # print(row.tolist())
        temp = reduce(operator.add, row.tolist())
        return list(set(temp))
        # return Counter(temp) # 不要去重,保存为字典形式

    # 这样做的话，用户数不一定全，因为可能存在用户自始至终没有产生过任何反馈
    userTagList = userKeywordTagDF.groupby(['userid'])["user_tag_list"].apply(lambda x: userTagKeywordCollect(x))
    userKeywordList = userKeywordTagDF.groupby(['userid'])["user_keyword_list"].apply(lambda x: userTagKeywordCollect(x))
    print(userTagList, userKeywordList)

    alluserDF = userdataDF[['userid']].drop_duplicates()

    userTagList = alluserDF.merge(userTagList, how="left", on="userid")
    userTagList['user_tag_list'].fillna([], inplace=True)

    userKeywordList = userTagList.merge(userKeywordList, how="left", on="userid")
    userKeywordList['user_keyword_list'].fillna([], inplace=True)
    userKeywordList.to_csv(FEATURE_PATH + "/user_keyword_tag.csv", index=True)
    print(userKeywordList)

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

    userDataDF, feedDataDF = loadFile()

    # user行为统计特征
    userStaticDF = user_statistics_feats(userDataDF)

    # feed统计特征
    feedStaticDF = feed_statistics_feat(userDataDF, feedDataDF)

    # feed 和 user 的tag和keyword
    feedKeywordDF = feed_keyword_process(feedDataDF)
    feedTagDF = feed_tag_process(feedDataDF)
    user_keyword_tag_process(userDataDF, feedKeywordDF, feedTagDF)

    totalFeats = userStaticDF.merge(feedStaticDF, how="left", on="feedid")
    totalFeats.to_csv(FEATURE_PATH + "/user_feed_feats.csv")

    # statis_feature(start_day=1, before_day=BEFOR_DAY, agg=['mean','sum','count'])
    # make_sample()
    # make_evaluate()
    # make_submit()
    print('Time cost: %.2f s' % (time.time() - t))


if __name__ == '__main__':
    main()