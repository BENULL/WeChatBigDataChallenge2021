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
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer


def create_dir():
    """
    创建所需要的目录
    """
    if not os.path.exists(ROOT_PATH):
        print('Create dir: %s' % ROOT_PATH)
        os.mkdir(ROOT_PATH)
    for need_dir in [FEATURE_PATH, MODEL_PATH, SUBMIT_PATH]:
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
    print("the number of feeds: ", len(feed_data["feedid"].unique()))
    print("the number of authorids: ", len(feed_data["authorid"].unique()))

    test_data = pd.read_csv(TEST_FILE)[TEST_COLUMNS]
    print("test description:\n ", test_data.describe())
    print("the number of feeds: ", len(test_data["feedid"].unique()))
    print("the number of users: ", len(test_data["userid"].unique()))

    return user_action_data, feed_data, test_data


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
    feed_keyword["feed_keyword_list"] = feed_keyword.apply(process_keywords, axis=1)  # 逐行处理

    # print(feed_keyword['feed_keyword_list'])
    # 对所有keyword都加1
    feed_keyword['feed_keyword_list'] = feed_keyword['feed_keyword_list'].apply(
        lambda row: list(map(lambda x: x + 1, row)))

    S = set()
    feed_keyword['feed_keyword_list'].apply(lambda row: getLength(row, S))
    print("the number of keywords embeddings: ", max(S) + 1)
    feed_keyword = feed_keyword[['feedid', 'feed_keyword_list']]  # 筛选出keyword列表
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

    feed_tag = feed_tag[['feedid', 'feed_tag_list']]  # 筛选出tag列表
    return feed_tag


def user_keyword_tag_process(userdataDF, feedkeywordDF, feedtagDF):
    """
    统计每个用户的keyword和tag列表
    """
    # 构造用户特征
    userdataDF = userdataDF.loc[:,
                 ["userid", "feedid", "read_comment", "comment", "like", "click_avatar", "forward", "follow",
                  "favorite"]]
    feedkwtagDF = feedkeywordDF.merge(feedtagDF, how="left", on="feedid")
    print("original feed keyword and tag:\n", feedkwtagDF)

    filename = FEATURE_PATH + "/feed_keyword_tag.csv"
    print('feed keyword and tag file save to: %s' % filename)
    feedkwtagDF.to_csv(filename, index=True)

    userKeywordTagDF = userdataDF.merge(feedkwtagDF, how='left', on='feedid')
    print("merge feed keyword and tag with user info:\n", userKeywordTagDF)

    # 筛选掉7种动作至少有一种不为0的用户行为
    userKeywordTagDF = userKeywordTagDF[(userKeywordTagDF["read_comment"] != 0) | (userKeywordTagDF["comment"] != 0) | (
                userKeywordTagDF["like"] != 0) | (userKeywordTagDF["click_avatar"] != 0) |
                                        (userKeywordTagDF["forward"] != 0) | (userKeywordTagDF["follow"] != 0) | (
                                                    userKeywordTagDF["favorite"] != 0)]
    print("select user action not all zero:\n", userKeywordTagDF)

    def userTagKeywordProcess(row, mode):
        if mode == 'keyword':
            return row['feed_keyword_list']
        elif mode == 'tag':
            return row['feed_tag_list']

    userKeywordTagDF['user_tag_list'] = userKeywordTagDF.apply(lambda row: userTagKeywordProcess(row, 'tag'), axis=1)
    userKeywordTagDF['user_keyword_list'] = userKeywordTagDF.apply(lambda row: userTagKeywordProcess(row, 'keyword'),
                                                                   axis=1)

    def userTagKeywordCollect(row):
        import operator
        from functools import reduce
        # print("row:\n ", row)
        # print(row.tolist())
        temp = reduce(operator.add, row.tolist())
        return list(set(temp))
        # return Counter(temp) # 不要去重,保存为字典形式

    # 这样做的话，用户数不一定全，因为可能存在用户自始至终没有产生过任何反馈
    userTagDF = userKeywordTagDF.groupby(['userid'])["user_tag_list"].apply(lambda x: userTagKeywordCollect(x))
    userKeywordDF = userKeywordTagDF.groupby(['userid'])["user_keyword_list"].apply(lambda x: userTagKeywordCollect(x))
    print("userTagDF:\n", userTagDF)
    print("userKeywordDF:\n", userKeywordDF)

    alluserDF = userdataDF[['userid']].drop_duplicates()

    userTagDF = alluserDF.merge(userTagDF, how="left", on="userid")
    userTagDF['user_tag_list'].fillna("[]", inplace=True)

    userKeywordTagFeat = userTagDF.merge(userKeywordDF, how="left", on="userid")
    userKeywordTagFeat['user_keyword_list'].fillna("[]", inplace=True)

    filename = FEATURE_PATH + "/user_keyword_tag.csv"
    print('user keyword and tag file save to: %s' % filename)

    userKeywordTagFeat.to_csv(filename, index=True)
    print("userKeywordTagFeat:\n", userKeywordTagFeat)
    return feedkwtagDF, userKeywordTagFeat

def make_keyword_tag(userDataDF, feedDataDF):
    # feed 和 user 的tag和keyword
    feedKeywordDF = feed_keyword_process(feedDataDF)
    feedTagDF = feed_tag_process(feedDataDF)
    feedKwTagDF, userKwTagDF = user_keyword_tag_process(userDataDF, feedKeywordDF, feedTagDF)
    return feedKwTagDF, userKwTagDF

def process_embed(pca=True, n_component=32):
    """
    处理feed_embed
    """
    feed_embed = pd.read_csv(FEED_EMBEDDINGS)
    feed_embed_array = np.zeros((feed_embed.shape[0], 512))
    for i in tqdm(range(feed_embed.shape[0])):
        x = feed_embed.loc[i, 'feed_embedding']
        if x != np.nan and x != '':
            y = [float(i) for i in str(x).strip().split(" ")]
        else:
            y = np.zeros((512,)).tolist()
        feed_embed_array[i] += y
    processed_feed_embed = pd.DataFrame(columns=[f"embed{i}" for i in range(512)], data=feed_embed_array)
    processed_feed_embed = pd.concat((feed_embed.feedid,processed_feed_embed),axis=1)
    # processed_feed_embed.to_csv(f'{FEATURE_PATH}/feed_embed.csv',index=False)
    if pca:
        feed_embed_pac(processed_feed_embed,n_component)

def feed_embed_pac(feed_embed,n_components):
    """
    feed_embed pca
    """
    data = np.array(feed_embed.iloc[:,1:])
    pca = PCA(n_components=n_components).fit(data)
    data_pca = pca.transform(data)
    pca_result = pd.DataFrame(data_pca, columns=[f'embed{i}' for i in range(pca.n_components_)])
    pca_result = pd.concat((feed_embed.feedid, pca_result), axis=1)
    pca_result.to_csv(f'{FEATURE_PATH}/feed_embed_pca_{pca.n_components_}.csv',index=False)


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
    return user_static

def feed_statistics_feats(userDataDF, feedDataDF):
    """
    处理feed统计特征
    """
    feed_static = userDataDF.groupby('feedid', as_index=False).agg(
        # 每个feed在过去一段时间内被曝光的总数
        feed_expo_cnt=('userid', 'count'),
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

    # 选取videoplayseconds列来计算平均播放和停留时间占总时长的比例
    feedDataDF = feedDataDF[["feedid", "videoplayseconds"]]
    feed_static = feed_static.merge(feedDataDF, how="left", on=["feedid"])

    feed_static["feed_play_rate"] = feed_static[["feed_play_avg", "videoplayseconds"]].apply(
        lambda x: x["feed_play_avg"] / 1000 / x["videoplayseconds"], axis=1)
    feed_static["feed_stay_rate"] = feed_static[["feed_stay_avg", "videoplayseconds"]].apply(
        lambda x: x["feed_stay_avg"] / 1000 / x["videoplayseconds"], axis=1)

    feed_static.drop(columns=['videoplayseconds'], inplace=True)
    return feed_static

def statis_feature(userDataDF, feedDataDF, start_day=2, before_day=BEFORE_DAY):
    """
    统计用户/feed 过去n天各类行为的次数
    :param start_day: Int. 起始日期
    :param before_day: Int. 时间范围（天数）
    :param agg: String. 统计方法
    """
    # userDataDF = userDataDF.drop(columns=['device'])
    userDataDF.sort_values(by="date_", inplace=True)
    # print("----original userData columns:\n", userDataDF.columns.values)

    user_arr, feed_arr = [], []
    for start in range(start_day, END_DAY + 1):
        # 不一定非得滑动窗口
        print("----the {}th day----".format(start))
        if start <= before_day:
            userActionData = userDataDF[userDataDF['date_'] < start]
        else:
            userActionData = userDataDF[
                ((userDataDF['date_']) > (start - BEFORE_DAY - 1)) & ((userDataDF['date_']) < start)]
        print("select data's shape: ", userActionData.shape)

        userStatic = user_statistics_feats(userActionData)
        print("userStatic data's shape: ", userStatic.shape)

        feedStatic = feed_statistics_feats(userActionData, feedDataDF)
        print("feedStatic data's shape: ", feedStatic.shape)

        userStatic['date_'] = start
        user_arr.append(userStatic)

        feedStatic['date_'] = start
        feed_arr.append(feedStatic)

    print("length of user_arr: ", len(user_arr))
    print("length of feed_arr: ", len(feed_arr))

    userStaticDF = pd.concat(user_arr)
    feedStaticDF = pd.concat(feed_arr)
    return userStaticDF, feedStaticDF

def merge_feature(userDataDF, feedDataDF, userStaticDF, feedStaticDF):
    print("userDataDF's shape and columns:\n ", userDataDF.shape, userDataDF.columns.values)
    print("feedDataDF's shape and columns:\n ", feedDataDF.shape, feedDataDF.columns.values)
    print("userStaticDF's shape and columns:\n ", userStaticDF.shape, userStaticDF.columns.values)
    print("feedStaticDF's shape and columns:\n ", feedStaticDF.shape, feedStaticDF.columns.values)

    feedDataDF = feedDataDF.set_index("feedid")
    userStaticDF = userStaticDF.set_index(["userid", "date_"])
    feedStaticDF = feedStaticDF.set_index(["feedid", "date_"])

    # merge with feedinfo
    TotalFeatDF = userDataDF.merge(feedDataDF, how="left", on="feedid")
    print("merge 1: ", TotalFeatDF.shape, TotalFeatDF.columns.values)

    # merge with user static info
    TotalFeatDF = TotalFeatDF.merge(userStaticDF, how="left", on=["userid", "date_"])
    print("merge 2: ", TotalFeatDF.shape, TotalFeatDF.columns.values)

    # merge with feed static info
    TotalFeatDF = TotalFeatDF.merge(feedStaticDF, how="left", on=["feedid", "date_"])
    print("merge 3: ", TotalFeatDF.shape, TotalFeatDF.columns.values)

    return TotalFeatDF

def fill_nan(totalFeatDF):
    feat_col = FEA_COLUMN_LIST
    user_feat_col = ['user_' + g + '_sum' for g in feat_col] + ['user_' + g + '_avg' for g in feat_col] \
                    + ['user_action_cnt', 'user_feed_cnt', 'user_stay_avg', 'user_play_avg']

    feed_feat_col = ['feed_' + g + '_sum' for g in feat_col] + ['feed_' + g + '_avg' for g in feat_col] \
                    + ['feed_expo_cnt', 'feed_user_cnt', 'feed_stay_avg', 'feed_play_avg', 'feed_play_rate',
                       'feed_stay_rate']

    print("user_feat_col: ", user_feat_col)
    print("feed_feat_col: ", feed_feat_col)

    totalFeatDF[user_feat_col] = totalFeatDF[user_feat_col].fillna(0.0)
    totalFeatDF[feed_feat_col] = totalFeatDF[feed_feat_col].fillna(0.0)

    totalFeatDF[["authorid", "bgm_song_id", "bgm_singer_id"]] += 1  # 0 用于填未知
    totalFeatDF[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]] = \
        totalFeatDF[["authorid", "bgm_song_id", "bgm_singer_id", "videoplayseconds"]].fillna(0)
    totalFeatDF[["authorid", "bgm_song_id", "bgm_singer_id"]] = \
        totalFeatDF[["authorid", "bgm_song_id", "bgm_singer_id"]].astype(int)

    totalFeatDF["videoplayseconds"] = np.log(totalFeatDF["videoplayseconds"] + 1.0)

    # check if there is NaN in dataframe
    # totalFeatDF.isnull().values.any()

    print(totalFeatDF)
    return totalFeatDF

def make_sample(userDataDF, feedDataDF):
    userStaticDF, feedStaticDF = statis_feature(userDataDF, feedDataDF)
    totalFeatDF = merge_feature(userDataDF, feedDataDF, userStaticDF, feedStaticDF)

    print("totalFeatDF's columns:\n ", totalFeatDF.columns.values)

    totalFeatDF = fill_nan(totalFeatDF)

    filename = FEATURE_PATH + "/total_feats.csv"
    print('total feature file save to: %s' % filename)

    totalFeatDF.to_csv(filename)
    return userStaticDF, feedStaticDF, totalFeatDF

def make_evaluate(totalFeatDF):
    # 筛选出date为14天的数据，作为验证集
    evaluateDF = totalFeatDF[totalFeatDF['date_'] == END_DAY - 1]

    filename = EVALUATE_PATH + "/evaluate_all_14.csv"
    print('evaluate file save to: %s' % filename)

    evaluateDF.to_csv(filename)
    return evaluateDF

def make_submit(testDataDF, feedDataDF, userStaticDF, feedStaticDF):
    print("testDataDF's shape and columns:\n ", testDataDF.shape, testDataDF.columns.values)
    print("feedDataDF's shape and columns:\n ", feedDataDF.shape, feedDataDF.columns.values)
    print("userStaticDF's shape and columns:\n ", userStaticDF.shape, userStaticDF.columns.values)
    print("feedStaticDF's shape and columns:\n ", feedStaticDF.shape, feedStaticDF.columns.values)

    # 设置日期为最后一天
    testDataDF["date_"] = END_DAY

    # 筛选出date为14天的数据，作为验证集
    userStaticDF = userStaticDF[userStaticDF['date_'] == END_DAY]
    feedStaticDF = feedStaticDF[feedStaticDF['date_'] == END_DAY]

    feedDataDF = feedDataDF.set_index("feedid")
    userStaticDF = userStaticDF.set_index(["userid", "date_"])
    feedStaticDF = feedStaticDF.set_index(["feedid", "date_"])

    # merge feed info
    testDataDF = testDataDF.merge(feedDataDF, how="left", on="feedid")

    # merge user and feed statistic info
    testDataDF = testDataDF.merge(userStaticDF, how="left", on=["userid", "date_"])
    testDataDF = testDataDF.merge(feedStaticDF, how="left", on=["feedid", "date_"])

    # fillNa
    testDataDF = fill_nan(testDataDF)

    filename = SUBMIT_PATH + "/submit_15.csv"
    print('submit file save to: %s' % filename)
    testDataDF.to_csv(filename)
    return testDataDF

def negative_sampling(totalFeatDF):
    for action in ACTION_LIST:
        negDF = totalFeatDF[totalFeatDF[action] == 0]
        posDF = totalFeatDF[totalFeatDF[action] == 1]

        negDF = negDF.sample(frac=ACTION_SAMPLE_RATE[action], random_state=SEED, replace=False)
        allDF = pd.concat([negDF, posDF])

        print("action {} positive samples length: {}".format(action, posDF.shape[0]))
        print("action {} negative samples length: {}".format(action, negDF.shape[0]))

        file_name = FEATURE_PATH + "/total_feats_" + action + ".csv"
        print('negative sample file save to: %s' % file_name)

        allDF.to_csv(file_name, index=False)

def user_tags_encoding():
    user_tags = pd.read_csv(f'{FEATURE_PATH}/user_tags.csv')
    tags = [eval(s) for s in user_tags['user_tag_list']]
    encoded_tags = MultiLabelBinarizer().fit_transform(tags)

    user_tags['user_tags_encode_list'] = [' '.join(map(str,encode)) for encode in encoded_tags]
    user_tags.to_csv(f'{FEATURE_PATH}/user_tags.csv',index=False)

    # df_encoded_tags = pd.DataFrame(encoded_tags, columns=[f'tags{i}' for i in range(encoded_tags.shape[1])])
    # df_user_tags = pd.concat((user_tags.userid, df_encoded_tags), axis=1)
    # df_user_tags.to_csv(f'{FEATURE_PATH}/user_tags_encoded_{encoded_tags.shape[1]}.csv', index=False)

def user_tags_pca(n_components=0.9):
    user_tags_encoded = pd.read_csv(f'{FEATURE_PATH}/user_tags_encoded_336.csv')
    data = np.array(user_tags_encoded.iloc[:,1:])
    pca = PCA(n_components).fit(data)
    data_pca = pca.transform(data)
    pca_result = pd.DataFrame(data_pca, columns=[f'tags{i}' for i in range(pca.n_components_)])
    pca_result = pd.concat((user_tags_encoded.userid, pca_result), axis=1)
    pca_result.to_csv(f'{FEATURE_PATH}/use_tags_pca_{pca.n_components_}.csv',index=False)
    # print(pca.explained_variance_ratio_)
    # print(pca.n_components_)


def main():
    t = time.time()
    create_dir()
    flag, not_exists_file = check_file()
    if not flag:
        print("files not exist: ", ",".join(not_exists_file))
        return

    # load original csv file
    userDataDF, feedDataDF, testDataDF = loadFile()

    # user feed tag and keyword process
    feedKeywordTagDF, userKeywordTagDF = make_keyword_tag(userDataDF, feedDataDF)
    print("feedKeyWordTagDF:\n", feedKeywordTagDF)
    print("userKeywordTagDF:\n", userKeywordTagDF)

    # drop unnecessary features
    feedDataDF.drop(columns=FEED_DROP_COLUMNS, inplace=True)

    # make train samples
    userStaticDF, feedStaticDF, totalFeatDF = make_sample(userDataDF, feedDataDF)
    print("userStaticDF:\n", userStaticDF)
    print("feedStaticDF:\n", feedStaticDF)
    print("totalFeatDF:\n", totalFeatDF)

    # negative sampling
    negative_sampling(totalFeatDF)

    # make evaluation samples
    evaluateDF = make_evaluate(totalFeatDF)
    print("evaluateDF:\n", evaluateDF)

    # make submit
    submitDF = make_submit(testDataDF, feedDataDF, userStaticDF, feedStaticDF)
    print("submitDF:\n", submitDF)
    print('Time cost: %.2f s' % (time.time() - t))

if __name__ == '__main__':
    pass
    # user_tags_encoding()
    processed_feed_embed= pd.read_csv(f'{FEATURE_PATH}/feed_embed.csv')

    feed_embed_pac(processed_feed_embed,n_components=0.8)
    # main()
    user_tags_pca(n_components=0.8)