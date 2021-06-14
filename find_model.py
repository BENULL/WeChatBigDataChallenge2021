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
from deepctr_torch.models.nfm import *
from deepctr_torch.models.afm import *
from deepctr_torch.models.dcn import *
from deepctr_torch.models.dcnmix import *
from deepctr_torch.models.xdeepfm import *
from deepctr_torch.models.autoint import *
from deepctr_torch.models.onn import *
from deepctr_torch.models.fibinet import *
from deepctr_torch.models.ifm import *
from config import *
import gc
import time
import numpy as np
from evaluation import uAUC, compute_weighted_score

if __name__ == "__main__":

    search = [
        """DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,task='binary',dnn_hidden_units=[256, 128],l2_reg_embedding=1e-1, device=device, gpus=[0, 1, 2, 3])""",
        """DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,task='binary',dnn_hidden_units=[256, 128],device=device, gpus=[0, 1, 2, 3])""",
        """DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,task='binary',dnn_hidden_units=[256,512,512,1024,1024,512,256,128],l2_reg_embedding=1e-1, device=device, gpus=[0, 1, 2, 3])""",
        """DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,task='binary',dnn_hidden_units=[256,512,512,1024,1024,512,256,128], device=device, gpus=[0, 1, 2, 3])""",
        """DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,task='binary',dnn_hidden_units=[256,512,512,1024,1024,512,256,128], dnn_dropout=0.3,l2_reg_embedding=1e-1,device=device, gpus=[0, 1, 2, 3])""",
        """DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,task='binary',dnn_hidden_units=[256,512,512,1024,1024,512,256,128], dnn_dropout=0.3,l2_reg_embedding=1e-1,device=device, gpus=[0, 1, 2, 3])""",
        """DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,device=device, gpus=[0, 1, 2, 3])""",
        """DIFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,device=device, gpus=[0, 1, 2, 3])""",
        """WDL(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,device=device, gpus=[0, 1, 2, 3])""",
        """NFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, device=device, gpus=[0, 1, 2, 3])""",
        """DCN(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,device=device, gpus=[0, 1, 2, 3])""",
        """DCNMix(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,device=device, gpus=[0, 1, 2, 3])""",
        """xDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,device=device, gpus=[0, 1, 2, 3])""",
        """AutoInt(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,device=device, gpus=[0, 1, 2, 3])""",
        """ONN(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,device=device, gpus=[0, 1, 2, 3])""",
        """FiBiNET(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,device=device, gpus=[0, 1, 2, 3])""",
        """IFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,device=device, gpus=[0, 1, 2, 3])""",
    ]

    # submit = pd.read_csv(f'{FEATURE_PATH}/test_data.csv')[['userid', 'feedid']]
    test = pd.read_csv(f'{EVALUATE_PATH}/evaluate_14.csv')

    train = pd.read_csv(FEATURE_PATH + f'/nosample_train_data_for_read_comment.csv')
    train = train[train['date_'] < 14].reset_index(drop=True)

    submit = test[['userid', 'feedid', 'read_comment']]

    test[ACTION_LIST] = 0

    # feed_embed
    # feed_embed = pd.read_csv(f'{FEATURE_PATH}/feed_embed.csv') # _pca_32

    # user_tags
    # user_tags = pd.read_csv(f'{FEATURE_PATH}/user_tags_encoded_336.csv')

    # 散装
    # train = pd.merge(train, feed_embed, on='feedid', how='left')
    # test = pd.merge(test, feed_embed, on='feedid', how='left')

    #
    # train = pd.merge(train, user_tags, on='userid', how='left')
    # test = pd.merge(test, user_tags, on='userid', how='left')

    train = train.sample(frac=1, random_state=42).reset_index(drop=True)

    data = pd.concat((train, test)).reset_index(drop=True)
    dense_features = ['videoplayseconds']

    sparse_features = ['userid', 'feedid', 'authorid','bgm_song_id', 'bgm_singer_id']

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
    # fixlen_feature_columns.extend([DenseFeat(f'embed{i}', 1) for i in range(512)])
    # fixlen_feature_columns.extend([DenseFeat(f'tags{i}', 1) for i in range(336)])

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    train, test = data.iloc[:train.shape[0]].reset_index(drop=True), data.iloc[train.shape[0]:].reset_index(
        drop=True)

    # print(feature_names)

    # 散装
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    eval_dict = {}
    for m in search[10:]:
        for action in ACTION_LIST[:1]:
            device = 'cpu'
            use_cuda = True
            if use_cuda and torch.cuda.is_available():
                # print('cuda ready...')
                device = 'cuda:0'

            # model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
            #                task='binary', dnn_hidden_units=[256, 128],
            #                l2_reg_embedding=1e-1, device=device, gpus=[0, 1, 2, 3])
            model = eval(m)

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
            pred_ans = model.predict(test_model_input, batch_size=1024)
            del model
            gc.collect()
            torch.cuda.empty_cache()

            labels = submit[action].values
            userid_list = submit.userid.astype(str).tolist()
            uauc = uAUC(labels.flatten(), pred_ans.flatten(), userid_list)
            print(m)
            print(f"{action} uAUC: ", uauc)


