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
from config import *
import time

if __name__ == "__main__":
    submit = pd.read_csv(ROOT_PATH + '/test_data.csv')[['userid', 'feedid']]

    for action in ACTION_LIST:
        USE_FEAT = ['userid', 'feedid', action] + FEA_FEED_LIST[1:]
        train = pd.read_csv(ROOT_PATH + f'/train_data_for_{action}.csv')[USE_FEAT]

        # shuffle..
        train = train.sample(frac=1, random_state=42).reset_index(drop=True)

        # 正例比
        print("posi prop:")
        print(sum((train[action] == 1) * 1) / train.shape[0])

        test = pd.read_csv(ROOT_PATH + '/test_data.csv')[[i for i in USE_FEAT if i != action]]
        target = [action]
        test[target[0]] = 0
        test = test[USE_FEAT]
        data = pd.concat((train, test)).reset_index(drop=True)
        dense_features = ['videoplayseconds']
        sparse_features = [i for i in USE_FEAT if i not in dense_features and i not in target]

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
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        feature_names = get_feature_names(
            linear_feature_columns + dnn_feature_columns)

        # 3.generate input data for model
        train, test = data.iloc[:train.shape[0]].reset_index(drop=True), data.iloc[train.shape[0]:].reset_index(
            drop=True)
        train_model_input = {name: train[name] for name in feature_names}
        test_model_input = {name: test[name] for name in feature_names}

        # 4.Define Model,train,predict and evaluate
        device = 'cpu'
        use_cuda = True
        if use_cuda and torch.cuda.is_available():
            print('cuda ready...')
            device = 'cuda:0'

        model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                       task='binary',
                       l2_reg_embedding=1e-1, device=device, gpus=[0, 1, 2, 3])

        model.compile("adagrad", "binary_crossentropy", metrics=["binary_crossentropy", "auc"], )
        model.fit(train_model_input, train[target].values, batch_size=512, epochs=5, verbose=1, validation_split=0.2)

        # evaluation

        # test
        pred_ans = model.predict(test_model_input, batch_size=128)
        submit[action] = pred_ans
        torch.save(model, f'{MODEL_PATH}/DeepFM_model_{action}.h5')
        torch.cuda.empty_cache()
    # 保存提交文件
    submit.to_csv(f'{SUBMIT_PATH}/DeepFM_submit_{time.time()}.csv', index=False)
