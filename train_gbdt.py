# encoding: utf-8
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error



def lgb_pred(tr_path, va_path, _sep = '\t', iter_num = 32):
    """
    @param tr_path: 训练集文件
    @param va_path: 验证集文件
    @param _sep: 分隔符
    @param iter_num: 迭代次数
    """

    """
    加载数据
    """
    print('Load data...')
    df_train = pd.read_csv(tr_path, header=None, sep=_sep)
    df_valid = pd.read_csv(va_path, header=None, sep=_sep)

    y_train = df_train[0].values  # 训练集标签
    y_valid = df_valid[0].values  # 验证集标签
    X_train = df_train.drop(0, axis=1).values  # 训练集数据
    X_valid = df_valid.drop(0, axis=1).values  # 验证集数据

    """
    转成 lightgbm 需要的数据格式
    """
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    """
    定义参数
    https://lightgbm.readthedocs.io/en/latest/Parameters.html
    """
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'l2', 'binary_logloss'},
        'num_leaves': 30,
        # 'max_depth': 7,
        'num_trees': 32,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    """
    训练
    """
    print('Start training...')
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=iter_num,
                    valid_sets=lgb_valid,
                    feature_name=['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'],
                    categorical_feature=['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21'],
                    early_stopping_rounds=5)

    """
    保存模型
    """
    print('Save model...')
    gbm.save_model('./model/lgb_model.model')

    """
    模型评估
    """
    print('Start predicting...')
    y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
    print('The rmse of prediction is:', mean_squared_error(y_valid, y_pred) ** 0.5)

    return gbm, y_pred, X_train, y_train


if __name__== '__main__':
    lgb_pred('./data/train_lgb.txt', './data/test_lgb.txt', _sep = '\t', iter_num = 32)

