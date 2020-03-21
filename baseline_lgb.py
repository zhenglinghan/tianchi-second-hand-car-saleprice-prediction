#!/usr/bin/env python 
# encoding: utf-8 

"""
@version: v1.0
@author: zhenglinghan
@contact: 422807471@qq.com
@software: PyCharm
@file: baselinelgb.py
@time: 2020/3/21 22:38
"""

import pandas as pd
import numpy as np
import os
import gc
import datetime as dt
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('max_colwidth', 200)
pd.set_option('display.width', 5000)

import lightgbm as lgb
from sklearn.model_selection import KFold


def func():
    pass


class tl:
    def __init__(self):
        pass


def get_MAE(y_true, y_pred):
    score = np.mean(np.abs(np.expm1(y_pred) - np.expm1(y_true)))
    return 'MAE_score', score, False


def get_MAE_local(y_true, y_pred):
    score = np.mean(np.abs(np.expm1(y_pred) - np.expm1(y_true)))
    return score


if __name__ == "__main__":

    train = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    # 删一些样本
    train = train.loc[train['price'] < 70000]
    train = train.loc[train['seller'] == 0]
    train['price'] = np.log1p(train['price'])
    test = pd.read_csv('used_car_testA_20200313.csv', sep=' ')
    data = pd.concat([train, test], axis=0, ignore_index=True)

    del data['seller'], data['offerType']

    data.loc[data['model'].isna(), 'model'] = -1
    for var in ['name', 'model', 'brand', 'regionCode']:
        data[var] = data[var].map(data[var].value_counts())

    cat_fe = ['fuelType', 'bodyType', 'gearbox', 'notRepairedDamage']
    data.loc[data['notRepairedDamage'] == '-', 'notRepairedDamage'] = '2'
    data.loc[data['fuelType'].isna(), 'fuelType'] = 10000
    data.loc[data['gearbox'].isna(), 'gearbox'] = 10000
    data.loc[data['bodyType'].isna(), 'bodyType'] = -1
    data['notRepairedDamage'] = data['notRepairedDamage'].astype('float').astype(int)

    for var in cat_fe:
        data[var] = data[var].astype('category')
    date_col = ['regDate', 'creatDate']


    def getdate(p):
        if int(str(p)[4:6]) > 0 and int(str(p)[4:6]) <= 12 \
                and int(str(p)[6:]) > 0 and int(str(p)[6:]) <= 31:
            return pd.to_datetime(str(p)[:4] + '-' + str(p)[4:6] + '-' + str(p)[6:])
        elif int(str(p)[4:6]) <= 0 or int(str(p)[4:6]) > 12:
            return pd.to_datetime(str(p)[:4] + '-01-' + str(p)[6:])
        elif int(str(p)[6:]) <= 0 and int(str(p)[6:]) > 31:
            return pd.to_datetime(str(p)[:4] + '-' + str(p)[4:6] + '-30')
        else:
            return pd.to_datetime(np.nan)


    for var in date_col:
        data['{}_date'.format(var)] = data[var].map(lambda x: getdate(x))
        data['{}_year_to_now'.format(var)] = 2020 - data['{}_date'.format(var)].map(lambda x: x.year)
        data['{}_month_to_now'.format(var)] = 12 * data['{}_year_to_now'.format(var)] + \
                                              12 - data['{}_date'.format(var)].map(lambda x: x.month)
        data['{}_weekofyear'.format(var)] = data['{}_date'.format(var)].map(lambda x: x.weekofyear)
        del data['{}_date'.format(var)]

    data['power_m_kilometer'] = data['power'] * data['kilometer']
    data['kilometer_d_regDate_month_to_now'] = data['kilometer'] / data['regDate_month_to_now']

    del data['creatDate_year_to_now']

    data.loc[data['kilometer'].isna(), 'kilometer'] = -1

    train = data.loc[~data['price'].isna()].copy().reset_index(drop=True)
    test = data.loc[data['price'].isna()].copy().reset_index(drop=True)

    features = [i for i in train.columns if i not in ['SaleID', 'price']]

    n_split = 5
    x_test = test[features].copy()
    x = train[features].copy()
    y = train['price'].copy()
    preds_train = np.zeros((train.shape[0],))
    preds = np.zeros((test.shape[0],))
    train_score = []
    val_score = []
    kfold = KFold(n_splits=n_split, shuffle=True, random_state=0)
    for i, (tr_idx, val_idx) in enumerate(kfold.split(x, y)):
        print('model_{}'.format(i + 1))
        x_train = x.iloc[tr_idx]
        y_train = y.iloc[tr_idx]
        x_val = x.iloc[val_idx]
        y_val = y.iloc[val_idx]
        print(x_train.shape, x_val.shape)
        model = lgb.LGBMRegressor(
            n_estimators=5000,
            learning_rate=0.1,
            objective='mae',
            random_state=0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.01,
            metric='None', n_jobs=32)
        model.fit(
            x_train, y_train,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            eval_metric=get_MAE,
            early_stopping_rounds=50, feature_name=features,
            verbose=100, categorical_feature=cat_fe)

        if i == 0:
            lgb_predictors = [i for i in x_train[features].columns]
            lgb_feat_imp = pd.Series(model.feature_importances_, lgb_predictors).sort_values(ascending=False)
            df_imp = lgb_feat_imp.reset_index()
            df_imp.columns = ['varName', 'imp']
            print(df_imp)

        preds_train[val_idx] = model.predict(x_val)
        preds += model.predict(x_test) / n_split

        trn_hat = model.predict(x_train)
        val_hat = model.predict(x_val)

        train_score.append(get_MAE_local(y_train, trn_hat))
        val_score.append(get_MAE_local(y_val, val_hat))

    print(np.mean(train_score), np.std(train_score), np.mean(val_score), np.std(val_score))
    print(get_MAE_local(y, preds_train))
    test['price'] = preds
    print('=' * 50)

    subsample = pd.read_csv('used_car_sample_submit.csv')
    del subsample['price']
    subsample = subsample.merge(test, on='SaleID', how='left')
    subsample['price'] = np.floor(np.expm1(subsample['price']))
    subsample[['SaleID', 'price']].to_csv('baseline0321.csv', index=False)