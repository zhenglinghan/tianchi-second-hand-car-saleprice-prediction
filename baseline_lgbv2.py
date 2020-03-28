#!/usr/bin/env python 
# encoding: utf-8 

"""
@version: v1.0
@author: zhenglinghan
@contact: 422807471@qq.com
@software: PyCharm
@file: baseline_lgbv2.py
@time: 2020/3/25 11:34
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
from tqdm import tqdm
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error


class feature_tool:

    def __init__(self):
        pass

    def AssignBin(self, x, cutOffPoints, special_attribute=[]):
        '''
        :param x: 某个变量的某个取值
        :param cutOffPoints: 上述变量的分箱结果，用切分点表示
        :param special_attribute:  不参与分箱的特殊取值
        :return: 分箱后的对应的第几个箱，从0开始
        for example, if cutOffPoints = [10,20,30], if x = 7, return Bin 0. If x = 35, return Bin 3
        '''
        numBin = len(cutOffPoints) + 1 + len(special_attribute)
        if x in special_attribute:
            i = special_attribute.index(x) + 1
            return 'Bin {}'.format(0 - i)
        if x <= cutOffPoints[0]:
            return 'Bin ' + str(cutOffPoints[0]) + '-'
        elif x > cutOffPoints[-1]:
            return 'Bin ' + str(cutOffPoints[-1]) + '+'
        else:
            for i in range(0, numBin - 1):
                if cutOffPoints[i] < x <= cutOffPoints[i + 1]:
                    return '(' + str(cutOffPoints[i]) + ', ' + str(cutOffPoints[i + 1]) + ']'

    def feature_psi_filter(self ,datatrain_,class_variables):
        '''

        :param datatrain_:
        :return:
        '''
        # check psi
        datatrain = datatrain_.copy()
        resultPSI = []
        for i in datatrain.columns:
            if i not in ['SaleID', 'price']+class_variables:
                # tobin
                dftrain = datatrain.loc[~datatrain['price'].isnull()]
                dftrain["{}_bin".format(i)] = pd.qcut(dftrain[i], 20, precision=4, duplicates='drop')
                group_by_bin = dftrain.groupby(["{}_bin".format(i)], as_index=True)
                df_min_max_bin = pd.DataFrame()
                df_min_max_bin["count_bin"] = group_by_bin["{}_bin".format(i)].count()
                list_bin = []
                for vari in range(df_min_max_bin.shape[0]):
                    list_bin.append(df_min_max_bin.index[vari].left)
                    list_bin.append(df_min_max_bin.index[vari].right)
                list_bin = sorted(list(set(list_bin)))
                dftest = datatrain.loc[datatrain['price'].isnull()]
                dftest['{}_bin'.format(i)] = dftest[i].map(lambda x: self.AssignBin(x, list_bin, special_attribute=[]))
                # cal psi
                gp0all = dftrain.groupby('{}_bin'.format(i))['SaleID'].count().reset_index()
                gp0all.columns = ['bin', 'total_train']
                gp1all = dftest.groupby('{}_bin'.format(i))['SaleID'].count().reset_index()
                gp1all.columns = ['bin', 'total_test']
                gp0all['bin'] = gp0all['bin'].astype(str)
                gp1all['bin'] = gp1all['bin'].astype(str)
                gp = pd.merge(gp0all, gp1all, on='bin', how='outer')
                gp['varName'] = i
                gp['total_train'] = gp['total_train'] / gp['total_train'].sum()
                gp['total_test'] = gp['total_test'] / gp['total_test'].sum()
                gp.fillna(0, inplace=True)
                gp['PSI'] = gp.apply(
                    lambda x: (x['total_test'] - x['total_train'] + 0.000000001) * np.log(
                        x['total_test'] / (x['total_train'] + 0.000000001)), axis=1)
                PSI = gp['PSI'].sum()
                print(i, PSI)
                resultPSI.append(gp)

            elif i in class_variables:
                # tobin
                dftrain = datatrain.loc[~datatrain['price'].isnull()]
                dftrain["{}_bin".format(i)] = dftrain[i].astype('str')
                group_by_bin = dftrain.groupby(["{}_bin".format(i)], as_index=True)
                df_min_max_bin = pd.DataFrame()
                df_min_max_bin["count_bin"] = group_by_bin["{}_bin".format(i)].count()
                dftest = datatrain.loc[datatrain['price'].isnull()]
                dftest['{}_bin'.format(i)] = dftest[i].astype('str')
                # cal psi
                gp0all = dftrain.groupby('{}_bin'.format(i))['SaleID'].count().reset_index()
                gp0all.columns = ['bin', 'total_train']
                gp1all = dftest.groupby('{}_bin'.format(i))['SaleID'].count().reset_index()
                gp1all.columns = ['bin', 'total_test']
                gp0all['bin'] = gp0all['bin'].astype(str)
                gp1all['bin'] = gp1all['bin'].astype(str)
                gp = pd.merge(gp0all, gp1all, on='bin', how='outer')
                gp['varName'] = i
                gp['total_train'] = gp['total_train'] / gp['total_train'].sum()
                gp['total_test'] = gp['total_test'] / gp['total_test'].sum()
                gp.fillna(0, inplace=True)
                gp['PSI'] = gp.apply(
                    lambda x: (x['total_test'] - x['total_train'] + 0.000000001) * np.log(
                        x['total_test'] / (x['total_train'] + 0.000000001)), axis=1)
                PSI = gp.loc[(gp['total_train']>0.000000001) &(gp['total_test']>0.000000001),'PSI'].sum()
                print(i, PSI)
                resultPSI.append(gp)
        psi_result=pd.concat(resultPSI, axis=0, ignore_index=True)
        q = psi_result.loc[(psi_result['total_train']>0.000000001) &(psi_result['total_test']>0.000000001)].groupby('varName')['PSI'].sum().reset_index()
        q_drop = q.loc[(q['PSI'] > 0.3)]
        print('droped!')
        print(q_drop)
        datatrain_.drop(list(q_drop['varName']), axis=1, inplace=True)
        return datatrain_,psi_result

    def null_importance_filter(self, data, id='group', label='type'):
        def get_feature_importances(data, shuffle, seed=None):
            # Gather real features
            train_features = [f for f in data if f not in [id, label]]
            # Go over fold and keep track of CV score (train and valid) and feature importances

            # Shuffle target if required
            y = data[label].copy()
            if shuffle:
                # Here you could as well use a binomial distribution
                y = data[label].copy().sample(frac=1.0)

            # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
            dtrain = lgb.Dataset(data[train_features], y, free_raw_data=False, silent=True)
            # lgb_params = {
            #     'objective': 'multiclass',
            #     'num_class': 3,
            #     'learning_rate': .03,
            #     'boosting_type': 'rf',
            #     "feature_fraction": 0.8,
            #     "bagging_freq": 1,
            #     "bagging_fraction": 0.8,
            #     'num_leaves': 50,
            #     'max_depth': -1,
            #     'seed': seed,
            #     'n_jobs': 3
            # }

            lgb_params = {
                'objective': 'regression',
                'learning_rate': .1,
                "feature_fraction": 0.8,
                "bagging_freq": 1,
                "bagging_fraction": 0.8,
                'num_leaves': 63,
                'max_depth': -1,
                'seed': seed,
                'n_jobs': 3
            }

            # Fit the model
            clf = lgb.train(params=lgb_params, train_set=dtrain, num_boost_round=500)

            # Get feature importances
            imp_df = pd.DataFrame()
            imp_df["feature"] = list(train_features)
            imp_df["importance_gain"] = clf.feature_importance(importance_type='gain')
            imp_df["importance_split"] = clf.feature_importance(importance_type='split')
            g = clf.predict(data[train_features])
            # print(g.shape)
            imp_df['trn_score'] = mean_squared_error(y, g, average='macro')

            return imp_df

        np.random.seed(123)
        # Get the actual importance, i.e. without shuffling
        actual_imp_df = get_feature_importances(data=train, shuffle=False)
        print(actual_imp_df)
        print('*' * 100)
        null_imp_df = pd.DataFrame()
        nb_runs = 30
        import time

        start = time.time()
        dsp = ''
        seeds =[i+2000 for i in range(nb_runs)]
        for i in range(nb_runs):
            # Get current run importances
            imp_df = get_feature_importances(data=train, shuffle=True, seed=seeds[i])
            imp_df['run'] = i + 1
            # Concat the latest importances with the old ones
            null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
            # Erase previous message
            for l in range(len(dsp)):
                print('\b', end='', flush=True)
            # Display current run and time used
            spent = (time.time() - start) / 60
            dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
            print(dsp, end='', flush=True)

        from matplotlib import pyplot as plt
        import seaborn as sns
        from matplotlib import gridspec

        def display_distributions(actual_imp_df_, null_imp_df_, feature_):
            plt.figure(figsize=(13, 6))
            gs = gridspec.GridSpec(1, 2)
            # Plot Split importances
            ax = plt.subplot(gs[0, 0])
            a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_split'].values,
                        label='Null importances')
            ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_split'].mean(),
                      ymin=0, ymax=np.max(a[0]), color='r', linewidth=10, label='Real Target')
            ax.legend()
            ax.set_title('Split Importance of %s' % feature_.upper(), fontweight='bold')
            plt.xlabel('Null Importance (split) Distribution for %s ' % feature_.upper())
            # Plot Gain importances
            ax = plt.subplot(gs[0, 1])
            a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_gain'].values,
                        label='Null importances')
            ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_gain'].mean(),
                      ymin=0, ymax=np.max(a[0]), color='r', linewidth=10, label='Real Target')
            ax.legend()
            ax.set_title('Gain Importance of %s' % feature_.upper(), fontweight='bold')
            plt.xlabel('Null Importance (gain) Distribution for %s ' % feature_.upper())

        feature_scores = []
        for _f in actual_imp_df['feature'].unique():
            f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
            f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
            gain_score = np.log(
                1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
            f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
            f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
            split_score = np.log(
                1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
            feature_scores.append((_f, split_score, gain_score))

        scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])

        plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(1, 2)
        # Plot Split importances
        ax = plt.subplot(gs[0, 0])
        sns.barplot(x='split_score', y='feature', data=scores_df.sort_values('split_score', ascending=False).iloc[0:70],
                    ax=ax)
        ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
        # Plot Gain importances
        ax = plt.subplot(gs[0, 1])
        sns.barplot(x='gain_score', y='feature', data=scores_df.sort_values('gain_score', ascending=False).iloc[0:70],
                    ax=ax)
        ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
        plt.tight_layout()

        correlation_scores = []
        for _f in actual_imp_df['feature'].unique():
            f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
            f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
            gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
            f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
            f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
            split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
            correlation_scores.append((_f, split_score, gain_score))

        corr_scores_df = pd.DataFrame(correlation_scores, columns=['feature', 'split_score', 'gain_score'])

        fig = plt.figure(figsize=(16, 16))
        gs = gridspec.GridSpec(1, 2)
        # Plot Split importances
        ax = plt.subplot(gs[0, 0])
        sns.barplot(x='split_score', y='feature',
                    data=corr_scores_df.sort_values('split_score', ascending=False).iloc[0:70], ax=ax)
        ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
        # Plot Gain importances
        ax = plt.subplot(gs[0, 1])
        sns.barplot(x='gain_score', y='feature',
                    data=corr_scores_df.sort_values('gain_score', ascending=False).iloc[0:70],
                    ax=ax)
        ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.suptitle("Features' split and gain scores", fontweight='bold', fontsize=16)
        fig.subplots_adjust(top=0.93)

        def score_feature_selection(df=None, train_features=None, cat_feats=None, target=None):
            # Fit LightGBM
            dtrain = lgb.Dataset(df[train_features], target, free_raw_data=False, silent=True)
            # lgb_params = {
            #     'objective': 'multiclass',
            #     'learning_rate': .03,
            #     'num_class': 3,
            #     'boosting_type': 'rf',
            #     "feature_fraction": 0.8,
            #     "bagging_freq": 1,
            #     "bagging_fraction": 0.8,
            #     'num_leaves': 50,
            #     'max_depth': -1,
            #     'seed': 0,
            #     'n_jobs': 3
            # }
            lgb_params = {
                'objective': 'regression',
                'learning_rate': .1,
                "feature_fraction": 0.8,
                "bagging_freq": 1,
                "bagging_fraction": 0.8,
                'num_leaves': 63,
                'max_depth': -1,
                'seed': 0,
                'n_jobs': 3
            }

            # Fit the model
            hist = lgb.cv(
                params=lgb_params,
                train_set=dtrain,
                num_boost_round=1000,
                categorical_feature=cat_feats,
                nfold=5,
                stratified=True,
                shuffle=True,
                early_stopping_rounds=100,
                verbose_eval=0,
                seed=0
            )
            # Return the last mean / std values
            return hist['multi_logloss-mean'][-1], hist['multi_logloss-stdv'][-1]

        # features = [f for f in data.columns if f not in ['SK_ID_CURR', 'TARGET']]
        # score_feature_selection(df=data[features], train_features=features, target=data['TARGET'])
        categorical_feats = []
        for threshold in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
            split_feats = [_f for _f, _score, _ in correlation_scores if _score >= threshold]
            split_cat_feats = [_f for _f, _score, _ in correlation_scores if
                               (_score >= threshold) & (_f in categorical_feats)]
            gain_feats = [_f for _f, _, _score in correlation_scores if _score >= threshold]
            gain_cat_feats = [_f for _f, _, _score in correlation_scores if
                              (_score >= threshold) & (_f in categorical_feats)]

            print('Results for threshold %3d' % threshold)
            split_results = score_feature_selection(df=train, train_features=split_feats, cat_feats=split_cat_feats,
                                                    target=train[label])
            print('\t SPLIT : %.6f +/- %.6f' % (split_results[0], split_results[1]))
            gain_results = score_feature_selection(df=train, train_features=gain_feats, cat_feats=gain_cat_feats,
                                                   target=train[label])
            print('\t GAIN  : %.6f +/- %.6f' % (gain_results[0], gain_results[1]))
        return gain_feats

def get_MAE(y_true, y_pred):
    score = np.mean(np.abs(np.expm1(y_pred) - np.expm1(y_true)))
    return 'MAE_score', score, False


def get_MAE_local(y_true, y_pred):
    score = np.mean(np.abs(np.expm1(y_pred) - np.expm1(y_true)))
    return score


if __name__ == "__main__":

    train = pd.read_csv('../used_car_train_20200313/used_car_train_20200313.csv', sep=' ')
    # 删一些样本
    train = train.loc[train['price'] < 70000]
    train = train.loc[train['seller'] == 0]
    train['price'] = np.log1p(train['price'])
    test = pd.read_csv('../used_car_testA_20200313/used_car_testA_20200313.csv', sep=' ')
    data = pd.concat([train, test], axis=0, ignore_index=True)

    del data['seller'], data['offerType']
    data.loc[data['model'].isna(), 'model'] = 167.0# 填入均值最接近的
    data.loc[data['model']==247, 'model'] = 237# test里没有
    data['model_last_2']=data['model'].map(lambda x:int(str(int(x))[-2:]))
    data['name_len']=data['name'].map(lambda x:len(str(x)))
    data.loc[data['regionCode'] == 1001, 'regionCode'] = 2029  # test里没有
    data.loc[data['regionCode'] == 1023, 'regionCode'] = 6852
    data.loc[data['regionCode'] == 1025, 'regionCode'] = 2255
    data.loc[data['regionCode'] == 1026, 'regionCode'] = 7410
    data.loc[data['regionCode'] == 1031, 'regionCode'] = 2698
    data.loc[data['regionCode'] == 1149, 'regionCode'] = 5871
    data.loc[data['regionCode'] == 115, 'regionCode'] = 1816
    data.loc[data['notRepairedDamage'] == '-', 'notRepairedDamage'] = '2'
    data['notRepairedDamage'] = data['notRepairedDamage'].astype('float').astype(int)

    # 查看psi
    class_variable=['name', 'model', 'brand', 'regionCode','fuelType', 'bodyType', 'gearbox', 'notRepairedDamage']
    tl=feature_tool()
    data,psi_result =tl.feature_psi_filter(data,class_variable)

    for var in ['name', 'model', 'brand', 'regionCode']:
        data[var] = data[var].map(data[var].value_counts())

    cat_fe = ['fuelType', 'bodyType', 'gearbox', 'notRepairedDamage']

    data.loc[data['fuelType'].isna(), 'fuelType'] = 10000
    data.loc[data['gearbox'].isna(), 'gearbox'] = 10000
    data.loc[data['bodyType'].isna(), 'bodyType'] = -1

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

    for var in ['name_len','model']:
        fe1=data.loc[~data['price'].isna()].groupby(var)['price'].mean().rename('{}_price_mean'.format(var)).reset_index()
        data=data.merge(fe1,on=var,how='left')


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
            reg_alpha=0.5,
            reg_lambda=0.01,
            metric='None', n_jobs=3)
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

    subsample = pd.read_csv('../used_car_sample_submit.csv')
    del subsample['price']
    subsample = subsample.merge(test, on='SaleID', how='left')
    subsample['price'] = np.floor(np.expm1(subsample['price']))
    subsample[['SaleID', 'price']].to_csv('baseline0325.csv', index=False)