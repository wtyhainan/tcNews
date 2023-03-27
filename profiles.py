import os
import time
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils import get_trn_val_tst_data, get_article_info_df, get_all_click_df, get_hist_and_last_click, get_all_click_sample
from utils import reduce_mem


data_path = 'E:\\tcNews\\datas\\'
save_path = '.\\result'


class MaxMinNormal(object):
    def __init__(self, cols):
        self.transformers = [MinMaxScaler() for i in range(len(cols))]
        self.cols = cols

    def fit_transform(self, data_df):
        for col_name, mm in zip(self.cols, self.transformers):
            data_df[col_name] = mm.fit_transform(data_df[[col_name]])
        return data_df

    def transform(self, data_df):
        for col_name, mm in zip(self.cols, self.transformers):
            data_df[col_name] = mm.transform(data_df[[col_name]])
        return data_df


def get_user_profile(data_df):
    '''用户画像
    1 点击文章字数
    2 点击文章主题
    3 点击文章时间
    4 用户活跃度
    5
    '''
    user_df = data_df.sort_values(by=['user_id', 'click_timestamp'])

    # 用户点击文章时间与文章创建时间差
    user_df['time_diff_click_created'] = user_df['click_timestamp'] - user_df['created_at_ts']

    user_df = user_df.groupby('user_id', as_index=False)[['click_article_id', 'click_timestamp', 'words_count',
                                                          'category_id', 'time_diff_click_created']]\
                    .agg({'click_article_id': np.size, 'click_timestamp': list, 'words_count': np.mean,
                          'category_id': list, 'time_diff_click_created': np.mean})

    # 用户活跃度
    def time_diff(l):
        if len(l) == 1:
            return 1
        else:
            return np.mean(np.diff(l))

    user_df['time_diff'] = user_df['click_timestamp'].apply(lambda x: time_diff(x))
    user_df['click_size'] = 1 / user_df['click_article_id']
    if os.path.exists('./result/user_normal.pkl'):
        print('加载用户归一化模型')
        transformers = pickle.load(open('./result/user_normal.pkl', 'rb'))
        user_df = transformers.transform(user_df)
    else:
        transformers = MaxMinNormal(cols=['time_diff', 'click_size', 'time_diff_click_created'])
        user_df = transformers.fit_transform(user_df)
        # pickle.dump(transformers, open('./result/user_normal.pkl', 'wb'))

    user_df['active_level'] = user_df['time_diff'] + user_df['click_size']
    user_df = user_df.rename(columns={'words_count': 'words_count_mean', 'category_id': 'habits'})
    user_df = user_df.drop(['click_article_id', 'click_timestamp'], axis=1)
    return user_df


def get_item_profile(all_data):
    '''文章画像
    1 文章时效性
    2 文章热度
    '''
    # 文章点击时间与创建时间差
    all_data['time_diff_click_created'] = all_data['click_timestamp'] - all_data['created_at_ts']

    # 文章热度（根据用户点击词数及点击时间差）
    item_df = all_data.groupby('article_id', as_index=False)[['user_id', 'click_timestamp', 'time_diff_click_created']].agg({'user_id': np.size,
                                                                                               'click_timestamp': list,
                                                                                               'time_diff_click_created': np.mean})
    print(item_df)
    def time_diff(l):
        if len(l) == 1:
            return 1
        else:
            return np.mean(np.diff(l))

    item_df['time_diff'] = item_df['click_timestamp'].apply(lambda x: time_diff(x))

    item_df['click_size'] = 1 / item_df['user_id']

    if os.path.exists('./result/hot_level.pkl'):
        transformers = pickle.load(open('./result/hot_level.pkl', 'rb'))
        item_df = transformers.transform(item_df)
    else:
        transformers = MaxMinNormal(cols=['time_diff', 'click_size', 'time_diff_click_created'])
        item_df = transformers.fit_transform(item_df)
        # pickle.dump(transformers, open('./result/hot_level.pkl', 'wb'))

    item_df['hot_level'] = item_df['time_diff'] + item_df['click_size']
    item_df = item_df.drop(['user_id', 'click_timestamp', 'time_diff', 'click_size'], axis=1)
    return item_df


if __name__ == '__main__':
    offline = False

    click_trn, click_val, val_ans, click_tst = get_trn_val_tst_data(data_path, offline)

    # 获取训练集用户历史点击和最有一次点击
    click_hist_df, click_last_df = get_hist_and_last_click(click_trn)

    # 获取文章信息
    articles_info_df = get_article_info_df(data_path)

    # 使用全量数据来生成用户画像和文章画像
    # 训练集和验证集不能包括最后一次点击
    all_data = click_hist_df.append(click_tst)
    if click_val is not None:
        all_data = all_data.append(click_val)
    all_data = reduce_mem(all_data)

    # 拼接文章信息。只能使用不包括最后一次点击的历史点击数据来做用户画像和文章画像。
    all_data = all_data.merge(articles_info_df, left_on='click_article_id', right_on='article_id')     # 只使用历史点击数据

    # 用户画像
    users_profile = get_user_profile(all_data)
    users_profile.to_csv(save_path+'/users_profile.csv', index=False)
    print(users_profile)
    # 文章画像
    articles_profile = get_item_profile(all_data)
    articles_profile.to_csv(save_path + '/articles_profile.csv', index=False)
    print(articles_profile)








