import os
import time
import math
import pickle
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from gensim.models import Word2Vec


np.random.seed(0)


def reduce_mem(df: pd.DataFrame):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem,
                                                                                                    100 * (start_mem - end_mem) / start_mem,
                                                                                                    (time.time() - starttime) / 60))
    return df


def get_all_click_sample(data_path, sample_nums=1000):
    all_click = pd.read_csv(data_path + 'train_click_log.csv')
    all_user_ids = all_click.user_id.unique()

    sample_user_ids = np.random.choice(all_user_ids, size=sample_nums, replace=False)
    all_click = all_click[all_click['user_id'].isin(sample_user_ids)]   # 从中选择一部分user
    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))     # 删除重复样本
    all_click['click_timestamp'] = all_click['click_timestamp'].apply(lambda x: x/1000)
    return all_click


def get_all_click_df(data_path, offline=True):
    if offline:
        all_click = pd.read_csv(data_path+'train_click_log.csv')
    else:
        trn_click = pd.read_csv(data_path+'train_click_log.csv')
        tst_click = pd.read_csv(data_path+'testA_click_log.csv')
        all_click = trn_click.append(tst_click)

    all_click = all_click.drop_duplicates((['user_id', 'click_article_id', 'click_timestamp']))
    reduce_mem(all_click)
    # 转化时间戳
    all_click['click_timestamp'] = all_click['click_timestamp'].apply(lambda x: x / 1000)
    return all_click


def get_article_info_df(data_path):
    article_info_df = pd.read_csv(data_path + 'articles.csv')
    article_info_df = article_info_df.drop_duplicates()     # 删除重复数据
    article_info_df = reduce_mem(article_info_df)
    # 转化时间戳
    article_info_df['created_at_ts'] = article_info_df['created_at_ts'].apply(lambda x: x / 1000)
    return article_info_df


def trn_val_split(all_click_df, sample_user_nums):
    all_click = all_click_df
    all_user_id = all_click['user_id'].unique()
    sample_user_ids = np.random.choice(all_user_id, size=sample_user_nums, replace=False)   # replace=False表示无放回抽样

    click_val = all_click[all_click['user_id'].isin(sample_user_ids)]       # 验证集
    click_trn = all_click[~all_click['user_id'].isin(sample_user_ids)]      # 训练集

    click_val = click_val.sort_values(['user_id', 'click_timestamp'])
    val_ans = click_val.groupby('user_id').tail(1)
    click_val = click_val.groupby('user_id').apply(lambda x: x[:-1]).reset_index(drop=True)     # 当只有一个点击的时候

    val_ans = val_ans[val_ans['user_id'].isin(click_val['user_id'].unique())]
    click_val = click_val[click_val['user_id'].isin(val_ans['user_id'].unique())]
    return click_trn, click_val, val_ans


def get_trn_val_tst_data(data_path, offline=True):
    if offline:
        click_trn_data = pd.read_csv(data_path+'train_click_log.csv')
        click_trn_data = reduce_mem(click_trn_data)
        click_trn, click_val, val_ans = trn_val_split(click_trn_data, sample_user_nums=1000)
    else:
        click_trn = pd.read_csv(data_path+'train_click_log.csv')
        click_trn = reduce_mem(click_trn)
        click_val = None
        val_ans = None
    click_tst = pd.read_csv(data_path+'testA_click_log.csv')
    return click_trn, click_val, val_ans, click_tst


def get_user_item_time(click_df):
    '''
    根据点击时间获取用户点击文章序列 {user1: [(item1, time1), (item2, time2)...]...}
    '''
    click_df = click_df.sort_values('click_timestamp')      # 根据文章被点击的时间进行排列

    def make_item_time_pair(df):
        return list(zip(df['click_article_id'], df['click_timestamp']))

    user_item_time_df = click_df.groupby('user_id')['click_article_id', 'click_timestamp'].apply(lambda x: make_item_time_pair(x)).reset_index().rename(columns={0: 'item_time_list'})

    user_item_time_dict = dict(zip(user_item_time_df['user_id'], user_item_time_df['item_time_list']))
    return user_item_time_dict


def get_item_topk_click(click_df, k):
    topk_click = click_df['click_article_id'].value_counts().index[:k]
    return topk_click


def itemcf_sim(df, item_created_time_dict):
    '''item之间的相似性矩阵计算'''
    user_item_time_dict = get_user_item_time(df)
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        for loc1, (i, i_click_time) in enumerate(item_time_list):  # 用户user点击的item序列，这里的i是item序列号
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_click_time) in enumerate(item_time_list):
                if i == j: continue
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                loc_weight = loc_alpha * (0.9 ** (np.abs(loc1 - loc2) - 1))       # 顺序权重
                click_time_weight = np.exp(0.7 ** np.abs(i_click_time - j_click_time))  # 点击时间差权重
                created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))  # item创建权重
                i2i_sim[i].setdefault(j, 0)
                i2i_sim[i][j] += loc_weight * click_time_weight * created_time_weight / math.log(len(item_time_list) + 1)      # 使用点击序列长度做归一化

    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j] + 1e-6)

    return i2i_sim_


def get_item_created_time_dict(data_path):
    df = pd.read_csv(data_path + 'articles.csv')
    df = df.drop_duplicates(['article_id', 'created_at_ts'])
    item_created_time_dict = dict(zip(df['article_id'], df['created_at_ts']))
    return item_created_time_dict


def get_hist_and_last_click(all_click):
    all_click = all_click.sort_values(by=['user_id', 'click_timestamp'])
    click_last_df = all_click.groupby('user_id').tail(1)

    def hist_func(user_df):
        if len(user_df) == 1:
            return user_df
        else:
            return user_df[:-1]

    click_hist_df = all_click.groupby('user_id').apply(hist_func).reset_index(drop=True)

    return click_hist_df, click_last_df


def train_item_word2vec(click_df, embed_size=64, save_name='item_w2v_emb.pkl', split_char=' '):
    click_df = click_df.sort_values('click_timestamp')
    # 只有转换成字符才可以进行训练
    click_df['click_article_id'] = click_df['click_article_id'].astype(str)
    # 转化成句子
    docs = click_df.groupby(['user_id'])['click_article_id'].apply(lambda x: list(x)).reset_index()
    docs = docs['click_article_id'].values.tolist()
    w2v = Word2Vec(docs, vector_size=16, sg=1, window=5, seed=2020, workers=4, min_count=1, epochs=1)

    item_w2v_emb_dict = {k: w2v.wv[k] for k in click_df['click_article_id']}
    pickle.dump(item_w2v_emb_dict, open('./'+save_name, 'wb'))
    return item_w2v_emb_dict


if __name__ == '__main__':
    from scipy import signal
    b, a = signal.butter(2, Wn=[0.5*2/100, 8*2/100], btype='pass')
    print(b)
    print(a)

