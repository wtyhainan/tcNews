'''
制作特征和标签，转成监督学习问题
目前可以使用的特征：
1、文章自身特征，包括文章类型、文章字数、文章建立时间（这个关系文章时效性）
2、文章的Embedding特性。在这里可以使用该Embedding特征，也可以不使用。或者使用W2V特征。
3、用户设备特征
上述特征在做完特征工程后可以直接加入。现在我们需要根据召回结果构造一些特征，然后制作特征标签。
根据召回结果，我们会得到一个{user1: [item1, item2, item3...]}的字典。那么我们可以对于
每个用户，每篇可能点击的文章构造一个监督测试集，比如对于用户user1，假设它的召回列表为
{user1:[item1, item2, item3...]}，我们就可以得到三行数据(user1, item1), (user1, item2), (user1, item3)
这就是监督数据集的前两列特征。
构造特征的思路是这样的，我们知道每个用户的点击文章是与其历史点击的文章信息有关的。比如一个主题，相似等等。所以
构造特征很重要的一系列特征是要结合用户的历史点击文章信息。我们已经得到每个用户及点击候选文章的两列的一个数据集，而我们
的目的是预测最后一次点击的文章，比较自然的一个思路就是和其最后几次点击的文章产生关系，这样既考虑了其历史点击文章信息，
又得离最后一次点击比较近，因为新闻很大的一个特性就是注重时效。


对样本进行负采样
通过召回我们将数据转换成三元组的形式（user1, item1, label）的形式，观察发现正负样本差距极度不平衡，我们可以先对负样本
进行下采样，下采样的目的一方面是缓解正负样本比例的问题，另一方面也减小做排序特征的压力。在做负采样的时候又有哪些东西需要
注意的呢：
1、只对负样本进行下采样（如果有比较好的正样本扩充方法其实也是可以考虑的）
2、负采样后，保证所有用户和文章仍然出现在采样之后的数据中
3、下采样比例可以根据实际情况人为控制
4、做完负采样后，更新此时新的用户召回文章列表，因为后续做特征的时候可能会用到相对位置的信息
'''
import os
import pickle
import time
import warnings

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils import get_item_topk_click, get_hist_and_last_click, get_all_click_df, get_user_item_time
from utils import itemcf_sim, trn_val_split, get_article_info_df, get_all_click_sample, get_trn_val_tst_data
from recall import metrics_recall, item_based_recommend

warnings.filterwarnings('ignore')
np.random.seed(10)


data_path = 'E:\\tcNews\\datas\\'
save_path = '.\\result'


def neg_sample_recall_data(recall_items_df, sample_rate=0.001):
    pos_data = recall_items_df[recall_items_df['label'] == 1]
    neg_data = recall_items_df[recall_items_df['label'] == 0]
    print('pos_data_num: ', len(pos_data), 'neg_data_num: ', len(neg_data), 'pos/neg: ', len(pos_data) / len(neg_data))

    def neg_sample_func(group_df):
        neg_num = len(group_df)
        sample_num = max(int(neg_num * sample_rate), 1)
        sample_num = min(sample_num, 5)
        return group_df.sample(n=sample_num, replace=True)
    neg_data_user_sample = neg_data.groupby('user_id', group_keys=False).apply(neg_sample_func)
    neg_data_item_sample = neg_data.groupby('sim_item', group_keys=False).apply(neg_sample_func)

    # 将两种情况的采样数据合并
    # neg_data_new = neg_data_user_sample.append(neg_data_item_sample)
    neg_data_new = pd.concat([neg_data_item_sample, neg_data_user_sample], ignore_index=True)
    neg_data_new = neg_data_new.sort_values(['user_id', 'score']).drop_duplicates(['user_id', 'sim_item'], keep='last')
    data_new = pd.concat([pos_data, neg_data_new], ignore_index=True)
    return data_new


def make_tuple_func(item_label_df):
    def _make_tuple_func(group_df):
        row_data = []
        for name, row_df in group_df.iterrows():
            row_data.append((row_df['sim_item'], row_df['score'], row_df['label']))
        return row_data

    return item_label_df.groupby('user_id').apply(_make_tuple_func)


def user_time_hob_fea(all_data, cols):
    user_time_hob_info = all_data[cols]
    mm = MinMaxScaler()
    user_time_hob_info['click_timestamp'] = mm.fit_transform(user_time_hob_info[['click_timestamp']])
    user_time_hob_info['created_at_ts'] = mm.fit_transform(user_time_hob_info[['created_at_ts']])
    user_time_hob_info = user_time_hob_info.groupby('user_id').agg('mean').reset_index()
    user_time_hob_info.rename(columns={'click_timestamp': 'user_time_hob1', 'created_at_ts': 'user_time_hob2'}, inplace=True)
    return user_time_hob_info


def recall_dict_2_df(recall_list_dict):
    df_row_list = []
    for user, recall_list in tqdm(recall_list_dict.items()):
        for item, score in recall_list:
            df_row_list.append((user, item, score))

    col_names = ['user_id', 'sim_item', 'score']
    recall_list_df = pd.DataFrame(df_row_list, columns=col_names)
    return recall_list_df


# def get_rank_label_df(recall_item_dict, user_last_click_df, is_test=False):
#     recall_item = []
#     for user, item_score_list in recall_item_dict.items():
#         for item, score in item_score_list:
#             recall_item.append((user, item, score))
#
#     recall_item_df = pd.DataFrame(recall_item, columns=['user_id', 'sim_item', 'score'])
#
#     if is_test:
#         recall_item_df['label'] = -1
#         return recall_item_df
#
#     user_last_click_df = user_last_click_df.rename(columns={'click_article_id': 'sim_item'})
#
#     recall_item_df_ = recall_item_df.merge(right=user_last_click_df[['user_id', 'sim_item', 'click_timestamp']],
#                                            how='left', on=['user_id', 'sim_item'])
#     recall_item_df_['label'] = recall_item_df_['click_timestamp'].apply(lambda x: 0.0 if np.isnan(x) else 1.0)
#     del recall_item_df_['click_timestamp']
#     return recall_item_df_


def get_rank_label_df(recall_item_dict, user_last_click_df, is_test=False):
    recall_item_dict_ = {}
    for user in user_last_click_df['user_id'].unique():
        recall_item_dict_[user] = recall_item_dict[user]

    recall_item = []
    for user, item_score_list in recall_item_dict_.items():
        for item, score in item_score_list:
            recall_item.append((user, item, score))

    recall_item_df = pd.DataFrame(recall_item, columns=['user_id', 'sim_item', 'score'])

    if is_test:
        recall_item_df['label'] = -1
        return recall_item_df

    user_last_click_df = user_last_click_df.rename(columns={'click_article_id': 'sim_item'})

    recall_item_df_ = recall_item_df.merge(right=user_last_click_df[['user_id', 'sim_item', 'click_timestamp']],
                                           how='left', on=['user_id', 'sim_item'])
    recall_item_df_['label'] = recall_item_df_['click_timestamp'].apply(lambda x: 0.0 if np.isnan(x) else 1.0)
    del recall_item_df_['click_timestamp']
    return recall_item_df_


def create_features(users_id, recall_list, click_hist_dict, articles_info_df, articles_emb, user_emb=None, N=1):

    all_user_feas = []
    for user_id in tqdm(users_id):
        user_hist_items = click_hist_dict[user_id][:N]      # 最后点击的N个item
        for rank, (article_id, score, label) in enumerate(recall_list[user_id]):        # 每一个召回item
            a_create_time = articles_info_df[articles_info_df['article_id'] == article_id]['created_at_ts'].values[0]
            a_words_count = articles_info_df[articles_info_df['article_id'] == article_id]['words_count'].values[0]
            single_user_fea = [user_id, article_id]
            sim_fea = []
            time_fea = []
            word_fea = []
            for (hist_item, click_item_time) in user_hist_items:    # 历史点击的item
                b_create_time = articles_info_df[articles_info_df['article_id'] == hist_item]['created_at_ts'].values[0]
                b_words_count = articles_info_df[articles_info_df['article_id'] == hist_item]['words_count'].values[0]
                if articles_emb is not None:        # 计算文章相似性
                    sim_fea.append(np.dot(articles_emb[hist_item], articles_emb[article_id]))
                time_fea.append(abs(a_create_time - b_create_time))
                word_fea.append(abs(a_words_count - b_words_count))

            single_user_fea.extend(sim_fea)
            single_user_fea.extend(time_fea)
            single_user_fea.extend(word_fea)
            if sim_fea:
                single_user_fea.extend([max(sim_fea), min(sim_fea), sum(sim_fea), sum(sim_fea)/len(sim_fea)])

            single_user_fea.extend([score, rank, label])
            all_user_feas.append(single_user_fea)

    id_cols = ['user_id', 'click_article_id']
    sim_cols = ['sim' + str(i) for i in range(N)] if articles_emb else []
    time_cols = ['time_diff' + str(i) for i in range(N)]
    word_cols = ['word_diff' + str(i) for i in range(N)]
    sat_cols = ['sim_max', 'sim_min', 'sim_sum', 'sim_mean'] if articles_emb else []
    user_item_sim_cols = ['user_item_sim'] if user_emb else []
    user_score_rank_label = ['score', 'rank', 'label']
    cols = id_cols + sim_cols + time_cols + word_cols + sat_cols + user_item_sim_cols + user_score_rank_label

    df = pd.DataFrame(all_user_feas, columns=cols)
    df['click_article_id'] = df['click_article_id'].astype('int')
    return df


if __name__ == '__main__':
    recall_item_num = 50

    offline = True     # 线下

    # 加载文章信息
    articles_info = get_article_info_df(data_path)

    # 加载文章画像
    articles_profile = pd.read_csv('./result/articles_profile.csv')

    # 拼接文章画像
    articles_info = articles_info.merge(articles_profile, left_on='article_id', right_on='article_id')

    # 加载用户画像
    users_profile = pd.read_csv('./result/users_profile.csv')
    users_profile['habits'] = users_profile['habits'].apply(lambda x: [int(i) for i in x.strip('[').strip(']').split(',')])

    # 加载并划分数据集
    click_trn, click_val, val_ans, click_tst = get_trn_val_tst_data(data_path, offline)

    # 文章创建时间
    item_created_time_dict = dict(zip(articles_info['article_id'], articles_info['created_at_ts']))

    # 加载召回数据
    user_recall_dict = pickle.load(open('./result/user_recall_dict.pkl', 'rb'))

    # # 获取训练集用户历史点击和最后一次点击
    # trn_user_hist_click_df, trn_user_last_click_df = get_hist_and_last_click(click_trn)
    #
    # # 用户点击历史字典{user: [(item1, time1), (item2, time2)...]...}
    # trn_user_item_time_dict = get_user_item_time(trn_user_hist_click_df)
    #
    # # 根据user最后一次点击给召回数据打标签。
    # # 召回数据格式：{user1: [(item1, score1), (item2, score)...]...}
    # recall_item_label_df = get_rank_label_df(recall_item_dict=user_recall_dict,
    #                                          user_last_click_df=trn_user_last_click_df,
    #                                          is_test=False)
    #
    # recall_item_label_df = neg_sample_recall_data(recall_item_label_df, sample_rate=0.1)        # 下采样
    # recall_item_label_df = make_tuple_func(recall_item_label_df)
    #
    # '''特征工程'''
    # # 训练集
    # train_data_df = create_features(click_trn['user_id'].unique(),
    #                                 recall_item_label_df,
    #                                 trn_user_item_time_dict,
    #                                 articles_info, articles_emb=None, user_emb=None, N=1)
    #
    # train_data_df = train_data_df.merge(users_profile, how='left', left_on='user_id', right_on='user_id')  # 拼接用户画像
    #
    # train_data_df = train_data_df.merge(articles_info[['article_id', 'words_count', 'hot_level', 'time_diff_click_created',
    #                                                    'created_at_ts', 'category_id']], how='left', left_on='click_article_id',
    #                                     right_on='article_id').drop(['article_id'], axis=1)  # 拼接文章画像
    #
    # train_data_df['is_category_hab'] = train_data_df.apply(lambda x: 1 if x.category_id in set(x.habits) else 0, axis=1)
    #
    # train_data_df.to_csv('./result/train_data.csv', index=False)

    # 验证集
    val_user_item_time_dict = get_user_item_time(click_val)
    metrics_recall(user_recall_dict, val_ans, topk=recall_item_num)
    recall_item_label_df = get_rank_label_df(recall_item_dict=user_recall_dict, user_last_click_df=val_ans, is_test=False)

    recall_item_label_df = neg_sample_recall_data(recall_item_label_df, sample_rate=0.1)  # 下采样

    recall_item_label_df = make_tuple_func(recall_item_label_df)

    val_data_df = create_features(click_val['user_id'].unique(), recall_item_label_df, val_user_item_time_dict,
                                  articles_info, articles_emb=None, user_emb=None, N=1)

    val_data_df = val_data_df.merge(users_profile, how='left', left_on='user_id', right_on='user_id')  # 拼接用户画像
    print(users_profile.columns)

    val_data_df = val_data_df.merge(articles_info[['article_id', 'words_count', 'hot_level', 'time_diff_click_created',
                                                   'created_at_ts', 'category_id']], how='left', left_on='click_article_id',
                                        right_on='article_id').drop(['article_id'], axis=1)  # 拼接文章画像
    print(val_data_df.columns)

    val_data_df['is_category_hab'] = val_data_df.apply(lambda x: 1 if x.category_id in set(x.habits) else 0, axis=1)
    print(val_data_df.columns)

    val_data_df.to_csv('./result/val_data.csv', index=False)

