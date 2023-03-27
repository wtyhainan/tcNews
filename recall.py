import os
import collections
import pickle
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import warnings

from tqdm import tqdm
import numpy as np

from utils import get_all_click_df, get_article_info_df, get_hist_and_last_click, itemcf_sim, get_item_topk_click
from utils import get_user_item_time, get_all_click_sample, get_trn_val_tst_data

warnings.filterwarnings('ignore')

data_path = 'E:\\tcNews\\datas\\'
save_path = '.\\result'


def item_based_recommend(user_id,
                         user_item_time_dict,       # 字典, 根据点击时间获取用户的点击文章序列   {user1: [(item1, time1), (item2, time2)..]...}
                         i2i_sim,                   # 字典，文章相似性矩阵
                         sim_item_topk,             # 整数， 选择与当前文章最相似的前k篇文章
                         recall_item_num,           # 整数， 最后的召回文章数量
                         item_topk_click,           # 列表，点击次数最多的文章列表，用户召回补全
                         item_created_time_dict):   # 文章创建时间
    # 获取用户历史交互的文章
    user_hist_items = user_item_time_dict[user_id]
    user_hist_items_ = {user_id for user_id, _ in user_hist_items}
    item_rank = {}
    for loc, (i, click_time) in enumerate(user_hist_items):
        # for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
        if i2i_sim.get(i, None) is not None:    # 避免新用户点击item不在i2i内
            for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
                if j in user_hist_items_:
                    continue
                # 文章创建时间差权重
                created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))
                # 相似文章和历史点击文章序列中历史文章所在的位置权重
                loc_weight = (0.9 ** (len(user_hist_items) - loc))
                item_rank.setdefault(j, 0)
                item_rank[j] += created_time_weight * loc_weight * wij
        else:   # 当用户点击的item不在i2i相似性矩阵内时，应该如果做召回？
            pass

    # 不足10个，用热门商品补全
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items() or item in user_hist_items_:  # 填充的item应该不在原来的列表中（也不能在用户点击过的item中）
                continue
            item_rank[item] = - i - 100  # 随便给个负数
            if len(item_rank) == recall_item_num:
                break

    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return item_rank


def item_embedding_sim_based_recommend():
    '''基于文章的Embedding相似性推荐'''
    pass


def itemcf_recall(users, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click, item_created_time_dict):
    user_recall_items_dict = {}
    for user in tqdm(users):
        user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click, item_created_time_dict)
    return user_recall_items_dict


def multiproc_itemcf_recall(users, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click, item_created_time_dict, workers=4):
    per = len(users) // 20
    pool = ProcessPoolExecutor(max_workers=workers)
    fn = partial(itemcf_recall,
                 user_item_time_dict=user_item_time_dict,
                 i2i_sim=i2i_sim,
                 sim_item_topk=sim_item_topk,
                 recall_item_num=recall_item_num,
                 item_topk_click=item_topk_click,
                 item_created_time_dict=item_created_time_dict)
    datas = []
    for i in range(20):
        u = users[i*per:(i+1)*per]
        if i == 20-1:
            u = users[i*per:]
        datas.append(u)

    results = pool.map(fn, datas)

    user_itemcf_recall_dict = {}
    for result in results:
        user_itemcf_recall_dict.update(result)

    return user_itemcf_recall_dict


def metrics_recall(user_recall_items_dict, last_click_df, topk=5):
    last_click_item_dict = dict(zip(last_click_df['user_id'], last_click_df['click_article_id']))
    user_num = len(last_click_df)
    for k in range(5, topk + 1, 5):
        hit_num = 0
        for user, click_item in last_click_item_dict.items():
            tmp_recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
            if click_item in set(tmp_recall_items):
                hit_num += 1
        hit_rate = round(hit_num * 1.0 / user_num, 5)
        print(' topk: ', k, ' : ', 'hit_num: ', hit_num, 'hit_rate: ', hit_rate, 'user_num : ', user_num)


if __name__ == '__main__':

    offline = False     # 应该设置为线上环境，使用全量数据来计算i2i_sim、用户画像和文章画像

    # click_trn = get_all_click_sample(data_path, 1000)
    # click_hist_df, click_last_df = get_hist_and_last_click(click_trn)
    #
    #
    click_trn, click_val, val_ans, click_tst = get_trn_val_tst_data(data_path, offline=offline)

    # # 训练集用户点击历史和最后一次点击
    click_hist_df, click_last_df = get_hist_and_last_click(click_trn)
    all_data = click_hist_df.append(click_tst)
    if click_val is not None:
        all_data.append(click_val)

    # 文章信息
    articles_info = get_article_info_df(data_path)

    # 文章创建时间
    item_created_time_dict = dict(zip(articles_info['article_id'], articles_info['created_at_ts']))

    # 在计算i2i_sim矩阵的时候，不应该包含用户最后一次点击数据。否则会造成数据泄露。
    if os.path.exists('./result/i2i_sim.pkl'):
        i2i_sim = pickle.load(open('./result/i2i_sim.pkl', 'rb'))
    else:
        i2i_sim = itemcf_sim(all_data, item_created_time_dict)  # 使用历史点击数据来计算i2i_sim
        pickle.dump(i2i_sim, open('./result/i2i_sim.pkl', 'wb'))    # 保存i2i_sim

    # 热门点击
    item_topk_click = get_item_topk_click(all_data, k=50)

    # 用户点击历史字典{user: [(item1, time1), (item2, time2)...]...}
    user_item_time_dict = get_user_item_time(all_data)

    sim_item_topk = 100
    recall_item_num = 50

    user_recall_dict = {}
    for user in tqdm(all_data['user_id'].unique()):     # 根据用户点击历史，使用i2i_sim协同召回
        user_recall_dict[user] = item_based_recommend(user_id=user, user_item_time_dict=user_item_time_dict,
                                                      i2i_sim=i2i_sim, sim_item_topk=sim_item_topk,
                                                      recall_item_num=recall_item_num, item_topk_click=item_topk_click,
                                                      item_created_time_dict=item_created_time_dict)

    print('召回评估')
    metrics_recall(user_recall_dict, click_last_df, topk=recall_item_num)       # 只对训练集的用户做召回评估？？？

    # 保存召回结果
    pickle.dump(user_recall_dict, open('./result/user_recall_dict.pkl', 'wb'))










    # 用户兴趣挖掘
    # def user_topk_article_preferences(user_df, k=3):
    #     user_df = user_df.sort_values('click_timestamp')
    #     if len(user_df) < k:    # 将用户最后一次点击的主题返回
    #         categories_ = user_df['category_id'].values.tolist()
    #     else:
    #         last_click_time = user_df['click_timestamp'].tail(1).values[0]      # 最后一次点击时间
    #         user_df['diff_time'] = 1/abs(user_df['click_timestamp'] - last_click_time + 1e-7)
    #         categories_ = user_df.groupby('category_id').agg({'diff_time': np.sum}).reset_index().sort_values('diff_time')['category_id'].values[-k:].tolist()
    #     categories = [categories_[-1]] * (k - len(categories_))
    #     categories.extend(categories_)
    #     return categories
    #
    #
    # df = click_hist_df.sort_values('click_timestamp')
    # users_preference = df.groupby('user_id')[['click_timestamp', 'category_id']].apply(lambda x: user_topk_article_preferences(x))
    # print(users_preference)




