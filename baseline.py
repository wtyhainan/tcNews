import time, math, os
from tqdm import tqdm
import gc
import pickle
import random
from datetime import datetime
from operator import itemgetter
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
import collections
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

from utils import get_all_click_sample, get_user_item_time, get_all_click_df, itemcf_sim, get_item_topk_click, reduce_mem
from utils import get_item_created_time_dict, get_hist_and_last_click
from recall import item_based_recommend


warnings.filterwarnings('ignore')


data_path = 'E:\\tcNews\\datas'
save_path = './result'


if __name__ == '__main__':

    import time
    start_time = time.time()
    df = get_all_click_df(data_path+'\\')
    # df = get_all_click_sample(data_path+'\\', sample_nums=80000)

    # 获取user历史点击和最后一次点击，用于召回评估
    user_hist_click, user_last_click = get_hist_and_last_click(df)

    # # 获取用户历史点击
    user_item_time_dict = get_user_item_time(user_hist_click)

    # 获取item创建时间
    item_created_time_dict = get_item_created_time_dict(data_path+'\\')

    # # 计算item相似性
    if os.path.exists(os.path.join(save_path, 'itemcf_i2i_sim.pkl')):
        i2i_sim = pickle.load(open(os.path.join(save_path, 'itemcf_i2i_sim.pkl'), 'rb'))
    else:
        i2i_sim = itemcf_sim(user_hist_click, item_created_time_dict)
        pickle.dump(i2i_sim, open(os.path.join(save_path, 'itemcf_i2i_sim.pkl'), 'wb'))


    # 获取点击次数最多的item
    item_topk_click = get_item_topk_click(user_hist_click, k=50)

    sim_item_topk = 100
    recall_item_num = 50


    from recall import multiproc_itemcf_recall
    multiproc_itemcf_recall(user_hist_click['user_id'].unique(),
                            user_item_time_dict,
                            i2i_sim,
                            sim_item_topk,
                            recall_item_num,
                            item_topk_click,
                            item_created_time_dict, workers=8)

    # 为每一个用户根据协同过滤推荐
    # user_recall_items_dict = collections.defaultdict(dict)
    # for user in tqdm(user_hist_click['user_id'].unique()):
    #     user_recall_items_dict[user] = item_based_recommend(user,
    #                                                         user_item_time_dict,
    #                                                         i2i_sim,
    #                                                         sim_item_topk=100,
    #                                                         recall_item_num=50,
    #                                                         item_topk_click=item_topk_click,
    #                                                         item_created_time_dict=item_created_time_dict)
    #
    # print(time.time() - start_time)

    # 召回评估
    # metrics_recall(user_recall_items_dict, user_last_click, topk=50)
    #
    # pickle.dump(user_recall_items_dict, open(os.path.join(save_path, 'itemcf_recall_item.pkl'), 'wb'))





