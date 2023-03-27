import warnings
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb

warnings.filterwarnings('ignore')

save_path = './result'


def submit(recall_df, topk=5, model_name=None):
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    # 判断是不是每个用户都有5篇文章及以上
    tmp = recall_df.groupby('user_id').apply(lambda x: x['rank'].max())
    assert tmp.min() >= topk

    del recall_df['pred_score']
    submit = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()

    submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]
    # 按照提交格式定义列名
    submit = submit.rename(columns={'': 'user_id', 1: 'article_1', 2: 'article_2',
                                    3: 'article_3', 4: 'article_4', 5: 'article_5'})

    save_name = save_path + model_name + '_' + datetime.today().strftime('%m-%d') + '.csv'
    submit.to_csv(save_name, index=False, header=True)


def norm_sim(sim_df, weight=0.0):
    min_sim = sim_df.min()
    max_sim = sim_df.max()
    if max_sim == min_sim:
        sim_df = sim_df.apply(lambda sim: 1.0)
    else:
        sim_df = sim_df.apply(lambda sim: 1.0 * (sim - min_sim) / (max_sim - min_sim))
    sim_df = sim_df.apply(lambda sim: sim + weight)
    return sim_df


if __name__ == '__main__':

    lgb_cols = ['time_diff0', 'word_diff0', 'score', 'words_count_mean', 'time_diff_click_created_x',
                'time_diff', 'click_size', 'active_level', 'words_count', 'hot_level',
                'time_diff_click_created_y', 'is_category_hab']

    trn_user_item_feats_df = pd.read_csv('./result/train_data.csv')
    trn_user_item_feats_df['click_article_id'] = trn_user_item_feats_df['click_article_id'].astype(int)
    val_user_item_feats_df = pd.read_csv('./result/val_data.csv')
    val_user_item_feats_df['click_article_id'] = val_user_item_feats_df['click_article_id'].astype(int)

    trn_user_item_feats_df_rank_model = trn_user_item_feats_df.copy()
    val_user_item_feats_df_rank_model = val_user_item_feats_df.copy()
    trn_user_item_feats_df_rank_model.sort_values(by=['user_id'], inplace=True)

    lgb_ranker = lgb.LGBMRanker(boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1, max_depth=-1,
                                n_estimators=100, subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
                                learning_rate=0.01, min_child_weight=50, random_state=2018, n_jobs=6)

    g_train = trn_user_item_feats_df_rank_model.groupby(by=['user_id'], as_index=False).count()['label'].values
    g_val = val_user_item_feats_df_rank_model.groupby(by=['user_id'], as_index=False).count()['label'].values

    lgb_ranker.fit(trn_user_item_feats_df_rank_model[lgb_cols], trn_user_item_feats_df_rank_model['label'], group=g_train,
                   eval_set=[(val_user_item_feats_df_rank_model[lgb_cols], val_user_item_feats_df_rank_model['label'])],
                   eval_group=[g_val], eval_at=[1, 2, 3, 4, 5], eval_metric=['ndcg'], early_stopping_rounds=50,)


