import random
import os
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import tensorflow as tf

from constants import LEFT_PAD_LEN, MAX_LEN, PAD_ID

def jaccard(str1, str2): 
    a = set(str(str1).lower().split()) 
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def get_score_(st_true_arr, st_pred_arr):
    return np.mean([jaccard(st_true, st_pred) for st_true, st_pred in zip(st_true_arr, st_pred_arr)])

def get_score(start_proba, end_proba, df, sample_ind2new_ind2old_ind):
    st_true_arr = df['selected_text']
    st_pred_arr = get_st_prediction(start_proba, end_proba, df, sample_ind2new_ind2old_ind)
    return get_score_(st_true_arr, st_pred_arr)

def get_proba_prediction(model, word_ids, mask, segm_ids):
    model = pred_wrapper(model)
    y_pred = model.predict([word_ids, mask, segm_ids], verbose=1)
    start_proba, end_proba = tuple(y_pred) # y_pred[:, :MAX_LEN], y_pred[:, MAX_LEN:]
    return start_proba, end_proba

def get_st_prediction(start_proba, end_proba, df, sample_ind2new_ind2old_ind, out_prefix=None):
    """ 
    Итерируемся по строкам df, независимо от индексов 
    (start_proba, end_proba есть выход модели и это проблематично и не нужно добавлять в них исходные индексы - 
    проверка корректности только по финальному скору, ведь если проблема с индексами, то это сразу будет заметно)
    """
    preds = {}
    n_samples = len(df)
    for ind, sample_ind in enumerate(df.index):
        text = df['text'][sample_ind]
        a, b = np.argmax(start_proba[ind,]), np.argmax(end_proba[ind,])
        if a > b: 
            pred = text
        else:
            new_ind2old_ind = sample_ind2new_ind2old_ind[sample_ind]

            new_inds_len = len(new_ind2old_ind.keys())
            min_board, max_board = LEFT_PAD_LEN, LEFT_PAD_LEN + new_inds_len - 1
            a, b = min(max(a, min_board), max_board), min(max(b, min_board), max_board)

            start_new_ind, end_new_ind = a - LEFT_PAD_LEN, b - LEFT_PAD_LEN
            start_old_ind, end_old_ind = new_ind2old_ind[start_new_ind], new_ind2old_ind[end_new_ind]

            words = np.array(text.split())
            pred = " " + " ".join(words[start_old_ind:end_old_ind + 1])

#             if sample_ind == 6:
#                 print("start_new_ind", start_new_ind)
#                 print("end_new_ind", end_new_ind)
#                 print("start_old_ind", start_old_ind)
#                 print("end_old_ind", end_old_ind)

        preds[sample_ind] = pred

    if out_prefix:
        DataFrame(start_proba).to_csv(f'{out_prefix}_start_prediction.csv')
        DataFrame(end_proba  ).to_csv(f'{out_prefix}_end_prediction.csv'  )
        df_pred = df.copy()
        df_pred["pred_selected_text"] = preds
        df_pred.to_csv(f'{out_prefix}_prediction.csv')

    return Series(preds)

def pred_wrapper(model):
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    padding = tf.cast(tf.equal(ids, PAD_ID), tf.int32)
    lens = MAX_LEN - tf.reduce_sum(padding, -1)
    max_len = tf.reduce_max(lens)

    x = model([ids, att, tok])

#     x1, x2 = x[:, :max_len], x[:, max_len:]
    x1, x2 = x[0], x[1]
    x1_padded = tf.pad(x1, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)
    x2_padded = tf.pad(x2, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)
#     out = tf.keras.layers.Concatenate(axis=1)([x1_padded, x2_padded])
    out = [x1_padded, x2_padded]

    padded_model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=out)
    return padded_model

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

def init_logging(params):
    # define log-folder
    log_dir_path = f'{params["base_path"]}/attempt_logs/{params["att_num"]}'
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)

    # define fold-folders log_path
    folder_path = f'{log_dir_path}/{params["n_fold"]}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # define & prefill log-file
    log_path = f'{log_dir_path}/log.txt'
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            for name, val in params.items():
                f.write(f'{name}: {val} \n')
            f.write("\n")
            
    return log_dir_path, log_path