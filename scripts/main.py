import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import StratifiedKFold
import random
import os
import sys
import gc
import time

from constants import *
from callback import CustomCallback
from base_model import get_base_model
from head_model import get_combined_model
from loss import get_loss
from data_reading import read_data
from data_preparation import get_train_data, get_test_data
from utils import *


def main(params):

    # Seed $ Logging
    seed_everything(SEED)
    log_dir_path, log_path = init_logging(params)

    with open(log_path, 'a') as f:
        f.write(f'\n[base_model] {params["base_model"]}')

    # Read  data
    print("Read data...")
    train_df, test_df, submission_df = read_data(params)

    # Splitter
    skf = StratifiedKFold(n_splits=N_SPLIT, shuffle=True, random_state=777)
    splits = list(skf.split(train_df.index.values, train_df.sentiment.values))
    tr_idx, val_idx = splits[params["n_fold"] - 1]
    test_idx = np.arange(N_TEST)

    if "debug" in params["mode"]:
        n_debug = int(params["mode"].split(":")[1])
        tr_idx, val_idx = tr_idx[:n_debug], val_idx[:n_debug]
        test_idx = test_idx[:n_debug]

    # Build & Compile model
    print("Build & Compile model...")
    tokenizer, base_model = get_base_model(params)
    combined_model = get_combined_model(base_model, params)

    opt = tf.keras.optimizers.Adam(learning_rate=params["lr"])
    loss = get_loss(params)
    combined_model.compile(loss=loss, optimizer=opt)

    # Prepare  data
    print("Prepare data...")
    known_idx = np.array(list(set(tr_idx) | set(val_idx)))
    input_ids, attention_mask, token_type_ids, start_tokens, end_tokens, train_sample_ind2new_ind2old_ind = get_train_data(train_df, tokenizer, idx=known_idx)
    test_word_ids, test_mask, test_segm_ids, test_sample_ind2new_ind2old_ind = get_test_data(test_df, tokenizer, idx=test_idx)

    # # Model hash
    # print(f'base_model hash: {np.array(base_model(test_word_ids[:16], test_mask[:16], test_segm_ids[:16])[0]).sum():.3}')
    # print(f'head_model hash: {combined_model.layers[-6].weights[0].numpy().sum():.3}')

    # Splitting data
    print("Splitting data...")
    tr_df = train_df.loc[tr_idx].reset_index(drop=True).set_index(tr_idx)
    val_df = train_df.loc[val_idx].reset_index(drop=True).set_index(val_idx)

    tr_word_ids, tr_mask, tr_segm_ids, tr_starts, tr_ends = input_ids[tr_idx,], attention_mask[tr_idx,], token_type_ids[tr_idx,], start_tokens[tr_idx,], end_tokens[tr_idx,]
    tr_targets = np.concatenate([tr_starts, tr_ends], axis=1)
    val_word_ids, val_mask, val_segm_ids, val_starts, val_ends = input_ids[val_idx,], attention_mask[val_idx,], token_type_ids[val_idx,], start_tokens[val_idx,], end_tokens[val_idx,]

    # Check Correcness
    print("Check Correcness...")
    tr_df["is_correct"] = tr_df.apply(lambda row: (" " + row.text + " ").find(" " + row.selected_text + " ") >= 0, axis=1)
    print(f'correct samples: {tr_df["is_correct"].mean():3f}')

    tr_df["recover_selected_text"] = get_st_prediction(tr_starts, tr_ends, tr_df, train_sample_ind2new_ind2old_ind)
    tr_df["recover_jaccard"] = tr_df.apply(lambda row: jaccard(row["recover_selected_text"], row["selected_text"]), axis=1)
    assert np.all(tr_df[tr_df["is_correct"]]["recover_jaccard"] == 1)
    print(f'preprocessing OK!')


    print(f'##### FOLD {params["n_fold"]} #####')
    gc.collect()

    # Model Paths & Pretraining (optional)
    best_weights_path = f'{log_dir_path}/{params["n_fold"]}/best_model.h5'
    pre_trained_weights_path = f'../attempt_logs/{params["weights_att_num"] or params["att_num"]}/{params["n_fold"]}/best_model.h5'

    pretrained_score = 0
    # if os.path.exists(pre_trained_weights_path):
    #     combined_model.load_weights(pre_trained_weights_path)
    #     start_proba, end_proba = get_proba_prediction(combined_model, val_word_ids, val_mask, val_segm_ids)
    #     pretrained_score = get_score(start_proba, end_proba, val_df, train_sample_ind2new_ind2old_ind)
    #     with open(log_path, 'a') as f:
    #         f.write(f'\nWeights PreTrained from {pre_trained_weights_path}, pretrained_score: {pretrained_score:.5f}')

    # Training (optional)
    if not params["wo_fitting"]:
        lr_scheduler = LearningRateScheduler(lambda epoch: 3e-5 * 0.2**epoch)
        custom_callback = CustomCallback(
            combined_model,
            val_word_ids, val_mask, val_segm_ids, val_df, train_sample_ind2new_ind2old_ind,
            params["n_fold"],
            params["start_epoch"],
            log_path,
            pretrained_score,
            best_weights_path
        )

        n_epoch = N_EPOCH - params["start_epoch"] + 1
        combined_model.fit(
            [tr_word_ids, tr_mask, tr_segm_ids], [tr_starts, tr_ends], #tr_targets,
            batch_size=BATCH_SIZE,
            epochs=n_epoch,
            callbacks=[
                custom_callback,
                lr_scheduler
            ],
            verbose=1,
        )

    combined_model.load_weights(best_weights_path)

    # scores = {}
    # for name, word_ids, mask, segm_ids, df, sample_ind2new_ind2old_ind in [
    #     ("train"      ,  tr_word_ids,   tr_mask,   tr_segm_ids,   tr_df, train_sample_ind2new_ind2old_ind),
    #     ("validation" , val_word_ids,  val_mask,  val_segm_ids,  val_df, train_sample_ind2new_ind2old_ind),
    #     ("test"      , test_word_ids, test_mask, test_segm_ids, test_df,  test_sample_ind2new_ind2old_ind)
    # ]:
    #     print(f'{name} prediction ...')
    #     start_proba, end_proba = get_proba_prediction(combined_model, word_ids, mask, segm_ids)
    #     if name != "test":
    #         scores[name] = get_score(start_proba, end_proba, df, sample_ind2new_ind2old_ind)

    # with open(log_path, 'a') as f:
    #     f.write(f'\n[fold: {params["n_fold"]}] Ensure Scores : train score: {scores["train"]:.5f}, validation score: {scores["validation"]:.5f}]')

params = {
    "base_path": sys.argv[1],
    "mode": sys.argv[2],
    "att_num": int(sys.argv[3]),
    "weights_att_num": sys.argv[4],
    "n_fold": int(sys.argv[5]),
    "start_epoch": int(sys.argv[6]),
    "wo_fitting": (sys.argv[7] == "True"),
    
    "base_model": sys.argv[8],
    "head_model": sys.argv[9],
    "lr": float(sys.argv[10]), # "base_lr": 0.00002, "max_lr": 0.00004,
    "loss": sys.argv[11],
    "label_smoothing": float(sys.argv[12])
}

main(params)
