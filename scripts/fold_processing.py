import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, CSVLogger
from sklearn.model_selection import StratifiedKFold
from transformers import *
import tokenizers
from tensorflow.keras.layers import *
from tensorflow.keras import models
import random
import os
from shutil import copyfile
import gc
import time
import sys


PAD_ID = 1

def fold_processing(params):
    
    ################################### INPUT DATA ###################################

    def read_train():
        train=pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
        train['text']=train['text'].astype(str)
        train['selected_text']=train['selected_text'].astype(str)
        return train

    def read_test():
        test=pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
        test['text']=test['text'].astype(str)
        return test

    def read_submission():
        test=pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')
        return test

    train_df = read_train()
    test_df = read_test()
    submission_df = read_submission()

    MAX_LEN = 96
    PATH = '../input/tf-roberta/'
    tokenizer = tokenizers.ByteLevelBPETokenizer(
        vocab_file=PATH+'vocab-roberta-base.json', 
        merges_file=PATH+'merges-roberta-base.txt', 
        lowercase=True,
        add_prefix_space=True
    )
    sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}

    ################################### TRAIN DATA ###################################

    N_TRAIN = train_df.shape[0]
    # args from https://huggingface.co/transformers/model_doc/roberta.html?highlight=tfrobertamodel#tfrobertamodel
    input_ids = np.ones((N_TRAIN, MAX_LEN), dtype='int32')       # token ids (pre-trained vocabulary & tokenizer)
    attention_mask = np.zeros((N_TRAIN, MAX_LEN), dtype='int32') # 0 in padding
    token_type_ids = np.zeros((N_TRAIN, MAX_LEN), dtype='int32') # 0 for A sentence, 1 - for B (there is only A sentence)

    start_tokens = np.zeros((N_TRAIN, MAX_LEN), dtype='int32')
    end_tokens = np.zeros((N_TRAIN, MAX_LEN), dtype='int32')

    for k in range(N_TRAIN):

        # FIND OVERLAP (mask with 1 including first whitespace)
        text1 = " " + " ".join(train_df.loc[k, 'text'].split())
        text2 = " ".join(train_df.loc[k, 'selected_text'].split())
        idx = text1.find(text2)
        chars = np.zeros((len(text1)))
        chars[idx:idx+len(text2)]=1
        if text1[idx - 1] == ' ': 
            chars[idx - 1] = 1

        # ID_OFFSETS (offsets = [(start1, finish1), .., (startN, finishN)])
        enc = tokenizer.encode(text1) 
        offsets = []
        idx = 0
        for t in enc.ids:
            w = tokenizer.decode([t])
            offsets.append((idx, idx + len(w)))
            idx += len(w)

        # START END TOKENS (toks - list of tokens from selected text)
        toks = []
        for i, (a,b) in enumerate(offsets):
            if np.sum(chars[a:b]) > 0:
                toks.append(i) 

        s_tok = sentiment_id[train_df.loc[k,'sentiment']]
    #         input_ids[k,:len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
    #         attention_mask[k,:len(enc.ids) + 5] = 1
        if params["label_consider_type"] == "right":
            LEFT_PAD_LEN = 1
            input_ids[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
            attention_mask[k,:len(enc.ids)+5] = 1
        elif params["label_consider_type"] == "left":
            LEFT_PAD_LEN = 2
            input_ids[k,:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]
            attention_mask[k,:len(enc.ids)+3] = 1
        else:
            assert False, "unknown label_consider_type param"

        if len(toks) > 0:
            start_tokens[k, toks[0] + LEFT_PAD_LEN] = 1
            end_tokens[k, toks[-1] + LEFT_PAD_LEN] = 1

    ################################### TEST DATA ###################################

    N_TEST = test_df.shape[0]
    test_word_ids = np.ones((N_TEST, MAX_LEN),dtype='int32')
    test_mask = np.zeros((N_TEST, MAX_LEN),dtype='int32')
    test_segm_ids = np.zeros((N_TEST, MAX_LEN),dtype='int32')

    for k in range(N_TEST):

        # INPUT_IDS
        text1 = " "+" ".join(test_df.loc[k,'text'].split())
        enc = tokenizer.encode(text1)                
        s_tok = sentiment_id[test_df.loc[k,'sentiment']]
        test_word_ids[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
        test_mask[k,:len(enc.ids)+5] = 1
        if params["label_consider_type"] == "right":
            LEFT_PAD_LEN = 1
            test_word_ids[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
            test_mask[k,:len(enc.ids)+5] = 1
        elif params["label_consider_type"] == "left":
            LEFT_PAD_LEN = 2
            test_word_ids[k,:len(enc.ids)+3] = [0, s_tok] + enc.ids + [2]
            test_mask[k,:len(enc.ids)+3] = 1
        else:
            assert False, "unknown label_consider_type param"


    ################################### POSTPROCESS FUNCTIONS ###################################

    def jaccard(str1, str2): 
        a = set(str(str1).lower().split()) 
        b = set(str(str2).lower().split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))

    def get_pred(start_proba, end_proba, df, tokenizer, out_prefix):
        pred = []
        n_samples = len(start_proba)
        for i in range(n_samples):
            text = df['text'][df.index[i]]
            a, b = np.argmax(start_proba[i,]), np.argmax(end_proba[i,])
            if a > b: 
                pred_ = text # IMPROVE CV/LB with better choice here
            else:
                cleaned_text = " " + " ".join(text.split())
                encoded_text = tokenizer.encode(cleaned_text)
                pred_ids = encoded_text.ids[a - LEFT_PAD_LEN: b - LEFT_PAD_LEN + 1]
                pred_ = tokenizer.decode(pred_ids)
            pred += [pred_]

        if out_prefix:
            DataFrame(start_proba).to_csv(f'{out_prefix}_start_prediction.csv')
            DataFrame(end_proba  ).to_csv(f'{out_prefix}_end_prediction.csv'  )
            df_pred = df.copy()
            df_pred["pred_selected_text"] = pred
            df_pred.to_csv(f'{out_prefix}_prediction.csv')

        return pred


    def get_metric(trues, preds):

        return np.mean([
            jaccard(pred, true)
            for true, pred in zip(trues, preds)
        ])

    def get_pred_and_score(start_proba, end_proba, df, tokenizer, out_prefix=None):
        pred = get_pred(start_proba, end_proba, df, tokenizer, out_prefix)
        metric = None
        if 'selected_text' in df:
            true = df['selected_text']
            metric = get_metric(true, pred)
        return pred, metric

    ################################### LOSSES ###################################

    def jaccard_expectation_loss(y_true, y_pred):
        batch_size, max_len = K.shape(y_pred)[0], K.shape(y_pred)[1] // 2
        start_true, end_true = y_true[:, :max_len], y_true[:, max_len:]
        start_pred, end_pred = y_pred[:, :max_len], y_pred[:, max_len:]

        # for true labels we can use argmax() function, cause labels don't involve in SGD
        x_start = K.cast(K.argmax(start_true, axis=1), dtype=tf.float32)
        x_end   = K.cast(K.argmax(end_true  , axis=1), dtype=tf.float32)
        l = x_end - x_start + 1

        # some magic for getting indices matrix like this: [[0, 1, 2, 3], [0, 1, 2, 3]] 
        ind_row = tf.range(0, max_len, dtype=tf.float32)
        ones_matrix = tf.ones([batch_size, max_len], dtype=tf.float32)
        ind_matrix = ind_row * ones_matrix

        # expectations for x_start^* (x_start_pred) and x_end^* (x_end_pred)
        x_start_pred = K.sum(start_pred * ind_matrix, axis=1)
        x_end_pred   = K.sum(end_pred   * ind_matrix, axis=1)

        relu11 = K.relu(x_start_pred - x_start)
        relu12 = K.relu(x_end   - x_end_pred  )
        relu21 = K.relu(x_start - x_start_pred)
        relu22 = K.relu(x_end_pred   - x_end  )

        intersection = l - relu11 - relu12
        union = l + relu21 + relu22
        jel = 1 - (intersection / union)

        return K.mean(jel)

    def smoothed_categorical_crossentropy(y_true, y_pred):
        # adjust the targets for sequence bucketing
        ll = tf.shape(y_pred)[1]
        y_true = y_true[:, :ll]
        if not params["label_smoothing"]:
            params["label_smoothing"] = 0.
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=params["label_smoothing"])

        return tf.reduce_mean(loss)

    ################################### MODEL ###################################

    def get_out_and_loss_fn(x1, x2):
        if params["loss"] == "JEL":
            out = Concatenate(axis=1)([x1, x2])
            loss_fn = jaccard_expectation_loss
        elif params["loss"] == "CCE":
            out = [x1, x2]
            loss_fn = smoothed_categorical_crossentropy
    #     elif params["loss"] == "BCE":
    #         out = [x1, x2]
    #         loss_fn = K.binary_crossentropy
        else:
            assert False, "unknown loss param"

        return out, loss_fn

    def build_model(opt):
        ids = Input((MAX_LEN,), dtype=tf.int32)
        att = Input((MAX_LEN,), dtype=tf.int32)
        tok = Input((MAX_LEN,), dtype=tf.int32)
        padding = tf.cast(tf.equal(ids, PAD_ID), tf.int32)

        lens = MAX_LEN - tf.reduce_sum(padding, -1)
        max_len = tf.reduce_max(lens)
        ids_ = ids[:, :max_len]
        att_ = att[:, :max_len]
        tok_ = tok[:, :max_len]

        config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')
        bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)
        # https://huggingface.co/transformers/model_doc/roberta.html?highlight=tfrobertamodel#tfrobertamodel
        x = bert_model(ids_, attention_mask=att_, token_type_ids=tok_)

        x1 = tf.keras.layers.Dropout(0.1)(x[0]) 
        x1 = tf.keras.layers.Conv1D(128, 2, padding='same')(x1)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.Conv1D(64, 2, padding='same')(x1)
        x1 = tf.keras.layers.Dense(1)(x1)
        x1 = tf.keras.layers.Flatten()(x1)
        x1 = tf.keras.layers.Activation('softmax')(x1)

        x2 = tf.keras.layers.Dropout(0.1)(x[0]) 
        x2 = tf.keras.layers.Conv1D(128, 2, padding='same')(x2)
        x2 = tf.keras.layers.LeakyReLU()(x2)
        x2 = tf.keras.layers.Conv1D(64, 2, padding='same')(x2)
        x2 = tf.keras.layers.Dense(1)(x2)
        x2 = tf.keras.layers.Flatten()(x2)
        x2 = tf.keras.layers.Activation('softmax')(x2)

        out, loss_fn = get_out_and_loss_fn(x1, x2)

        model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=out)
        model.compile(loss=loss_fn, optimizer=opt)

        return model

    def build_model2(opt):
        ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
        att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
        tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
        padding = tf.cast(tf.equal(ids, PAD_ID), tf.int32)

        lens = MAX_LEN - tf.reduce_sum(padding, -1)
        max_len = tf.reduce_max(lens)
        ids_ = ids[:, :max_len]
        att_ = att[:, :max_len]
        tok_ = tok[:, :max_len]

        config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')
        bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)
        x = bert_model(ids_,attention_mask=att_,token_type_ids=tok_)

        x1 = tf.keras.layers.Dropout(0.1)(x[0])
        x1 = tf.keras.layers.Conv1D(768, 2,padding='same')(x1)
        x1 = tf.keras.layers.LeakyReLU()(x1)
        x1 = tf.keras.layers.Dense(1)(x1)
        x1 = tf.keras.layers.Flatten()(x1)
        x1 = tf.keras.layers.Activation('softmax')(x1)

        x2 = tf.keras.layers.Dropout(0.1)(x[0]) 
        x2 = tf.keras.layers.Conv1D(768, 2, padding='same')(x2)
        x2 = tf.keras.layers.LeakyReLU()(x2)
        x2 = tf.keras.layers.Dense(1)(x2)
        x2 = tf.keras.layers.Flatten()(x2)
        x2 = tf.keras.layers.Activation('softmax')(x2)

        out, loss_fn = get_out_and_loss_fn(x1, x2)

        model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=out)   
        model.compile(loss=loss_fn, optimizer=opt)

        return model

    def pred_wrapper(model):
        ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
        att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
        tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

        padding = tf.cast(tf.equal(ids, PAD_ID), tf.int32)
        lens = MAX_LEN - tf.reduce_sum(padding, -1)
        max_len = tf.reduce_max(lens)

        x = model([ids, att, tok])
        if params["loss"] == "JEL":
            x1, x2 = x[:, :max_len], x[:, max_len:]
        elif params["loss"] == "CCE":
            x1, x2 = tuple(x)
        else:
            assert False, "unknown loss param"

        x1_padded = tf.pad(x1, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)
        x2_padded = tf.pad(x2, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)

        out, _ = get_out_and_loss_fn(x1_padded, x2_padded)

        padded_model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=out)
        return padded_model


    ################################### CALLBACKS ###################################

    def get_prediction(model, word_ids, mask, segm_ids, verbose=1):
        pred_model = pred_wrapper(model)
        if params["loss"] == "JEL":
            y_pred = pred_model.predict([word_ids, mask, segm_ids], verbose)
            start_proba, end_proba = y_pred[:, :MAX_LEN], y_pred[:, MAX_LEN:]

        elif params["loss"] == "CCE":
            start_proba, end_proba = tuple(pred_model.predict([word_ids, mask, segm_ids], verbose))
        else:
            assert False, "unknown loss param"
        return start_proba, end_proba


    class CustomCallback(Callback):
        def __init__(self, model, word_ids, mask, segm_ids, start, end, df, tokenizer, n_fold, start_epoch, log_path, start_score, best_weights_path):
            self.model = model

            self.word_ids = word_ids
            self.mask = mask
            self.segm_ids = segm_ids
            self.start = start
            self.end = end

            self.df = df
            self.tokenizer = tokenizer

            self.start_epoch = start_epoch
            self.n_fold = n_fold
            self.log_path = log_path
            self.best_weights_path = best_weights_path

            self.best = start_score
            self.checkpoint = time.time()

        def on_epoch_end(self, epoch, logs):
            # Validation

    #         start_proba, end_proba = tuple(pred_model.predict([self.word_ids, self.mask, self.segm_ids], verbose=1))
            start_proba, end_proba = get_prediction(model, self.word_ids, self.mask, self.segm_ids)
            _, current = get_pred_and_score(start_proba, end_proba, self.df, self.tokenizer)

            # Save best model
            if current > self.best:
                self.best = current
                self.model.save_weights(self.best_weights_path, overwrite=True)

            # Log score info
            abs_epoch = self.start_epoch + epoch
            with open(log_path, 'a') as f:
                f.write(f'\n[fold: {self.n_fold}, epoch: {abs_epoch}] Val Score : {current:.5f} (time: {(time.time() - self.checkpoint) // 60} min.)')
            self.checkpoint = time.time()


    def scheduler(epoch):

        return 3e-5 * 0.2**epoch

    lr_scheduler = LearningRateScheduler(scheduler)

    ################################### SEED & SESSION ###################################

    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        tf.random.set_seed(seed)

    seed_everything(params["seed"])
    #     K.clear_session()
    #     config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=4)
    #     sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
    #     tf.compat.v1.keras.backend.set_session(sess)

    ################################### ARGS ###################################

    if params["opt_name"] == "Adam":
        opt = tf.keras.optimizers.Adam(learning_rate=params["lr"])
    else:
        assert False, "unknown opt_name"

    if params["lr_schedule_name"] == "default":
        lr_scheduler = lr_scheduler
    else:
        assert False, "unknown lr_schedule_name"

    ################################### LOGGING ###################################

    # define log-folder
    log_dir_path = f'../attempt_logs/{params["att_num"]}'
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


    ################################### OPTIMIZATION PROCESSING ###################################

    # Fold Splitter
    skf = StratifiedKFold(n_splits=params["n_split"], shuffle=True, random_state=777)
    splits = list(skf.split(input_ids, train_df.sentiment.values))

    # Splitting
    tr_idx, val_idx = splits[params["n_fold"] - 1]

    train_df_ = train_df.loc[tr_idx].reset_index(drop=True)
    train_df_.index = tr_idx

    val_df = train_df.loc[val_idx].reset_index(drop=True)
    val_df.index = val_idx

    tr_word_ids, tr_mask, tr_segm_ids, tr_starts, tr_ends = input_ids[tr_idx,], attention_mask[tr_idx,], token_type_ids[tr_idx,], start_tokens[tr_idx,], end_tokens[tr_idx,]
    val_word_ids, val_mask, val_segm_ids, val_starts, val_ends = input_ids[val_idx,], attention_mask[val_idx,], token_type_ids[val_idx,], start_tokens[val_idx,], end_tokens[val_idx,]


#     # Test sample for debug
#     N_TEST = 16 * 10
#     tr_word_ids, tr_mask, tr_segm_ids, tr_starts, tr_ends = tr_word_ids[:N_TEST], tr_mask[:N_TEST], tr_segm_ids[:N_TEST], tr_starts[:N_TEST], tr_ends[:N_TEST]
#     tr_idx = tr_idx[:N_TEST]
#     train_df_ = train_df_[:N_TEST]

#     val_word_ids, val_mask, val_segm_ids, val_starts, val_ends = tr_word_ids, tr_mask, tr_segm_ids, tr_starts, tr_ends
#     val_idx = tr_idx
#     val_df = train_df_

#     test_word_ids, test_mask, test_segm_ids = test_word_ids[16:32], test_mask[16:32], test_segm_ids[16:32]
#     test_df = test_df[16:32]


    print(f'##### FOLD {params["n_fold"]} #####')
    gc.collect()

    # Model Defining
    best_weights_path = f'{log_dir_path}/{params["n_fold"]}/best_model.h5'
    if params["model_name"] == "default":
        model = build_model(opt)
    elif params["model_name"] == "v2.0":
        model = build_model2(opt)
    else:
        assert False, "unknown model_name param"


    # Model Pretraining [optional]
    pre_trained_weights_path = f'../attempt_logs/{params["weights_att_num"] or params["att_num"]}/{params["n_fold"]}/best_model.h5'
    pretrained_score = 0
    if os.path.exists(pre_trained_weights_path):
        model.load_weights(pre_trained_weights_path)
        start_proba, end_proba = get_prediction(model, val_word_ids, val_mask, val_segm_ids)
        _, pretrained_score = get_pred_and_score(start_proba, end_proba, val_df, tokenizer)
        with open(log_path, 'a') as f:
            f.write(f'\nWeights PreTrained from {pre_trained_weights_path}, pretrained_score: {pretrained_score:.5f}')

    # Model Pretraining [optional, when we need only ensure total scores]
    if not params["wo_fitting"]:
        custom_callback = CustomCallback(
            model,
            val_word_ids, val_mask, val_segm_ids, val_starts, val_ends, val_df,
            tokenizer, 
            params["n_fold"],
            params["start_epoch"],
            log_path,
            pretrained_score,
            best_weights_path
        )


        n_remain_epoch = params["n_epoch"] - params["start_epoch"] + 1

        tr_out, _ = get_out_and_loss_fn(tr_starts, tr_ends)

        model.fit(
            [tr_word_ids, tr_mask, tr_segm_ids], tr_out,
            batch_size=params["batch_size"],
            epochs=n_remain_epoch,
            callbacks=[
                custom_callback,
                lr_scheduler
            ],
            verbose=1,
        )

    model.load_weights(best_weights_path)

    scores = {}
    for name, word_ids, mask, segm_ids, df in [
        ("train", tr_word_ids, tr_mask, tr_segm_ids, train_df_),
        ("validation", val_word_ids, val_mask, val_segm_ids, val_df),
        ("test", test_word_ids, test_mask, test_segm_ids, test_df)
    ]:
        print(f'{name} prediction ...')
        start_proba, end_proba = get_prediction(model, word_ids, mask, segm_ids)
        _, scores[name] = get_pred_and_score(start_proba, end_proba, df, tokenizer, out_prefix=f'{log_dir_path}/{params["n_fold"]}/{name}')

    with open(log_path, 'a') as f:
        f.write(f'\n[fold: {params["n_fold"]}] Ensure Scores : train score: {scores["train"]:.5f}, validation score: {scores["validation"]:.5f}]')

params = {
    "seed": int(sys.argv[1]),
    "n_split": int(sys.argv[2]),
    "n_epoch": int(sys.argv[3]),
    "batch_size": int(sys.argv[4]),
    "att_num": int(sys.argv[5]),                                               # Number of current attempt
    "weights_att_num": int(sys.argv[6]) if not sys.argv[6] == "None" else None,  # Number of attempt for pre-define model
    "model_name": sys.argv[7],                                                 # ML model name
    "opt_name": sys.argv[8],                                                   # Optimizer (custom filling opt_name)
    "lr": float(sys.argv[9]),                                                  # Initial LR
    "lr_schedule_name": sys.argv[10],                                          # LR checduling (custom filling opt_name)
    "n_fold": int(sys.argv[11]),                                               # fold for training
    "start_epoch": int(sys.argv[12]),                                          # start epoch for training
    "wo_fitting": (sys.argv[13] == "True"),
    "label_consider_type": sys.argv[14],                                       # way to using positive/nutral/negative label
    "loss": sys.argv[15],
    "label_smoothing": float(sys.argv[16]) if not sys.argv[16] == "None" else None
}


fold_processing(params)