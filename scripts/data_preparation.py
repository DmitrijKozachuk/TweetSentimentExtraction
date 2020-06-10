import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import tqdm
from tqdm import tqdm_notebook

from constants import N_TRAIN, N_TEST, SENTIMENT_ID, MAX_LEN, LEFT_PAD_LEN

def get_train_data(train_df, tokenizer, idx=None):
    if idx is None:
        idx = range(len(train_df))

    input_ids = np.ones((N_TRAIN, MAX_LEN), dtype='int32')       # token ids (pre-trained vocabulary & tokenizer)
    attention_mask = np.zeros((N_TRAIN, MAX_LEN), dtype='int32') # 0 in padding
    token_type_ids = np.zeros((N_TRAIN, MAX_LEN), dtype='int32') # 0 for A sentence, 1 - for B (there is only A sentence)

    start_tokens = np.zeros((N_TRAIN, MAX_LEN), dtype='int32')
    end_tokens = np.zeros((N_TRAIN, MAX_LEN), dtype='int32')

    train_sample_ind2new_ind2old_ind = {}

    for k in tqdm_notebook(idx):
        text = " " + " ".join(train_df.loc[k, 'text'].split())
        selected_text = " ".join(train_df.loc[k, 'selected_text'].split())

        l_ind = (" " + text + " ").find(" " + selected_text + " ")
        if l_ind < 0:
            l_ind = (text).find(selected_text)
        else:
            l_ind += 1

        old_start_ind = len(text[:l_ind + 1].split()) - 1
        old_end_ind = old_start_ind + (len(selected_text.split()) - 1)

        tokens = []
        new_ind, new_start_ind, new_end_ind = 0, None, None
        new_ind2old_ind = {}
        for old_ind, word in enumerate(text.split()):
            word_tokens = tokenizer.encode(word)
            
#             if k == 6:
#                 print("old_ind", old_ind)
#                 print("word_tokens", word_tokens)

            if old_ind == old_start_ind:
                new_start_ind = new_ind

            for word_token in word_tokens:
                tokens += [word_token]
                new_ind2old_ind[new_ind] = old_ind
                new_ind += 1

            if old_ind == old_end_ind:
                new_end_ind = new_ind - 1

        train_sample_ind2new_ind2old_ind[k] = new_ind2old_ind.copy()
        
#         if k == 6:
#             print("text", text)
#             print("selected_text", selected_text)
#             print("old_start_ind", old_start_ind)
#             print("old_end_ind", old_end_ind)
#             print("new_ind2old_ind", new_ind2old_ind)

        s_tok = SENTIMENT_ID[train_df.loc[k,'sentiment']]
        input_ids[k,:len(tokens) + 5] = [0] + tokens + [2,2] + [s_tok] + [2]
        attention_mask[k,:len(tokens) + 5] = 1

        start_tokens[k, new_start_ind + LEFT_PAD_LEN] = 1
        end_tokens  [k, new_end_ind   + LEFT_PAD_LEN] = 1
        
    return input_ids, attention_mask, token_type_ids, start_tokens, end_tokens, train_sample_ind2new_ind2old_ind


def get_test_data(test_df, tokenizer, idx=None):
    if idx is None:
        idx = range(len(train_df))

    test_word_ids = np.ones((N_TEST, MAX_LEN),dtype='int32')
    test_mask = np.zeros((N_TEST, MAX_LEN),dtype='int32')
    test_segm_ids = np.zeros((N_TEST, MAX_LEN),dtype='int32')

    test_sample_ind2new_ind2old_ind = {}

    for k in tqdm_notebook(idx):
        text = " " + " ".join(test_df.loc[k, 'text'].split()) + " "

        tokens = []
        new_ind = 0
        new_ind2old_ind = {}
        for old_ind, word in enumerate(text.split()):
            word_tokens = tokenizer.encode(word)

            for word_token in word_tokens:
                tokens += [word_token]
                new_ind2old_ind[new_ind] = old_ind
                new_ind += 1

        test_sample_ind2new_ind2old_ind[k] = new_ind2old_ind.copy()

        s_tok = SENTIMENT_ID[test_df.loc[k,'sentiment']]
        test_word_ids[k,:len(tokens) + 3] = [0, s_tok] + tokens + [2]
        test_mask    [k,:len(tokens) + 3] = 1
        
    return test_word_ids, test_mask, test_segm_ids, test_sample_ind2new_ind2old_ind

# if params["label_consider_type"] == "right":
#     LEFT_PAD_LEN = 1
#     input_ids[k,:len(tokens) + 5] = [0] + tokens + [2,2] + [s_tok] + [2]
#     attention_mask[k,:len(tokens) + 5] = 1
# elif params["label_consider_type"] == "left":
#     LEFT_PAD_LEN = 2
#     input_ids[k,:len(tokens) + 3] = [0, s_tok] + tokens + [2]
#     attention_mask[k,:len(tokens) + 3] = 1
# else:
#     assert False, "unknown label_consider_type param"