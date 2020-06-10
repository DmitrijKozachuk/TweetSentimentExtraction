import ast
import tensorflow as tf
import tensorflow.keras.backend as K

from constants import MAX_LEN

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

# def smoothed_cce_loss(y_true, y_pred, params):
# #     print("y_true", y_true)
# #     print("y_pred", y_pred)
#     ls = params["label_smoothing"] or 0

#     max_len = K.shape(y_pred)[1] // 2
#     start_pred, end_pred = y_pred[:, :max_len], y_pred[:, max_len:]
#     start_true, end_true = y_true[:, :max_len], y_true[:, MAX_LEN:MAX_LEN + max_len]
# #     start_true, end_true = y_pred[:, :max_len], y_pred[:, max_len:]
# #     print("start_pred", start_pred)
# #     print("end_pred", end_pred)
# #     print("start_true", start_true)
# #     print("end_true", end_true)
    

#     start_loss = tf.keras.losses.categorical_crossentropy(start_true, start_pred, label_smoothing=ls)
#     end_loss =   tf.keras.losses.categorical_crossentropy(  end_true,   end_pred, label_smoothing=ls)
#     loss = tf.reduce_mean(start_loss + end_loss)

#     return loss

def smoothed_cce_loss(y_true, y_pred, params):
    ls = params["label_smoothing"] or 0
    
    max_len = K.shape(y_pred)[1]
    y_true = y_true[:, :max_len]

    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=ls)

    return tf.reduce_mean(loss)

# def smoothed_cce_loss(y_true, y_pred, params):
#     # adjust the targets for sequence bucketing
#     LABEL_SMOOTHING = 0.1
#     ll = tf.shape(y_pred)[1]
#     y_true = y_true[:, :ll]
#     loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred,
#         from_logits=False, label_smoothing=LABEL_SMOOTHING)
#     loss = tf.reduce_mean(loss)
#     return loss

def get_custom_dist(start_proba, end_proba, max_len):
    weights = tf.range(0, max_len, dtype=tf.float32)
    start_pos = K.sum(start_proba * weights, axis=1)
    end_pos = K.sum(end_proba * weights, axis=1)
    diff = end_pos - start_pos

    return diff

def public_loss(y_true, y_pred, scale=1):
    max_len = K.shape(y_pred)[1] // 2
    start_true, end_true = y_true[:, :max_len], y_true[:, max_len:]
    start_pred, end_pred = y_pred[:, :max_len], y_pred[:, max_len:]
    
    dist_pred = get_custom_dist(start_true, start_pred, max_len)
    dist_true = get_custom_dist(end_true, end_pred, max_len)
    diff = (dist_pred - dist_true)

    loss = K.sqrt(diff * diff)

    return scale * loss

def get_loss(params):
    losses = {
        "CCE": lambda y_true, y_pred: smoothed_cce_loss(y_true, y_pred, params),
        "JEL": jaccard_expectation_loss
    }
    if params["loss"] in losses.keys():
        return losses[params["loss"]]
    
    loss_dict = ast.literal_eval(params["loss"])
    return lambda y_true, y_pred: tf.add_n([
        coeff * losses[loss_name](y_true, y_pred)
        for loss_name, coeff in loss_dict.items()
    ])