import ast
import tensorflow as tf

from constants import MAX_LEN, PAD_ID


def build_one_head_baseline_model(x):
    x1 = tf.keras.layers.Dropout(0.1)(x)
    x1 = tf.keras.layers.Conv1D(768, 2, padding='same')(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Dense(1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation('softmax')(x1)

    return x1


def get_activation(act_name):
    activation_layer = None
    if act_name == 'relu':
        activation_layer = tf.keras.layers.ReLU()
    elif act_name == 'leaky_relu':
        activation_layer = tf.keras.layers.LeakyReLU()
    elif act_name == 'tanh':
        activation_layer = tf.keras.activations.tanh
    else:
        assert False, "unknown act_name"
    return activation_layer


def make_aggregation(x_arr, mode="average"):
    if mode == 'average':
        x = tf.keras.layers.Average()(x_arr)
    elif mode == 'concat':
        x = tf.keras.layers.Concatenate()(x_arr)
    else:
        assert False, "unknown mode"
    return x


def cnn_block(x, filters, kernel_size, dropout=None, batch_norm=False, act_name="relu"):
    x = tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="same")(x) ### fix this
    if dropout:
        x = tf.keras.layers.Dropout(dropout)(x) 
    if batch_norm:
        x = BatchNormalization()(x)
    if act_name:
        x = get_activation(act_name)(x)
    return x


def dense_block(x, dropout=None, batch_norm=False, dense_unit=1, act_name="relu"):
    if dropout:
        x = tf.keras.layers.Dropout(dropout)(x) 
    if batch_norm:
        x = BatchNormalization()(x)
    x = tf.keras.layers.Dense(dense_unit)(x)
    if act_name:
        x = get_activation(act_name)(x)
    return x


def cnn_kernel_improved_block(x, filters_arr, kernel_size_arr, dropout=None, batch_norm=False, act_name="relu", mode="average"):
    x_arr = []
    for filters, kernel_size in zip(filters_arr, kernel_size_arr):
        x_arr += [cnn_block(x, filters, kernel_size, dropout, batch_norm, act_name)]
    return make_aggregation(x_arr, mode)


def cnn_act_improved_block(x, filters, kernel_size, dropout=None, batch_norm=False, act_name_arr=["tanh", "relu"], mode="average"):
    x_arr = []
    for act_name in act_name_arr:
        x_arr += [cnn_block(x, filters, kernel_size, dropout, batch_norm, act_name)]
    return make_aggregation(x_arr, mode)


def build_one_head_improved_model(
    x,
    filters_arr1=[128, 64],
    kernel_size_arr1=[2, 2],
    dropout_arr1=[0.1, 0.1],
    batch_norm_arr1=[False, False],
    act_name_arr1=['leaky_relu', 'leaky_relu'],
    res_mode_arr1=[False, False],
    kernel_mode_arr1=[None, None],
    act_mode_arr1=[None, None],
    dropout_arr2=[0.1],
    batch_norm_arr2=[False],
    dense_units_arr2=[1],
    act_name_arr2=[None],
):
    # CNN part
    for filters, kernel_size, dropout, batch_norm, act_name, res_mode, kernel_mode, act_mode in zip(
        filters_arr1,
        kernel_size_arr1,
        dropout_arr1,
        batch_norm_arr1,
        act_name_arr1,
        
        res_mode_arr1,
        kernel_mode_arr1,
        act_mode_arr1
    ):
        if kernel_mode:
            x1 = cnn_kernel_improved_block(x, filters, kernel_size, dropout, batch_norm, act_name, kernel_mode)
        elif act_mode:
            x1 = cnn_act_improved_block(x, filters, kernel_size, dropout, batch_norm, act_name, act_mode)
        else:
            x1 = cnn_block(x, filters, kernel_size, dropout, batch_norm, act_name)
           
        if res_mode:
            x = tf.keras.layers.Add()([x, x1])
        else:
            x = x1
        
    # Dense part
    assert act_name_arr2[-1] is None
    for dropout, batch_norm, dense_units, act_name in zip(
        dropout_arr2,
        batch_norm_arr2,
        dense_units_arr2,
        act_name_arr2
    ):
        x = dense_block(x, dropout, batch_norm, dense_units, act_name)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Activation('softmax')(x)

    return x


def get_combined_model_(base_model, is_default_head=False, **model_params):
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    padding = tf.cast(tf.equal(ids, PAD_ID), tf.int32)
    lens = MAX_LEN - tf.reduce_sum(padding, -1)
    max_len = tf.reduce_max(lens)

    ids_ = ids[:, :max_len]
    att_ = att[:, :max_len]
    tok_ = tok[:, :max_len]

    x = base_model(ids_, att_, tok_)
    
    if is_default_head:
        x1 = build_one_head_baseline_model(x[0])
        x2 = build_one_head_baseline_model(x[0])
    else:
        x1 = build_one_head_improved_model(x[0], **model_params)
        x2 = build_one_head_improved_model(x[0], **model_params)

#     x1 = tf.pad(x1, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)
#     x2 = tf.pad(x2, [[0, 0], [0, MAX_LEN - max_len]], constant_values=0.)

#     out = tf.keras.layers.Concatenate(axis=1)([x1, x2])
    out = [x1, x2]

    head_model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=out)
    return head_model

# def get_combined_model_(base_model, is_default_head=False, **model_params):
#     ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
#     att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
#     tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
#     padding = tf.cast(tf.equal(ids, PAD_ID), tf.int32)

#     lens = MAX_LEN - tf.reduce_sum(padding, -1)
#     max_len = tf.reduce_max(lens)
#     ids_ = ids[:, :max_len]
#     att_ = att[:, :max_len]
#     tok_ = tok[:, :max_len]

# #     config = RobertaConfig.from_pretrained(PATH+'config-roberta-base.json')
# #     bert_model = TFRobertaModel.from_pretrained(PATH+'pretrained-roberta-base.h5',config=config)
#     x = base_model(ids_, att_, tok_)
    
#     x1 = tf.keras.layers.Dropout(0.1)(x[0])
#     x1 = tf.keras.layers.Conv1D(768, 2,padding='same')(x1)
#     x1 = tf.keras.layers.LeakyReLU()(x1)
#     x1 = tf.keras.layers.Dense(1)(x1)
#     x1 = tf.keras.layers.Flatten()(x1)
#     x1 = tf.keras.layers.Activation('softmax')(x1)
    
#     x2 = tf.keras.layers.Dropout(0.1)(x[0]) 
#     x2 = tf.keras.layers.Conv1D(768, 2,padding='same')(x2)
#     x2 = tf.keras.layers.LeakyReLU()(x2)
#     x2 = tf.keras.layers.Dense(1)(x2)
#     x2 = tf.keras.layers.Flatten()(x2)
#     x2 = tf.keras.layers.Activation('softmax')(x2)

#     head_model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
#     return head_model

def get_combined_model(base_model, params):
    head_model = params["head_model"]
    if head_model == "default":
        return get_combined_model_(base_model, is_default_head=True)
    model_params = ast.literal_eval(head_model)
    return get_combined_model_(base_model, **model_params)
