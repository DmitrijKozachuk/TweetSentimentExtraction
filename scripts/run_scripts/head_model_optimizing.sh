#!/bin/bash

base_path=../..

# Testing on 1600 samples and 3 epochs
mode=debug:1600
start_epoch=4
wo_fitting=False

# Number of attempt for pre-define model
weights_att_num=None

# Base Model
base_model=distilbert-base-cased-distilled-squad

# LR scheduling (custom filling opt_name)
lr=0.00003

# Loss type ["JEL", "CCE", "{'CCE':1, 'JEL':0.1}"]
loss=CCE

# Label smoothing (for model_name=padded_with_smoothing, in other cases ignored)
label_smoothing=0.3

# code head
att_num=16
n_fold=1
for head_model in "{'filters_arr1':[768],'kernel_size_arr1':[2],'dropout_arr1':[0.1],'batch_norm_arr1':[False],'act_name_arr1':['leaky_relu'],'res_mode_arr1':[False],'kernel_mode_arr1':[None],'act_mode_arr1':[None],'dropout_arr2':[None],'batch_norm_arr2':[False],'dense_units_arr2':[1],'act_name_arr2':[None]}"
do
    ~/anaconda3/envs/tf/bin/python reset_gpu.py
    ~/anaconda3/envs/tf/bin/python ../main.py $base_path $mode $att_num $weights_att_num $n_fold $start_epoch $wo_fitting $base_model $head_model $lr $loss $label_smoothing
done

























