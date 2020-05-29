#!/bin/bash

seed=42

n_folds=5

n_epochs=7

batch_size=8

# Number of current attempt
att_num=8

# Number of attempt for pre-define model
weights_att_num=None

# Model (custom filling model_name) [default, v2.0]
model_name=default

# Optimizer (custom filling opt_name)
opt_name=Adam

# LR scheduling (custom filling opt_name)
lr=0.00003

# LR checduling (custom filling opt_name)
lr_schedule_name=default

# Way to using positive/nutral/negative label [left, right]
label_consider_type=left

# Loss type [JEL, CCE]
loss=JEL

# Label smoothing (for model_name=padded_with_smoothing, in other cases ignored)
label_smoothing=0.1

str_start_point=$(~/anaconda3/envs/tf/bin/python get_starts.py $att_num $n_epochs)
IFS=', '
read -r -a array <<< "$str_start_point"
start_fold="${array[0]}"
start_epoch="${array[1]}"
wo_fitting="${array[2]}"

# echo $start_fold
# echo $start_epoch
# echo $wo_fitting

for ((n_fold = $start_fold ; n_fold <= $n_folds ; n_fold++)); do
    echo "####### n_fold: " $n_fold "start_epoch: " $start_epoch "#######"

    ~/anaconda3/envs/tf/bin/python reset_gpu.py
    ~/anaconda3/envs/tf/bin/python fold_processing.py $seed $n_folds $n_epochs $batch_size $att_num $weights_att_num $model_name $opt_name $lr $lr_schedule_name $n_fold $start_epoch $wo_fitting $label_consider_type $loss $label_smoothing
    
    start_epoch=1
    wo_fitting=False
done


