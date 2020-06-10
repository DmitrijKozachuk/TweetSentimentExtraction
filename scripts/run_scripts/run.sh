#!/bin/bash

# Training Mode ("prod" or "debug:[num_samples]")
mode=debug:1600

# Number of attempt for pre-define model
att_num=14

# Number of attempt for pre-define model
weights_att_num=None

# Model (custom filling model_name) [roberta-base]
base_model=roberta-base

# Model (custom filling model_name) [default]
head_model=default

# LR scheduling (custom filling opt_name)
lr=0.00003

# Loss type ["JEL", "CCE", "{'CCE':1, 'JEL':0.1}"]
loss=CCE

# Label smoothing (for model_name=padded_with_smoothing, in other cases ignored)
label_smoothing=0.3
 

str_start_point=$(~/anaconda3/envs/tf/bin/python get_starts.py $att_num 5)
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
    ~/anaconda3/envs/tf/bin/python ../main.py $mode $att_num $weights_att_num $n_fold $start_epoch $wo_fitting $base_model $head_model $lr $loss $label_smoothing
    
    start_epoch=1
    wo_fitting=False
done


