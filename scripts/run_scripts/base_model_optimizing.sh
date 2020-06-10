#!/bin/bash

base_path=../..

# Testing on 1600 samples and 3 epochs
mode=debug:1600
start_epoch=4
wo_fitting=False

# Number of attempt for pre-define model
weights_att_num=None

# Model (custom filling model_name) [default]
head_model=default

# LR scheduling (custom filling opt_name)
lr=0.00003

# Loss type ["JEL", "CCE", "{'CCE':1, 'JEL':0.1}"]
loss=CCE

# Label smoothing (for model_name=padded_with_smoothing, in other cases ignored)
label_smoothing=0.3

# code head
att_num=15
for ((n_fold = 2 ; n_fold <= 5 ; n_fold++))
do
    for base_model in roberta-base distilbert-base-cased bert-base-cased bert-base-uncased distilbert-base-uncased distilbert-base-cased-distilled-squad distilroberta-base albert-base-v1 distilbert-base-uncased-distilled-squad bert-base-uncased bert-base-cased-finetuned-mrpc bert-base-uncased
    do
        ~/anaconda3/envs/tf/bin/python reset_gpu.py
        ~/anaconda3/envs/tf/bin/python ../main.py $base_path $mode $att_num $weights_att_num $n_fold $start_epoch $wo_fitting $base_model $head_model $lr $loss $label_smoothing
    done
done


# att_num=15
# n_fold=2
# for ((n_fold = 2 ; n_fold <= 5 ; n_fold++)); do
#     for base_model in roberta-base distilbert-base-cased bert-base-cased bert-base-uncased distilbert-base-uncased distilbert-base-cased-distilled-squad distilroberta-base albert-base-v1 distilbert-base-uncased-distilled-squad bert-base-uncased bert-base-cased-finetuned-mrpc bert-base-uncased do
#         ~/anaconda3/envs/tf/bin/python reset_gpu.py
#         ~/anaconda3/envs/tf/bin/python ../main.py $base_path $mode $att_num $weights_att_num $n_fold $start_epoch $wo_fitting $base_model $head_model $lr $loss $label_smoothing
#     done
# done


########## CHAMPIONS LIGUE : preparation stage : code head ##########
# att_num=14 
# n_fold=1
# for base_model in roberta-base bert-base-chinese bert-base-uncased bert-base-uncased bert-base-finnish-uncased-v1 bert-base-dutch-cased bert-base-cased-finetuned-mrpc bert-base-finnish-cased-v1 bert-base-german-cased bert-base-cased distilroberta-base xlnet-base-cased xlm-mlm-ende-1024 xlm-mlm-enfr-1024 xlm-mlm-enro-1024 xlm-clm-ende-1024 xlm-clm-enfr-1024 distilbert-base-uncased-distilled-squad distilbert-base-multilingual-cased distilbert-base-uncased distilbert-base-cased-distilled-squad distilbert-base-cased albert-base-v2 albert-large-v2 albert-xlarge-v2 albert-base-v1 albert-large-v1 google/electra-large-discriminator google/electra-small-discriminator google/electra-small-generator google/electra-base-generator google/electra-base-discriminator google/electra-large-generator do
#         ~/anaconda3/envs/tf/bin/python reset_gpu.py
#         ~/anaconda3/envs/tf/bin/python ../main.py $base_path $mode $att_num $weights_att_num $n_fold $start_epoch $wo_fitting $base_model $head_model $lr $loss $label_smoothing
#         done

########## CHAMPIONS LIGUE : preparation stage : results ##########

# [base_model] roberta-base
# [fold: 1, epoch: 4] Val Score : 0.57691 (time: 1.0 min.)
# [fold: 1, epoch: 5] Val Score : 0.57824 (time: 1.0 min.)
# [base_model] roberta-base
# [fold: 1, epoch: 4] Val Score : 0.59846 (time: 2.0 min.)
# [fold: 1, epoch: 5] Val Score : 0.59825 (time: 2.0 min.)

# 1. distilbert-base-cased
# [fold: 1, epoch: 4] Val Score : 0.62467 (time: 1.0 min.)
# [fold: 1, epoch: 5] Val Score : 0.65771 (time: 0.0 min.)

# 2. bert-base-cased
# [fold: 1, epoch: 4] Val Score : 0.63260 (time: 1.0 min.)
# [fold: 1, epoch: 5] Val Score : 0.65459 (time: 1.0 min.)

# 3. bert-base-uncased
# [fold: 1, epoch: 4] Val Score : 0.64043 (time: 2.0 min.)
# [fold: 1, epoch: 5] Val Score : 0.65487 (time: 2.0 min.)
# [base_model] bert-base-uncased
# [fold: 1, epoch: 4] Val Score : 0.62592 (time: 2.0 min.)
# [fold: 1, epoch: 5] Val Score : 0.65443 (time: 1.0 min.)

# 4. distilbert-base-uncased
# [fold: 1, epoch: 4] Val Score : 0.64100 (time: 1.0 min.)
# [fold: 1, epoch: 5] Val Score : 0.65395 (time: 0.0 min.)

# 5. distilbert-base-cased-distilled-squad
# [fold: 1, epoch: 4] Val Score : 0.63863 (time: 1.0 min.)
# [fold: 1, epoch: 5] Val Score : 0.65365 (time: 0.0 min.)

# 6. distilroberta-base
# [fold: 1, epoch: 4] Val Score : 0.60188 (time: 1.0 min.)
# [fold: 1, epoch: 5] Val Score : 0.64048 (time: 0.0 min.)

# 7. albert-base-v1
# [fold: 1, epoch: 4] Val Score : 0.63602 (time: 1.0 min.)
# [fold: 1, epoch: 5] Val Score : 0.64619 (time: 1.0 min.)

# 8. distilbert-base-uncased-distilled-squad
# [fold: 1, epoch: 4] Val Score : 0.63624 (time: 1.0 min.)
# [fold: 1, epoch: 5] Val Score : 0.64630 (time: 0.0 min.)

# 9. bert-base-uncased
# [fold: 1, epoch: 4] Val Score : 0.63172 (time: 2.0 min.)
# [fold: 1, epoch: 5] Val Score : 0.64393 (time: 1.0 min.)

# 10. bert-base-cased-finetuned-mrpc
# [fold: 1, epoch: 4] Val Score : 0.63991 (time: 1.0 min.)
# [fold: 1, epoch: 5] Val Score : 0.64386 (time: 1.0 min.)

# 11. bert-base-uncased
# [fold: 1, epoch: 4] Val Score : 0.62854 (time: 2.0 min.)
# [fold: 1, epoch: 5] Val Score : 0.64327 (time: 2.0 min.)


########## CHAMPIONS LIGUE : final : code head ##########
# att_num=15 
# for ((n_fold = 2 ; n_fold <= 5 ; n_fold++)); do
#     for base_model in roberta-base distilbert-base-cased bert-base-cased bert-base-uncased distilbert-base-uncased distilbert-base-cased-distilled-squad distilroberta-base albert-base-v1 distilbert-base-uncased-distilled-squad bert-base-uncased bert-base-cased-finetuned-mrpc bert-base-uncased do
#         ~/anaconda3/envs/tf/bin/python reset_gpu.py
#         ~/anaconda3/envs/tf/bin/python ../main.py $base_path $mode $att_num $weights_att_num $n_fold $start_epoch $wo_fitting $base_model $head_model $lr $loss $label_smoothing
#         done
#     done


########## CHAMPIONS LIGUE : final : results ##########
[base_model] roberta-base (0.597)
[fold: 1, epoch: 4] Val Score : 0.59846 (time: 2.0 min.)
[fold: 2, epoch: 5] Val Score : 0.61619 (time: 1.0 min.)
[fold: 3, epoch: 5] Val Score : 0.59026 (time: 2.0 min.)
[fold: 4, epoch: 5] Val Score : 0.58957 (time: 2.0 min.)
[fold: 5, epoch: 5] Val Score : 0.59352 (time: 2.0 min.)


1. distilbert-base-cased-distilled-squad (0.661)
[fold: 1, epoch: 5] Val Score : 0.65365 (time: 0.0 min.)
[fold: 2, epoch: 5] Val Score : 0.66989 (time: 1.0 min.)
[fold: 3, epoch: 5] Val Score : 0.65700 (time: 1.0 min.)
[fold: 4, epoch: 5] Val Score : 0.65822 (time: 1.0 min.)
[fold: 5, epoch: 5] Val Score : 0.66956 (time: 1.0 min.)


2. distilbert-base-uncased-distilled-squad (0.658)
[fold: 1, epoch: 5] Val Score : 0.64630 (time: 0.0 min.)
[fold: 2, epoch: 5] Val Score : 0.67183 (time: 1.0 min.)
[fold: 3, epoch: 5] Val Score : 0.65721 (time: 1.0 min.)
[fold: 4, epoch: 5] Val Score : 0.65206 (time: 1.0 min.)
[fold: 5, epoch: 5] Val Score : 0.66409 (time: 0.0 min.)


3. distilbert-base-cased (0.656)
[fold: 1, epoch: 5] Val Score : 0.65771 (time: 0.0 min.)
[fold: 2, epoch: 5] Val Score : 0.66407 (time: 0.0 min.)
[fold: 3, epoch: 5] Val Score : 0.65056 (time: 1.0 min.)
[fold: 4, epoch: 5] Val Score : 0.64867 (time: 1.0 min.)
[fold: 5, epoch: 5] Val Score : 0.66164 (time: 1.0 min.)


4. bert-base-cased (0.656)
[fold: 1, epoch: 5] Val Score : 0.65459 (time: 1.0 min.)
[fold: 2, epoch: 5] Val Score : 0.66789 (time: 2.0 min.)
[fold: 3, epoch: 5] Val Score : 0.65161 (time: 2.0 min.)
[fold: 4, epoch: 5] Val Score : 0.65596 (time: 2.0 min.)
[fold: 5, epoch: 5] Val Score : 0.65033 (time: 2.0 min.)


5. bert-base-cased-finetuned-mrpc (0.655)
[fold: 1, epoch: 5] Val Score : 0.64386 (time: 1.0 min.)
[fold: 2, epoch: 5] Val Score : 0.66784 (time: 2.0 min.)
[fold: 3, epoch: 5] Val Score : 0.65062 (time: 2.0 min.)
[fold: 4, epoch: 5] Val Score : 0.66057 (time: 2.0 min.)
[fold: 5, epoch: 5] Val Score : 0.65602 (time: 1.0 min.)




6. distilroberta-base (0.653)
[fold: 1, epoch: 5] Val Score : 0.64048 (time: 0.0 min.)
[fold: 2, epoch: 5] Val Score : 0.67419 (time: 1.0 min.)
[fold: 3, epoch: 5] Val Score : 0.65740 (time: 1.0 min.)
[fold: 4, epoch: 5] Val Score : 0.63931 (time: 1.0 min.)
[fold: 5, epoch: 5] Val Score : 0.65636 (time: 1.0 min.)


7. distilbert-base-uncased (0.652)
[fold: 1, epoch: 5] Val Score : 0.65395 (time: 0.0 min.)
[fold: 2, epoch: 5] Val Score : 0.66788 (time: 1.0 min.)
[fold: 3, epoch: 5] Val Score : 0.64615 (time: 1.0 min.)
[fold: 4, epoch: 5] Val Score : 0.63201 (time: 1.0 min.)
[fold: 5, epoch: 5] Val Score : 0.66315 (time: 1.0 min.)


8. bert-base-uncased (0.652)
[fold: 1, epoch: 5] Val Score : 0.65487 (time: 2.0 min.)
[fold: 2, epoch: 5] Val Score : 0.66263 (time: 2.0 min.)
[fold: 3, epoch: 5] Val Score : 0.65088 (time: 2.0 min.)
[fold: 4, epoch: 5] Val Score : 0.64941 (time: 2.0 min.)
[fold: 5, epoch: 5] Val Score : 0.64542 (time: 2.0 min.)


9. albert-base-v1
[fold: 1, epoch: 4] Val Score : 0.63602 (time: 1.0 min.)
[fold: 1, epoch: 5] Val Score : 0.64619 (time: 1.0 min.)
[base_model] albert-base-v1
[fold: 2, epoch: 4] Val Score : 0.64983 (time: 1.0 min.)
[fold: 2, epoch: 5] Val Score : 0.67572 (time: 1.0 min.)
[base_model] albert-base-v1
[base_model] albert-base-v1
[fold: 4, epoch: 4] Val Score : 0.62741 (time: 1.0 min.)
[fold: 4, epoch: 5] Val Score : 0.62674 (time: 1.0 min.)
[base_model] albert-base-v1
[fold: 5, epoch: 4] Val Score : 0.62254 (time: 1.0 min.)
[fold: 5, epoch: 5] Val Score : 0.64814 (time: 1.0 min.)

























