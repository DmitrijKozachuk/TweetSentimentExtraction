import os
import re
import sys

def get_starts(att_num, n_epochs):
    
    pair_pattern = '\[fold: \d\d*, epoch: \d\d*\]'
    max_pair = (1, 0)
    
    fold_pattern = '\[fold: \d\d*]'
    max_entire_fold = 1
    
    log_path = "../attempt_logs/{}/log.txt".format(att_num)
    if not os.path.exists(log_path):
        print(1, 1, False)
        return

    with open(log_path, "r") as f:
        for line in f:
            if re.search(pair_pattern, line):
                fold = int(line.split("fold:")[1].split(",")[0])
                epoch = int(line.split("epoch:")[1].split("]")[0])
                max_pair = max(max_pair, (fold, epoch))
            if re.search(fold_pattern, line):
                entire_fold = int(line.split("fold:")[1].split("]")[0])
                max_entire_fold = max(max_entire_fold, entire_fold)
                
    last_fold, last_epoch = max_pair
    if (last_epoch == n_epochs) and (max_entire_fold < last_fold):
        print(last_fold, n_epochs, True)
        return

    if last_epoch != n_epochs:
        next_fold, next_epoch = last_fold, last_epoch + 1
    else:
        next_fold, next_epoch = last_fold + 1, 1

    print(next_fold, next_epoch, False)

att_num = int(sys.argv[1])
n_epochs = int(sys.argv[2])

get_starts(att_num, n_epochs)