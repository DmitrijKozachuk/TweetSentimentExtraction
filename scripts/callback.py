import time
from tensorflow.keras.callbacks import Callback

from utils import get_proba_prediction, get_score

class CustomCallback(Callback):
    def __init__(
        self,
        model,
        word_ids,
        mask,
        segm_ids,
        df,
        sample_ind2new_ind2old_ind,
        n_fold,
        start_epoch,
        log_path,
        start_score,
        best_weights_path
    ):
        self.model = model

        self.word_ids = word_ids
        self.mask = mask
        self.segm_ids = segm_ids

        self.df = df
        self.sample_ind2new_ind2old_ind = sample_ind2new_ind2old_ind

        self.start_epoch = start_epoch
        self.n_fold = n_fold
        self.log_path = log_path
        self.best_weights_path = best_weights_path

        self.best = start_score
        self.checkpoint = time.time()

    def on_epoch_end(self, epoch, logs):
        # Validation
        start_proba, end_proba = get_proba_prediction(self.model, self.word_ids, self.mask, self.segm_ids)
        current = get_score(start_proba, end_proba, self.df, self.sample_ind2new_ind2old_ind)

        # Save best model
        if current > self.best:
            self.best = current
            self.model.save_weights(self.best_weights_path, overwrite=True)

        # Log score info
        abs_epoch = self.start_epoch + epoch
        with open(self.log_path, 'a') as f:
            f.write(f'\n[fold: {self.n_fold}, epoch: {abs_epoch}] Val Score : {current:.5f} (time: {(time.time() - self.checkpoint)// 60 } min.)')
        self.checkpoint = time.time()
