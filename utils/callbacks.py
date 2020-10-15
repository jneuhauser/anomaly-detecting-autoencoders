import sys
assert sys.version_info >= (3, 0), "Python 3.0 or greater required"
import logging
logger   = logging.getLogger(__name__)
debug    = logger.debug
info     = logger.info
warning  = logger.warning
error    = logger.error
critical = logger.critical

import tensorflow as tf
import numpy as np


class ADModelEvaluator(tf.keras.callbacks.Callback):
    def __init__(self, test_count, model_dir=None, early_stopping=None):
        super().__init__()

        self.model_dir = model_dir

        # AUROC as fixed metric
        self.metric = tf.keras.metrics.AUC(
            curve='ROC',
            summation_method='interpolation'
        )

        self.test_labels = np.zeros((test_count, 1), dtype=np.float32)
        self.test_losses = np.zeros((test_count, 1), dtype=np.float32)
        self.test_results_idx_start = 0
        self.test_results_idx_end = 0

        self.test_ptp_loss = 0.
        self.test_min_loss = 0.
        self.test_result = 0.
        self.test_results = []

        self.best_ptp_loss = 0.
        self.best_min_loss = 0.
        self.best_result = 0.
        self.best_epoch = 0
        self.best_weights = None

        self.early_stopping = early_stopping

    def on_epoch_end(self, epoch, logs=None):
        # Save epoch result history
        self.test_results.append(self.test_result)
        # Keep track of best metric and save best model
        if self.test_result >= self.best_result:
            self.best_epoch = epoch
            self.best_result = self.test_result
            self.best_ptp_loss = self.test_ptp_loss
            self.best_min_loss = self.test_min_loss
            self.best_weights = self.model.get_weights()
            if self.model_dir:
                self.model.save_weights(path=self.model_dir)
        # Print determined values
        info("Curr Epoch {:02d}: AUC(ROC): {:.5f}, ptp_loss: {:.5f}, min_loss: {:.5f}".format(
            epoch+1,
            self.test_result,
            self.test_ptp_loss,
            self.test_min_loss
        ))
        info("Best Epoch {:02d}: AUC(ROC): {:.5f}, ptp_loss: {:.5f}, min_loss: {:.5f}".format(
            self.best_epoch+1,
            self.best_result,
            self.best_ptp_loss,
            self.best_min_loss
        ))
        #debug("TP: {}\nTN: {}\nFP: {}\nFN: {}\nTH: {}".format(
        #    self.metric.true_positives,
        #    self.metric.true_negatives,
        #    self.metric.false_positives,
        #    self.metric.false_negatives,
        #    self.metric.thresholds
        #))
        if (
            self.early_stopping and
            self.early_stopping > 0 and
            (epoch - self.best_epoch) >= self.early_stopping
        ):
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_begin(self, logs=None):
        self.stopped_epoch = 0

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            info("Epoch {:02d}: early stopping after {} epoch{} of no improvement".format(
                self.stopped_epoch + 1,
                self.early_stopping,
                's' if self.early_stopping > 1 else ''
            ))

    def on_test_begin(self, logs=None):
        # Prepare for new evaluation
        self.metric.reset_states()
        self.test_results_idx_start = 0
        self.test_results_idx_end = 0

    def on_test_end(self, logs=None):
        # Raise error if not enoght results are collected
        if self.test_results_idx_end != self.test_losses.shape[0]:
            raise ValueError("collected results count of {} doesn't match expected count of {}"
                .format(self.test_results_idx_end, self.test_losses.shape[0]))
        # Scale losses between [0, 1]
        self.test_min_loss = np.min(self.test_losses)
        self.test_ptp_loss = np.max(self.test_losses) - self.test_min_loss
        self.test_losses -= self.test_min_loss
        self.test_losses /= self.test_ptp_loss
        # Calculate metric AUROC
        self.metric.update_state(self.test_labels, self.test_losses)
        self.test_result = self.metric.result().numpy()

    def on_test_batch_end(self, batch, logs=None):
        # Gather all per batch losses and labels
        labels = logs.get('labels')
        losses = logs.get('losses')
        batch_size = losses.shape[0]
        self.test_results_idx_end += batch_size
        self.test_labels[self.test_results_idx_start:self.test_results_idx_end] = labels
        self.test_losses[self.test_results_idx_start:self.test_results_idx_end] = losses
        self.test_results_idx_start += batch_size
