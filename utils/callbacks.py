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
    """ADModelEvaluator is a Keras evaluation callback for detecting anomalies with autoencoders.
    The evaluation is performed with binary labels against the reconstruction error,
    for example with MSE, of the reconstructed data using the AUROC metric.

    This callback also implements some functions from the callbacks "EarlyStopping"
    and "ReduceLROnPlateau", because in this callback the evaluated metric is calculated
    and therefore not available for other callbacks.

    This callback requires a specially designed test_step(data) function because
    the labels must be returned from there to make them available within this callback.

    A minimal example for the test_step(data) function is:

    def test_step(self, data):
        _, y, _ = tf.keras.utils.unpack_x_y_sample_weight(data)
        return { *super().test_step(data), 'labels': y }

    Args:
        test_count (int): The numer of test samples. Needed for allocation of the used numpy arrays.
        model_dir (str, optional): Output directory for storing the best model weights. Defaults to None.
        early_stopping_patience (int, optional): Number of epochs without improvment to stop training. Defaults to None.
        reduce_lr_cooldown (int, optional): Number of epochs to wait before resuming normal operation after lr has been reduced. Defaults to reduce_lr_patience if None.
        reduce_lr_factor (float, optional): Factor by which the learning rate will be reduced. new_lr = lr * factor. Defaults to 0.1.
        reduce_lr_min_lr (float, optional): Lower bound on the learning rate. Defaults to 1e-6.
        reduce_lr_patience (int, optional): Number of epochs with no improvement after which learning rate will be reduced. Defaults to None.
    """
    def __init__(self, test_count: int, model_dir: str=None,
        early_stopping_patience: int=None,
        reduce_lr_cooldown: int=None,
        reduce_lr_factor: float=0.1,
        reduce_lr_min_lr: float=1e-6,
        reduce_lr_patience: int=None):
        super().__init__()

        self.model_dir = model_dir

        # AUROC as fixed metric
        self._metric = tf.keras.metrics.AUC(
            curve='ROC',
            summation_method='interpolation'
        )

        self._test_labels = np.zeros((test_count, 1), dtype=np.float32)
        self._test_losses = np.zeros((test_count, 1), dtype=np.float32)
        self._test_results_idx_start = 0
        self._test_results_idx_end = 0

        self._test_ptp_loss = 0.
        self._test_min_loss = 0.
        self._test_result = 0.

        self.test_ptp_losses = []
        self.test_min_losses = []
        self.test_results = []

        self.best_ptp_loss = 0.
        self.best_min_loss = 0.
        self.best_result = 0.
        self.best_epoch = 0
        self.best_weights = None

        self.early_stopping_patience = early_stopping_patience

        self.reduce_lr_patience = reduce_lr_patience
        self.reduce_lr_factor = reduce_lr_factor
        self.reduce_lr_min_lr = reduce_lr_min_lr
        self.reduce_lr_cooldown = reduce_lr_cooldown or reduce_lr_patience

    def _handle_best_epoch(self, epoch):
        # Keep track of best metric and save best model
        if (self._test_result - 1e-7) > self.best_result:
            self.best_epoch = epoch
            self.best_result = self._test_result
            self.best_ptp_loss = self._test_ptp_loss
            self.best_min_loss = self._test_min_loss
            self.best_weights = self.model.get_weights()
            if self.model_dir:
                self.model.save_weights(path=self.model_dir)
            self._log_best_epoch()

    def _handle_early_stopping(self, epoch):
        if (
            self.early_stopping_patience and
            self.early_stopping_patience > 0 and
            (epoch - self.best_epoch) >= self.early_stopping_patience
        ):
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def _handle_reduce_lr(self, epoch):
        # TODO: Keep track of test_result based on last time lr reduced???
        if (
            self.reduce_lr_patience and
            self.reduce_lr_patience > 0 and
            (epoch - self.best_epoch) >= self.reduce_lr_patience and
            self._cooldown <= 0
        ):
            # Reduce learning rate of all optimizers
            for var_name, optimizer in self._model_optimizers.items():
                old_lr = float(tf.keras.backend.get_value(optimizer.lr))
                if old_lr <= self.reduce_lr_min_lr:
                    continue
                new_lr = old_lr * self.reduce_lr_factor
                if new_lr < self.reduce_lr_min_lr:
                    new_lr = self.reduce_lr_min_lr
                info("Setting {}.lr to {}".format(var_name, new_lr))
                tf.keras.backend.set_value(optimizer.lr, new_lr)
            self._cooldown = self.reduce_lr_cooldown
        else:
            self._cooldown -= 1

    def _log_best_epoch(self):
        info("Best Epoch {:02d}: AUC(ROC): {:.5f}, ptp_loss: {:.5f}, min_loss: {:.5f}".format(
            self.best_epoch+1,
            self.best_result,
            self.best_ptp_loss,
            self.best_min_loss
        ))

    def _log_current_epoch(self, epoch):
        info("Curr Epoch {:02d}: AUC(ROC): {:.5f}, ptp_loss: {:.5f}, min_loss: {:.5f}".format(
            epoch+1,
            self._test_result,
            self._test_ptp_loss,
            self._test_min_loss
        ))

        debug("TP: {}\nTN: {}\nFP: {}\nFN: {}\nTH: {}".format(
            self._metric.true_positives,
            self._metric.true_negatives,
            self._metric.false_positives,
            self._metric.false_negatives,
            self._metric.thresholds
        ))

    def on_epoch_end(self, epoch, logs=None):
        # Log determined values
        self._log_current_epoch(epoch)

        # Save epoch result history
        self.test_ptp_losses.append(self._test_ptp_loss)
        self.test_min_losses.append(self._test_min_loss)
        self.test_results.append(self._test_result)

        # Handle epoch based callback features
        self._handle_best_epoch(epoch)
        self._handle_early_stopping(epoch)
        self._handle_reduce_lr(epoch)

    def on_train_begin(self, logs=None):
        self.stopped_epoch = 0
        self._cooldown = 0
        # collect all variables of type Optimizer from model object
        self._model_optimizers = {
            k: v for k, v in self.model.__dict__.items()
            if isinstance(v, tf.keras.optimizers.Optimizer)
        }

    def on_train_end(self, logs=None):
        self._log_best_epoch()
        if self.stopped_epoch > 0:
            info("Epoch {:02d}: early stopping after {} epoch{} of no improvement".format(
                self.stopped_epoch + 1,
                self.early_stopping_patience,
                's' if self.early_stopping_patience > 1 else ''
            ))

    def on_test_begin(self, logs=None):
        # Prepare for new evaluation
        self._metric.reset_states()
        self._test_results_idx_start = 0
        self._test_results_idx_end = 0

    def on_test_end(self, logs=None):
        # Raise error if not enoght results are collected
        if self._test_results_idx_end != self._test_losses.shape[0]:
            raise ValueError("collected results count of {} doesn't match expected count of {}"
                .format(self._test_results_idx_end, self._test_losses.shape[0]))
        # Scale losses between [0, 1]
        self._test_min_loss = np.min(self._test_losses)
        self._test_ptp_loss = np.max(self._test_losses) - self._test_min_loss
        self._test_losses -= self._test_min_loss
        self._test_losses /= self._test_ptp_loss
        # Clip values to [0, 1] to work around float inaccuracy (inplace)
        np.clip(self._test_losses, 0, 1, out=self._test_losses)
        # Calculate metric AUROC
        try:
            self._metric.update_state(self._test_labels, self._test_losses)
            self._test_result = self._metric.result().numpy()
        except Exception as e:
            print("test_labels:", self._test_labels)
            print("test_losses:", self._test_losses)
            raise e

    def on_test_batch_end(self, batch, logs=None):
        # Gather all per batch losses and labels
        labels = logs.get('labels')
        losses = logs.get('losses')
        batch_size = losses.shape[0]
        self._test_results_idx_end += batch_size
        self._test_labels[self._test_results_idx_start:self._test_results_idx_end] = labels
        self._test_losses[self._test_results_idx_start:self._test_results_idx_end] = losses
        self._test_results_idx_start += batch_size
