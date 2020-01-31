from ml4ir.model.losses.loss_base import PointwiseLossBase
from tensorflow import math
from tensorflow.keras import losses
import tensorflow as tf


class SigmoidCrossEntropy(PointwiseLossBase):
    def _make_loss_fn(self, **kwargs):
        """
        Define a sigmoid cross entropy loss
        Additionally can pass in record positions to handle positional bias

        NOTE:
        Should handle different types of scoring functions.
        Keeping it simple for now.
        """
        cce = losses.CategoricalCrossentropy()

        def _loss_fn(y_true, y_pred):
            # Mask the predictions to ignore padded records
            mask = kwargs.get("mask")
            y_true = math.multiply(tf.cast(mask, tf.float32), tf.cast(y_true, tf.float32))
            y_pred = math.multiply(tf.cast(mask, tf.float32), tf.cast(y_pred, tf.float32))

            return cce(y_true, y_pred)

        return _loss_fn
