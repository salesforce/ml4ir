from tensorflow.keras import losses
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.losses import Reduction

from ml4ir.base.model.losses.loss_base import RelevanceLossBase


class SigmoidCrossEntropy(losses.BinaryCrossentropy, RelevanceLossBase):
    def __init__(self, reduction=Reduction.SUM_OVER_BATCH_SIZE, **kwargs):
        super(SigmoidCrossEntropy, self).__init__(reduction=reduction, **kwargs)

    def __call__(self, y_true, y_pred, features, **kwargs):
        """
        Define a sigmoid cross entropy loss
        Additionally can pass in record positions to handle positional bias

        """
        # Mask the predictions to ignore padded records
        mask = features["mask"]
        masked_indices = tf.where(tf.equal(tf.cast(mask, tf.float32), tf.constant(1.0)))
        y_true = tf.gather_nd(y_true, masked_indices)
        y_pred = tf.gather_nd(y_pred, masked_indices)

        return super().__call__(y_true, y_pred)

    def get_final_activation_op(self, output_name):
        # Pointwise sigmoid loss
        return lambda logits, mask: layers.Activation("sigmoid", name=output_name)(logits)
