import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow.keras.losses import Reduction

from ml4ir.base.model.losses.loss_base import RelevanceLossBase


class RankOneListNet(losses.BinaryCrossentropy, RelevanceLossBase):
    def __init__(self, reduction=Reduction.SUM, **kwargs):
        super(RankOneListNet, self).__init__(reduction=reduction, **kwargs)

    def __call__(self, y_true, y_pred, features, **kwargs):
        """
        Define a rank 1 ListNet loss
        Additionally can pass in record positions to handle positional bias

        """
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)

        # Mask the padded records
        mask = features["mask"]
        masked_indices = tf.where(tf.equal(tf.cast(mask, tf.float32), tf.constant(1.0)))
        y_true = tf.gather_nd(y_true, masked_indices)
        y_pred = tf.gather_nd(y_pred, masked_indices)

        # Reshape the tensors
        y_true = tf.expand_dims(tf.squeeze(y_true), axis=-1)
        y_pred = tf.expand_dims(tf.squeeze(y_pred), axis=-1)

        # Compute binary cross entropy
        bce = super().__call__(y_true, y_pred, **kwargs)
        return tf.math.divide(bce, batch_size)

    def get_final_activation_op(self, output_name):
        softmax_op = layers.Softmax(axis=-1, name=output_name)

        # Listwise Top 1 RankNet Loss
        def masked_softmax(logits, mask):
            """
            NOTE:
            Tried to manually compute softmax with tf operations,
            but tf.keras.layers.Softmax() is more stable when working with
            cross_entropy layers
            """
            logits = tf.where(
                tf.equal(mask, tf.constant(1.0)), logits, tf.constant(tf.float32.min)
            )

            return softmax_op(logits)

        return masked_softmax
