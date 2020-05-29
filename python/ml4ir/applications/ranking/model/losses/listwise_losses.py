import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow.keras.losses import Reduction

from ml4ir.applications.ranking.model.losses.loss_base import ListwiseLossBase


class RankOneListNet(ListwiseLossBase):
    def get_loss_fn(self, **kwargs):
        """
        Define a rank 1 ListNet loss
        Additionally can pass in record positions to handle positional bias

        """
        bce = losses.BinaryCrossentropy(reduction=Reduction.SUM)
        mask = kwargs.get("mask")

        def _loss_fn(y_true, y_pred):
            batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)

            # Mask the padded records
            y_true = tf.gather_nd(y_true, tf.where(tf.equal(mask, tf.constant(1.0))))
            y_pred = tf.gather_nd(y_pred, tf.where(tf.equal(mask, tf.constant(1.0))))

            # Reshape the tensors
            y_true = tf.expand_dims(tf.squeeze(y_true), axis=-1)
            y_pred = tf.expand_dims(tf.squeeze(y_pred), axis=-1)

            return tf.math.divide(bce(y_true, y_pred), batch_size)

        return _loss_fn

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
