from ml4ir.model.losses.loss_base import ListwiseLossBase
import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow.keras.losses import Reduction


class RankOneListNet(ListwiseLossBase):
    def _make_loss_fn(self, **kwargs):
        """
        Define a rank 1 ListNet loss
        Additionally can pass in record positions to handle positional bias

        """
        bce = losses.BinaryCrossentropy(reduction=Reduction.SUM)
        mask = kwargs.get("mask")
        # cce = losses.CategoricalCrossentropy(reduction=Reduction.SUM_OVER_BATCH_SIZE)

        def _loss_fn(y_true, y_pred):
            # Mask the padded records
            y_true = tf.gather_nd(y_true, tf.where(tf.equal(mask, tf.constant(1.0))))
            y_pred = tf.gather_nd(y_pred, tf.where(tf.equal(mask, tf.constant(1.0))))

            # Reshape the tensors
            y_true = tf.expand_dims(tf.squeeze(y_true), axis=-1)
            y_pred = tf.expand_dims(tf.squeeze(y_pred), axis=-1)

            y_true = tf.clip_by_value(y_true, tf.constant(1e-9), tf.constant(1.0))

            return bce(y_true, y_pred)
            # return cce(y_true, y_pred)

        return _loss_fn

    def _final_activation_op(self):
        # Without masking padded records
        softmax = layers.Softmax(axis=-1, name="ranking_scores")

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

            return softmax(logits)

        return masked_softmax
