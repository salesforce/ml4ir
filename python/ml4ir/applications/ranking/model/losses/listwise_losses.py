import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow.keras.losses import Reduction

from ml4ir.applications.ranking.model.losses.loss_base import ListwiseLossBase


class RankOneListNet(ListwiseLossBase):
    def get_loss_fn(self, **kwargs):
        """
        Define a masked rank 1 ListNet loss
        Additionally can pass in record positions to handle positional bias

        Returns
        -------
        function
            Function to compute top 1 listnet loss

        Notes
        -----
            Uses `mask` field to exclude padded records from contributing
            to the loss
        """
        bce = losses.BinaryCrossentropy(reduction=Reduction.SUM)
        mask = kwargs.get("mask")

        def _loss_fn(y_true, y_pred):
            """
            Shapes
            ------
            y_true : [batch_size, num_classes, 1]
            y_pred : [batch_size, num_classes, 1]
            mask : [batch_size, num_classes, 1]
            """
            batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)

            # Mask the padded records
            y_true = tf.gather_nd(y_true, tf.where(tf.equal(mask, tf.constant(1.0))))
            y_pred = tf.gather_nd(y_pred, tf.where(tf.equal(mask, tf.constant(1.0))))

            # Reshape the tensors so that we sum the losses from each record
            y_true = tf.expand_dims(tf.squeeze(y_true), axis=-1)
            y_pred = tf.expand_dims(tf.squeeze(y_pred), axis=-1)

            # Scale the sum of losses down by number of queries in the batch
            return tf.math.divide(bce(y_true, y_pred), batch_size)

        return _loss_fn

    def get_final_activation_op(self, output_name):
        """
        Define a masked softmax activation function

        Parameters
        ----------
        output_name : str
            Name of the output to apply softmax activation on

        Returns
        -------
        function
            Function to compute masked softmax

        Notes
        -----
            Uses `mask` field to exclude padded records from contributing
            to the softmax activation
        """
        softmax_op = layers.Softmax(axis=-1, name=output_name)

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
