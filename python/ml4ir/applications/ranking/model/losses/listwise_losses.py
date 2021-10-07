import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow.keras.losses import Reduction

from ml4ir.applications.ranking.model.losses.loss_base import ListwiseLossBase


class SoftmaxCrossEntropy(ListwiseLossBase):
    def get_loss_fn(self, **kwargs):
        """
        Define a softmax cross entropy loss

        Returns
        -------
        function
            Function to compute softmax cross entropy loss

        Notes
        -----
            Uses `mask` field to exclude padded records from contributing
            to the loss
        """
        cce = losses.CategoricalCrossentropy()
        mask = kwargs.get("mask")

        def _loss_fn(y_true, y_pred):
            """
            Shapes
            ------
            y_true : [batch_size, num_classes]
            y_pred : [batch_size, num_classes]
            mask : [batch_size, num_classes]
            """
            return cce(y_true, tf.math.multiply(y_pred, mask))

        return _loss_fn

    def get_final_activation_op(self, output_name):
        """
        Define a masked softmax activation function.
        This is one of the simplest and most effective loss functions
        for training ranking models with single click label.

        Ref -> https://dl.acm.org/doi/10.1145/3341981.3344221

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


class RankOneListNet(SoftmaxCrossEntropy):
    def get_loss_fn(self, **kwargs):
        """
        Define a masked rank 1 ListNet loss.
        This loss is useful for multi-label classification when we have multiple
        click labels per document. This is because the loss breaks down the comparison
        between y_pred and y_true into individual binary assessments.

        Ref -> https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf

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
            y_true : [batch_size, num_classes]
            y_pred : [batch_size, num_classes]
            mask : [batch_size, num_classes]
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