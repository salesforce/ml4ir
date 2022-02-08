import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow.keras.losses import Reduction

from ml4ir.base.config.keys import FeatureTypeKey
from ml4ir.applications.ranking.model.losses.loss_base import ListwiseLossBase


class SoftmaxCrossEntropy(ListwiseLossBase):

    def __init__(self, loss_key="pointwise", scoring_type="", output_name="score", **kwargs):
        super().__init__(loss_key=loss_key, scoring_type=scoring_type, output_name=output_name)

        self.final_activation_fn = layers.Softmax(axis=-1, name=output_name)

        self.loss_fn = losses.CategoricalCrossentropy()

    def call(self, inputs, y_true, y_pred, training=None):
        """
        Get the sigmoid cross entropy loss
        Additionally can pass in record positions to handle positional bias

        Parameters
        ----------
        inputs: dict of dict of tensors
            Dictionary of input feature tensors
        y_true: tensor
            True labels
        y_pred: tensor
            Predicted scores
        training: boolean
            Boolean indicating whether the layer is being used in training mode

        Returns
        -------
        tensor
            Scalar sigmoid cross entropy loss tensor

        Notes
        -----
            Uses `mask` field to exclude padded records from contributing
            to the loss
        """
        mask = inputs[FeatureTypeKey.MASK]

        return self.loss_fn(y_true, tf.math.multiply(y_pred, mask))

    def final_activation_op(self, inputs, training=None):
        """
        Get sigmoid activated scores on logits

        Parameters
        ----------
        inputs: dict of dict of tensors
            Dictionary of input feature tensors

        Returns
        -------
        tensor
            sigmoid activated scores

        Notes
        -----
            Uses `mask` field to exclude padded records from contributing
            to the softmax activation
        """
        mask = inputs[FeatureTypeKey.MASK]
        logits = inputs[FeatureTypeKey.LOGITS]

        # NOTE:
        # Tried to manually compute softmax with tf operations,
        # but tf.keras.layers.Softmax() is more stable when working with
        # cross_entropy layers
        logits = tf.where(
            tf.equal(mask, tf.constant(1.0)), logits, tf.constant(tf.float32.min)
        )

        return self.final_activation_fn(logits)


class RankOneListNet(SoftmaxCrossEntropy):

    def __init__(self, loss_key="pointwise", scoring_type="", output_name="score", **kwargs):
        super().__init__(loss_key=loss_key, scoring_type=scoring_type, output_name=output_name)

        self.loss_fn = losses.BinaryCrossentropy(reduction=Reduction.SUM)

    def call(self, inputs, y_true, y_pred, training=None):
        """
        Define a masked rank 1 ListNet loss.
        This loss is useful for multi-label classification when we have multiple
        click labels per document. This is because the loss breaks down the comparison
        between y_pred and y_true into individual binary assessments.

        Ref -> https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf

        Parameters
        ----------
        inputs: dict of dict of tensors
            Dictionary of input feature tensors
        y_true: tensor
            True labels
        y_pred: tensor
            Predicted scores
        training: boolean
            Boolean indicating whether the layer is being used in training mode

        Returns
        -------
        tensor
            Scalar sigmoid cross entropy loss tensor

        Notes
        -----
            Uses `mask` field to exclude padded records from contributing
            to the loss
        """
        mask = inputs[FeatureTypeKey.MASK]
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)

        # Mask the padded records
        y_true = tf.gather_nd(y_true, tf.where(tf.equal(mask, tf.constant(1.0))))
        y_pred = tf.gather_nd(y_pred, tf.where(tf.equal(mask, tf.constant(1.0))))

        # Reshape the tensors so that we sum the losses from each record
        y_true = tf.expand_dims(tf.squeeze(y_true), axis=-1)
        y_pred = tf.expand_dims(tf.squeeze(y_pred), axis=-1)

        # Scale the sum of losses down by number of queries in the batch
        return tf.math.divide(self.loss_fn(y_true, y_pred), batch_size)
