from tensorflow.keras import losses
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.losses import Reduction

from ml4ir.base.config.keys import FeatureTypeKey
from ml4ir.applications.ranking.model.losses.loss_base import PointwiseLossBase


class SigmoidCrossEntropy(PointwiseLossBase):

    def __init__(self,
                 loss_key="pointwise",
                 scoring_type="",
                 output_name="score",
                 **kwargs):
        super().__init__(loss_key=loss_key, scoring_type=scoring_type, output_name=output_name)

        self.final_activation_fn = layers.Activation("sigmoid", name=output_name)

        self.loss_fn = losses.BinaryCrossentropy(reduction=Reduction.SUM_OVER_BATCH_SIZE)

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

        mask = tf.cast(inputs[FeatureTypeKey.MASK], y_pred.dtype)

        y_true = tf.gather_nd(y_true, tf.where(tf.equal(mask, tf.constant(1.0))))
        y_pred = tf.gather_nd(y_pred, tf.where(tf.equal(mask, tf.constant(1.0))))

        return self.loss_fn(y_true, y_pred)

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
        """
        mask = inputs[FeatureTypeKey.METADATA][FeatureTypeKey.MASK]
        logits = inputs[FeatureTypeKey.LOGITS]

        logits = tf.where(
            tf.equal(mask, tf.constant(1.0)), logits, tf.constant(tf.float32.min)
        )

        return self.final_activation_fn(logits)