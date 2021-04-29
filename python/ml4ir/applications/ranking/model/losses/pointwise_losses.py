from tensorflow.keras import losses
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.losses import Reduction

from ml4ir.applications.ranking.model.losses.loss_base import PointwiseLossBase


class SigmoidCrossEntropy(PointwiseLossBase):
    def get_loss_fn(self, **kwargs):
        """
        Define a sigmoid cross entropy loss
        Additionally can pass in record positions to handle positional bias

        Returns
        -------
        function
            Function to compute sigmoid cross entropy loss

        Notes
        -----
            Uses `mask` field to exclude padded records from contributing
            to the loss
        """
        bce = losses.BinaryCrossentropy(reduction=Reduction.SUM_OVER_BATCH_SIZE)
        mask = kwargs.get("mask")

        def _loss_fn(y_true, y_pred):
            """
            Shapes
            ------
            y_true : [batch_size, num_classes, 1]
            y_pred : [batch_size, num_classes, 1]
            mask : [batch_size, num_classes, 1]
            """
            # Mask the padded records
            y_true = tf.gather_nd(y_true, tf.where(tf.equal(mask, tf.constant(1.0))))
            y_pred = tf.gather_nd(y_pred, tf.where(tf.equal(mask, tf.constant(1.0))))

            return bce(y_true, y_pred)

        return _loss_fn

    def get_final_activation_op(self, output_name):
        """
        Define a sigmoid activation function

        Parameters
        ----------
        output_name : str
            Name of the output score node in the tensorflow model

        Returns
        -------
        function
            Function to apply sigmoid activation to the output score
        """
        return lambda logits, mask: layers.Activation("sigmoid", name=output_name)(logits)
