import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow.keras.losses import Reduction

from ml4ir.base.config.keys import FeatureTypeKey
from ml4ir.applications.ranking.config.keys import LossKey, ScoringTypeKey
from ml4ir.applications.ranking.model.losses.loss_base import ListwiseLossBase


class SoftmaxCrossEntropy(ListwiseLossBase):

    def __init__(self,
                 loss_key: str = LossKey.SOFTMAX_CROSS_ENTROPY,
                 scoring_type: str = ScoringTypeKey.LISTWISE,
                 output_name: str = "score",
                 **kwargs):
        """
        Parameters
        ----------
        loss_key : str
            Name of the loss function as specified by LossKey
        scoring_type : str
            Type of scoring function - pointwise, pairwise, groupwise
        output_name: str
            Name of the output node for the predicted scores
        """
        super().__init__(loss_key=loss_key, scoring_type=scoring_type, output_name=output_name)

        self.final_activation_fn = layers.Softmax(axis=-1, name=output_name)

        self.loss_fn = losses.CategoricalCrossentropy()

    def call(self, inputs, y_true, y_pred, training=None):
        """
        Get the softmax cross entropy loss

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
            Scalar softmax cross entropy loss tensor

        Notes
        -----
            Uses `mask` field to exclude padded records from contributing
            to the loss
        """
        mask = tf.cast(inputs[FeatureTypeKey.MASK], y_pred.dtype)
        y_true = tf.cast(y_true, y_pred.dtype)

        return self.loss_fn(y_true=tf.math.multiply(y_true, mask),
                            y_pred=tf.math.multiply(y_pred, mask))

    def final_activation_op(self, inputs, training=None):
        """
        Get softmax activated scores on logits

        Parameters
        ----------
        inputs: dict of dict of tensors
            Dictionary of input feature tensors

        Returns
        -------
        tensor
            softmax activated scores

        Notes
        -----
            Uses `mask` field to exclude padded records from contributing
            to the softmax activation
        """
        mask = inputs[FeatureTypeKey.METADATA][FeatureTypeKey.MASK]
        logits = inputs[FeatureTypeKey.LOGITS]

        logits = tf.where(
            tf.equal(mask, tf.constant(1.0)), logits, tf.constant(tf.float32.min)
        )

        return self.final_activation_fn(logits)


class RankOneListNet(SoftmaxCrossEntropy):

    def __init__(self,
                 loss_key: str = LossKey.RANK_ONE_LISTNET,
                 scoring_type: str = ScoringTypeKey.LISTWISE,
                 output_name: str = "score",
                 **kwargs):
        """
        Parameters
        ----------
        loss_key : str
            Name of the loss function as specified by LossKey
        scoring_type : str
            Type of scoring function - pointwise, pairwise, groupwise
        output_name: str
            Name of the output node for the predicted scores
        """
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
        mask = tf.cast(inputs[FeatureTypeKey.MASK], y_pred.dtype)
        y_true = tf.cast(y_true, y_pred.dtype)
        batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)

        # Mask the padded records
        y_true = tf.gather_nd(y_true, tf.where(tf.equal(mask, tf.constant(1.0))))
        y_pred = tf.gather_nd(y_pred, tf.where(tf.equal(mask, tf.constant(1.0))))

        # Reshape the tensors so that we sum the losses from each record
        y_true = tf.expand_dims(tf.squeeze(y_true), axis=-1)
        y_pred = tf.expand_dims(tf.squeeze(y_pred), axis=-1)

        # Scale the sum of losses down by number of queries in the batch
        return tf.math.divide(self.loss_fn(y_true, y_pred), batch_size)