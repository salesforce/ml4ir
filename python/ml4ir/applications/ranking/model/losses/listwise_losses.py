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


class AuxiliaryOneHotCrossEntropy(SoftmaxCrossEntropy):
    """
    Compute the one-hot softmax cross entropy loss on the auxiliary label
    """

    def call(self, inputs, y_true, y_pred, training=None):
        """
        Get the softmax cross entropy loss on the auxiliary label

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
        - Uses `mask` field to exclude padded records from contributing
        to the loss
        - Queries with ties in the highest scores would have multiple one's in the 1-hot vector.
        - Queries with all zeros for y_true would have all ones as their 1-hot vector.
        - A simple remedy is to scale down the loss by the number of ties per query.
        """
        mask = tf.cast(inputs[FeatureTypeKey.MASK], y_pred.dtype)
        y_true = tf.cast(y_true, y_pred.dtype)

        # Convert y_true to 1-hot labels
        y_true_one_hot = tf.equal(y_true, tf.expand_dims(tf.math.reduce_max(y_true, axis=1), axis=1))
        y_true_one_hot = tf.cast(y_true_one_hot, dtype=y_pred.dtype)

        # Scale down the loss of a query by 1 / (number of ties)
        sample_weight = tf.math.divide(tf.constant(1, dtype=tf.float32), tf.reduce_sum(y_true_one_hot, axis=1))

        return self.loss_fn(y_true=tf.math.multiply(y_true_one_hot, mask),
                            y_pred=tf.math.multiply(y_pred, mask),
                            sample_weight=sample_weight)


class AuxiliarySoftmaxCrossEntropy(SoftmaxCrossEntropy):
    """
    Compute the softmax cross entropy loss on auxiliary label
    FIXME: Add difference between variants of losses
    """

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
            Scalar basic cross entropy loss tensor

        Notes
        -----
        - Uses `mask` field to exclude padded records from contributing to the loss
        """
        mask = tf.cast(inputs[FeatureTypeKey.MASK], y_pred.dtype)
        y_true = tf.cast(y_true, y_pred.dtype)

        # Convert y_true to a probability distribution
        y_true_softmax = self.final_activation_op({
            FeatureTypeKey.METADATA: {
                FeatureTypeKey.MASK: mask
            },
            FeatureTypeKey.LOGITS: y_true
        }, training=training)

        return self.loss_fn(y_true=tf.math.multiply(y_true_softmax, mask),
                            y_pred=tf.math.multiply(y_pred, mask))


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