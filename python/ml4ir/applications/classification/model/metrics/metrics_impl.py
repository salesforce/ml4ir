import tensorflow as tf
from tensorflow.keras import metrics


class Top5CategoricalAccuracy(metrics.TopKCategoricalAccuracy):
    """
    Custom metric class to compute the Top K Categorical Accuracy.

    Currently a wrapper around tf.keras.metrics.TopKCategoricalAccuracy that
    squeezes one dimension.
    It maintains consistency of arguments to __init__
    """

    def __init__(
        self,
        name="top_5_categorical_accuracy",
        **kwargs
    ):
        """
        Creates a CategoricalAccuracy instance

        Parameters
        ----------
        name : str
            Name of the metric
        """
        super(Top5CategoricalAccuracy, self).__init__(name=name, k=5)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Squeeze the second dimension(axis=1) and compute top K categorical accuracy

        Parameters
        ----------
        y_true : Tensor object
            Tensor containing true class labels
            Shape : [batch_size, 1, num_classes]
        y_pred : Tensor object
            Tensor containing predicted scores for the classes
            Shape : [batch_size, 1, num_classes]
        sample_weight : dict
            Dictionary containing weights for the classes to measure weighted average metric

        Returns
        -------
        Tensor object
            Top K categorical accuracy computed on y_true and y_pred

        Notes
        -----
        Input shape is a 3 dimensional tensor of size
        (batch_size, 1, num_classes). We are squeezing
        the second dimension to follow the API of tf.keras.metrics.TopKCategoricalAccuracy

        Axis 1 of y_true and y_pred must be of size 1, otherwise `tf.squeeze`
        will throw error.
        """
        return super(Top5CategoricalAccuracy, self).update_state(
            tf.squeeze(y_true), tf.squeeze(y_pred), sample_weight=sample_weight
        )
