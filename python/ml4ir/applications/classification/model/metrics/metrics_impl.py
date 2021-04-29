import tensorflow as tf
from tensorflow.keras import metrics

from ml4ir.base.model.metrics.metrics_impl import MetricState
from ml4ir.base.features.feature_config import FeatureConfig

from typing import Optional, Dict


class CategoricalAccuracy(metrics.CategoricalAccuracy):
    """
    Custom metric class to compute the Categorical Accuracy.

    Currently just a wrapper around tf.keras.metrics.CategoricalAccuracy
    to maintain consistency of arguments to __init__
    """

    def __init__(
        self,
        feature_config: FeatureConfig,
        metadata_features: Dict,
        name="categorical_accuracy",
        state=MetricState.NEW,
        **kwargs
    ):
        """
        Creates a CategoricalAccuracy instance

        Parameters
        ----------
        feature_config : FeatureConfig object
            FeatureConfig object that defines the configuration for each model
            feature
        metadata_features : dict
            Dictionary of metadata feature tensors that can be used to compute
            custom metrics
        name : str
            Name of the metric
        state : {"new", "old"}
            State of the metric
        """
        super(CategoricalAccuracy, self).__init__(name=name)


class Top5CategoricalAccuracy(metrics.TopKCategoricalAccuracy):
    """
    Custom metric class to compute the Top K Categorical Accuracy.

    Currently a wrapper around tf.keras.metrics.TopKCategoricalAccuracy that
    squeezes one dimension.
    It maintains consistency of arguments to __init__
    """

    def __init__(
        self,
        feature_config: Optional[FeatureConfig] = None,
        metadata_features: Dict = {},
        name="top_5_categorical_accuracy",
        state=MetricState.NEW,
        **kwargs
    ):
        """
        Creates a CategoricalAccuracy instance

        Parameters
        ----------
        feature_config : FeatureConfig object
            FeatureConfig object that defines the configuration for each model
            feature
        metadata_features : dict
            Dictionary of metadata feature tensors that can be used to compute
            custom metrics
        name : str
            Name of the metric
        state : {"new", "old"}
            State of the metric
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
