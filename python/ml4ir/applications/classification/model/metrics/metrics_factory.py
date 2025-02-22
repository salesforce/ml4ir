import tensorflow as tf
from tensorflow.keras.metrics import Metric

from ml4ir.applications.classification.config.keys import MetricKey
from ml4ir.applications.classification.model.metrics.metrics_impl import Top5CategoricalAccuracy


def get_metric(metric_key: str) -> Metric:
    """
    Factory method to get Metric class

    Parameters
    ----------
    metric_key : str
        Name of the metric class to retrieve

    Returns
    -------
    Metric class
        Class defining the metric computation logic
    """
    if metric_key == MetricKey.TOP_5_CATEGORICAL_ACCURACY:
        return Top5CategoricalAccuracy(name="top_5_categorical_accuracy")
    elif metric_key == MetricKey.CATEGORICAL_ACCURACY:
        return tf.keras.metrics.CategoricalAccuracy(name=metric_key)
    else:
        # For out-of-the-box keras metric classes
        return tf.keras.metrics.get(metric_key)
