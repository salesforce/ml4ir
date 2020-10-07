from tensorflow.keras.metrics import Metric

from ml4ir.applications.ranking.config.keys import MetricKey
from ml4ir.applications.ranking.model.metrics.metrics_impl import MRR, ACR, CategoricalAccuracy, Top5CategoricalAccuracy


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
    if metric_key == MetricKey.MRR:
        return MRR
    elif metric_key == MetricKey.ACR:
        return ACR
    elif metric_key == MetricKey.NDCG:
        raise NotImplementedError
    elif metric_key == MetricKey.CATEGORICAL_ACCURACY:
        return CategoricalAccuracy
    elif metric_key == MetricKey.TOP_5_CATEGORICAL_ACCURACY:
        return Top5CategoricalAccuracy
    else:
        return metric_key
