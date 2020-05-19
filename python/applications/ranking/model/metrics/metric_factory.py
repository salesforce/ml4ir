# type: ignore
# TODO: Fix typing

from applications.ranking.config.keys import MetricKey
from applications.ranking.model.metrics.metrics_impl import MRR, ACR, CategoricalAccuracy
from tensorflow.keras.metrics import Metric


def get_metric(metric_key: str) -> Metric:
    if metric_key == MetricKey.MRR:
        return MRR
    elif metric_key == MetricKey.ACR:
        return ACR
    elif metric_key == MetricKey.NDCG:
        raise NotImplementedError
    elif metric_key == MetricKey.CATEGORICAL_ACCURACY:
        return CategoricalAccuracy
    else:
        return metric_key
