from ml4ir.config.keys import MetricKey
from ml4ir.model.metrics.metrics_impl import MRR, ACR, CategoricalAccuracy
from typing import Type, List
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
        raise NotImplementedError


def get_metric_impl(metric: Type[Metric], **kwargs) -> List[Metric]:
    return [
        metric(rerank=False, **kwargs),
        metric(rerank=True, **kwargs),
    ]
