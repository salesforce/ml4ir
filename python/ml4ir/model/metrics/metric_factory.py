from ml4ir.config.keys import MetricKey
from ml4ir.model.metrics.metrics_impl import MRR, ACR, MeanMetricWrapper
from typing import Union, Type
from tensorflow.keras.metrics import Metric


def get_metric(metric_key):
    if metric_key == MetricKey.MRR:
        return MRR
    elif metric_key == MetricKey.ACR:
        return ACR
    elif metric_key == MetricKey.NDCG:
        raise NotImplementedError
    elif metric_key == MetricKey.CATEGORICAL_ACCURACY:
        return "categorical_accuracy"
    else:
        raise NotImplementedError


def get_metric_impl(metric: Union[str, Type[MeanMetricWrapper]], **kwargs) -> Union[str, Metric]:
    if isinstance(metric, str):
        return metric
    else:
        return metric(**kwargs)
