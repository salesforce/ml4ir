from tensorflow.keras.metrics import Metric

from ml4ir.applications.ranking.config.keys import MetricKey
from ml4ir.applications.ranking.model.metrics.metrics_impl import MRR, ACR, CategoricalAccuracy, Top_K_Categorical_Accuracy


def get_metric(metric_key: str) -> Metric:
    if metric_key == MetricKey.MRR:
        return MRR
    elif metric_key == MetricKey.ACR:
        return ACR
    elif metric_key == MetricKey.NDCG:
        raise NotImplementedError
    elif metric_key == MetricKey.CATEGORICAL_ACCURACY:
        return CategoricalAccuracy
    elif metric_key == MetricKey.TOP_K_CATEGORICAL_ACCURACY:
        return Top_K_Categorical_Accuracy
    else:
        return metric_key
