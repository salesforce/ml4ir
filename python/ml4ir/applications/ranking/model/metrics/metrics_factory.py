from tensorflow.keras.metrics import Metric

from ml4ir.applications.ranking.config.keys import MetricKey
from ml4ir.applications.ranking.model.metrics.metrics_impl import MRR, ACR
from ml4ir.applications.ranking.model.metrics.aux_metrics_impl import RankMatchFailure


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
        return MRR(name="MRR")
    elif metric_key == MetricKey.ACR:
        return ACR(name="ACR")
    elif metric_key == MetricKey.RANK_MATCH_FAILURE:
        return RankMatchFailure(name="AuxRankMF")
    elif metric_key == MetricKey.NDCG:
        raise NotImplementedError
    else:
        return metric_key