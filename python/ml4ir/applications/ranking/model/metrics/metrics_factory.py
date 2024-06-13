import tensorflow as tf
from tensorflow.keras.metrics import Metric

from ml4ir.applications.ranking.config.keys import MetricKey
from ml4ir.applications.ranking.model.metrics.metrics_impl import MRR, SegmentMRR, MacroMRR, ACR, NDCG
from ml4ir.applications.ranking.model.metrics.aux_metrics_impl import RankMatchFailure

from typing import Optional, List


def get_metric(metric_key: str, segments: Optional[List[str]] = None) -> Metric:
    """
    Factory method to get Metric class

    Parameters
    ----------
    metric_key : str
        Name of the metric class to retrieve
    segments: list of strings
        List of segment names to be used to compute group metrics

    Returns
    -------
    Metric class
        Class defining the metric computation logic
    """
    if metric_key == MetricKey.MRR:
        return MRR(name="MRR")
    if metric_key == MetricKey.SEGMENT_MRR:
        if not segments:
            raise ValueError("segments must be specified in the evaluation config to use SegmentMRR")
        return SegmentMRR(name="SegmentMRR", segments=segments)
    if metric_key == MetricKey.MACRO_MRR:
        if not segments:
            raise ValueError("segments must be specified in the evaluation config to use MacroMRR")
        return MacroMRR(name="MacroMRR", segments=segments)
    elif metric_key == MetricKey.ACR:
        return ACR(name="ACR")
    elif metric_key == MetricKey.RANK_MATCH_FAILURE:
        return RankMatchFailure(name="AuxRankMF")
    elif metric_key == MetricKey.NDCG:
        return NDCG(name="NDCG")
    elif metric_key == MetricKey.CATEGORICAL_ACCURACY:
        return tf.keras.metrics.CategoricalAccuracy(name=metric_key)
    else:
        # For out-of-the-box keras metric classes
        return tf.keras.metrics.get(metric_key)