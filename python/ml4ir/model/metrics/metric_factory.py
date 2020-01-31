from ml4ir.config.keys import MetricKey

# import tensorflow as tf


def get_metric(metric_key):
    if metric_key == MetricKey.MRR:
        raise NotImplementedError
    elif metric_key == MetricKey.ACR:
        raise NotImplementedError
    elif metric_key == MetricKey.NDCG:
        raise NotImplementedError
    elif metric_key == MetricKey.CATEGORICAL_ACCURACY:
        return "categorical_accuracy"
    else:
        raise NotImplementedError
