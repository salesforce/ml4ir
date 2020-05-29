from typing import Type, List, Union
from tensorflow.keras.metrics import Metric

from ml4ir.base.features.feature_config import FeatureConfig

from typing import Dict


class MetricState:
    OLD = "old"
    NEW = "new"


def get_metrics_impl(
    metrics: List[Union[str, Type[Metric]]],
    feature_config: FeatureConfig,
    metadata_features: Dict,
    **kwargs
) -> List[Union[Metric, str]]:

    metrics_impl: List[Union[Metric, str]] = list()

    for metric in metrics:
        if isinstance(metric, str):
            # If metric is specified as a string, then do nothing
            metrics_impl.append(metric)
        else:
            # If metric is a class of type Metric
            try:
                metrics_impl.extend(
                    [
                        metric(
                            state=MetricState.OLD,
                            feature_config=feature_config,
                            metadata_features=metadata_features,
                            **kwargs
                        ),
                        metric(
                            state=MetricState.NEW,
                            feature_config=feature_config,
                            metadata_features=metadata_features,
                            **kwargs
                        ),
                    ]
                )
            except TypeError:
                metrics_impl.append(metric())

    return metrics_impl
