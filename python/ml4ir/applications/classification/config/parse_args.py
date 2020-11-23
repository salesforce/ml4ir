from argparse import Namespace

from ml4ir.base.config.keys import TFRecordTypeKey

from ml4ir.applications.classification.config.keys import LossKey, MetricKey
from ml4ir.base.config.parse_args import RelevanceArgParser

from typing import List


class ClassificationArgParser(RelevanceArgParser):
    """
    States default arguments for classification model.
    """

    def set_default_args(self):
        super().set_default_args()
        self.set_defaults(
            tfrecord_type=TFRecordTypeKey.EXAMPLE,
            loss_key=LossKey.CATEGORICAL_CROSS_ENTROPY,
            metrics_keys=[MetricKey.CATEGORICAL_ACCURACY,
                          MetricKey.TOP_5_CATEGORICAL_ACCURACY],
            monitor_metric=MetricKey.CATEGORICAL_ACCURACY,
            monitor_mode="max",
            group_metrics_min_queries=25,
            output_name="category_label",
        )


def get_args(args: List[str]) -> Namespace:
    return ClassificationArgParser().parse_args(args)
