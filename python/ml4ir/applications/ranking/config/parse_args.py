from argparse import Namespace

from ml4ir.base.config.parse_args import RelevanceArgParser
from ml4ir.applications.ranking.config.keys import ScoringTypeKey, LossTypeKey, LossKey, MetricKey

from typing import List


class RankingArgParser(RelevanceArgParser):
    def define_args(self):
        super().define_args()

        self.add_argument(
            "--scoring_type",
            type=str,
            choices=LossTypeKey.get_all_keys(),
            default=LossTypeKey.POINTWISE,
            help="Scoring technique to use. Has to be one of the scoring types in ScoringTypeKey in "
                 "applications/ranking/config/keys.py",
        )

        self.add_argument(
            "--loss_type",
            type=str,
            choices=ScoringTypeKey.get_all_keys(),
            default=ScoringTypeKey.LISTWISE,
            help="Loss technique to use. Has to be one of the loss types in LossTypeKey in "
                 "applications/ranking/config/keys.py",
        )

    def set_default_args(self):
        super().set_default_args()

        self.set_defaults(
            tfrecord_type="sequence_example",
            loss_key="sigmoid_cross_entropy",
            metrics_keys=[MetricKey.MRR, MetricKey.ACR],
            monitor_metric=MetricKey.MRR,
            monitor_mode="max",
            max_sequence_size=25,
            group_metrics_min_queries=25,
            output_name="ranking_score",
            aux_loss_key=LossKey.AUX_SOFTMAX_CROSS_ENTROPY,
            aux_metrics_keys=[MetricKey.RANK_MATCH_FAILURE]
        )


def get_args(args: List[str]) -> Namespace:
    return RankingArgParser().parse_args(args)