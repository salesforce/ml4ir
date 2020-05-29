from argparse import Namespace

from ml4ir.base.config.parse_args import RelevanceArgParser

from typing import List


class RankingArgParser(RelevanceArgParser):
    def define_args(self):
        super().define_args()

        self.add_argument(
            "--scoring_type",
            type=str,
            default="pointwise",
            help="Scoring technique to use. Has to be one of the scoring types in ScoringTypeKey in "
            "applications/ranking/config/keys.py",
        )

        self.add_argument(
            "--loss_type",
            type=str,
            default="listwise",
            help="Loss technique to use. Has to be one of the loss types in LossTypeKey in "
            "applications/ranking/config/keys.py",
        )

    def set_default_args(self):
        super().set_default_args()

        self.set_defaults(
            tfrecord_type="sequence_example",
            loss_key="sigmoid_cross_entropy",
            metrics_keys="['MRR', 'ACR']",
            monitor_metric="new_MRR",
            monitor_mode="max",
            max_sequence_size=25,
            group_metrics_min_queries=25,
            output_name="ranking_score",
        )


def get_args(args: List[str]) -> Namespace:
    return RankingArgParser().parse_args(args)
