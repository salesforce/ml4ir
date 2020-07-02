import os
import numpy as np

from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.applications.ranking.model.ranking_model import RankingModel
from ml4ir.base.features.feature_config import FeatureConfig, parse_config
from ml4ir.applications.ranking.tests.test_base import RankingTestBase


# Constants
GOLD_METRICS = {
    "query_count": 1408.0,
    "old_ACR": 1.65056,
    "new_ACR": 2.19460,
    "old_MRR": 0.78588,
    "new_MRR": 0.64859,
    "mean_old_name_match_failure_all": 0.06392,
    "mean_new_name_match_failure_all": 0.09517,
    "mean_old_name_match_failure_any": 0.11221,
    "mean_new_name_match_failure_any": 0.19744,
    "mean_old_name_match_failure_all_rank": 0.148437,
    "mean_new_name_match_failure_all_rank": 0.217329,
    "mean_old_name_match_failure_any_rank": 0.352272,
    "mean_new_name_match_failure_any_rank": 0.684659,
    "mean_old_name_match_failure_any_count": 0.166903,
    "mean_new_name_match_failure_any_count": 0.311789,
    "mean_old_name_match_failure_any_fraction": 0.088986,
    "mean_new_name_match_failure_any_fraction": 0.148283,
    "perc_improv_ACR": -32.960413,
    "perc_improv_MRR": -17.469067,
    "perc_improv_mean_name_match_failure_all": -0.488888,
    "perc_improv_mean_name_match_failure_any": -0.759493,
    "perc_improv_mean_name_match_failure_all_rank": -0.464114,
    "perc_improv_mean_name_match_failure_any_rank": -0.943548,
    "perc_improv_mean_name_match_failure_any_count": -0.868085,
    "perc_improv_mean_name_match_failure_any_fraction": -0.666356,
}


class RankingModelTest(RankingTestBase):
    def run_default_pipeline(self, data_dir: str, data_format: str, feature_config_path: str):
        """Train a model with the default set of args"""
        feature_config: FeatureConfig = parse_config(
            tfrecord_type=self.args.tfrecord_type,
            feature_config=feature_config_path,
            logger=self.logger,
        )
        data_dir = os.path.join(self.root_data_dir, "tfrecord")
        data_format = "tfrecord"

        metrics_keys = ["categorical_accuracy", "MRR", "ACR"]

        relevance_dataset = RelevanceDataset(
            data_dir=data_dir,
            data_format=data_format,
            feature_config=feature_config,
            tfrecord_type=self.args.tfrecord_type,
            max_sequence_size=self.args.max_sequence_size,
            batch_size=self.args.batch_size,
            preprocessing_keys_to_fns={},
            train_pcent_split=self.args.train_pcent_split,
            val_pcent_split=self.args.val_pcent_split,
            test_pcent_split=self.args.test_pcent_split,
            use_part_files=self.args.use_part_files,
            parse_tfrecord=True,
            logger=self.logger,
        )

        ranking_model: RankingModel = self.get_ranking_model(
            loss_key=self.args.loss_key, feature_config=feature_config, metrics_keys=metrics_keys
        )

        overall_metrics, _ = ranking_model.evaluate(
            test_dataset=relevance_dataset.test, logs_dir=self.args.logs_dir,
        )

        return overall_metrics.to_dict()

    def test_model_training(self):
        """
        Test model training and evaluate the performance metrics
        """

        # Test model training on TFRecord SequenceExample data
        data_dir = os.path.join(self.root_data_dir, "tfrecord")
        feature_config_path = os.path.join(self.root_data_dir, "config", self.feature_config_fname)

        metrics = self.run_default_pipeline(
            data_dir=data_dir, data_format="tfrecord", feature_config_path=feature_config_path
        )

        # Compare the metrics to gold metrics
        for gold_metric_name, gold_metric_val in GOLD_METRICS.items():
            assert gold_metric_name in metrics
            assert np.isclose(metrics[gold_metric_name], gold_metric_val, rtol=0.01)
