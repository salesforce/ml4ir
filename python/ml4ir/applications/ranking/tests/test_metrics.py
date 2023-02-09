import os
import numpy as np
import unittest

import pandas as pd
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.applications.ranking.model.ranking_model import RankingModel
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.applications.ranking.tests.test_base import RankingTestBase
from ml4ir.applications.ranking.model.metrics.metrics_impl import MRR, ACR
from ml4ir.applications.ranking.model.metrics.helpers import metrics_helper

# Constants
GOLD_METRICS = {
    "query_count": 1500.0,
    "old_ACR": 1.656,
    "new_ACR": 2.410,
    "old_MRR": 0.783,
    "new_MRR": 0.597,
    "old_AuxAllFailure": 0.061,
    "old_AuxIntrinsicFailure": 0.154,
    "new_AuxAllFailure": 0.086,
    "new_AuxIntrinsicFailure": 0.153,
    "perc_improv_ACR": -45.513,
    "perc_improv_MRR": -23.760,
    "perc_improv_AuxAllFailure": -40.217,
    "perc_improv_AuxIntrinsicFailure": 0.235
}


class RankingModelTest(RankingTestBase):
    """End-to-End tests for Ranking models"""

    def run_default_pipeline(self, data_dir: str, data_format: str, feature_config_path: str):
        """Train a model with the default set of args"""
        feature_config: FeatureConfig = FeatureConfig.get_instance(
            tfrecord_type=self.args.tfrecord_type,
            feature_config_dict=self.file_io.read_yaml(feature_config_path),
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
            file_io=self.file_io,
            logger=self.logger,
        )

        ranking_model: RankingModel = self.get_ranking_model(
            loss_key=self.args.loss_key, feature_config=feature_config, metrics_keys=metrics_keys
        )

        overall_metrics, _, _ = ranking_model.evaluate(
            test_dataset=relevance_dataset.test, logs_dir=self.args.logs_dir,
        )

        return overall_metrics.to_dict()

    def test_model_training(self):
        """
        Test model training and evaluate the performance metrics
        """

        # Test model training on TFRecord SequenceExample data
        data_dir = os.path.join(self.root_data_dir, "tfrecord")
        feature_config_path = os.path.join(self.root_data_dir, "configs", self.feature_config_fname)

        metrics = self.run_default_pipeline(
            data_dir=data_dir, data_format="tfrecord", feature_config_path=feature_config_path
        )

        # Compare the metrics to gold metrics
        for gold_metric_name, gold_metric_val in GOLD_METRICS.items():
            assert gold_metric_name in metrics
            assert np.isclose(metrics[gold_metric_name], gold_metric_val, atol=0.05)


class RankingMetricsTest(unittest.TestCase):
    """Unit tests for ml4ir.applications.ranking.model.metrics"""

    def test_mrr(self):
        self.assertEquals(MRR()([[1, 0, 0], [0, 0, 1]], [[0.3, 0.6, 0.1], [0.2, 0.2, 0.3]]).numpy(),
                          0.75)

    def test_acr(self):
        self.assertEquals(ACR()([[1, 0, 0], [0, 0, 1]], [[0.3, 0.6, 0.1], [0.2, 0.2, 0.3]]).numpy(),
                          1.5)


class MetricHelperTest(unittest.TestCase):
    """Unit tests for metric helper functions"""

    def test_generate_stat_sig_based_metrics(self):
        metric = "m1"
        group_keys = ["col1", "col2"]
        metrics_dict = {}
        r1 = {
            "is_" + metric + "_lift_stat_sig": True,
            "perc_improv_" + metric: 7.14,
            "old_" + metric: 0.7,
            "new_" + metric: 0.75,
            str(group_keys): "('g1', 'g2')"
        }
        r2 = {
            "is_" + metric + "_lift_stat_sig": True,
            "perc_improv_" + metric: -7.14,
            "old_" + metric: 0.75,
            "new_" + metric: 0.7,
            str(group_keys): "('g3', 'g4')"
        }
        r3 = {
            "is_" + metric + "_lift_stat_sig": False,
            "perc_improv_" + metric: -7.14,
            "old_" + metric: 0.75,
            "new_" + metric: 0.7,
            str(group_keys): "('g5', 'g6')"
        }
        r4 = {
            "is_" + metric + "_lift_stat_sig": False,
            "perc_improv_" + metric: 1.4,
            "old_" + metric: 0.7,
            "new_" + metric: 0.71,
            str(group_keys): "('g7', 'g8')"
        }
        df = pd.DataFrame([r1, r2, r3, r4])
        stat_sig = metrics_helper.generate_stat_sig_based_metrics(df, metric, str(group_keys), metrics_dict)
        assert metrics_dict["stat_sig_" + metric + "_improved_groups"] == 1
        assert metrics_dict["stat_sig_" + metric + "_degraded_groups"] == 1
        assert metrics_dict["stat_sig_" + metric + "_group_improv_perc"] == 0