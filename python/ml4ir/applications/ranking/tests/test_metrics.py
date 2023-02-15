import os
import numpy as np
import unittest
import pathlib

import pandas as pd
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.applications.ranking.model.ranking_model import RankingModel
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.applications.ranking.tests.test_base import RankingTestBase
from ml4ir.applications.ranking.model.metrics.metrics_impl import MRR, ACR
from ml4ir.applications.ranking.model.metrics.helpers import metrics_helper
from ml4ir.applications.ranking.config.parse_args import get_args
from ml4ir.applications.ranking.pipeline import RankingPipeline
from testfixtures import TempDirectory

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

    def train_ml4ir(self, data_dir, feature_config, model_config, eval_config, logs_dir, aux_loss):
        argv = [
            "--data_dir",
            data_dir,
            "--feature_config",
            feature_config,
            "--evaluation_config",
            eval_config,
            "--loss_type",
            "listwise",
            "--scoring_type",
            "listwise",
            "--run_id",
            "test_aux_loss",
            "--data_format",
            "tfrecord",
            "--execution_mode",
            "train_evaluate",
            "--loss_key",
            "softmax_cross_entropy",
            "--aux_loss_key",
            aux_loss,
            "--aux_loss_weight",
            "0.4",
            "--num_epochs",
            "2",
            "--model_config",
            model_config,
            "--batch_size",
            "32",
            "--logs_dir",
            logs_dir,
            "--max_sequence_size",
            "25",
            "--train_pcent_split",
            "0.7",
            "--val_pcent_split",
            "0.15",
            "--test_pcent_split",
            "0.15",
            "--early_stopping_patience",
            "25",
            "--metrics_keys",
            "MRR",
            "categorical_accuracy",
            "--monitor_metric",
            "categorical_accuracy",
        ]
        args = get_args(argv)
        rp = RankingPipeline(args=args)
        rp.run()

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

    def test_stat_sig_evaluation(self):
        # FIXME: Avoid end to end test
        """testing ml4ir stat sig computation end-to-end"""
        self.dir = pathlib.Path(__file__).parent
        self.working_dir = TempDirectory()
        self.log_dir = self.working_dir.makedir("logs")

        ROOT_DATA_DIR = "ml4ir/applications/ranking/tests/data"
        feature_config_path = os.path.join(
            ROOT_DATA_DIR, "configs", "feature_config_aux_loss.yaml"
        )
        model_config_path = os.path.join(
            ROOT_DATA_DIR, "configs", "model_config_set_rank.yaml")
        eval_config_path = os.path.join(
            ROOT_DATA_DIR, "configs", "aux_metrics_evaluation_config.yaml")

        data_dir = os.path.join(ROOT_DATA_DIR, "tfrecord")
        aux_loss = "aux_one_hot_cross_entropy"
        self.train_ml4ir(data_dir, feature_config_path,
                    model_config_path, eval_config_path, self.log_dir, aux_loss)

        results_dict = pd.read_csv(
            os.path.join(self.log_dir, "test_aux_loss", "_SUCCESS"), header=None
        ).set_index(0).to_dict()[1]

        expected_metrics = {
            'stat_sig_MRR_improved_groups': '0', 'stat_sig_MRR_degraded_groups': '5', 'stat_sig_MRR_group_improv_perc': '-11.131335125195395',
            'stat_sig_AuxIntrinsicFailure_improved_groups': '5', 'stat_sig_AuxIntrinsicFailure_degraded_groups': '0',
            'stat_sig_AuxIntrinsicFailure_group_improv_perc': '92.81582417518467',
            'stat_sig_AuxRankMF_improved_groups': '1', 'stat_sig_AuxRankMF_degraded_groups': '0',
            'stat_sig_AuxRankMF_group_improv_perc': '78.30481111176042'
        }
        for metric in expected_metrics:
            np.isclose(float(expected_metrics[metric]), float(results_dict[metric]), atol=0.0001)
        TempDirectory.cleanup_all()

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