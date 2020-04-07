from ml4ir.tests.test_base import RankingTestBase
from ml4ir.data.ranking_dataset import RankingDataset
from ml4ir.model.ranking_model import RankingModel
from ml4ir.features.feature_config import FeatureConfig, parse_config
import os
import numpy as np


# Constants
GOLD_METRICS = {
    "old_ACR": 1.6669034,
    "new_ACR": 2.3998579,
    "old_MRR": 0.7806604,
    "new_MRR": 0.6045914,
    "perc_improv_ACR": -43.9710,
    "perc_improv_MRR": -22.5538,
    "query_count": 1408.0,
}


class RankingModelTest(RankingTestBase):
    def run_default_pipeline(self, data_dir: str, data_format: str, feature_config_path: str):
        """Train a model with the default set of args"""
        feature_config: FeatureConfig = parse_config(feature_config_path)

        self.args.metrics = ["categorical_accuracy", "MRR", "ACR"]

        ranking_dataset = RankingDataset(
            data_dir=data_dir,
            data_format=data_format,
            feature_config=feature_config,
            max_num_records=self.args.max_num_records,
            loss_key=self.args.loss,
            scoring_key=self.args.scoring,
            batch_size=self.args.batch_size,
            train_pcent_split=self.args.train_pcent_split,
            val_pcent_split=self.args.val_pcent_split,
            test_pcent_split=self.args.test_pcent_split,
            logger=self.logger,
        )
        ranking_model = RankingModel(
            model_config=self.model_config,
            loss_key=self.args.loss,
            scoring_key=self.args.scoring,
            metrics_keys=self.args.metrics,
            optimizer_key=self.args.optimizer,
            feature_config=feature_config,
            max_num_records=self.args.max_num_records,
            model_file=self.args.model_file,
            learning_rate=self.args.learning_rate,
            learning_rate_decay=self.args.learning_rate_decay,
            learning_rate_decay_steps=self.args.learning_rate_decay_steps,
            gradient_clip_value=self.args.gradient_clip_value,
            compute_intermediate_stats=self.args.compute_intermediate_stats,
            compile_keras_model=self.args.compile_keras_model,
            logger=self.logger,
        )

        overall_metrics, _ = ranking_model.evaluate(
            test_dataset=ranking_dataset.test, logs_dir=self.args.logs_dir,
        )

        return overall_metrics.to_dict()

    def test_model_training(self):
        """
        Test model training and evaluate the performance metrics
        """

        # Test model training on TFRecord SequenceExample data
        data_dir = os.path.join(self.root_data_dir, "tfrecord")
        feature_config_path = os.path.join(
            self.root_data_dir, "tfrecord", self.feature_config_fname
        )

        metrics = self.run_default_pipeline(
            data_dir=data_dir, data_format="tfrecord", feature_config_path=feature_config_path
        )

        # Compare the metrics to gold metrics
        for gold_metric_name, gold_metric_val in GOLD_METRICS.items():
            assert gold_metric_name in metrics
            assert np.isclose(metrics[gold_metric_name], gold_metric_val, rtol=0.01)
