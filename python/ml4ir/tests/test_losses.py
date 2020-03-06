from ml4ir.tests.test_base import RankingTestBase
from ml4ir.data.ranking_dataset import RankingDataset
from ml4ir.model.ranking_model import RankingModel
from ml4ir.config.features import FeatureConfig, parse_config
import os
import numpy as np


class RankingModelTest(RankingTestBase):
    def run_default_pipeline(self, loss_key: str):
        """Train a model with the default set of args"""
        feature_config_path = os.path.join(
            self.root_data_dir, "tfrecord", self.feature_config_fname
        )
        feature_config: FeatureConfig = parse_config(feature_config_path)
        data_dir = os.path.join(self.root_data_dir, "tfrecord")
        data_format = "tfrecord"

        self.args.metrics = ["MRR"]

        ranking_dataset = RankingDataset(
            data_dir=data_dir,
            data_format=data_format,
            feature_config=feature_config,
            max_num_records=self.args.max_num_records,
            loss_key=loss_key,
            scoring_key=self.args.scoring,
            batch_size=self.args.batch_size,
            train_pcent_split=self.args.train_pcent_split,
            val_pcent_split=self.args.val_pcent_split,
            test_pcent_split=self.args.test_pcent_split,
            logger=self.logger,
        )
        ranking_model = RankingModel(
            model_config=self.model_config,
            loss_key=loss_key,
            scoring_key=self.args.scoring,
            metrics_keys=self.args.metrics,
            optimizer_key=self.args.optimizer,
            feature_config=feature_config,
            max_num_records=self.args.max_num_records,
            model_file=self.args.model_file,
            learning_rate=self.args.learning_rate,
            learning_rate_decay=self.args.learning_rate_decay,
            compute_intermediate_stats=self.args.compute_intermediate_stats,
            logger=self.logger,
        )

        metrics = ranking_model.evaluate(
            ranking_dataset.test, models_dir=self.args.models_dir, logs_dir=self.args.logs_dir
        )

        return metrics["loss"]

    def test_sigmoid_cross_entropy(self):
        """
        Test model training and evaluate Sigmoid CrossEntropy loss
        """

        loss = self.run_default_pipeline(loss_key="sigmoid_cross_entropy")

        assert np.isclose(loss, 0.68705, rtol=0.05)

    def test_rank_one_listnet(self):
        """
        Test model training and evaluate Rank One ListNet loss
        """

        loss = self.run_default_pipeline(loss_key="rank_one_listnet")

        assert np.isclose(loss, 2.00879, rtol=0.05)
