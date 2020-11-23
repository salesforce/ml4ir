import os
import numpy as np

from ml4ir.applications.ranking.tests.test_base import RankingTestBase
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.applications.ranking.model.ranking_model import RankingModel
from ml4ir.base.features.feature_config import FeatureConfig


class RankingModelTest(RankingTestBase):
    def run_default_pipeline(self, loss_key: str):
        """Train a model with the default set of args"""
        feature_config_path = os.path.join(self.root_data_dir, "configs", self.feature_config_fname)
        feature_config: FeatureConfig = FeatureConfig.get_instance(
            tfrecord_type=self.args.tfrecord_type,
            feature_config_dict=self.file_io.read_yaml(feature_config_path),
            logger=self.logger,
        )
        data_dir = os.path.join(self.root_data_dir, "tfrecord")
        data_format = "tfrecord"

        metrics_keys = ["MRR"]

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
            loss_key=loss_key, feature_config=feature_config, metrics_keys=metrics_keys
        )

        metrics = ranking_model.model.evaluate(relevance_dataset.test)
        return dict(zip(ranking_model.model.metrics_names, metrics))["loss"]

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
