import os
import numpy as np
import random
import tensorflow as tf

from ml4ir.applications.ranking.tests.test_base import RankingTestBase
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.applications.ranking.model.ranking_model import RankingModel
from ml4ir.base.features.feature_config import parse_config, FeatureConfig


class RankingModelTest(RankingTestBase):
    def run_default_pipeline(self, data_dir: str, data_format: str, feature_config_path: str):
        """Train a model with the default set of args"""
        metrics_keys = ["MRR"]

        # Fix random seed values for repeatability
        tf.keras.backend.clear_session()
        np.random.seed(123)
        tf.random.set_seed(123)
        random.seed(123)

        feature_config: FeatureConfig = parse_config(
            tfrecord_type=self.args.tfrecord_type,
            feature_config=feature_config_path,
            logger=self.logger,
        )

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

        ranking_model.fit(dataset=relevance_dataset, num_epochs=1, models_dir=self.output_dir)

        loss = dict(
            zip(
                ranking_model.model.metrics_names,
                ranking_model.model.evaluate(relevance_dataset.test),
            )
        )["loss"]
        new_MRR = ranking_model.evaluate(
            test_dataset=relevance_dataset.test, logs_dir=self.args.logs_dir,
        )[0]["new_MRR"]

        return loss, new_MRR

    def test_csv_and_tfrecord(self):
        """
        Test model training and evaluate the performance metrics between CSV and TFRecord data
        """

        # Test model training on CSV data
        data_dir = os.path.join(self.root_data_dir, "csv")
        feature_config_path = os.path.join(self.root_data_dir, "csv", self.feature_config_fname)

        csv_loss, csv_mrr = self.run_default_pipeline(
            data_dir=data_dir, data_format="csv", feature_config_path=feature_config_path
        )

        # Check if the loss and accuracy on the test set is the same
        assert np.isclose(csv_loss, 0.58629, rtol=0.01)
        assert np.isclose(csv_mrr, 0.69231, rtol=0.01)

        # Test model training on TFRecord SequenceExample data
        data_dir = os.path.join(self.root_data_dir, "tfrecord")
        feature_config_path = os.path.join(
            self.root_data_dir, "tfrecord", self.feature_config_fname
        )

        tfrecord_loss, tfrecord_mrr = self.run_default_pipeline(
            data_dir=data_dir, data_format="tfrecord", feature_config_path=feature_config_path
        )

        # Check if the loss and accuracy on the test set is the same
        assert np.isclose(tfrecord_loss, 0.58629, rtol=0.01)
        assert np.isclose(tfrecord_mrr, 0.69231, rtol=0.01)

        # Compare CSV and TFRecord loss and accuracies
        assert np.isclose(tfrecord_loss, csv_loss, rtol=0.01)
        assert np.isclose(tfrecord_mrr, csv_mrr, rtol=0.01)
