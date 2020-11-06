import os

from ml4ir.applications.ranking.tests.test_base import RankingTestBase
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.base.features.feature_config import FeatureConfig


class RankingDatasetTest(RankingTestBase):
    def validate_dataset(self, ranking_dataset):
        # Check if the datasets were populated
        assert ranking_dataset.train is not None
        assert ranking_dataset.validation is not None
        assert ranking_dataset.test is not None

        # Check if data batch size is correct
        def check_shape(d):
            X, y = list(d)[0]
            assert y.shape[0] == self.args.batch_size
            for feature in X.keys():
                assert X[feature].shape[0] == self.args.batch_size

        map(
            check_shape,
            [
                ranking_dataset.train.take(1),
                ranking_dataset.validation.take(1),
                ranking_dataset.test.take(1),
            ],
        )

    def get_ranking_dataset(self, data_dir: str, data_format: str, feature_config_path: str):

        feature_config: FeatureConfig = FeatureConfig.get_instance(
            tfrecord_type=self.args.tfrecord_type,
            feature_config_dict=self.file_io.read_yaml(feature_config_path),
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
            file_io=self.file_io,
            logger=self.logger,
        )

        return relevance_dataset

    def test_csv_dataset(self):
        """
        Test ranking dataset creation from CSV files
        using native tensorflow methods
        """

        data_dir = os.path.join(self.root_data_dir, "csv")
        feature_config_path = os.path.join(self.root_data_dir, "configs", self.feature_config_fname)

        ranking_dataset = self.get_ranking_dataset(
            data_dir=data_dir, data_format="csv", feature_config_path=feature_config_path
        )

        self.validate_dataset(ranking_dataset)

    def test_tfrecord_dataset(self):
        """
        Test ranking dataset creation from TFRecord files
        using pandas data load methods
        """

        data_dir = os.path.join(self.root_data_dir, "tfrecord")
        feature_config_path = os.path.join(self.root_data_dir, "configs", self.feature_config_fname)

        ranking_dataset = self.get_ranking_dataset(
            data_dir=data_dir, data_format="tfrecord", feature_config_path=feature_config_path
        )

        self.validate_dataset(ranking_dataset)
