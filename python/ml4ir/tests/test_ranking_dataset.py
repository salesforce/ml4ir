from ml4ir.tests.test_base import RankingTestBase
from ml4ir.data.ranking_dataset import RankingDataset
from ml4ir.config.features import FeatureConfig, parse_config
import os


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

        feature_config: FeatureConfig = parse_config(feature_config_path)

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

        return ranking_dataset

    def test_csv_dataset(self):
        """
        Test ranking dataset creation from CSV files
        using native tensorflow methods
        """

        data_dir = os.path.join(self.root_data_dir, "csv")
        feature_config_path = os.path.join(self.root_data_dir, "csv", self.feature_config_fname)

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
        feature_config_path = os.path.join(
            self.root_data_dir, "tfrecord", self.feature_config_fname
        )

        ranking_dataset = self.get_ranking_dataset(
            data_dir=data_dir, data_format="tfrecord", feature_config_path=feature_config_path
        )

        self.validate_dataset(ranking_dataset)
