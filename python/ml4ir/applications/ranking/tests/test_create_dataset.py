# Run from ml4ir/python directory as: python3 -m pytest ml4ir/applications/ranking/tests/test_create_dataset.py

import unittest
import os
import warnings
from ml4ir.base.io.local_io import LocalIO
from ml4ir.base.io.logging_utils import setup_logging
from ml4ir.applications.ranking.data.scripts.create_dataset import run_dataset_creation

warnings.filterwarnings("ignore")

ROOT_DATA_DIR = "ml4ir/applications/ranking/tests/data/csv/train"
FEATURE_CONFIG = "ml4ir/applications/ranking/tests/data/configs/feature_config.yaml"
OUTPUT_DIR = "ml4ir/applications/ranking/tests/test_output/synthetic"
LOG_DIR = "ml4ir/applications/ranking/tests/test_output_log/"


class RankingCreateDatasetTest(unittest.TestCase):
    def setUp(
        self,
        root_data_dir: str = ROOT_DATA_DIR,
        feature_config: str = FEATURE_CONFIG,
        output_dir: str = OUTPUT_DIR,
        log_dir: str = LOG_DIR,
    ):
        self.root_data_dir = root_data_dir
        self.feature_config = feature_config
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.file_io = LocalIO()

        # Set up logging
        self.file_io.make_directory(self.log_dir, clear_dir=True)
        outfile: str = os.path.join(self.log_dir, "output_log.csv")
        self.logger = setup_logging(reset=True, file_name=outfile, log_to_file=True)

    def test_synthetic_data(self):

        feature_highval = {"text_match_bool": [0, 1]}
        max_num_records = 20
        num_samples = 10

        df = run_dataset_creation(
            self.root_data_dir,
            self.output_dir,
            self.feature_config,
            feature_highval,
            max_num_records,
            num_samples,
            random_state=123,
        )
        assert len(df) == 32
        assert df.query_id.nunique() == num_samples
        assert df.num_results_calc.max() <= max_num_records
        assert "text_match_bool" in list(df.columns)
        assert list(df.text_match_bool.unique()) == [0, 1]

        df_2 = run_dataset_creation(
            self.root_data_dir,
            self.output_dir,
            self.feature_config,
            feature_highval,
            max_num_records=2,
            num_samples=10,
            random_state=123,
        )
        assert len(df_2) == 20

    def tearDown(self):
        # Delete output directory
        self.file_io.rm_dir(self.output_dir)
        self.file_io.rm_dir(self.log_dir)


if __name__ == "__main__":
    unittest.main()
