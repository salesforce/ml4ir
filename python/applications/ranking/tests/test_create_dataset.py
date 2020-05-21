# Run from ml4ir/python directory as: python3 -m pytest applications/ranking/tests/test_create_dataset.py

import unittest
import os
import random
from argparse import Namespace
import warnings
warnings.filterwarnings("ignore")

from ml4ir.io import file_io
from ml4ir.io.logging_utils import setup_logging
from applications.ranking.data.scripts.create_dataset import create_dataset


OUTPUT_DIR = "applications/ranking/tests/test_output/synthetic"
ROOT_DATA_DIR = "applications/ranking/tests/data/csv/synthetic"
FEATURE_CONFIG_FNAME = "feature_config.yaml"


class RankingCreateDatasetTest(unittest.TestCase):
    def setUp(
            self,
            output_dir: str = OUTPUT_DIR,
            root_data_dir: str = ROOT_DATA_DIR,
            feature_config_fname: str = FEATURE_CONFIG_FNAME,
    ):
        self.output_dir = output_dir
        self.root_data_dir = root_data_dir
        self.feature_config_fname = feature_config_fname

        # Make temp output directory
        file_io.make_directory(self.output_dir, clear_dir=True)

        # Setup logging
        outfile: str = os.path.join(self.output_dir, "output_log.csv")
        self.logger = setup_logging(reset=True, file_name=outfile, log_to_file=True)

    def test_synthetic_data(self):

        feature_config = os.path.join(self.root_data_dir, self.feature_config_fname)
        max_num_records = 10
        num_samples = 10
        random_state = 123
        df = create_dataset(self.root_data_dir, self.output_dir, feature_config, max_num_records, num_samples, random_state)
        assert len(df) == 37
        assert df.query_str.nunique() == num_samples
        assert df.num_sequences.max() <= max_num_records
        assert 'high_val_feature_1' in list(df.columns)
        assert list(df.high_val_feature_1.unique()) == [0,1]


#    def tearDown(self):
        # Delete output directory
#        file_io.rm_dir(self.output_dir)

if __name__ == "__main__":
    unittest.main()
