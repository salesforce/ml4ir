# type: ignore
# TODO: Fix typing

import unittest
import tensorflow as tf
import numpy as np
import os
import random
from argparse import Namespace

from ml4ir.base.io import file_io
from ml4ir.base.io.logging_utils import setup_logging
from ml4ir.base.config.parse_args import define_args

import warnings

warnings.filterwarnings("ignore")


OUTPUT_DIR = "ml4ir/tests/test_output"
ROOT_DATA_DIR = "ml4ir/tests/data"
FEATURE_CONFIG_FNAME = "feature_config.yaml"


class RankingTestBase(unittest.TestCase):
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

        # Fix random seed values for repeatability
        tf.keras.backend.clear_session()
        np.random.seed(123)
        tf.random.set_seed(123)
        random.seed(123)

        # Setup arguments
        self.args: Namespace = define_args().parse_args([])
        self.args.models_dir = output_dir
        self.args.logs_dir = output_dir

        # Load model_config
        self.model_config = file_io.read_yaml(self.args.model_config)

        # Setup logging
        outfile: str = os.path.join(self.args.logs_dir, "output_log.csv")

        self.logger = setup_logging(reset=True, file_name=outfile, log_to_file=True)

    def tearDown(self):
        # Delete output directory
        file_io.rm_dir(self.output_dir)

        # Delete other temp directories
        file_io.rm_dir(os.path.join(self.root_data_dir, "csv", "tfrecord"))


if __name__ == "__main__":
    unittest.main()
