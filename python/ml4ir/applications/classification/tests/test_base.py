import unittest
import tensorflow as tf
import numpy as np
import os
import random
import gc
from argparse import Namespace

from ml4ir.base.io.local_io import LocalIO
from ml4ir.base.io.logging_utils import setup_logging

from ml4ir.applications.classification.config.parse_args import get_args

import warnings

warnings.filterwarnings("ignore")


OUTPUT_DIR = "ml4ir/applications/classification/tests/test_output"
ROOT_DATA_DIR = "ml4ir/applications/classification/tests/data"
FEATURE_CONFIG_FNAME = "feature_config.yaml"
MODEL_CONFIG_FNAME = "model_config.yaml"


class ClassificationTestBase(unittest.TestCase):
    """
    Setting default arguments and context for tests .../classification/tests folder.
    """
    def setUp(
        self,
        output_dir: str = OUTPUT_DIR,
        root_data_dir: str = ROOT_DATA_DIR,
        feature_config_fname: str = FEATURE_CONFIG_FNAME,
        model_config_fname: str = MODEL_CONFIG_FNAME,
    ):
        self.output_dir = output_dir
        self.root_data_dir = root_data_dir
        self.feature_config_fname = feature_config_fname
        self.model_config_fname = model_config_fname
        self.file_io = LocalIO()

        # Make temp output directory
        self.file_io.make_directory(self.output_dir, clear_dir=True)

        # Fix random seed values for repeatability
        tf.keras.backend.clear_session()
        np.random.seed(123)
        tf.random.set_seed(123)
        random.seed(123)

        # Setup arguments
        self.args: Namespace = get_args([])
        self.args.models_dir = output_dir
        self.args.logs_dir = output_dir

        # Setting small batch size less than testing data size
        self.args.batch_size = 32

        # Load feature config
        self.args.feature_config = os.path.join(self.root_data_dir, "configs", self.feature_config_fname)
        self.feature_config = self.file_io.read_yaml(self.args.feature_config)

        # Load model_config
        self.args.model_config = os.path.join(self.root_data_dir, "configs", self.model_config_fname)
        self.model_config = self.file_io.read_yaml(self.args.model_config)

        # Setup logging
        outfile: str = os.path.join(self.args.logs_dir, "output_log.csv")

        self.logger = setup_logging(reset=True, file_name=outfile, log_to_file=True)

    def tearDown(self):
        # Delete output directory
        self.file_io.rm_dir(self.output_dir)

        # Delete other temp directories
        self.file_io.rm_dir(os.path.join(self.root_data_dir, "csv", "tfrecord"))

        # Clear memory
        tf.keras.backend.clear_session()
        gc.collect()

    def get_overridden_args(self, data_format:str = "tfrecord"):
        """Overriding test default setup args from parameters."""
        data_dir = os.path.join(self.root_data_dir, data_format)
        # Fix random seed values for repeatability

        args: Namespace = self.args
        # Overriding test default setup args from parameters.
        args.data_dir = data_dir
        args.data_format = data_format
        return args


    @staticmethod
    def set_seeds():
        tf.keras.backend.clear_session()
        np.random.seed(123)
        tf.random.set_seed(123)
        random.seed(123)
        return


if __name__ == "__main__":
    unittest.main()
