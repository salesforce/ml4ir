import unittest
import tensorflow as tf
import numpy as np
import os
import random
import gc
from argparse import Namespace
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Optimizer

from ml4ir.base.model.relevance_model import RelevanceModel
from ml4ir.base.model.losses.loss_base import RelevanceLossBase
from ml4ir.base.model.scoring.scoring_model import ScorerBase, RelevanceScorer
from ml4ir.base.model.scoring.interaction_model import InteractionModel, UnivariateInteractionModel
from ml4ir.base.model.optimizers.optimizer import get_optimizer
from ml4ir.base.io.local_io import LocalIO
from ml4ir.base.io.logging_utils import setup_logging
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.config.keys import ArchitectureKey
from ml4ir.applications.ranking.model.ranking_model import RankingModel, LinearRankingModel
from ml4ir.applications.ranking.model.losses import loss_factory
from ml4ir.applications.ranking.model.metrics import metric_factory
from ml4ir.applications.ranking.config.parse_args import get_args

import warnings
from typing import Union, List, Type

warnings.filterwarnings("ignore")


OUTPUT_DIR = "ml4ir/applications/ranking/tests/test_output"
ROOT_DATA_DIR = "ml4ir/applications/ranking/tests/data"
FEATURE_CONFIG_FNAME = "feature_config.yaml"


class RelevanceTestBase(unittest.TestCase):
    """
    This is the base test class for the common relevance code under ml4ir/base/

    Inherit this class to define tests which need the default pipeline args and configs.
    """
    def setUp(
        self,
        output_dir: str = OUTPUT_DIR,
        root_data_dir: str = ROOT_DATA_DIR,
        feature_config_fname: str = FEATURE_CONFIG_FNAME,
    ):
        self.output_dir = output_dir
        self.root_data_dir = root_data_dir
        self.feature_config_fname = feature_config_fname
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

        self.load_model_config(self.args.model_config)

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

    def load_model_config(self, model_config_path:str):
        """Load the model config dictionary"""
        self.model_config = self.file_io.read_yaml(model_config_path)

if __name__ == "__main__":
    unittest.main()
