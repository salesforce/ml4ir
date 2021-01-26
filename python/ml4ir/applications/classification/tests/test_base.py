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
from ml4ir.applications.classification.pipeline import ClassificationPipeline
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.base.model.relevance_model import RelevanceModel

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

    @classmethod
    def setUpClass(
        cls,
        output_dir: str = OUTPUT_DIR,
        root_data_dir: str = ROOT_DATA_DIR,
        feature_config_fname: str = FEATURE_CONFIG_FNAME,
        model_config_fname: str = MODEL_CONFIG_FNAME,
    ):
        cls.output_dir = output_dir
        cls.root_data_dir = root_data_dir
        cls.feature_config_fname = feature_config_fname
        cls.model_config_fname = model_config_fname
        cls.file_io = LocalIO()

        # Make temp output directory
        cls.file_io.make_directory(cls.output_dir, clear_dir=True)

        # Fix random seed values for repeatability
        tf.keras.backend.clear_session()
        np.random.seed(123)
        tf.random.set_seed(123)
        random.seed(123)

        # Setup arguments
        cls.args: Namespace = get_args([])
        cls.args.models_dir = output_dir
        cls.args.logs_dir = output_dir

        # Setting small batch size less than testing data size
        cls.args.batch_size = 32

        # Load feature config
        cls.args.feature_config = os.path.join(
            cls.root_data_dir, "configs", cls.feature_config_fname
        )
        cls.feature_config = cls.file_io.read_yaml(cls.args.feature_config)

        # Load model_config
        cls.args.model_config = os.path.join(
            cls.root_data_dir, "configs", cls.model_config_fname
        )
        cls.model_config = cls.file_io.read_yaml(cls.args.model_config)

        # Setup logging
        outfile: str = os.path.join(cls.args.logs_dir, "output_log.csv")

        cls.logger = setup_logging(reset=True,
                                   file_name=outfile,
                                   log_to_file=True)
        cls.run_default_pipeline(data_format="csv")

    @classmethod
    def run_default_pipeline(cls, data_format: str):
        """Train a model with the default set of args"""
        # Fix random seed values for repeatability
        cls.set_seeds()
        args: Namespace = cls.get_overridden_args(data_format)

        cls.classification_pipeline: ClassificationPipeline = ClassificationPipeline(args=args)
        cls.relevance_dataset: RelevanceDataset = cls.classification_pipeline.get_relevance_dataset()
        cls.classification_model: RelevanceModel = cls.classification_pipeline.get_relevance_model()

        cls.train_metrics = cls.classification_model.fit(dataset=cls.relevance_dataset,
                                                         num_epochs=3,
                                                         models_dir=cls.output_dir)

        cls.global_metrics, cls.grouped_metrics, cls.metrics_dict = \
            cls.classification_model.evaluate(test_dataset=cls.relevance_dataset.test,
                                              logs_dir=cls.args.logs_dir,
                                              group_metrics_min_queries=0)
        cls.predictions = cls.classification_model.predict(test_dataset=cls.relevance_dataset.test)

    @classmethod
    def tearDownClass(cls):
        # Delete output directory
        cls.file_io.rm_dir(cls.output_dir)

        # Delete other temp directories
        cls.file_io.rm_dir(os.path.join(cls.root_data_dir, "csv", "tfrecord"))

        # Clear memory
        tf.keras.backend.clear_session()
        gc.collect()

    @classmethod
    def get_overridden_args(cls, data_format: str = "tfrecord"):
        """Overriding test default setup args from parameters."""
        data_dir = os.path.join(cls.root_data_dir, data_format)
        # Fix random seed values for repeatability

        args: Namespace = cls.args
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
