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

    def get_ranking_model(
        self,
        loss_key: str,
        metrics_keys: List,
        feature_config: FeatureConfig,
        model_config: dict = {},
        feature_layer_keys_to_fns={},
        initialize_layers_dict={},
        freeze_layers_list=[],
    ) -> RelevanceModel:
        """
        Creates RankingModel

        NOTE: Override this method to create custom loss, scorer, model objects
        """

        # Define interaction model
        interaction_model: InteractionModel = UnivariateInteractionModel(
            feature_config=feature_config,
            feature_layer_keys_to_fns=feature_layer_keys_to_fns,
            tfrecord_type=self.args.tfrecord_type,
            max_sequence_size=self.args.max_sequence_size,
            file_io=self.file_io,
        )

        # Define loss object from loss key
        loss: RelevanceLossBase = loss_factory.get_loss(
            loss_key=loss_key, scoring_type=self.args.scoring_type
        )

        # Define scorer
        scorer: ScorerBase = RelevanceScorer(
            feature_config=feature_config,
            model_config=self.model_config,
            interaction_model=interaction_model,
            loss=loss,
            output_name=self.args.output_name,
            logger=self.logger,
            file_io=self.file_io,
        )

        # Define metrics objects from metrics keys
        metrics: List[Union[Type[Metric], str]] = [
            metric_factory.get_metric(metric_key=metric_key) for metric_key in metrics_keys
        ]

        # Define optimizer
        optimizer: Optimizer = get_optimizer(
            model_config=self.file_io.read_yaml(self.args.model_config),
        )

        # Combine the above to define a RelevanceModel
        if self.model_config["architecture_key"] == ArchitectureKey.LINEAR:
            RankingModelClass = LinearRankingModel
        else:
            RankingModelClass = RankingModel
        relevance_model: RelevanceModel = RankingModelClass(
            feature_config=feature_config,
            tfrecord_type=self.args.tfrecord_type,
            scorer=scorer,
            metrics=metrics,
            optimizer=optimizer,
            model_file=self.args.model_file,
            initialize_layers_dict=initialize_layers_dict,
            freeze_layers_list=freeze_layers_list,
            compile_keras_model=self.args.compile_keras_model,
            output_name=self.args.output_name,
            logger=self.logger,
            file_io=self.file_io,
        )

        return relevance_model


if __name__ == "__main__":
    unittest.main()
