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
from ml4ir.base.model.scoring.scoring_model import RelevanceScorer
from ml4ir.base.model.scoring.interaction_model import InteractionModel, UnivariateInteractionModel
from ml4ir.base.model.optimizers.optimizer import get_optimizer
from ml4ir.base.io.local_io import LocalIO
from ml4ir.base.io.logging_utils import setup_logging
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.config.keys import ArchitectureKey
from ml4ir.base.tests.test_base import RelevanceTestBase
from ml4ir.applications.ranking.model.ranking_model import RankingModel, LinearRankingModel
from ml4ir.applications.ranking.model.losses import loss_factory
from ml4ir.applications.ranking.model.metrics import metrics_factory
from ml4ir.applications.ranking.config.parse_args import get_args

import warnings
from typing import Union, List, Type

warnings.filterwarnings("ignore")


OUTPUT_DIR = "ml4ir/applications/ranking/tests/test_output"
ROOT_DATA_DIR = "ml4ir/applications/ranking/tests/data"
FEATURE_CONFIG_FNAME = "feature_config.yaml"


class RankingTestBase(RelevanceTestBase):

    def get_ranking_model(
            self,
            loss_key: str,
            metrics_keys: List,
            feature_config: FeatureConfig,
            model_config: dict = None,
            feature_layer_keys_to_fns={},
            initialize_layers_dict={},
            freeze_layers_list=[],
    ) -> RelevanceModel:
        """
        Creates RankingModel

        NOTE: Override this method to create custom loss, scorer, model objects
        """
        self.model_config = model_config if model_config else self.model_config

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
            loss_key=loss_key,
            scoring_type=self.args.scoring_type,
            output_name=self.args.output_name
        )

        # Define scorer
        scorer: RelevanceScorer = RelevanceScorer(
            feature_config=feature_config,
            model_config=self.model_config,
            interaction_model=interaction_model,
            loss=loss,
            output_name=self.args.output_name,
            logger=self.logger,
            file_io=self.file_io,
            logs_dir=self.args.logs_dir
        )

        # Define metrics objects from metrics keys
        metrics: List[Union[Type[Metric], str]] = [
            metrics_factory.get_metric(metric_key=metric_key) for metric_key in metrics_keys
        ]

        # Define optimizer
        optimizer: Optimizer = get_optimizer(
            model_config=self.model_config,
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