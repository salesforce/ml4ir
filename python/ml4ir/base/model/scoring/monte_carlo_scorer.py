import logging
from logging import Logger
from typing import Dict, Optional, Union, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import Metric

from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.io.file_io import FileIO
from ml4ir.base.model.losses.loss_base import RelevanceLossBase
from ml4ir.base.model.scoring.interaction_model import InteractionModel
from ml4ir.base.model.scoring.scoring_model import RelevanceScorer
from ml4ir.base.config.keys import MonteCarloInferenceKey
from ml4ir.applications.ranking.model.layers.masking import QueryFeatureMask


class MonteCarloScorer(RelevanceScorer):
    """
    Monte Carlo class that defines the neural network layers that convert
    the input features into scores by using monte carlo trials in inference.
    The actual masking of features is done by a masking layer called:
    `RecordFeatureMask` or `QueryFeatureMask` or by using fixed masks
    """

    def __init__(
            self,
            model_config: dict,
            feature_config: FeatureConfig,
            interaction_model: InteractionModel,
            loss: RelevanceLossBase,
            file_io: FileIO,
            aux_loss: Optional[RelevanceLossBase] = None,
            aux_loss_weight: float = 0.0,
            aux_metrics: Optional[List[Union[Metric, str]]] = None,
            output_name: str = "score",
            logger: Optional[Logger] = None,
            logs_dir: Optional[str] = "",
            **kwargs
    ):
        """
        Constructor method for creating a RelevanceScorer object

        Parameters
        ----------
        model_config : dict
            Dictionary defining the model layer configuration
        feature_config : `FeatureConfig` object
            FeatureConfig object defining the features and their configurations
        interaction_model : `InteractionModel` object
            InteractionModel that defines the feature transformation layers
            on the input model features
        loss : `RelevanceLossBase` object
            Relevance loss object that defines the final activation layer
            and the loss function for the model
        file_io : `FileIO` object
            FileIO object that handles read and write
        aux_loss : `RelevanceLossBase` object
            Auxiliary loss to be used in conjunction with the primary loss
        aux_loss_weight: float
            Floating point number in [0, 1] to indicate the proportion of the auxiliary loss
            in the total final loss value computed using a linear combination
            total loss = (1 - aux_loss_weight) * loss + aux_loss_weight * aux_loss
        aux_metrics: List of keras.metrics.Metric
            Keras metric list to be computed on the aux label
        output_name : str, optional
            Name of the output that captures the score computed by the model
        logger : Logger, optional
            Logging handler
        logs_dir : str, optional
            Path to the logging directory

        Notes
        -----
        logs_dir : Used to point model architectures to local logging directory,
            primarily for saving visualizations.
        """
        super().__init__(feature_config=feature_config,
                         model_config=model_config,
                         interaction_model=interaction_model,
                         loss=loss,
                         aux_loss=aux_loss,
                         aux_loss_weight=aux_loss_weight,
                         aux_metrics=aux_metrics,
                         output_name=output_name,
                         logger=logger,
                         file_io=file_io,
                         logs_dir=logs_dir,
                         **kwargs)

        self.use_fixed_mask_in_training = bool(self.model_config[MonteCarloInferenceKey.MONTE_CARLO_TRIALS].get(
            MonteCarloInferenceKey.USE_FIXED_MASK_IN_TRAINING, False))
        QueryFeatureMask.masking_dict[MonteCarloInferenceKey.USE_FIXED_MASK_IN_TRAINING] = self.use_fixed_mask_in_training
        if not self.use_fixed_mask_in_training:
            self.monte_carlo_training_trials = self.model_config[MonteCarloInferenceKey.MONTE_CARLO_TRIALS][
                MonteCarloInferenceKey.NUM_TRAINING_TRIALS]
            QueryFeatureMask.masking_dict[MonteCarloInferenceKey.NUM_TRAINING_TRIALS] = self.monte_carlo_training_trials

        self.use_fixed_mask_in_testing = bool(self.model_config[MonteCarloInferenceKey.MONTE_CARLO_TRIALS].get(
            MonteCarloInferenceKey.USE_FIXED_MASK_IN_TESTING, False))
        QueryFeatureMask.masking_dict[MonteCarloInferenceKey.USE_FIXED_MASK_IN_TESTING] = self.use_fixed_mask_in_testing
        if not self.use_fixed_mask_in_testing:
            self.monte_carlo_test_trials = self.model_config[MonteCarloInferenceKey.MONTE_CARLO_TRIALS][
                MonteCarloInferenceKey.NUM_TEST_TRIALS]
            QueryFeatureMask.masking_dict[MonteCarloInferenceKey.NUM_TEST_TRIALS] = self.monte_carlo_test_trials

    def call(self, inputs: Dict[str, tf.Tensor], training=None):
        """
        Compute score from input features

        Parameters
        --------
        inputs : dict of tensors
            Dictionary of input feature tensors

        Returns
        -------
        scores : dict of tensor object
            Tensor object of the score computed by the model
        """
        if training:
            if not self.use_fixed_mask_in_training:
                monte_carlo_trials = self.monte_carlo_training_trials
            else:
                monte_carlo_trials = QueryFeatureMask.masking_dict[MonteCarloInferenceKey.FIXED_MASK_COUNT] - 1
        else:
            if not self.use_fixed_mask_in_testing:
                monte_carlo_trials = self.monte_carlo_test_trials
            else:
                monte_carlo_trials = QueryFeatureMask.masking_dict[MonteCarloInferenceKey.FIXED_MASK_COUNT] - 1

        scores = super().call(inputs, training=training)[self.output_name]
        for _ in range(monte_carlo_trials):
            scores += super().call(inputs, training=training)[self.output_name]
        scores = tf.divide(scores, tf.constant(monte_carlo_trials + 1, dtype=tf.float32))
        return {self.output_name: scores}