import logging
from logging import Logger
import traceback
from typing import Dict, Optional, Union, List

import tensorflow as tf
import itertools
from tensorflow import keras
from tensorflow.keras.metrics import Metric

from ml4ir.base.config.keys import FeatureTypeKey
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.io.file_io import FileIO
from ml4ir.base.model.architectures import architecture_factory
from ml4ir.base.model.losses.loss_base import RelevanceLossBase
from ml4ir.base.model.scoring.interaction_model import InteractionModel
from ml4ir.base.model.scoring.scoring_model import RelevanceScorer
from ml4ir.base.config.keys import MonteCarloInferenceKey


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
        if not self.use_fixed_mask_in_training:
            self.monte_carlo_training_trials = self.model_config[MonteCarloInferenceKey.MONTE_CARLO_TRIALS][
                MonteCarloInferenceKey.NUM_TRAINING_TRIALS]
            # Adding 1 here to account of the extra inference run with training=False.
            self.monte_carlo_training_trials_tf = tf.constant(self.monte_carlo_training_trials + 1, dtype=tf.float32)

        self.use_fixed_mask_in_testing = bool(self.model_config[MonteCarloInferenceKey.MONTE_CARLO_TRIALS].get(
            MonteCarloInferenceKey.USE_FIXED_MASK_IN_TESTING, False))
        if not self.use_fixed_mask_in_testing:
            self.monte_carlo_test_trials = self.model_config[MonteCarloInferenceKey.MONTE_CARLO_TRIALS][
                MonteCarloInferenceKey.NUM_TEST_TRIALS]
            # Adding 1 here to account of the extra inference run with training=False.
            self.monte_carlo_test_trials_tf = tf.constant(self.monte_carlo_test_trials + 1, dtype=tf.float32)

        self.features_with_fixed_masks = self.model_config[MonteCarloInferenceKey.MONTE_CARLO_TRIALS].get(MonteCarloInferenceKey.FEATURES_WITH_FIXED_MASKS, [])
        if self.features_with_fixed_masks:
            self.features_with_fixed_masks = [f.strip() for f in self.features_with_fixed_masks.split()]
            self.fixed_feature_count = len(self.features_with_fixed_masks)
            self.fixed_mask = list(itertools.product([0, 1], repeat=self.fixed_feature_count))
            self.monte_carlo_fixed_trials_count = tf.constant(len(self.fixed_mask), dtype=tf.float32)

    def mask_inputs(self, inputs, mask_count):
        # Apply masks in a vectorized manner
        masked_inputs_list = []
        for i in range(mask_count):
            masked_inputs = {}
            for key, value in inputs.items():
                masked_value = value
                for j in range(self.fixed_feature_count):
                    if key == self.features_with_fixed_masks[j]:
                        mask = tf.constant(self.fixed_mask[i][j], dtype=tf.float32)
                        masked_value = tf.multiply(masked_value, mask)
                masked_inputs[key] = masked_value
            masked_inputs_list.append(masked_inputs)
        return masked_inputs_list

    def score_with_fixed_mask(self, inputs, masked_inputs_list, training):
        # Stack the masked inputs to create a batch of different masked inputs
        stacked_inputs = {key: tf.stack([masked_inputs[key] for masked_inputs in masked_inputs_list])
                          for key in inputs.keys()}

        # Flatten the batch dimension with the mask dimension
        flattened_inputs = {}
        for key, value in stacked_inputs.items():
            value_shape = tf.shape(value)
            flattened_shape = tf.concat([[value_shape[0] * value_shape[1]], value_shape[2:]], axis=0)
            flattened_inputs[key] = tf.reshape(value, flattened_shape)

        # Compute scores for all masked inputs in one go
        all_scores = super().call(flattened_inputs, training=False)[self.output_name]
        return all_scores

    def reshape_and_normalize(self, all_scores, mask_count, batch_size):
        all_scores_shape = tf.shape(all_scores)
        all_scores = tf.reshape(all_scores, tf.concat([[mask_count, batch_size], all_scores_shape[1:]], axis=0))

        # Aggregate the scores across the mask dimension
        all_scores = tf.reduce_sum(all_scores, axis=0)
        all_scores = tf.divide(all_scores, self.monte_carlo_fixed_trials_count)
        return all_scores

    def deterministic_call(self, inputs: Dict[str, tf.Tensor], training=None):
        batch_size = tf.shape(list(inputs.values())[0])[0]
        mask_count = len(self.fixed_mask)

        # Apply fixed masks
        masked_inputs_list = self.mask_inputs(inputs, mask_count)
        # score with the fixed mask
        all_scores = self.score_with_fixed_mask(inputs, masked_inputs_list, training)
        # Reshape all_scores to separate the mask dimension and the original batch dimension
        all_scores = self.reshape_and_normalize(all_scores, mask_count, batch_size)
        return {self.output_name: all_scores}

    def stochastic_call(self, inputs: Dict[str, tf.Tensor], monte_carlo_trials, monte_carlo_trials_tf, training=None):
        scores = super().call(inputs, training=False)[self.output_name]
        for _ in range(monte_carlo_trials):
            scores += super().call(inputs, training=True)[self.output_name]
        scores = tf.divide(scores, monte_carlo_trials_tf)
        return {self.output_name: scores}

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
            if self.use_fixed_mask_in_training:
                return self.deterministic_call(inputs, training=training)
            else:
                return self.stochastic_call(inputs, self.monte_carlo_training_trials,
                                        self.monte_carlo_training_trials_tf, training=training)
        else:
            if self.use_fixed_mask_in_testing:
                return self.deterministic_call(inputs, training=training)
            else:
                return self.stochastic_call(inputs, self.monte_carlo_test_trials, self.monte_carlo_test_trials_tf,
                                        training=training)


