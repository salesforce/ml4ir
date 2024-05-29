
import logging
from logging import Logger
import traceback
from typing import Dict, Optional, Union, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import Metric

from ml4ir.base.config.keys import FeatureTypeKey
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.io.file_io import FileIO
from ml4ir.base.model.architectures import architecture_factory
from ml4ir.base.model.losses.loss_base import RelevanceLossBase
from ml4ir.base.model.scoring.interaction_model import InteractionModel
from ml4ir.base.model.scoring.scoring_model import RelevanceScorer


class MonteCarloScorer(RelevanceScorer):
    """
    Monte Carlo class that defines the neural network layers that convert
    the input features into scores by using monte carlo trials in inference.

    Defines the feature transformation layer(InteractionModel), dense
    neural network layers combined with activation layers and the loss function

    Notes
    -----
    - This is a Keras model subclass and is built recursively using keras Layer instances
    - This is an abstract class. In order to use a Scorer, one must define
      and override the `architecture_op` and the `final_activation_op` functions
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
            monte_carlo_inference_trials: int = 0,
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
        monte_carlo_inference_trials: int, optinal
            The number of monte carlo inference trials

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

        self.monte_carlo_inference_trials = monte_carlo_inference_trials

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
        scores = self.make_pred(inputs, training=False)
        for _ in range(self.monte_carlo_inference_trials):
            scores += self.make_pred(inputs, training=True)
        scores = tf.divide(scores, tf.constant(self.monte_carlo_inference_trials + 1, dtype=tf.float32))
        return {self.output_name: scores}

    def make_pred(self, inputs: Dict[str, tf.Tensor], training=None):
        # Apply feature layer and transform inputs
        features = self.interaction_model(inputs, training=training)

        # Apply architecture op on train_features
        features[FeatureTypeKey.LOGITS] = self.architecture_op(features, training=training)

        # Apply final activation layer
        scores = self.loss_op.final_activation_op(features, training=training)

        return scores