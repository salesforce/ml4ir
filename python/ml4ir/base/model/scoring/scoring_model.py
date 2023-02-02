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


class RelevanceScorer(keras.Model):
    """
    Base Scorer class that defines the neural network layers that convert
    the input features into scores

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
        super().__init__(**kwargs)

        self.logger = logger

        self.model_config = model_config
        self.feature_config = feature_config
        self.interaction_model = interaction_model

        self.loss_op = loss
        self.aux_loss_op = aux_loss
        self.aux_loss_weight = aux_loss_weight
        self.aux_label = self.__get_aux_label()
        self.aux_metrics = aux_metrics

        self.loss_metric = None
        self.aux_loss_metric = None
        self.primary_loss_metric = None

        self.file_io = file_io
        self.output_name = output_name
        self.logs_dir = logs_dir
        self.architecture_op = self.get_architecture_op()
        self.plot_abstract_model()

    @classmethod
    def from_model_config_file(
            cls,
            model_config_file: str,
            interaction_model: InteractionModel,
            loss: RelevanceLossBase,
            file_io: FileIO,
            aux_loss: Optional[RelevanceLossBase] = None,
            aux_loss_weight: float = 0.0,
            output_name: str = "score",
            feature_config: Optional[FeatureConfig] = None,
            logger: Optional[Logger] = None,
            **kwargs
    ):
        """
        Get a Scorer object from a YAML model config file

        Parameters
        ----------
        model_config_file : str
            Path to YAML file defining the model layer configuration
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
        output_name : str, optional
            Name of the output that captures the score computed by the model
        logger: Logger, optional
            Logging handler

        Returns
        -------
        `RelevanceScorer` object
            RelevanceScorer object that computes the scores from the input features of the model
        """
        model_config = file_io.read_yaml(model_config_file)

        return cls(
            model_config=model_config,
            feature_config=feature_config,
            interaction_model=interaction_model,
            loss=loss,
            file_io=file_io,
            aux_loss=aux_loss,
            aux_loss_weight=aux_loss_weight,
            output_name=output_name,
            logger=logger,
            **kwargs
        )

    def plot_abstract_model(self):
        """Visualize the model architecture if defined by the architecture op"""
        if hasattr(self.architecture_op, "plot_abstract_model"):
            self.architecture_op.plot_abstract_model(self.logs_dir)
        else:
            self.file_io.log(f"plot_abstract_model method is not defined for {self.architecture_op.__class__.__name__}",
                             logging.DEBUG)

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
        # Apply feature layer and transform inputs
        features = self.interaction_model(inputs, training=training)

        # Apply architecture op on train_features
        features[FeatureTypeKey.LOGITS] = self.architecture_op(features, training=training)

        # Apply final activation layer
        scores = self.loss_op.final_activation_op(features, training=training)

        return {self.output_name: scores}

    def get_architecture_op(self):
        """Get the model architecture instance based on the configs"""
        # NOTE: Override self.architecture_op with any tensorflow network for customization
        return architecture_factory.get_architecture(
            model_config=self.model_config,
            feature_config=self.feature_config,
            file_io=self.file_io,
        )

    def compile(self, **kwargs):
        """Compile the keras model and defining a loss metrics to track any custom loss"""
        # Define metric to track loss
        self.loss_metric = keras.metrics.Mean(name="loss")

        # Define metric to track an auxiliary loss if needed
        if self.aux_loss_weight > 0:
            self.primary_loss_metric = keras.metrics.Mean(name="primary_loss")
            self.aux_loss_metric = keras.metrics.Mean(name="aux_loss")

        super().compile(**kwargs)

    def __get_aux_label(self):
        """
        Get aux_label from FeatureConfig

        Returns
        -------
        str
            Name of the aux label if present
        """
        aux_label = None
        if self.aux_loss_weight > 0:
            try:
                aux_label = self.feature_config.get_aux_label("node_name")
            except AttributeError:
                self.logger.error("There was an error while loading the aux_label info. "
                                  "Did you set the is_aux_label flag to true in the FeatureConfig YAML?")
                self.logger.error(traceback.format_exc())
        return aux_label

    def __update_loss(self, inputs, y_true, y_pred):
        """
        Compute loss value

        Parameters
        ----------
        inputs: dict of tensors
            Dictionary of input feature tensors
        y_true: tensor
            True labels
        y_pred: tensor
            Predicted scores

        Returns
        -------
        tensor
            scalar loss value tensor
        """
        if isinstance(self.loss, str):
            loss_value = self.compiled_loss(y_true=y_true, y_pred=y_pred)
        elif isinstance(self.loss, RelevanceLossBase):
            loss_value = self.loss_op(inputs=inputs, y_true=y_true, y_pred=y_pred)
        elif isinstance(self.loss, keras.losses.Loss):
            loss_value = self.compiled_loss(y_true=y_true, y_pred=y_pred)
        else:
            raise KeyError("Unknown Loss encountered in RelevanceScorer")

        # If auxiliary loss is present, add it to compute final loss
        if self.aux_loss_weight > 0:
            aux_loss_value = self.aux_loss_op(inputs=inputs,
                                              y_true=inputs[self.aux_label],
                                              y_pred=y_pred)

            self.primary_loss_metric.update_state(loss_value)
            self.aux_loss_metric.update_state(aux_loss_value)

            loss_value = tf.math.multiply((1. - self.aux_loss_weight), loss_value) + \
                         tf.math.multiply(self.aux_loss_weight, aux_loss_value)

        # Update loss metric
        self.loss_metric.update_state(loss_value)

        return loss_value

    def __update_metrics(self, inputs, y_true, y_pred):
        """
        Compute metric value

        Parameters
        ----------
        inputs: dict of tensors
            Dictionary of input feature tensors
        y_true: tensor
            True labels
        y_pred: tensor
            Predicted scores

        Notes
        -----
        - Currently only support pre-compiled Keras metrics
        """
        # Compute metrics on primary label
        self.compiled_metrics.update_state(y_true, y_pred)

        # Compute metrics on auxiliary label
        if self.aux_label:
            y_aux = inputs[self.aux_label]
            y_true_ranks = inputs[self.feature_config.get_rank("node_name")]
            mask = inputs[self.feature_config.get_mask("node_name")]
            for metric in self.aux_metrics:
                # TODO: The function definition could be made more generic
                #       to accommodate more metrics in the future,
                #       but this is sufficient for now
                metric.update_state(y_true, y_pred, y_aux, y_true_ranks, mask)

    def train_step(self, data):
        """
        Defines the operations performed within a single training step.
        Called implicitly by tensorflow-keras when using model.fit()

        Parameters
        ----------
        data: tuple of tensor objects
            Tuple of features and corresponding labels to be used to learn the
            model weights

        Returns
        -------
        dict
            Dictionary of metrics and loss computed for this training step
        """
        X, y = data

        with tf.GradientTape() as tape:
            y_pred = self(X, training=True)[self.output_name]
            loss_value = self.__update_loss(inputs=X, y_true=y, y_pred=y_pred)

        # Compute gradients
        gradients = tape.gradient(loss_value, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.__update_metrics(inputs=X, y_true=y, y_pred=y_pred)

        # Return a dict mapping metric names to current value
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        """
        Defines the operations performed within a single prediction or evaluation step.
        Called implicitly by tensorflow-keras when using model.predict() or model.evaluate()

        Parameters
        ----------
        data: tuple of tensor objects
            Tuple of features and corresponding labels to be used to evaluate the model

        Returns
        -------
        dict
            Dictionary of metrics and loss computed for this evaluation step
        """
        X, y = data

        y_pred = self(X, training=False)[self.output_name]

        # Update loss metric
        self.__update_loss(inputs=X, y_true=y, y_pred=y_pred)

        # Update metrics
        self.__update_metrics(inputs=X, y_true=y, y_pred=y_pred)

        # Return a dict mapping metric names to current value
        return {metric.name: metric.result() for metric in self.metrics}

    @property
    def metrics(self):
        """Get the metrics for the keras model along with the custom loss metric"""
        metrics = [self.loss_metric] + super().metrics

        if self.aux_loss_weight > 0:
            # Add aux loss values
            metrics += [self.primary_loss_metric, self.aux_loss_metric]

            # Add aux metrics
            metrics += self.aux_metrics

        return metrics
