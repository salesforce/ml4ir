import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics

from ml4ir.base.config.keys import FeatureTypeKey
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.model.architectures import architecture_factory
from ml4ir.base.model.scoring.interaction_model import InteractionModel
from ml4ir.base.model.losses.loss_base import RelevanceLossBase
from ml4ir.base.io.file_io import FileIO
from logging import Logger

from typing import Dict, Optional


class ScorerBase(Model):
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
        output_name: str = "score",
        logger: Optional[Logger] = None,
        **kwargs
    ):
        """
        Constructor method for creating a ScorerBase object

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
        output_name : str, optional
            Name of the output that captures the score computed by the model
        logger : Logger, optional
            Logging handler
        """
        super().__init__(**kwargs)

        self.model_config = model_config
        self.feature_config = feature_config
        self.interaction_model = interaction_model
        self.loss = loss
        self.file_io = file_io
        self.output_name = output_name

    @classmethod
    def from_model_config_file(
        cls,
        model_config_file: str,
        interaction_model: InteractionModel,
        loss: RelevanceLossBase,
        file_io: FileIO,
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
        output_name : str, optional
            Name of the output that captures the score computed by the model
        logger: Logger, optional
            Logging handler

        Returns
        -------
        `ScorerBase` object
            ScorerBase object that computes the scores from the input features of the model
        """
        model_config = file_io.read_yaml(model_config_file)

        return cls(
            model_config=model_config,
            feature_config=feature_config,
            interaction_model=interaction_model,
            loss=loss,
            file_io=file_io,
            output_name=output_name,
            logger=logger,
            **kwargs
        )

    def call(self, inputs: Dict[str, tf.Tensor], training=None):
        """
        Compute score from input features

        Parameters
        --------
        inputs : dict of tensors
            Dictionary of input feature tensors

        Returns
        -------
        scores : Tensor object
            Tensor object of the score computed by the model
        """
        # Apply feature layer and transform inputs
        features = self.interaction_model(inputs, training=training)

        # Apply architecture op on train_features
        features[FeatureTypeKey.LOGITS] = self.architecture_op(features, training=training)

        # Apply final activation layer
        scores = self.loss.final_activation_op(features, training=training)

        return scores


class RelevanceScorer(ScorerBase):

    def __init__(
        self,
        model_config: dict,
        feature_config: FeatureConfig,
        interaction_model: InteractionModel,
        loss: RelevanceLossBase,
        file_io: FileIO,
        output_name: str = "score",
        logger: Optional[Logger] = None,
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
        output_name : str, optional
            Name of the output that captures the score computed by the model
        logger : Logger, optional
            Logging handler
        """
        super().__init__(model_config=model_config,
                         feature_config=feature_config,
                         interaction_model=interaction_model,
                         loss=loss,
                         file_io=file_io,
                         output_name=output_name,
                         logger=logger,
                         **kwargs)

        # NOTE: Override self.architecture_op with any tensorflow network for customization
        self.architecture_op = architecture_factory.get_architecture(
            model_config=self.model_config,
            feature_config=self.feature_config,
            file_io=self.file_io,
        )

    def compile(self, **kwargs):
        """Compile the keras model"""
        # Define metric to track loss
        self.loss_metric = keras.metrics.Mean(name="loss")
        super().compile(**kwargs)

    def __get_loss_value(self, inputs, y_true, y_pred):
        """
        Compute loss value

        Parameters
        ----------
        inputs: dict of dict of tensors
            Dictionary of input feature tensors
        y_true: tensor
            True labels
        y_pred: tensor
            Predicted scores
        training: boolean
            Boolean indicating whether the layer is being used in training mode

        Returns
        -------
        tensor
            scalar loss value tensor
        """
        if isinstance(self.loss, str):
            loss_value = self.compiled_loss(y_true=y_true, y_pred=y_pred)
        elif isinstance(self.loss, RelevanceLossBase):
            loss_value = self.loss(inputs=inputs, y_true=y_true, y_pred=y_pred)
            # Update loss metric
            self.loss_metric.update_state(loss_value)
        elif isinstance(self.loss, keras.losses.Loss):
            loss_value = self.compiled_loss(y_true=y_true, y_pred=y_pred)
        else:
            raise KeyError("Unknown Loss encountered in RelevanceScorer")

        return loss_value

    def train_step(self, data):
        """
        TODO: Add docs
        """
        X, y = data

        with tf.GradientTape() as tape:
            y_pred = self(X, training=True)
            loss_value = self.__get_loss_value(inputs=X, y_true=y, y_pred=y_pred)

        # Compute gradients
        gradients = tape.gradient(loss_value, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        # self.compiled_metrics.update_state(y, y_pred, features=X)

        # Return a dict mapping metric names to current value
        return {metric.name: metric.result() for m in self.metrics}

    def test_step(self, data):
        """
        TODO: Add docs
        """
        X, y = data

        y_pred = self(X, training=False)

        # Update loss metric
        self.get_loss_value(y_true=y, y_pred=y_pred, features=X)

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {metric.name: metric.result() for m in self.metrics}

    @property
    def metrics(self):
        """Add loss metric to keras model metrics"""
        return [self.loss_metric] + super().metrics
