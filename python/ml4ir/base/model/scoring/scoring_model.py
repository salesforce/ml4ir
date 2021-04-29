from tensorflow.keras import Input

from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.model.architectures import architecture_factory
from ml4ir.base.model.scoring.interaction_model import InteractionModel
from ml4ir.base.model.losses.loss_base import RelevanceLossBase
from ml4ir.base.io.file_io import FileIO
from logging import Logger

from typing import Dict, Optional


class ScorerBase(object):
    """
    Base Scorer class that defines the neural network layers that convert
    the input features into scores

    Defines the feature transformation layer(InteractionModel), dense
    neural network layers combined with activation layers and the loss function

    Notes
    -----
    This is an abstract class. In order to use a Scorer, one must define
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
            logger=logger
        )

    def __call__(self, inputs: Dict[str, Input]):
        """
        Compute score from input features

        Parameters
        --------
        inputs : dict
            Dictionary of input feature tensors

        Returns
        -------
        scores : Tensor object
            Tensor object of the score computed by the model
        train_features : Tensor object
            Dense feature tensor object that is used to compute the model score
        metadata_features : dict of tensor objects
            Dictionary of tensor objects that are not used for training,
            but can be used for computing loss and metrics
        """
        # Apply feature layer and transform inputs
        train_features, metadata_features = self.interaction_model(inputs)

        # Apply architecture op on train_features
        scores = self.architecture_op(train_features, metadata_features)

        # Apply final activation layer
        scores = self.final_activation_op(scores, metadata_features)

        return scores, train_features, metadata_features

    def architecture_op(self, train_features, metadata_features):
        """
        Define and apply the architecture of the model to produce scores from
        transformed features produced by the InteractionModel

        Parameters
        ----------
        train_features : Tensor object
            Dense feature tensor object that is used to compute the model score
        metadata_features : dict of tensor objects
            Dictionary of tensor objects that are not used for training,
            but can be used for computing loss and metrics

        Returns
        -------
        scores : Tensor object
            Tensor object of the score computed by the model
        """
        raise NotImplementedError

    def final_activation_op(self, scores, metadata_features):
        """
        Define and apply the final activation layer to the scores

        Parameters
        ----------
        scores : Tensor object
            Tensor object of the score computed by the model
        metadata_features : dict of tensor objects
            Dictionary of tensor objects that are not used for training,
            but can be used for computing loss and metrics

        Returns
        -------
        scores : Tensor object
            Tensor object produced by applying the final activation function
            to the scores computed by the model
        """
        raise NotImplementedError


class RelevanceScorer(ScorerBase):
    def architecture_op(self, train_features, metadata_features):
        """
        Define and apply the architecture of the model to produce scores from
        transformed features produced by the InteractionModel

        Parameters
        ----------
        train_features : Tensor object
            Dense feature tensor object that is used to compute the model score
        metadata_features : dict of tensor objects
            Dictionary of tensor objects that are not used for training,
            but can be used for computing loss and metrics

        Returns
        -------
        scores : Tensor object
            Tensor object of the score computed by the model
        """
        return architecture_factory.get_architecture(
            model_config=self.model_config,
            feature_config=self.feature_config,
            file_io=self.file_io,
        )(train_features)

    def final_activation_op(self, scores, metadata_features):
        """
        Define and apply the final activation layer to the scores

        Parameters
        ----------
        scores : Tensor object
            Tensor object of the score computed by the model
        metadata_features : dict of tensor objects
            Dictionary of tensor objects that are not used for training,
            but can be used for computing loss and metrics

        Returns
        -------
        scores : Tensor object
            Tensor object produced by applying the final activation function
            to the scores computed by the model
        """
        return self.loss.get_final_activation_op(self.output_name)(
            scores, mask=metadata_features.get("mask")
        )
