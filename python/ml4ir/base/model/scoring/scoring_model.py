from tensorflow.keras import Input

from ml4ir.base.model.architectures import architecture_factory
from ml4ir.base.model.scoring.interaction_model import InteractionModel
from ml4ir.base.model.losses.loss_base import RelevanceLossBase
from ml4ir.base.io import file_io

from typing import Dict


class ScorerBase(object):
    def __init__(
        self,
        model_config: dict,
        interaction_model: InteractionModel,
        loss: RelevanceLossBase,
        output_name: str = "score",
    ):
        self.interaction_model = interaction_model
        self.loss = loss
        self.model_config = model_config
        self.output_name = output_name

    @classmethod
    def from_model_config_file(
        cls,
        model_config_file: str,
        interaction_model: InteractionModel,
        loss: RelevanceLossBase,
        output_name: str,
    ):
        model_config = file_io.read_yaml(model_config_file)

        return cls(
            model_config=model_config,
            interaction_model=interaction_model,
            loss=loss,
            output_name=output_name,
        )

    def __call__(self, inputs: Dict[str, Input]):
        # Apply feature layer and transform inputs
        train_features, metadata_features = self.interaction_model(inputs)

        # Apply architecture op on train_features
        scores = self.architecture_op(train_features, metadata_features)

        # Apply final activation layer
        scores = self.final_activation_op(scores, metadata_features)

        return scores, train_features, metadata_features

    def architecture_op(self, train_features, metadata_features):
        """Define architecture of the model to produce scores from transformed features"""
        raise NotImplementedError

    def final_activation_op(self, scores, metadata_features):
        """Define final activation layer"""
        raise NotImplementedError


class RelevanceScorer(ScorerBase):
    def architecture_op(self, train_features, metadata_features):
        return architecture_factory.get_architecture(model_config=self.model_config)(
            train_features
        )

    def final_activation_op(self, scores, metadata_features):
        return self.loss.get_final_activation_op(self.output_name)(
            scores, mask=metadata_features.get("mask")
        )
