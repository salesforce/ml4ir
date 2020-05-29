from tensorflow.keras import Input
import tensorflow as tf

from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.features.feature_layer import FeatureLayerMap
from ml4ir.base.config.keys import TFRecordTypeKey
from ml4ir.base.features.feature_layer import define_example_feature_layer
from ml4ir.base.features.feature_layer import define_sequence_example_feature_layer

from typing import Dict


class InteractionModel:
    """
    Define an Interaction Model that acts on input features to
    convert them into numeric features to be fed into further neural network layers
    """

    def __call__(self, inputs: Dict[str, Input]):
        train_features, metadata_features = self.feature_layer_op(inputs)
        train_features, metadata_features = self.transform_features_op(
            train_features, metadata_features
        )

        return train_features, metadata_features

    def feature_layer_op(self, inputs: Dict[str, Input]):
        raise NotImplementedError

    def transform_features_op(self, train_features, metadata_features):
        raise NotImplementedError


class UnivariateInteractionModel(InteractionModel):
    """
    Defines an interaction model that configures feature layer operations
    on individual features
    """

    def __init__(
        self,
        feature_config: FeatureConfig,
        feature_layer_keys_to_fns: dict,
        tfrecord_type: str,
        max_sequence_size: int = 0,
    ):
        self.feature_config = feature_config
        self.tfrecord_type = tfrecord_type
        self.max_sequence_size = max_sequence_size
        self.feature_layer_map = FeatureLayerMap()
        self.feature_layer_map.add_fns(feature_layer_keys_to_fns)

    def feature_layer_op(self, inputs: Dict[str, Input]):
        if self.tfrecord_type == TFRecordTypeKey.EXAMPLE:
            train_features, metadata_features = define_example_feature_layer(
                feature_config=self.feature_config, feature_layer_map=self.feature_layer_map
            )(inputs)
        else:
            train_features, metadata_features = define_sequence_example_feature_layer(
                feature_config=self.feature_config,
                feature_layer_map=self.feature_layer_map,
                max_sequence_size=self.max_sequence_size,
            )(inputs)

        return train_features, metadata_features

    def transform_features_op(self, train_features, metadata_features):
        # TODO: Make train_features a dictionary
        train_features = tf.concat(train_features, axis=-1, name="train_features")

        return train_features, metadata_features
