from tensorflow.keras import Input
import tensorflow as tf

from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.features.feature_layer import FeatureLayerMap
from ml4ir.base.features.feature_layer import define_feature_layer
from ml4ir.base.io.file_io import FileIO

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
        tfrecord_type: str,
        feature_layer_keys_to_fns: dict = {},
        max_sequence_size: int = 0,
        file_io: FileIO = None,
    ):
        self.feature_config = feature_config
        self.tfrecord_type = tfrecord_type
        self.max_sequence_size = max_sequence_size
        self.feature_layer_map = FeatureLayerMap()
        self.feature_layer_map.add_fns(feature_layer_keys_to_fns)
        self.file_io = file_io

    def feature_layer_op(self, inputs: Dict[str, Input]):
        """
        Apply feature layer functions on each of the tf.keras.Input

        Args:
            inputs: dictionary of keras input symbolic tensors

        Returns:
            train_features: dictionary of feature tensors that can be used for training
            metadata_features: dictionary of feature tensors that can be used as additional metadata
        """
        train_features, metadata_features = define_feature_layer(
            feature_config=self.feature_config,
            tfrecord_type=self.tfrecord_type,
            feature_layer_map=self.feature_layer_map,
            file_io=self.file_io,
        )(inputs)

        return train_features, metadata_features

    def transform_features_op(
        self, train_features: Dict[str, tf.Tensor], metadata_features: Dict[str, tf.Tensor]
    ):
        """
        Transform train_features and metadata_features after the
        univariate feature_layer fns have been applied.

        Args:
            train_features: dictionary of feature tensors that can be used for training
            metadata_features: dictionary of feature tensors that can be used as additional metadata

        Returns:
            train_features: single dense trainable feature tensor
            metadata_features: dictionary of metadata feature tensors
        """

        # Sorting the train features dictionary so that we control the order
        train_features_list = [train_features[k] for k in sorted(train_features)]

        # Concat all train features to get a dense feature vector
        train_features_transformed = tf.concat(train_features_list, axis=-1, name="train_features")

        return train_features_transformed, metadata_features
