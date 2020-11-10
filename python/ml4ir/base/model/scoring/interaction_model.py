from tensorflow.keras import Input
import tensorflow as tf

from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.features.feature_layer import FeatureLayerMap
from ml4ir.base.features.feature_layer import define_feature_layer
from ml4ir.base.io.file_io import FileIO

from typing import Dict


class InteractionModel:
    """
    InteractionModel class that defines tensorflow layers that act on input features to
    convert them into numeric features to be fed into further neural network layers
    """

    def __call__(self, inputs: Dict[str, Input]):
        """
        Convert input tensorflow features into numeric
        train features and metadata features by applying respective
        feature transformation functions as specified in the FeatureConfig

        Parameters
        ----------
        inputs : dict
            Dictionary of the inputs to the tensorflow keras model

        Returns
        -------
        train_features : `tf.Tensor`
            Dense tensor object that is used for training
        metadata_features : dict
            Dictionary of feature tensors that can be used for
            computing custom metrics and losses
        """
        train_features, metadata_features = self.feature_layer_op(inputs)
        train_features, metadata_features = self.transform_features_op(
            train_features, metadata_features
        )

        return train_features, metadata_features

    def feature_layer_op(self, inputs: Dict[str, Input]):
        """
        Convert input tensorflow features into numeric
        train features and metadata features by applying respective feature
        transformation functions as specified in the FeatureConfig

        Parameters
        ----------
        inputs : dict
            Dictionary of the inputs to the tensorflow keras model

        Returns
        -------
        train_features : dict
            Dict of feature tensors that are used for training
        metadata_features : dict
            Dictionary of feature tensors that can be used for
            computing custom metrics and losses
        """
        raise NotImplementedError

    def transform_features_op(self, train_features, metadata_features):
        """
        Convert train and metadata features which have feature layer
        functions applied to them into dense numeric tensors

        Parameters
        ----------
        inputs : dict
            Dictionary of the inputs to the tensorflow keras model

        Returns
        -------
        train_features : `tf.Tensor`
            Dense tensor object that is used for training
        metadata_features : dict
            Dictionary of feature tensors that can be used for
            computing custom metrics and losses
        """
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
        """
        Constructor for instantiating a UnivariateInteractionModel

        Parameters
        ----------
        feature_config : `FeatureConfig` object
            FeatureConfig object that defines list of model features
            and the feature transformation functions to be used on each
        tfrecord_type : {"example", "sequence_example"}
            Type of TFRecord protobuf being used for model training
        feature_layer_keys_to_fns : dict
            Dictionary of custom feature transformation functions to be applied
            on the input features
        max_sequence_size : int, optional
            Maximum size of the sequence in SequenceExample protobuf
        file_io : FileIO object
            `FileIO` object that handles read write operations
        """
        self.feature_config = feature_config
        self.tfrecord_type = tfrecord_type
        self.max_sequence_size = max_sequence_size
        self.feature_layer_map = FeatureLayerMap()
        self.feature_layer_map.add_fns(feature_layer_keys_to_fns)
        self.file_io = file_io

    def feature_layer_op(self, inputs: Dict[str, Input]):
        """
        Convert input tensorflow features into numeric
        train features and metadata features by applying respective feature
        transformation functions as specified in the FeatureConfig

        Parameters
        ----------
        inputs : dict
            Dictionary of the inputs to the tensorflow keras model

        Returns
        -------
        train_features : dict
            Dict of feature tensors that are used for training
        metadata_features : dict
            Dictionary of feature tensors that can be used for
            computing custom metrics and losses
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
        Convert train and metadata features which have feature layer
        functions applied to them into dense numeric tensors.
        Sorts the features by name and concats the individual features
        into a dense tensor.

        Parameters
        ----------
        inputs : dict
            Dictionary of the inputs to the tensorflow keras model

        Returns
        -------
        train_features : `tf.Tensor`
            Dense tensor object that is used for training
        metadata_features : dict
            Dictionary of feature tensors that can be used for
            computing custom metrics and losses
        """
        # Sorting the train features dictionary so that we control the order
        train_features_list = [train_features[k] for k in sorted(train_features)]

        # Concat all train features to get a dense feature vector
        train_features_transformed = tf.concat(train_features_list, axis=-1, name="train_features")

        return train_features_transformed, metadata_features
