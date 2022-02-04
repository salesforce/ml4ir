import tensorflow as tf

from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.config.keys import SequenceExampleTypeKey
from ml4ir.base.config.keys import TFRecordTypeKey
from ml4ir.base.features.feature_fns.sequence import bytes_sequence_to_encoding_bilstm
from ml4ir.base.features.feature_fns.sequence import global_1d_pooling
from ml4ir.base.features.feature_fns.categorical import categorical_embedding_to_encoding_bilstm
from ml4ir.base.features.feature_fns.categorical import categorical_embedding_with_hash_buckets
from ml4ir.base.features.feature_fns.categorical import categorical_embedding_with_indices
from ml4ir.base.features.feature_fns.categorical import categorical_embedding_with_vocabulary_file
from ml4ir.base.features.feature_fns.categorical import (
    categorical_embedding_with_vocabulary_file_and_dropout,
)
from ml4ir.base.features.feature_fns.tf_native import tf_native_op
from ml4ir.base.io.file_io import FileIO


class FeatureLayerMap:
    """Class defining mapping from keys to feature layer functions"""

    def __init__(self):
        """
        Define ml4ir's predefined feature transformation functions
        """
        self.key_to_fn = {
            bytes_sequence_to_encoding_bilstm.__name__: bytes_sequence_to_encoding_bilstm,
            global_1d_pooling.__name__: global_1d_pooling,
            categorical_embedding_to_encoding_bilstm.__name__: categorical_embedding_to_encoding_bilstm,
            categorical_embedding_with_hash_buckets.__name__: categorical_embedding_with_hash_buckets,
            categorical_embedding_with_indices.__name__: categorical_embedding_with_indices,
            categorical_embedding_with_vocabulary_file.__name__: categorical_embedding_with_vocabulary_file,
            categorical_embedding_with_vocabulary_file_and_dropout.__name__: categorical_embedding_with_vocabulary_file_and_dropout,
            tf_native_op.__name__: tf_native_op
        }

    def add_fn(self, key, fn):
        """
        Add custom new function to the FeatureLayerMap

        Parameters
        ----------
        key : str
            name of the feature transformation function
        fn : tf.function
            tensorflow function that transforms input features
        """

        self.key_to_fn[key] = fn

    def add_fns(self, keys_to_fns_dict):
        """
        Add custom new functions to the FeatureLayerMap

        Parameters
        ----------
        keykeys_to_fns_dict : dict
            Dictionary with name and definition of custom
            tensorflow functions that transform input features
        """
        self.key_to_fn.update(keys_to_fns_dict)

    def get_fns(self):
        """
        Get all feature transformation functions

        Returns
        -------
        dict
            Dictionary of feature transformation functions
        """
        return self.key_to_fn

    def get_fn(self, key):
        """
        Get feature transformation function using the key

        Parameters
        ----------
        key : str
            Name of the feature transformation function to be fetched

        Returns
        -------
        tf.function
            Feature transformation function
        """
        return self.key_to_fn.get(key)

    def pop_fn(self, key):
        """
        Get feature transformation function using the key and remove
        from FeatureLayerMap

        Parameters
        ----------
        key : str
            Name of the feature transformation function to be fetched

        Returns
        -------
        tf.function
            Feature transformation function
        """
        self.key_to_fn.pop(key)

from tensorflow.keras import layers


class FeatureLayer(layers.Layer):
    """Keras layer that applies in-graph transformations to input feature tensors"""

    def __init__(self,
                 feature_config: FeatureConfig,
                 tfrecord_type: str,
                 feature_layer_map: FeatureLayerMap,
                 file_io: FileIO):
        """Initialize the FeatureLayer object"""
        self.feature_config = feature_config
        self.tfrecord_type = tfrecord_type
        self.feature_layer_map = feature_layer_map
        self.file_io = file_io

    



def define_feature_layer(
    feature_config: FeatureConfig,
    tfrecord_type: str,
    feature_layer_map: FeatureLayerMap,
    file_io: FileIO,
):
    """
    Defines a feature layer function that works on keras.Inputs

    Parameters
    ----------
    feature_config : `FeatureConfig` object
        FeatureConfig object that defines the feature transformation
        function to be applied to each feature
    tfrecord_type : {"example", "sequence_example"}
        String defining the TFRecord type of the data
    feature_layer_map : `FeatureLayerMap`
        `FeatureLayerMap` object mapping custom function names to function definition

    Returns
    -------
        tensorflow op that is the feature layer by collecting all the
        feature transformation functions assigned to the respective
        keras.Inputs features as specified by the FeatureConfig

    Notes
    -----
    Use feature_layer_map to define custom functions when using ml4ir as a library
    """

    def feature_layer_op(inputs):
        """
        Apply feature transformation functions and reshape the input
        feature tensors as specified by the FeatureConfig
        """
        train_features = dict()
        metadata_features = dict()

        # Define a dynamic tensor tiling shape
        # NOTE: Can not be hardcoded as we allow for varying sequence_size at inference time
        if tfrecord_type == TFRecordTypeKey.SEQUENCE_EXAMPLE:
            train_tile_shape = tf.shape(
                tf.expand_dims(
                    tf.expand_dims(tf.gather(inputs["mask"], indices=0), axis=0), axis=-1
                )
            )
            metadata_tile_shape = tf.shape(
                tf.expand_dims(tf.gather(inputs["mask"], indices=0), axis=0)
            )

        for feature_info in feature_config.get_all_features(include_label=False):
            feature_node_name = feature_info.get("node_name", feature_info["name"])
            feature_layer_info = feature_info["feature_layer_info"]
            feature_tensor = inputs[feature_node_name]

            if "fn" in feature_layer_info:
                feature_fn = feature_layer_map.get_fn(feature_layer_info["fn"])
                if not feature_fn:
                    raise RuntimeError(
                        "Unsupported feature function: {}".format(feature_layer_info["fn"])
                    )
                feature_tensor = feature_fn(
                    feature_tensor=feature_tensor, feature_info=feature_info, file_io=file_io
                )
            elif feature_info["trainable"]:
                # Default feature layer
                feature_tensor = tf.expand_dims(feature_tensor, axis=-1, name="{}_expanded".format(feature_node_name))

            """
            NOTE: If the trainable feature is of type context, then we tile/duplicate
            the values for all examples of the sequence
            """
            if (
                tfrecord_type == TFRecordTypeKey.SEQUENCE_EXAMPLE
                and feature_info["tfrecord_type"] == SequenceExampleTypeKey.CONTEXT
            ):
                if feature_info["trainable"]:
                    feature_tensor = tf.tile(feature_tensor, train_tile_shape)
                else:
                    feature_tensor = tf.tile(feature_tensor, metadata_tile_shape)

            if feature_info["trainable"]:
                train_features[feature_node_name] = tf.cast(feature_tensor, tf.float32)
            else:
                if feature_info["dtype"] == tf.int64:
                    feature_tensor = tf.cast(feature_tensor, tf.float32)
                metadata_features[feature_node_name] = feature_tensor

        return train_features, metadata_features

    return feature_layer_op


from tensorflow.keras import Input
import tensorflow as tf

from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.features.feature_layer import FeatureLayerMap
from ml4ir.base.features.feature_layer import define_feature_layer
from ml4ir.base.io.file_io import FileIO

from typing import Dict


class FeatureLayer:
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

