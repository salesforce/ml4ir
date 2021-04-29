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
