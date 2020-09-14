import tensorflow as tf

from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.config.keys import SequenceExampleTypeKey
from ml4ir.base.config.keys import TFRecordTypeKey
from ml4ir.base.features.feature_fns.sequence import bytes_sequence_to_encoding_bilstm
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
        self.key_to_fn = {
            bytes_sequence_to_encoding_bilstm.__name__: bytes_sequence_to_encoding_bilstm,
            categorical_embedding_to_encoding_bilstm.__name__: categorical_embedding_to_encoding_bilstm,
            categorical_embedding_with_hash_buckets.__name__: categorical_embedding_with_hash_buckets,
            categorical_embedding_with_indices.__name__: categorical_embedding_with_indices,
            categorical_embedding_with_vocabulary_file.__name__: categorical_embedding_with_vocabulary_file,
            categorical_embedding_with_vocabulary_file_and_dropout.__name__: categorical_embedding_with_vocabulary_file_and_dropout,
        }

    def add_fn(self, key, fn):
        self.key_to_fn[key] = fn

    def add_fns(self, keys_to_fns_dict):
        self.key_to_fn.update(keys_to_fns_dict)

    def get_fns(self):
        return self.key_to_fn

    def get_fn(self, key):
        return self.key_to_fn.get(key)

    def pop_fn(self, key):
        self.key_to_fn.pop(key)


def define_feature_layer(
    feature_config: FeatureConfig,
    tfrecord_type: str,
    feature_layer_map: FeatureLayerMap,
    file_io: FileIO,
):
    """
    Defines a feature layer function that works on keras.Inputs

    Args:
        - feature_config: FeatureConfig object
        - tfrecord_type: string defining the TFRecord type of the data
        - feature_layer_map: dictionary mapping custom function names to function definition

    Returns:
        feature layer function that works on keras.Inputs

    NOTE:
        - Use feature_layer_map to define custom functions when using ml4ir as a library
        - Currently implements a 1:1 feature layer map
    """

    def feature_layer_op(inputs):
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
                feature_tensor = feature_layer_map.get_fn(feature_layer_info["fn"])(
                    feature_tensor=feature_tensor, feature_info=feature_info, file_io=file_io
                )
            elif feature_info["trainable"]:
                # Default feature layer
                feature_tensor = tf.expand_dims(feature_tensor, axis=-1)

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
