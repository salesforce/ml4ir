# flake8: noqa
# TODO: Fix complexity

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import io

from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.config.keys import FeatureTypeKey
from ml4ir.base.config.keys import SequenceExampleTypeKey
from ml4ir.base.config.keys import TFRecordTypeKey


class FeatureLayerMap:
    """Class defining mapping from keys to feature layer functions"""

    # Constants
    GET_SEQUENCE_ENCODINg = "preprocess_text"

    def __init__(self):
        self.key_to_fn = {
            get_sequence_encoding.__name__: get_sequence_encoding,
            get_categorical_embedding.__name__: get_categorical_embedding,
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


def get_sequence_encoding(input_feature, feature_info):
    """Encode a sequence of numbers into a fixed size tensor"""
    feature_layer_info = feature_info["feature_layer_info"]

    input_feature = tf.reshape(input_feature, [-1, feature_layer_info["args"]["max_length"]])
    if "embedding_size" in feature_layer_info["args"]:
        char_embedding = layers.Embedding(
            input_dim=256,
            output_dim=feature_layer_info["args"]["embedding_size"],
            mask_zero=True,
            input_length=feature_layer_info["args"]["max_length"],
        )(input_feature)
    else:
        char_embedding = tf.one_hot(input_feature, depth=256)

    encoding = layers.Bidirectional(
        layers.LSTM(int(feature_layer_info["args"]["encoding_size"] / 2), return_sequences=False,),
        merge_mode="concat",
    )(char_embedding)
    encoding = tf.expand_dims(encoding, axis=1)
    # if feature_info.get("tfrecord_type") == SequenceExampleTypeKey.CONTEXT:
    #     # If feature is a context feature then tile it for all records
    #     encoding = tf.expand_dims(encoding, axis=1)
    # else:
    #     # If sequence feature, then reshape back to original shape
    #     # FIXME
    #     encoding = tf.reshape(
    #         encoding, [-1, encoding, feature_layer_info["args"]["encoding_size"]],
    #     )

    return encoding


def get_categorical_embedding(input_feature, feature_info):
    """Embedding lookup for categorical features"""

    # Numeric input features
    if feature_info["dtype"] in (tf.float32, tf.int64):
        raise NotImplementedError

    # String input features
    elif feature_info["dtype"] in (tf.string,):
        feature_layer_info = feature_info.get("feature_layer_info")
        embeddings_list = list()
        for i in range(feature_layer_info["args"]["num_categorical_features"]):
            # augmented_string = tf.strings.join([input_feature, tf.strings.as_string(tf.constant(i))])
            augmented_string = layers.Lambda(lambda x: tf.add(x, str(i)))(input_feature)

            hash_bucket = tf.strings.to_hash_bucket_fast(
                augmented_string, num_buckets=feature_layer_info["args"]["num_hash_buckets"]
            )
            embeddings_list.append(
                layers.Embedding(
                    input_dim=feature_layer_info["args"]["num_hash_buckets"],
                    output_dim=feature_layer_info["args"]["embedding_size"],
                    name="categorical_embedding_{}_{}".format(feature_info.get("name"), i),
                )(hash_bucket)
            )

        if feature_layer_info["args"]["merge_mode"] == "mean":
            return tf.reduce_mean(
                embeddings_list,
                axis=0,
                name="categorical_embedding_{}".format(feature_info.get("name")),
            )
        elif feature_layer_info["args"]["merge_mode"] == "sum":
            return tf.reduce_sum(
                embeddings_list,
                axis=0,
                name="categorical_embedding_{}".format(feature_info.get("name")),
            )
        elif feature_layer_info["args"]["merge_mode"] == "concat":
            return tf.concat(
                embeddings_list,
                axis=-1,
                name="categorical_embedding_{}".format(feature_info.get("name")),
            )


def define_example_feature_layer(
    feature_config: FeatureConfig, feature_layer_map: FeatureLayerMap
):
    """
    Add feature layer by processing the inputs
    NOTE: Embeddings or any other in-graph preprocessing goes here
    """

    def feature_layer(inputs):
        train_features = list()
        metadata_features = dict()

        for feature_info in feature_config.get_all_features(include_label=False):
            feature_name = feature_info["name"]
            feature_node_name = feature_info.get("node_name", feature_name)
            feature_layer_info = feature_info["feature_layer_info"]

            if feature_layer_info["type"] == FeatureTypeKey.NUMERIC:
                # Numeric input features
                if feature_info["dtype"] in (tf.float32, tf.int64):
                    dense_feature = inputs[feature_node_name]

                    if "fn" in feature_layer_info:
                        dense_feature = feature_layer_map.get_fn(feature_layer_info["fn"])(
                            dense_feature, feature_info
                        )
                    else:
                        dense_feature = tf.expand_dims(tf.cast(dense_feature, tf.float32), axis=-1)

                    if feature_info["trainable"]:
                        train_features.append(dense_feature)
                    else:
                        metadata_features[feature_node_name] = tf.cast(dense_feature, tf.float32)

                # String input features
                elif feature_info["dtype"] in (tf.string,):
                    if feature_info["trainable"]:
                        decoded_string_tensor = io.decode_raw(
                            inputs[feature_node_name],
                            out_type=tf.uint8,
                            fixed_length=feature_layer_info["args"]["max_length"],
                        )
                        if "fn" in feature_layer_info:
                            dense_feature = feature_layer_map.get_fn(feature_layer_info["fn"])(
                                decoded_string_tensor, feature_info
                            )
                        # encoding = get_sequence_encoding(decoded_string_tensor, feature_info)
                        """
                        Creating a tensor [1, sequence_size, 1] dynamically
                        NOTE:
                        Tried multiple methods using `convert_to_tensor`, `concat`, with no results
                        """
                        train_features.append(dense_feature)
            elif feature_layer_info["type"] == FeatureTypeKey.STRING:
                if feature_info["trainable"]:
                    raise ValueError(
                        "Can not train on string tensors directly. Please use a feature layer"
                    )
                else:
                    metadata_features[feature_node_name] = inputs[feature_node_name]
            elif feature_layer_info["type"] == FeatureTypeKey.CATEGORICAL:
                if feature_info["trainable"]:
                    if "fn" in feature_layer_info:
                        dense_feature = feature_layer_map.get_fn(feature_layer_info["fn"])(
                            inputs[feature_node_name], feature_info
                        )

                    train_features.append(dense_feature)
                else:
                    raise NotImplementedError
            else:
                raise Exception(
                    "Unknown feature type {} for feature : {}".format(
                        feature_layer_info["type"], feature_name
                    )
                )

        """
        Reshape ranking features to create features of shape
        [batch, max_sequence_size, num_features]
        """
        train_features = tf.concat(train_features, axis=-1, name="train_features")

        return train_features, metadata_features

    return feature_layer


def define_sequence_example_feature_layer(
    feature_config: FeatureConfig, feature_layer_map: FeatureLayerMap, max_sequence_size: int
):
    """
    Add feature layer by processing the inputs
    NOTE: Embeddings or any other in-graph preprocessing goes here
    """

    def feature_layer(inputs):
        train_features = list()
        metadata_features = dict()

        numeric_tile_shape = tf.shape(tf.expand_dims(tf.gather(inputs["mask"], indices=0), axis=0))

        for feature_info in feature_config.get_all_features(include_label=False):
            feature_name = feature_info["name"]
            feature_node_name = feature_info.get("node_name", feature_name)
            feature_layer_info = feature_info["feature_layer_info"]

            if feature_layer_info["type"] == FeatureTypeKey.NUMERIC:
                # Numeric input features
                if feature_info["dtype"] in (tf.float32, tf.int64):
                    dense_feature = inputs[feature_node_name]

                    if feature_info["tfrecord_type"] == SequenceExampleTypeKey.CONTEXT:
                        dense_feature = tf.tile(dense_feature, numeric_tile_shape)

                    if "fn" in feature_layer_info:
                        dense_feature = feature_layer_map.get_fn(feature_layer_info["fn"])(
                            dense_feature, feature_info
                        )
                    elif feature_info["trainable"]:
                        dense_feature = tf.expand_dims(tf.cast(dense_feature, tf.float32), axis=-1)

                    if feature_info["trainable"]:
                        train_features.append(dense_feature)
                    else:
                        metadata_features[feature_node_name] = tf.cast(dense_feature, tf.float32)

                # String input features
                elif feature_info["dtype"] in (tf.string,):
                    if feature_info["trainable"]:
                        decoded_string_tensor = io.decode_raw(
                            inputs[feature_node_name],
                            out_type=tf.uint8,
                            fixed_length=feature_layer_info["args"]["max_length"],
                        )
                        if "fn" in feature_layer_info:
                            dense_feature = feature_layer_map.get_fn(feature_layer_info["fn"])(
                                decoded_string_tensor, feature_info
                            )

                        """
                        Creating a tensor [1, sequence_size, 1] dynamically
                        NOTE:
                        Tried multiple methods using `convert_to_tensor`, `concat`, with no results
                        """
                        if feature_info["tfrecord_type"] == SequenceExampleTypeKey.CONTEXT:
                            tile_dims = tf.shape(
                                tf.expand_dims(
                                    tf.expand_dims(tf.gather(inputs["mask"], indices=0), axis=0),
                                    axis=-1,
                                )
                            )
                            dense_feature = tf.tile(dense_feature, tile_dims)

                        train_features.append(dense_feature)

            elif feature_layer_info["type"] == FeatureTypeKey.STRING:
                if feature_info["trainable"]:
                    raise ValueError(
                        "Can not train on string tensors directly. Please use a feature layer"
                    )
                else:
                    metadata_features[feature_node_name] = inputs[feature_node_name]
            elif feature_layer_info["type"] == FeatureTypeKey.CATEGORICAL:
                if feature_info["trainable"]:
                    if "fn" in feature_layer_info:
                        dense_feature = feature_layer_map.get_fn(feature_layer_info["fn"])(
                            inputs[feature_node_name], feature_info
                        )

                    if feature_info["tfrecord_type"] == SequenceExampleTypeKey.CONTEXT:
                        tile_dims = tf.shape(
                            tf.expand_dims(
                                tf.expand_dims(tf.gather(inputs["mask"], indices=0), axis=0),
                                axis=-1,
                            )
                        )
                        dense_feature = tf.tile(dense_feature, tile_dims)

                    train_features.append(dense_feature)
                else:
                    raise NotImplementedError
            else:
                raise Exception(
                    "Unknown feature type {} for feature : {}".format(
                        feature_layer_info["type"], feature_name
                    )
                )

        return train_features, metadata_features

    return feature_layer
