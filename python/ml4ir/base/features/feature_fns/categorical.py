import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import feature_column
from tensorflow import lookup

import copy

from ml4ir.base.io import file_io


def categorical_embedding_with_hash_buckets(feature_tensor, feature_info):
    """Embedding lookup for categorical features"""

    # Numeric input features
    if feature_info["dtype"] in (tf.float32, tf.int64):
        raise NotImplementedError

    # String input features
    elif feature_info["dtype"] in (tf.string,):
        feature_layer_info = feature_info.get("feature_layer_info")
        embeddings_list = list()
        for i in range(feature_layer_info["args"]["num_categorical_features"]):
            # augmented_string = tf.strings.join([feature_tensor, tf.strings.as_string(tf.constant(i))])
            augmented_string = layers.Lambda(lambda x: tf.add(x, str(i)))(feature_tensor)

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

        embedding = None
        if feature_layer_info["args"]["merge_mode"] == "mean":
            embedding = tf.reduce_mean(
                embeddings_list,
                axis=0,
                name="categorical_embedding_{}".format(feature_info.get("name")),
            )
        elif feature_layer_info["args"]["merge_mode"] == "sum":
            embedding = tf.reduce_sum(
                embeddings_list,
                axis=0,
                name="categorical_embedding_{}".format(feature_info.get("name")),
            )
        elif feature_layer_info["args"]["merge_mode"] == "concat":
            embedding = tf.concat(
                embeddings_list,
                axis=-1,
                name="categorical_embedding_{}".format(feature_info.get("name")),
            )

        # embedding = tf.expand_dims(embedding, axis=-1)

        return embedding


def categorical_embedding_with_indices(feature_tensor, feature_info):
    """Embedding lookup for categorical features which already are converted to numeric indices"""
    CATEGORICAL_VARIABLE = "categorical_variable"
    feature_layer_info = feature_info.get("feature_layer_info")

    categorical_fc = feature_column.categorical_column_with_identity(
        CATEGORICAL_VARIABLE,
        num_buckets=feature_layer_info["args"]["num_buckets"],
        default_value=feature_layer_info["args"].get("default_value", None),
    )
    embedding_fc = feature_column.embedding_column(
        categorical_fc, dimension=feature_layer_info["args"]["embedding_size"]
    )

    embedding = layers.DenseFeatures(
        embedding_fc,
        name="{}_embedding".format(feature_info.get("node_name", feature_info["name"])),
    )({CATEGORICAL_VARIABLE: feature_tensor})
    embedding = tf.expand_dims(embedding, axis=1)

    return embedding


class VocabLookup(layers.Layer):
    """
    Defines a keras layer around a tf lookup table using the given vocabulary list

    NOTE:
    Issue[1] with using LookupTable with keras symbolic tensors; expects eager tensors.

    Ref: https://github.com/tensorflow/tensorflow/issues/38305
    """

    def __init__(self, vocabulary_list, num_oov_buckets, feature_name):
        super(VocabLookup, self).__init__(trainable=False, dtype=tf.int64)
        self.vocabulary_list = vocabulary_list
        self.vocabulary_size = len(vocabulary_list)
        self.num_oov_buckets = num_oov_buckets
        self.feature_name = feature_name

    def build(self, input_shape):
        table_init = lookup.KeyValueTensorInitializer(
            keys=self.vocabulary_list,
            values=range(self.vocabulary_size),
            key_dtype=tf.string,
            value_dtype=tf.int64,
        )
        self.lookup_table = lookup.StaticVocabularyTable(
            initializer=table_init,
            num_oov_buckets=self.num_oov_buckets,
            name="{}_lookup_table".format(self.feature_name),
        )
        self.built = True

    def call(self, input_text):
        return self.lookup_table.lookup(input_text)

    def get_config(self):
        config = super(VocabLookup, self).get_config()
        config.update(
            {
                "vocabulary_list": self.vocabulary_list,
                "vocabulary_size": self.vocabulary_size,
                "num_oov_buckets": self.num_oov_buckets,
                "feature_name": self.feature_name,
            }
        )
        return config


def categorical_embedding_with_vocabulary_file(feature_tensor, feature_info):
    """
    Embedding lookup for string features with a vocabulary file to index

    NOTE:
    Current bug[1] with saving a Keras model when using
    feature_column.categorical_column_with_vocabulary_list.
    Tracking the issue currently and should be able to upgrade
    to current latest stable release 2.2.0 to test.

    Can not use TF2.1.0 due to issue[2] regarding saving Keras models with
    custom loss, metric layers

    Can not use TF2.2.0 due to issues[3, 4] regarding incompatibility of
    Keras Functional API models and Tensorflow

    References:
    [1] https://github.com/tensorflow/tensorflow/issues/31686
    [2] https://github.com/tensorflow/tensorflow/issues/36954
    [3] https://github.com/tensorflow/probability/issues/519
    [4] https://github.com/tensorflow/tensorflow/issues/35138
    """
    feature_layer_info = feature_info.get("feature_layer_info")
    vocabulary_list = file_io.read_list(feature_layer_info["args"]["vocabulary_file"])

    #
    ##########################################################################
    #
    # CATEGORICAL_VARIABLE = "categorical_variable"
    # categorical_fc = feature_column.categorical_column_with_vocabulary_list(
    #     CATEGORICAL_VARIABLE,
    #     vocabulary_list=vocabulary_list,
    #     default_value=feature_layer_info["args"].get("default_value", -1),
    #     num_oov_buckets=feature_layer_info["args"].get("num_oov_buckets", 0),
    # )
    # embedding_fc = feature_column.embedding_column(
    #     categorical_fc, dimension=feature_layer_info["args"]["embedding_size"]
    # )

    # embedding = layers.DenseFeatures(
    #     embedding_fc,
    #     name="{}_embedding".format(feature_info.get("node_name", feature_info["name"])),
    # )({CATEGORICAL_VARIABLE: feature_tensor})
    # embedding = tf.expand_dims(embedding, axis=1)
    #
    ##########################################################################
    #

    # Define a lookup table to convert categorical variables to IDs
    num_oov_buckets = feature_layer_info["args"].get("num_oov_buckets", 1)
    vocabulary_size = len(vocabulary_list)
    lookup_table = VocabLookup(
        vocabulary_list=vocabulary_list,
        num_oov_buckets=num_oov_buckets,
        feature_name=feature_info.get("node_name", feature_info["name"]),
    )
    feature_tensor_indices = lookup_table(feature_tensor)

    feature_info_new = copy.deepcopy(feature_info)
    feature_info_new["feature_layer_info"]["args"]["num_buckets"] = (
        vocabulary_size + num_oov_buckets
    )
    feature_info_new["feature_layer_info"]["args"]["default_value"] = vocabulary_size

    embedding = categorical_embedding_with_indices(
        feature_tensor=feature_tensor_indices, feature_info=feature_info_new
    )

    return embedding