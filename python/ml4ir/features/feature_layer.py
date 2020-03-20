import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.keras import backend as K

from ml4ir.features.feature_config import FeatureConfig
from ml4ir.config.keys import FeatureTypeKey, EmbeddingTypeKey, TFRecordTypeKey


def get_dense_feature(inputs, feature, shape=(1,)):
    """
    Convert an input into a dense numeric feature

    NOTE: Can remove this in the future and
          pass inputs[feature] directly
    """
    feature_col = feature_column.numeric_column(feature, shape=shape)
    dense_feature = layers.DenseFeatures(feature_col)(inputs)

    return dense_feature


def get_sequence_embedding(input, feature_info, max_num_records):
    feature_layer_info = feature_info["feature_layer_info"]
    preprocessing_info = feature_info.get("preprocessing_info", {})

    input_feature = tf.cast(input, tf.uint8)
    input_feature = tf.reshape(input_feature, [-1, preprocessing_info["max_length"]])
    if feature_layer_info["embedding_size"]:
        char_embedding = layers.Embedding(
            input_dim=256,
            output_dim=feature_layer_info["embedding_size"],
            mask_zero=True,
            input_length=preprocessing_info["max_length"],
        )(input_feature)
    else:
        char_embedding = tf.one_hot(input_feature, depth=256)

    encoding = layers.Bidirectional(
        layers.LSTM(int(feature_layer_info["encoding_size"] / 2), return_sequences=False,),
        merge_mode="concat",
    )(char_embedding)
    if feature_info["tfrecord_type"] == TFRecordTypeKey.CONTEXT:
        # If feature is a context feature then tile it for all records
        encoding = tf.expand_dims(encoding, axis=1)
        encoding = K.repeat_elements(encoding, rep=max_num_records, axis=1)

    else:
        # If sequence feature, then reshape back to original shape
        encoding = tf.reshape(encoding, [-1, encoding, feature_layer_info["embedding_size"]],)

    return encoding


def define_feature_layer(feature_config: FeatureConfig, max_num_records: int):
    """
    Add feature layer by processing the inputs
    NOTE: Embeddings or any other in-graph preprocessing goes here
    """

    def feature_layer(inputs):
        ranking_features = list()
        metadata_features = dict()

        for feature_info in feature_config.get_all_features(include_label=False):
            feature_name = feature_info["name"]
            feature_node_name = feature_info.get("node_name", feature_name)
            feature_layer_info = feature_info["feature_layer_info"]

            if feature_layer_info["type"] == FeatureTypeKey.NUMERIC:
                dense_feature = get_dense_feature(
                    inputs, feature_node_name, shape=(max_num_records, 1)
                )
                if feature_info["trainable"]:
                    dense_feature = tf.expand_dims(tf.cast(dense_feature, tf.float32), axis=-1)
                    ranking_features.append(dense_feature)
                else:
                    metadata_features[feature_node_name] = tf.cast(dense_feature, tf.float32)
            elif feature_layer_info["type"] == FeatureTypeKey.STRING:
                if feature_info["trainable"]:
                    if feature_layer_info["encoding_type"] == EmbeddingTypeKey.BILSTM:
                        embedding = get_sequence_embedding(
                            inputs[feature_node_name], feature_info, max_num_records
                        )

                        ranking_features.append(embedding)
                    else:
                        raise NotImplementedError
            elif feature_layer_info["type"] == FeatureTypeKey.CATEGORICAL:
                # TODO: Add embedding layer with vocabulary here
                raise NotImplementedError
            else:
                raise Exception(
                    "Unknown feature type {} for feature : {}".format(
                        feature_layer_info["type"], feature_name
                    )
                )

        """
        Reshape ranking features to create features of shape
        [batch, max_num_records, num_features]
        """
        ranking_features = tf.concat(ranking_features, axis=-1)

        return ranking_features, metadata_features

    return feature_layer
