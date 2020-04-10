import tensorflow as tf
from tensorflow.keras import layers

from ml4ir.features.feature_config import FeatureConfig
from ml4ir.config.keys import FeatureTypeKey, EncodingTypeKey, TFRecordTypeKey


def get_sequence_encoding(input, feature_info):
    feature_layer_info = feature_info["feature_layer_info"]
    preprocessing_info = feature_info.get("preprocessing_info", {})

    input_feature = tf.cast(input, tf.uint8)
    input_feature = tf.reshape(input_feature, [-1, preprocessing_info["max_length"]])
    if "embedding_size" in feature_layer_info:
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
    else:
        # If sequence feature, then reshape back to original shape
        # FIXME
        encoding = tf.reshape(encoding, [-1, encoding, feature_layer_info["encoding_size"]],)

    return encoding


def define_feature_layer(feature_config: FeatureConfig, max_num_records: int):
    """
    Add feature layer by processing the inputs
    NOTE: Embeddings or any other in-graph preprocessing goes here
    """

    def feature_layer(inputs):
        ranking_features = list()
        metadata_features = dict()

        numeric_tile_shape = tf.shape(tf.expand_dims(tf.gather(inputs["mask"], indices=0), axis=0))

        for feature_info in feature_config.get_all_features(include_label=False):
            feature_name = feature_info["name"]
            feature_node_name = feature_info.get("node_name", feature_name)
            feature_layer_info = feature_info["feature_layer_info"]

            if feature_layer_info["type"] == FeatureTypeKey.NUMERIC:
                dense_feature = inputs[feature_node_name]

                if feature_info["tfrecord_type"] == TFRecordTypeKey.CONTEXT:
                    dense_feature = tf.tile(dense_feature, numeric_tile_shape, name="broooo")

                if feature_info["trainable"]:
                    dense_feature = tf.expand_dims(tf.cast(dense_feature, tf.float32), axis=-1)
                    ranking_features.append(dense_feature)
                else:
                    metadata_features[feature_node_name] = tf.cast(dense_feature, tf.float32)
            elif feature_layer_info["type"] == FeatureTypeKey.STRING:
                if feature_info["trainable"]:
                    if feature_layer_info["encoding_type"] == EncodingTypeKey.BILSTM:
                        encoding = get_sequence_encoding(inputs[feature_node_name], feature_info)
                        """
                        Creating a tensor [1, num_records, 1] dynamically

                        NOTE:
                        Tried multiple methods using `convert_to_tensor`, `concat`, with no results
                        """
                        tile_dims = tf.shape(
                            tf.expand_dims(
                                tf.expand_dims(tf.gather(inputs["mask"], indices=0), axis=0),
                                axis=-1,
                            )
                        )
                        encoding = tf.tile(encoding, tile_dims)

                        ranking_features.append(encoding)
                    else:
                        raise NotImplementedError
                # pass
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
        ranking_features = tf.concat(ranking_features, axis=-1, name="ranking_features")

        return ranking_features, metadata_features

    return feature_layer
