import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import io

from ml4ir.features.feature_config import FeatureConfig
from ml4ir.config.keys import FeatureTypeKey, EncodingTypeKey, TFRecordTypeKey


def get_sequence_encoding(input_feature, feature_info):
    """Encode a sequence of numbers into a fixed size tensor"""
    feature_layer_info = feature_info["feature_layer_info"]

    input_feature = tf.reshape(input_feature, [-1, feature_layer_info["max_length"]])
    if "embedding_size" in feature_layer_info:
        char_embedding = layers.Embedding(
            input_dim=256,
            output_dim=feature_layer_info["embedding_size"],
            mask_zero=True,
            input_length=feature_layer_info["max_length"],
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


def get_categorical_embedding(input_feature, feature_info):
    """Embedding lookup for categorical features"""

    # Numeric input features
    if feature_info["dtype"] in (tf.float32, tf.int64):
        raise NotImplementedError

    # String input features
    elif feature_info["dtype"] in (tf.string,):
        feature_layer_info = feature_info.get("feature_layer_info")
        embeddings_list = list()
        for i in range(feature_layer_info["num_categorical_features"]):
            # augmented_string = tf.strings.join([input_feature, tf.strings.as_string(tf.constant(i))])
            augmented_string = layers.Lambda(lambda x: tf.add(x, str(i)))(input_feature)

            hash_bucket = tf.strings.to_hash_bucket_fast(
                augmented_string, num_buckets=feature_layer_info["num_hash_buckets"]
            )
            embeddings_list.append(
                layers.Embedding(
                    input_dim=feature_layer_info["num_hash_buckets"],
                    output_dim=feature_layer_info["embedding_size"],
                    name="categorical_embedding_{}_{}".format(feature_info.get("name"), i),
                )(hash_bucket)
            )

        if feature_layer_info["merge_mode"] == "mean":
            return tf.reduce_mean(
                embeddings_list,
                axis=0,
                name="categorical_embedding_{}".format(feature_info.get("name")),
            )
        elif feature_layer_info["merge_mode"] == "sum":
            return tf.reduce_sum(
                embeddings_list,
                axis=0,
                name="categorical_embedding_{}".format(feature_info.get("name")),
            )
        elif feature_layer_info["merge_mode"] == "concat":
            return tf.concat(
                embeddings_list,
                axis=-1,
                name="categorical_embedding_{}".format(feature_info.get("name")),
            )


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
                # Numeric input features
                if feature_info["dtype"] in (tf.float32, tf.int64):
                    dense_feature = inputs[feature_node_name]

                    if feature_info["tfrecord_type"] == TFRecordTypeKey.CONTEXT:
                        dense_feature = tf.tile(dense_feature, numeric_tile_shape)

                    if feature_info["trainable"]:
                        dense_feature = tf.expand_dims(tf.cast(dense_feature, tf.float32), axis=-1)
                        ranking_features.append(dense_feature)
                    else:
                        metadata_features[feature_node_name] = tf.cast(dense_feature, tf.float32)

                # String input features
                elif feature_info["dtype"] in (tf.string,):
                    if feature_info["trainable"]:
                        if feature_layer_info["encoding_type"] == EncodingTypeKey.BILSTM:
                            decoded_string_tensor = io.decode_raw(
                                inputs[feature_node_name],
                                out_type=tf.uint8,
                                fixed_length=feature_layer_info["max_length"],
                            )
                            encoding = get_sequence_encoding(decoded_string_tensor, feature_info)
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
            elif feature_layer_info["type"] == FeatureTypeKey.STRING:
                pass
            elif feature_layer_info["type"] == FeatureTypeKey.CATEGORICAL:
                if feature_info["trainable"]:
                    categorical_embedding = get_categorical_embedding(
                        inputs[feature_node_name], feature_info
                    )

                    tile_dims = tf.shape(
                        tf.expand_dims(
                            tf.expand_dims(tf.gather(inputs["mask"], indices=0), axis=0), axis=-1,
                        )
                    )
                    categorical_embedding = tf.tile(categorical_embedding, tile_dims)

                    ranking_features.append(categorical_embedding)
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
