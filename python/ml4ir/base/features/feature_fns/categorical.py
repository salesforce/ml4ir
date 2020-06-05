import tensorflow as tf
from tensorflow.keras import layers


def categorical_embedding_with_hash_buckets(input_feature, feature_info):
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


def categorical_embedding_with_indices(input_feature, feature_info):
    """Embedding lookup for categorical features which already are converted to numeric indices"""

    feature_layer_info = feature_info.get("feature_layer_info")
    return layers.Embedding(
        input_dim=feature_layer_info["args"]["vocabulary_size"],
        output_dim=feature_layer_info["args"]["embedding_size"],
        name="categorical_embedding_{}".format(feature_info.get("name")),
    )(input_feature)


def categorical_embedding_with_vocabulary_file(input_feature, feature_info):
    """
    Embedding lookup for string features with a vocabulary file to index
    """
    raise NotImplementedError
