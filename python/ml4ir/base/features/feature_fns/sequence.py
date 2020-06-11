import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import io


def bytes_sequence_to_encoding(feature_tensor, feature_info):
    """Encode a sequence of numbers into a fixed size tensor"""
    feature_layer_info = feature_info["feature_layer_info"]

    # Decode string tensor to bytes
    feature_tensor = io.decode_raw(
        feature_tensor,
        out_type=tf.uint8,
        fixed_length=feature_layer_info["args"].get("max_length", None),
    )

    feature_tensor = tf.reshape(feature_tensor, [-1, feature_layer_info["args"]["max_length"]])
    if "embedding_size" in feature_layer_info["args"]:
        char_embedding = layers.Embedding(
            input_dim=256,
            output_dim=feature_layer_info["args"]["embedding_size"],
            mask_zero=True,
            input_length=feature_layer_info["args"]["max_length"],
        )(feature_tensor)
    else:
        char_embedding = tf.one_hot(feature_tensor, depth=256)

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
