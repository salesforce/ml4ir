import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import io


def bytes_sequence_to_encoding_bilstm(feature_tensor, feature_info):
    """
    Encode a string tensor into an encoding.
    Works by converting the string into a bytes sequence and then generating
    a categorical/char embedding for each of the 256 bytes. The char/byte embeddings
    are then combined using a biLSTM

    Args:
        feature_tensor: String feature tensor that is to be encoded
        feature_info: Dictionary representing the feature_config for the input feature

    Returns:
        Encoded feature tensor

    Args under feature_layer_info:
        max_length: int; max length of bytes sequence
        embedding_size: int; dimension size of the embedding;
                        if null, then the tensor is just converted to its one-hot representation
        encoding_size: int: dimension size of the sequence encoding computed using a biLSTM

    NOTE:
        The input dimension for the embedding is fixed to 256 because the string is
        converted into a bytes sequence.
    """
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

    return encoding
