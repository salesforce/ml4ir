import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import io

from ml4ir.base.io.file_io import FileIO


def bytes_sequence_to_encoding_bilstm(feature_tensor, feature_info, file_io: FileIO):
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
    args = feature_info["feature_layer_info"]["args"]

    # Decode string tensor to bytes
    feature_tensor = io.decode_raw(
        feature_tensor, out_type=tf.uint8, fixed_length=args.get("max_length", None),
    )

    feature_tensor = tf.squeeze(feature_tensor, axis=1)
    if "embedding_size" in args:
        char_embedding = layers.Embedding(
            name="{}_bytes_embedding".format(
                feature_info.get("node_name", feature_info.get("name"))
            ),
            input_dim=256,
            output_dim=args["embedding_size"],
            mask_zero=True,
            input_length=args.get("max_length", None),
        )(feature_tensor)
    else:
        char_embedding = tf.one_hot(feature_tensor, depth=256)

    encoding = get_bilstm_encoding(char_embedding, int(args["encoding_size"] / 2))

    return encoding


def get_bilstm_encoding(embedding, units):
    """
    Builds a bilstm on to on the embedding passed as input.
    """
    encoding = layers.Bidirectional(
        layers.LSTM(units=units, return_sequences=False), merge_mode="concat",
    )(embedding)
    encoding = tf.expand_dims(encoding, axis=1)
    return encoding
