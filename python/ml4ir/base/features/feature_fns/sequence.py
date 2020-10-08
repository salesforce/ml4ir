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

    Parameters
    ----------
    feature_tensor : Tensor object
        String feature tensor that is to be encoded
    feature_info : dict
        Dictionary representing the feature_config for the input feature
    file_io : FileIO object
        FileIO handler object for reading and writing

    Returns
    -------
    Tensor object
        Encoded feature tensor

    Notes
    -----
    Args under `feature_layer_info`:
        max_length : int
            max length of bytes sequence
        embedding_size : int
            dimension size of the embedding;
            if null, then the tensor is just converted to its one-hot representation
        encoding_size : int
            dimension size of the sequence encoding computed using a biLSTM

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

    kernel_initializer = args.get("lstm_kernel_initializer", "glorot_uniform")
    encoding = get_bilstm_encoding(
        embedding=char_embedding,
        lstm_units=int(args["encoding_size"] / 2),
        kernel_initializer=kernel_initializer,
    )
    return encoding


def get_bilstm_encoding(embedding, lstm_units, kernel_initializer="glorot_uniform"):
    """
    Convert sequence into encoding by passing through bidirectional LSTM

    Parameters
    ----------
    sequence_tensor : Tensor object
        Sequence tensor with representations for each time step
    lstm_units : int
        Number of units in the LSTM
    kernel_initializer : str
        Any supported tf.keras.initializers
        e.g., 'ones', 'glorot_uniform', 'lecun_normal' ...

    Returns
    -------
    Tensor object
        Encoded feature tensor
    """
    encoding = layers.Bidirectional(
        layers.LSTM(
            units=lstm_units, return_sequences=False, kernel_initializer=kernel_initializer
        ),
        merge_mode="concat",
    )(embedding)
    encoding = tf.expand_dims(encoding, axis=1)
    return encoding
