import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import io
import numpy as np

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


def global_1d_pooling(feature_tensor, feature_info, file_io: FileIO):
    """
    1D pooling to reduce a variable length sequence feature into a scalar
    value. This method optionally allows users to add multiple such pooling
    operations to produce a fixed dimensional feature vector as well.

    Parameters
    ----------
    feature_tensor : Tensor object
        String feature tensor that is to be aggregated/pooled
        Dimensions -> [batch_size, max_sequence_size, max_len]
    feature_info : dict
        Dictionary representing the feature_config for the input feature
    file_io : FileIO object
        FileIO handler object for reading and writing

    Returns
    -------
    Tensor object
        Global pooled/aggregated feature vector

    Notes
    -----
    Args under `feature_layer_info`:
        fns : list of str
            List of string pooling operations that should be applied.
            Must be one of ["sum", "mean", "max", "min", "count_nonzero"]
        padded_val : int/float
            Value to be ignored from the pooling operations.
        masked_max_val : int/float
            Value used to mask the padded values for computing the max and min
            pooling operations. This allows us to ignore these values in the min
            and max pool operations. For example, if all the values in the tensor
            are in [0., 1.], then a masked_max_val of > 1. will make sure we do
            not pick padded values in the min/max pooling ops.
            Default value: 2
    """
    args = feature_info["feature_layer_info"]["args"]
    pooled_tensors = list()

    # Check if at least one pooling fn is specified
    if len(args["fns"]) == 0:
        raise ValueError(
            "At least 1 pooling function should be specified. Found : {}".format(len(args["fns"]))
        )

    # Define the masking value for each pooling op
    DEFAULT_MASKED_MAX_VAL = 2.0
    masked_val_lookup = {
        "sum": 0.0,
        "mean": 0.0,
        "max": - args.get("masked_max_val", DEFAULT_MASKED_MAX_VAL),
        "min": args.get("masked_max_val", DEFAULT_MASKED_MAX_VAL),
        "count_nonzero": 0.0
    }

    padded_val_tensor = None
    if "padded_val" in args:
        padded_val_tensor = tf.constant(args["padded_val"], dtype=feature_tensor.dtype)

    # Apply the 1D global pooling on the last dimension
    for fn in args["fns"]:
        # If padded_val is passed, mask the padded values appropriately
        if "padded_val" in args:
            masked_feature_tensor = tf.where(
                tf.equal(feature_tensor, padded_val_tensor), masked_val_lookup[fn], feature_tensor
            )
        else:
            masked_feature_tensor = feature_tensor

        # Apply each pooling op
        if fn == "sum":
            pooled_tensors.append(tf.math.reduce_sum(masked_feature_tensor, axis=-1))

        elif fn == "mean":
            if "padded_val" in args:
                # NOTE: To avoid division by zero, we set 0 to a small value of 1e-10
                pooled_tensors.append(
                    tf.math.divide(
                        tf.math.reduce_sum(masked_feature_tensor, axis=-1),
                        tf.math.reduce_sum(
                            tf.where(
                                tf.equal(feature_tensor, padded_val_tensor),
                                tf.constant(1e-10, dtype=feature_tensor.dtype),
                                tf.constant(1, dtype=feature_tensor.dtype),
                            ),
                            axis=-1,
                        ),
                    )
                )
            else:
                pooled_tensors.append(tf.math.reduce_mean(masked_feature_tensor, axis=-1))

        elif fn == "max":
            pooled_tensors.append(tf.math.reduce_max(masked_feature_tensor, axis=-1))

        elif fn == "min":
            pooled_tensors.append(tf.math.reduce_min(masked_feature_tensor, axis=-1))

        elif fn == "count_nonzero":
            pooled_tensors.append(
                tf.cast(
                    tf.math.count_nonzero(masked_feature_tensor, axis=-1),
                    dtype=masked_feature_tensor.dtype,
                )
            )

        else:
            raise KeyError(
                "{} pooling function not supported. Please use one of sum, mean, max, min, count_nonzero.".format(
                    fn
                )
            )

    # Stack all the pooled tensors to get one fixed dimensional dense vector
    return tf.stack(
        pooled_tensors,
        axis=-1,
        name="global_1d_pooling_{}".format(
            feature_info.get("node_name", feature_info.get("name"))
        ),
    )
