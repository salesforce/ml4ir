import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import io
import numpy as np

from ml4ir.base.io.file_io import FileIO
from ml4ir.base.features.feature_fns.base import BaseFeatureLayerOp


class BytesSequenceToEncodingBiLSTM(BaseFeatureLayerOp):
    """
    Encode a string tensor into an encoding.
    Works by converting the string into a bytes sequence and then generating
    a categorical/char embedding for each of the 256 bytes. The char/byte embeddings
    are then combined using a biLSTM
    """
    LAYER_NAME = "bytes_sequence_to_encoding_bilstm"

    MAX_LENGTH = "max_length"
    EMBEDDING_SIZE = "embedding_size"
    LSTM_KERNEL_INITIALIZER = "lstm_kernel_initializer"
    ENCODING_SIZE = "encoding_size"

    def __init__(self, feature_info: dict, file_io: FileIO, **kwargs):
        """
        Initialize a feature layer to convert string tensor to bytes encoding

        Parameters
        ----------
        feature_info : dict
            Dictionary representing the feature_config for the input feature
        file_io : FileIO object
            FileIO handler object for reading and writing

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
        super().__init__(feature_info=feature_info, file_io=file_io, **kwargs)

        self.max_length = self.feature_layer_args.get(self.MAX_LENGTH, None)
        self.embedding_size = self.feature_layer_args.get(self.EMBEDDING_SIZE)
        self.encoding_size = self.feature_layer_args[self.ENCODING_SIZE]
        self.kernel_initializer = self.feature_layer_args.get(self.LSTM_KERNEL_INITIALIZER, "glorot_uniform")

        if self.EMBEDDING_SIZE in self.feature_layer_args:
            self.char_embedding = layers.Embedding(
                name="{}_bytes_embedding".format(self.feature_name),
                input_dim=256,
                output_dim=self.embedding_size,
                mask_zero=True,
                input_length=self.max_length,
            )
        else:
            self.char_embedding = lambda feature_tensor: tf.one_hot(
                feature_tensor,
                depth=256,
                name="{}_bytes_embedding".format(self.feature_name))

        self.encoding_op = layers.Bidirectional(
            layers.LSTM(
                units=int(self.encoding_size / 2),
                return_sequences=False,
                kernel_initializer=self.kernel_initializer
            ),
            merge_mode="concat",
            name="{}_bilstm_encoding".format(self.feature_name)
        )

    def call(self, inputs, training=None):
        """
        Defines the forward pass for the layer on the inputs tensor

        Parameters
        ----------
        inputs: tensor
            Input tensor on which the feature transforms are applied
        training: boolean
            Boolean flag indicating if the layer is being used in training mode or not

        Returns
        -------
        tf.Tensor
            Resulting tensor after the forward pass through the feature transform layer
        """
        # Decode string tensor to bytes
        feature_tensor = io.decode_raw(inputs, out_type=tf.uint8, fixed_length=self.max_length)
        feature_tensor = tf.squeeze(feature_tensor, axis=1)

        feature_tensor = self.char_embedding(feature_tensor, training=training)
        feature_tensor = self.encoding_op(feature_tensor, training=training)

        feature_tensor = tf.expand_dims(feature_tensor, axis=1)

        return feature_tensor


class Global1dPooling(BaseFeatureLayerOp):
    """
    1D pooling to reduce a variable length sequence feature into a scalar
    value. This method optionally allows users to add multiple such pooling
    operations to produce a fixed dimensional feature vector as well.
    """
    LAYER_NAME = "global_1d_pooling"

    FNS = "fns"
    PADDED_VAL = "padded_val"
    MASKED_MAX_VAL = "masked_max_val"
    DEFAULT_MASKED_MAX_VAL = 2.0

    def __init__(self, feature_info: dict, file_io: FileIO, **kwargs):
        """
        Initialize a feature layer to apply global 1D pooling operation on input tensor

        Parameters
        ----------
        feature_info : dict
            Dictionary representing the feature_config for the input feature
        file_io : FileIO object
            FileIO handler object for reading and writing

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
        super().__init__(feature_info=feature_info, file_io=file_io, **kwargs)

        self.fns = self.feature_layer_args[self.FNS]
        # Check if at least one pooling fn is specified
        if len(self.fns) == 0:
            raise ValueError(
                "At least 1 pooling function should be specified. Found : {}".format(len(self.fns))
            )

        # Define the masking value for each pooling op
        self.masked_val_lookup = {
            "sum": 0.0,
            "mean": 0.0,
            "max": - self.feature_layer_args.get(self.MASKED_MAX_VAL, self.DEFAULT_MASKED_MAX_VAL),
            "min": self.feature_layer_args.get(self.MASKED_MAX_VAL, self.DEFAULT_MASKED_MAX_VAL),
            "count_nonzero": 0.0
        }

        self.padded_val = self.feature_layer_args.get(self.PADDED_VAL)

    def call(self, inputs, training=None):
        """
        Defines the forward pass for the layer on the inputs tensor

        Parameters
        ----------
        inputs: tensor
            Input tensor on which the feature transforms are applied
        training: boolean
            Boolean flag indicating if the layer is being used in training mode or not

        Returns
        -------
        tf.Tensor
            Resulting tensor after the forward pass through the feature transform layer
        """
        pooled_tensors = list()
        padded_val_tensor = None

        if self.padded_val:
            padded_val_tensor = tf.constant(self.padded_val, dtype=inputs.dtype)

        # Apply the 1D global pooling on the last dimension
        for fn in self.fns:
            # If padded_val is passed, mask the padded values appropriately
            if self.padded_val:
                masked_feature_tensor = tf.where(
                    tf.equal(inputs, padded_val_tensor), self.masked_val_lookup[fn], inputs
                )
            else:
                masked_feature_tensor = inputs

            # Apply each pooling op
            if fn == "sum":
                pooled_tensors.append(tf.math.reduce_sum(masked_feature_tensor, axis=-1))

            elif fn == "mean":
                if "padded_val" in self.feature_layer_args:
                    # NOTE: To avoid division by zero, we set 0 to a small value of 1e-10
                    pooled_tensors.append(
                        tf.math.divide(
                            tf.math.reduce_sum(masked_feature_tensor, axis=-1),
                            tf.math.reduce_sum(
                                tf.where(
                                    tf.equal(inputs, padded_val_tensor),
                                    tf.constant(1e-10, dtype=inputs.dtype),
                                    tf.constant(1, dtype=inputs.dtype),
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
            name="global_1d_pooling_{}".format(self.feature_name),
        )
