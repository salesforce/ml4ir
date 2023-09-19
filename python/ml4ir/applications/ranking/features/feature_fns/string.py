import tensorflow as tf

from ml4ir.base.features.feature_fns.base import BaseFeatureLayerOp
from ml4ir.base.io.file_io import FileIO


class QueryLength(BaseFeatureLayerOp):
    """
    Compute the length of the query string context feature
    """
    LAYER_NAME = "query_length"

    TOKENIZE = "tokenize"
    SEPARATOR = "sep"

    def __init__(self, feature_info: dict, file_io: FileIO, **kwargs):
        """
        Initialize layer to define a theoretical min max normalization

        Parameters
        ----------
        feature_info : dict
            Dictionary representing the configuration parameters for the specific feature from the FeatureConfig
        file_io : FileIO object
            FileIO handler object for reading and writing

        Notes
        -----
        Args under feature_layer_info:
            tokenize: boolean
                Whether to tokenize the string before counting length
                Defaults to true
            sep : string
                String char used to split the query, to count number of tokens
                Defaults to space " "
        TODO: In the future, we might want to support custom tokenizers to split the string.
        """
        super().__init__(feature_info=feature_info, file_io=file_io, **kwargs)

        self.tokenize = self.feature_layer_args.get(self.TOKENIZE, True)
        self.sep = self.feature_layer_args.get(self.SEPARATOR, " ")


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
        if self.tokenize:
            query_len = tf.strings.split(inputs, sep=self.sep).row_lengths(axis=-1).to_tensor()
        else:
            query_len = tf.strings.length(inputs)

        query_len = tf.expand_dims(tf.cast(query_len, tf.float32), axis=-1)
        return query_len