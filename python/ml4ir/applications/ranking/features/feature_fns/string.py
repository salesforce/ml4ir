import string
import re
import tensorflow as tf

from ml4ir.base.features.feature_fns.base import BaseFeatureLayerOp
from ml4ir.base.io.file_io import FileIO
from ml4ir.applications.ranking.features.feature_fns.categorical import CategoricalVector


class QueryLength(BaseFeatureLayerOp):
    """
    Compute the length of the query string context feature
    """
    LAYER_NAME = "query_length"

    TOKENIZE = "tokenize"
    SEPARATOR = "sep"

    def __init__(self, feature_info: dict, file_io: FileIO, **kwargs):
        """
        Initialize layer to define a query length feature transform

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


class QueryTypeVector(BaseFeatureLayerOp):
    """
    Compute the length of the query string context feature
    """
    LAYER_NAME = "query_type_vector"

    REMOVE_SPACES = "remove_spaces"
    REMOVE_QUOTES = "remove_quotes"
    OUTPUT_MODE = "output_mode"
    EMBEDDING_SIZE = "embedding_size"

    ALPHA_QUERY_TYPE = "alpha"
    NUMERIC_QUERY_TYPE = "num"
    PUNCTUATION_QUERY_TYPE = "punct"

    ALPHA_REGEX = r"^.*[a-zA-Z]+.*$"
    NUMERIC_REGEX = r"^.*[0-9]+.*$"
    PUNCTUATION_REGEX = "^.*[" + "".join([re.escape(c) for c in list(string.punctuation)]) + "]+.*$"


    VOCABULARY = [
        ALPHA_QUERY_TYPE,
        NUMERIC_QUERY_TYPE,
        PUNCTUATION_QUERY_TYPE,
        ALPHA_QUERY_TYPE + NUMERIC_QUERY_TYPE,
        ALPHA_QUERY_TYPE + PUNCTUATION_QUERY_TYPE,
        NUMERIC_QUERY_TYPE + PUNCTUATION_QUERY_TYPE,
        ALPHA_QUERY_TYPE + NUMERIC_QUERY_TYPE + PUNCTUATION_QUERY_TYPE
    ]

    def __init__(self, feature_info: dict, file_io: FileIO, **kwargs):
        """
        Initialize layer to define a query type vectorizer

        Parameters
        ----------
        feature_info : dict
            Dictionary representing the configuration parameters for the specific feature from the FeatureConfig
        file_io : FileIO object
            FileIO handler object for reading and writing

        Notes
        -----
        Args under feature_layer_info:
            remove_spaces: boolean
                Whether to remove spaces from the string tensors
                Defaults to true
            remove_quotes : string
                Whether to remove quotes from the string tensors
                Defaults to true
            output_mode : str
                the type of vector representation to compute
                currently supports either embedding or one_hot
            embedding_size : int
                dimension size of categorical embedding
        TODO: In the future, we might want to support custom tokenizers to split the string.
        """
        super().__init__(feature_info=feature_info, file_io=file_io, **kwargs)

        self.remove_spaces = self.feature_layer_args.get(self.REMOVE_SPACES, True)
        self.remove_quotes = self.feature_layer_args.get(self.REMOVE_QUOTES, True)

        feature_info["feature_layer_info"]["args"]["vocabulary"] = self.VOCABULARY
        self.categorical_vector_op = CategoricalVector(feature_info, file_io, **kwargs)

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
        # Replace quotes and white spaces
        if self.remove_spaces:
            inputs = tf.strings.regex_replace(inputs, """[\s]""", "")
        if self.remove_quotes:
            inputs = tf.strings.regex_replace(inputs, """["']""", "")

        # Assign query type
        query_type = tf.strings.join([
            tf.where(tf.strings.regex_full_match(inputs, self.ALPHA_REGEX), self.ALPHA_QUERY_TYPE, ""),
            tf.where(tf.strings.regex_full_match(inputs, self.NUMERIC_REGEX), self.NUMERIC_QUERY_TYPE, ""),
            tf.where(tf.strings.regex_full_match(inputs, self.PUNCTUATION_REGEX), self.PUNCTUATION_QUERY_TYPE, ""),
        ])

        # Vectorize the query type to either dense embedding or sparse one-hot
        query_type_vector = self.categorical_vector_op(query_type, training=training)

        return query_type_vector
