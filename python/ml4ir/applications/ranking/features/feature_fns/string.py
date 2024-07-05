import string
import re
import tensorflow as tf
import numpy as np

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
    ONE_HOT_VECTOR = "one_hot_vector"
    MAX_LENGTH = "max_length"  # define max length for one hot encoding.

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
        self.one_hot = self.feature_layer_args.get(self.ONE_HOT_VECTOR, False)
        self.max_length = self.feature_layer_args.get(self.MAX_LENGTH, 10)

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

        if self.one_hot:
            # Clip the query lengths to the max_length
            query_len = tf.clip_by_value(query_len, 0, self.max_length)
            # Convert to one-hot encoding
            query_len_one_hot = tf.one_hot(query_len, depth=self.max_length + 1)
            return query_len_one_hot
        else:
            query_len = tf.expand_dims(tf.cast(query_len, tf.float32), axis=-1)
            return query_len


class QueryTypeVector(BaseFeatureLayerOp):
    """
    Compute the length of the query string context feature
    """
    LAYER_NAME = "query_type_vector"

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
            remove_quotes : string
                Whether to remove quotes from the string tensors
                Defaults to true
            output_mode : str
                the type of vector representation to compute
                currently supports either embedding or one_hot
            embedding_size : int
                dimension size of categorical embedding
        """
        super().__init__(feature_info=feature_info, file_io=file_io, **kwargs)

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


class GloveQueryEmbeddingVector(BaseFeatureLayerOp):
    """
    A feature layer operation to define a query embedding vectorizer using pre-trained word embeddings.

    Attributes
    ----------
    LAYER_NAME : str
        Name of the layer, set to "query_embedding_vector".
    feature_info : dict
        Configuration parameters for the specific feature from the FeatureConfig.
    file_io : FileIO
        FileIO handler object for reading and writing.
    embedding_size : int
        Dimension size of categorical embedding.
    glove_path : str
        Path to the pre-trained GloVe embeddings file.
    max_entries : int
        Maximum number of entries to load from the GloVe embeddings file.
    stop_words : set
        Set of English stopwords.
    word_vectors : dict
        Dictionary to store word embeddings.
    embedding_dim : int
        Dimension of the embeddings.
    """

    LAYER_NAME = "glove_query_embedding_vector"

    def __init__(self, feature_info: dict, file_io: FileIO, **kwargs):
        """
        Initialize layer to define a query embedding vectorizer using pre-trained word embeddings.

        Parameters
        ----------
        feature_info : dict
            Dictionary representing the configuration parameters for the specific feature from the FeatureConfig.
        file_io : FileIO
            FileIO handler object for reading and writing.

        Notes
        -----
        Args under feature_layer_info:
            remove_quotes : string
                Whether to remove quotes from the string tensors. Defaults to true.
            output_mode : str
                The type of vector representation to compute. Currently supports either embedding or one_hot.
            embedding_size : int
                Dimension size of categorical embedding.
            glove_path : str
                Path to the pre-trained GloVe embeddings file.
        """
        super().__init__(feature_info=feature_info, file_io=file_io, **kwargs)
        self.feature_info = feature_info
        self.file_io = file_io
        self.embedding_size = feature_info["feature_layer_info"]["args"]["embedding_size"]
        self.glove_path = feature_info["feature_layer_info"]["args"]["glove_path"]
        self.max_entries = feature_info["feature_layer_info"]["args"]["max_entries"]
        self.stop_words = {"you're", 'itself', 'but', 'against', 'until', 'where', 'as', 'from', 'own', 'again', 's', "wasn't", 'about', 'out', 'his', 'an', 'those', 've', 'should', 'doing', 'ourselves', 'or', 'down', 'such', "she's", 't', 're', 'me', 'what', 'to', 'didn', "wouldn't", 'hers', 'been', 'which', 'further', 'there', "shouldn't", 'them', "couldn't", 'is', 'wouldn', 'he', 'over', "hasn't", 'their', 'after', 'during', 'few', 'up', 'ma', 'yourselves', 'i', 'themselves', "won't", 'having', "you'll", 'these', 'were', 'most', "isn't", 'how', 'ours', 'y', 'and', 'if', 'not', 'between', 'its', "that'll", 'then', 'that', 'above', 'hadn', 'can', 'each', 'aren', 'whom', 'don', 'we', 'won', 'who', 'be', 'here', 'in', 'our', 'any', 'your', 'shan', 'all', 'd', 'same', 'you', 'nor', 'theirs', 'am', 'isn', 'below', 'o', 'couldn', 'into', "hadn't", 'shouldn', 'very', 'haven', 'it', 'wasn', 'other', 'they', 'are', 'both', 'no', 'through', 'at', 'now', 'himself', 'was', 'off', 'herself', 'doesn', 'mightn', "weren't", "you've", 'too', "mustn't", 'when', 'only', 'on', 'him', 'by', 'hasn', 'once', "haven't", 'yourself', 'have', "you'd", 'a', "doesn't", 'll', 'so', "should've", 'does', 'had', 'my', 'yours', 'she', 'than', 'some', 'why', 'with', 'the', 'will', 'needn', 'did', 'mustn', "needn't", 'more', 'her', 'before', 'for', 'has', 'because', 'of', 'do', "didn't", 'myself', "mightn't", 'just', 'weren', "aren't", 'this', 'ain', "don't", 'while', 'under', 'm', 'being', "it's", "shan't"}

        self.word_vectors = {}

        # Load GloVe embeddings with a limit on the number of entries
        with open(self.glove_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if self.max_entries is not None and idx >= self.max_entries:
                    break
                values = line.split()
                word = values[0]
                word = tf.constant(word, dtype=tf.string)
                vector = np.array(values[1:], dtype='float32')
                self.word_vectors[str(word)] = vector

        # Determine the dimension of the embeddings
        self.embedding_dim = len(vector)

    def preprocess_text(self, text):
        """
        Preprocess the input text by converting to lowercase, removing punctuation, tokenizing,
        and filtering out stopwords.

        Parameters
        ----------
        text : tf.Tensor
            Input text tensor.

        Returns
        -------
        tf.RaggedTensor
            Preprocessed and tokenized text.
        """
        # Convert to lowercase
        text = tf.strings.lower(text)

        # Remove punctuation
        text = tf.strings.regex_replace(text, f"[{string.punctuation}]", " ")

        # Tokenize the text
        tokens = tf.strings.split(text)

        # Filter out stopwords
        def filter_stopwords(tokens):
            return tf.ragged.boolean_mask(tokens,
                                          ~tf.reduce_any(tf.strings.regex_full_match(tokens, '|'.join(self.stop_words)),
                                                         axis=-1))

        tokens = filter_stopwords(tokens)
        return tokens

    def word_lookup(self, word):
        """
        Look up the word embedding for a given word.

        Parameters
        ----------
        word : str
            The word to look up.

        Returns
        -------
        np.ndarray
            The embedding vector for the word, or a zero vector if the word is not found.
        """
        return self.word_vectors.get(str(word), np.zeros((self.embedding_dim), dtype=np.float32))

    def build_embeddings(self, query):
        """
        Build the embedding for a given query by summing the embeddings of its words.

        Parameters
        ----------
        query : tf.Tensor
            Tensor containing the words of the query.

        Returns
        -------
        tf.Tensor
            Tensor of shape (embedding_dim,) containing the summed word embeddings.
        """
        if query.shape[0] == 1:
            word_embeddings = tf.map_fn(lambda word: self.word_lookup(word), query[0], dtype=tf.float32)
            query_embedding = tf.reduce_sum(word_embeddings, axis=0)
            return query_embedding
        else:
            return np.zeros((self.embedding_dim), dtype=np.float32)

    def call(self, queries, training=None):
        """
        Defines the forward pass for the layer on the input queries tensor.

        Parameters
        ----------
        queries : tf.Tensor
            Input tensor containing the queries.
        training : bool, optional
            Boolean flag indicating if the layer is being used in training mode or not.

        Returns
        -------
        tf.Tensor
            Resulting tensor after the forward pass through the feature transform layer.
        """
        inputs = self.preprocess_text(queries)
        query_embeddings = tf.map_fn(lambda query: self.build_embeddings(query), inputs.to_tensor(),
                                     dtype=tf.float32)
        query_embeddings = tf.expand_dims(query_embeddings, axis=1)
        return query_embeddings

