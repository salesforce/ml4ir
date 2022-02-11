import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import feature_column

import copy

from ml4ir.base.features.feature_fns.base import BaseFeatureLayerOp
from ml4ir.base.features.feature_fns.utils import get_vocabulary_info
from ml4ir.base.features.feature_fns.utils import VocabLookup, CategoricalDropout
from ml4ir.base.features.feature_fns.utils import CategoricalIndicesFromVocabularyFile
from ml4ir.base.io.file_io import FileIO


CATEGORICAL_VARIABLE = "categorical_variable"


# TODO
# Tensorflow has a new recommended (and more stable) alternative to feature_columns called
# Keras Preprocessing Layer. This helps with saving and serving the model without errors.
#
# Reference -> https://github.com/tensorflow/community/blob/master/rfcs/20191212-keras-categorical-inputs.md
# Issue -> https://github.com/tensorflow/tensorflow/issues/43628

class CategoricalEmbeddingWithHashBuckets(BaseFeatureLayerOp):
    """
    Converts a string feature tensor into a categorical embedding.
    Works by first converting the string into num_hash_buckets buckets
    each of size hash_bucket_size, then converting each hash bucket into
    a categorical embedding of dimension embedding_size. Finally, these embeddings
    are combined either through mean, sum or concat operations to generate the final
    embedding based on the feature_info.
    """
    LAYER_NAME = "categorical_embedding_with_hash_buckets"

    NUM_HASH_BUCKETS = "num_hash_buckets"
    HASH_BUCKET_SIZE = "hash_bucket_size"
    EMBEDDING_SIZE = "embedding_size"
    MERGE_MODE = "merge_mode"

    def __init__(self, feature_info: dict, file_io: FileIO, **kwargs):
        """
        Initialize the layer to get categorical embedding with hash buckets

        Parameters
        ----------
        feature_info : dict
            Dictionary representing the configuration parameters for the specific feature from the FeatureConfig
        file_io : FileIO object
            FileIO handler object for reading and writing

        Notes
        -----
        Args under feature_layer_info:
            num_hash_buckets : int
                number of different hash buckets to convert the input string into
            hash_bucket_size : int
                the size of each hash bucket
            embedding_size : int
                dimension size of the categorical embedding
            merge_mode : str
                can be one of "mean", "sum", "concat" representing the mode of combining embeddings from each categorical embedding
        """
        super().__init__(feature_info=feature_info, file_io=file_io, **kwargs)

        self.num_hash_buckets = self.feature_layer_args[self.NUM_HASH_BUCKETS]
        self.hash_bucket_size = self.feature_layer_args[self.HASH_BUCKET_SIZE]
        self.embedding_size = self.feature_layer_args[self.EMBEDDING_SIZE]
        self.merge_mode = self.feature_layer_args[self.MERGE_MODE]

        self.embeddings_op_list = list()
        for i in range(self.num_hash_buckets):
            self.embeddings_op_list.append(
                layers.Embedding(
                    input_dim=self.hash_bucket_size,
                    output_dim=self.embedding_size,
                    name="categorical_embedding_{}_{}".format(self.feature_name, i),
                )
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
        embeddings_list = list()
        for i in range(self.num_hash_buckets):
            augmented_string = tf.add(inputs, str(i))
            hash_bucket = tf.strings.to_hash_bucket_fast(
                augmented_string, num_buckets=self.hash_bucket_size
            )
            embeddings_list.append(self.embeddings_op_list[i](hash_bucket, training=training))

        embedding = None
        if self.merge_mode == "mean":
            embedding = tf.reduce_mean(
                embeddings_list,
                axis=0,
                name="categorical_embedding_{}".format(self.feature_name),
            )
        elif self.merge_mode == "sum":
            embedding = tf.reduce_sum(
                embeddings_list,
                axis=0,
                name="categorical_embedding_{}".format(self.feature_name),
            )
        elif self.merge_mode == "concat":
            embedding = tf.concat(
                embeddings_list,
                axis=-1,
                name="categorical_embedding_{}".format(self.feature_name),
            )
        else:
            raise KeyError(
                "The merge_mode currently supported under categorical_embedding_with_hash_buckets are ['mean', 'sum', 'concat']. merge_mode specified in the feature config: {}".format(
                    self.merge_mode
                )
            )

        embedding = tf.expand_dims(embedding, axis=1)

        return embedding


class CategoricalEmbeddingWithIndices(BaseFeatureLayerOp):
    """
    Converts input integer tensor into categorical embedding.
    Works by converting the categorical indices in the input feature_tensor,
    represented as integer values, into categorical embeddings based on the feature_info.
    """
    LAYER_NAME = "categorical_embedding_with_indices"

    NUM_BUCKETS = "num_buckets"
    DEFAULT_VALUE = "default_value"
    EMBEDDING_SIZE = "embedding_size"

    def __init__(self, feature_info: dict, file_io: FileIO, **kwargs):
        """
        Initialize feature layer to convert categorical feature into embedding based on indices

        Parameters
        ----------
        feature_info : dict
            Dictionary representing the configuration parameters for the specific feature from the FeatureConfig
        file_io : FileIO object
            FileIO handler object for reading and writing

        Notes
        -----
        Args under feature_layer_info:
            num_buckets : int
                Maximum number of categorical values
            default_value : int
                default value to be assigned to indices out of the num_buckets range
            embedding_size : int
                dimension size of the categorical embedding
        """
        super().__init__(feature_info=feature_info, file_io=file_io, **kwargs)

        self.num_buckets = self.feature_layer_args[self.NUM_BUCKETS]
        self.default_value = self.feature_layer_args.get(self.DEFAULT_VALUE, None)
        self.embedding_size = self.feature_layer_args[self.EMBEDDING_SIZE]

        categorical_fc = feature_column.categorical_column_with_identity(
            CATEGORICAL_VARIABLE,
            num_buckets=self.num_buckets,
            default_value=self.default_value,
        )
        embedding_fc = feature_column.embedding_column(
            categorical_fc, dimension=self.embedding_size, trainable=True
        )

        self.embedding_op = layers.DenseFeatures(
            embedding_fc,
            name="{}_embedding".format(self.feature_name),
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
        embedding = self.embedding_op({CATEGORICAL_VARIABLE: inputs}, training=training)
        embedding = tf.expand_dims(embedding, axis=1)

        return embedding


class CategoricalEmbeddingToEncodingBiLSTM(BaseFeatureLayerOp):
    """
    Encode a string tensor into categorical embedding.
    Works by converting the string into a word sequence and then generating a categorical/char embedding for each words
    based on the List of strings that form the vocabulary set of categorical values, defined by the argument
    vocabulary_file.
    The char/byte embeddings are then combined using a biLSTM.
    """
    LAYER_NAME = "categorical_embedding_to_encoding_bilstm"

    VOCABULARY_FILE = "vocabulary_file"
    MAX_LENGTH = "max_length"
    EMBEDDING_SIZE = "embedding_size"
    ENCODING_SIZE = "encoding_size"
    LSTM_KERNEL_INITIALIZER = "lstm_kernel_initializer"

    def __init__(self, feature_info: dict, file_io: FileIO, **kwargs):
        """
        Initialize the layer to convert input string tensor into an encoding using
        categorical embeddings

        Parameters
        ----------
        feature_info : dict
            Dictionary representing the configuration parameters for the specific feature from the FeatureConfig
        file_io : FileIO object
            FileIO handler object for reading and writing


        Notes
        -----
        Args under feature_layer_info:
            vocabulary_file : string
                path to vocabulary CSV file for the input tensor containing the vocabulary to look-up.
                uses the "key" named column as vocabulary of the 1st column if no "key" column present.
            max_length: int
                max number of rows to consider from the vocabulary file.
                            if null, considers the entire file vocabulary.
            embedding_size : int
                dimension size of the embedding;
                            if null, then the tensor is just converted to its one-hot representation
            encoding_size : int
                dimension size of the sequence encoding computed using a biLSTM

        The input dimension for the embedding is fixed to 256 because the string is
        converted into a bytes sequence.
        """
        super().__init__(feature_info=feature_info, file_io=file_io, **kwargs)

        self.embedding_size = self.feature_layer_args[self.EMBEDDING_SIZE]
        self.encoding_size = self.feature_layer_args[self.ENCODING_SIZE]
        self.kernel_initializer = self.feature_layer_args.get(
            self.LSTM_KERNEL_INITIALIZER, "glorot_uniform")

        self.categorical_indices_op = CategoricalIndicesFromVocabularyFile(
            feature_info, file_io, **kwargs
        )
        self.vocabulary_size = self.categorical_indices_op.vocabulary_size
        self.num_oov_buckets = self.categorical_indices_op.num_oov_buckets

        self.input_dim = self.vocabulary_size + \
            self.num_oov_buckets if self.num_oov_buckets else self.vocabulary_size
        self.categorical_embedding_op = layers.Embedding(
            input_dim=self.input_dim,
            output_dim=self.embedding_size,
            mask_zero=True,
            input_length=self.feature_layer_args.get(self.MAX_LENGTH),
        )

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
        categorical_indices = self.categorical_indices_op(inputs, training=training)

        categorical_embeddings = self.categorical_embedding_op(categorical_indices, training=training)
        categorical_embeddings = tf.squeeze(categorical_embeddings, axis=1)

        encoding = self.encoding_op(categorical_embeddings, training=training)
        encoding = tf.expand_dims(encoding, axis=1)

        return encoding


class CategoricalEmbeddingWithVocabularyFile(BaseFeatureLayerOp):
    """
    Converts a string tensor into a categorical embedding representation.
    Works by using a vocabulary file to convert the string tensor into categorical indices
    and then converting the categories into embeddings based on the feature_info.
    """
    LAYER_NAME = "categorical_embedding_with_vocabulary_file"

    VOCABULARY_FILE = "vocabulary_file"
    MAX_LENGTH = "max_length"
    NUM_OOV_BUCKETS = "num_oov_buckets"
    NUM_BUCKETS = "num_buckets"
    EMBEDDING_SIZE = "embedding_size"
    DEFAULT_VALUE = "default_value"

    def __init__(self, feature_info: dict, file_io: FileIO, **kwargs):
        """
        Initialize layer to define a categorical embedding using a vocabulary file

        Parameters
        ----------
        feature_info : dict
            Dictionary representing the configuration parameters for the specific feature from the FeatureConfig
        file_io : FileIO object
            FileIO handler object for reading and writing

        Notes
        -----
        Args under feature_layer_info:
            vocabulary_file : string
                path to vocabulary CSV file for the input tensor containing the vocabulary to look-up.
                            uses the "key" named column as vocabulary of the 1st column if no "key" column present.
            max_length : int
                max number of rows to consider from the vocabulary file.
                            if null, considers the entire file vocabulary.
            num_oov_buckets : int
                number of out of vocabulary buckets/slots to be used to
                             encode strings into categorical indices
            embedding_size : int
                dimension size of categorical embedding

        The vocabulary CSV file must contain two columns - key, id,
        where the key is mapped to one id thereby resulting in a
        many-to-one vocabulary mapping.
        If id field is absent, a unique whole number id is assigned by default
        resulting in a one-to-one mapping
        """
        super().__init__(feature_info=feature_info, file_io=file_io, **kwargs)

        self.categorical_indices_op = CategoricalIndicesFromVocabularyFile(
            feature_info, file_io, **kwargs
        )
        self.vocabulary_size = self.categorical_indices_op.vocabulary_size
        self.num_oov_buckets = self.categorical_indices_op.num_oov_buckets

        feature_info_new = copy.deepcopy(feature_info)
        feature_info_new["feature_layer_info"]["args"][self.NUM_BUCKETS] = (
            self.vocabulary_size + self.num_oov_buckets
        )
        feature_info_new["feature_layer_info"]["args"][self.DEFAULT_VALUE] = self.vocabulary_size

        self.embedding_op = CategoricalEmbeddingWithIndices(
            feature_info=feature_info_new, file_io=file_io, **kwargs
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
        categorical_indices = self.categorical_indices_op(inputs, training=training)
        embedding = self.embedding_op(categorical_indices, training=training)

        return embedding


class CategoricalEmbeddingWithVocabularyFileAndDropout(BaseFeatureLayerOp):
    """
    Converts a string tensor into a categorical embedding representation.
    Works by using a vocabulary file to convert the string tensor into categorical indices
    and then converting the categories into embeddings based on the feature_info.
    Also uses a dropout to convert categorical indices to the OOV index of 0 at a rate of dropout_rate
    """
    LAYER_NAME = "categorical_embedding_with_vocabulary_file_and_dropout"

    VOCABULARY_FILE = "vocabulary_file"
    DROPOUT_RATE = "dropout_rate"
    EMBEDDING_SIZE = "embedding_size"
    NUM_BUCKETS = "num_buckets"
    DEFAULT_VALUE = "default_value"

    def __init__(self, feature_info: dict, file_io: FileIO, **kwargs):
        """
        Parameters
        ----------
        feature_info : dict
            Dictionary representing the configuration parameters for the specific feature from the FeatureConfig
        file_io : FileIO object
            FileIO handler object for reading and writing

        Notes
        -----
        Args under feature_layer_info:
            vocabulary_file : str
                path to vocabulary CSV file for the input tensor
            dropout_rate : float
                rate at which to convert categorical indices to OOV
            embedding_size : int
                dimension size of categorical embedding

        The vocabulary CSV file must contain two columns - key, id,
        where the key is mapped to one id thereby resulting in a
        many-to-one vocabulary mapping.
        If id field is absent, a unique natural number id is assigned by default
        resulting in a one-to-one mapping

        OOV index will be set to 0
        num_oov_buckets will be 0
        """
        super().__init__(feature_info=feature_info, file_io=file_io, **kwargs)

        self.dropout_rate = self.feature_layer_args[self.DROPOUT_RATE]

        self.categorical_indices_op = CategoricalIndicesFromVocabularyFile(
            feature_info, file_io, **kwargs
        )
        self.vocabulary_size = self.categorical_indices_op.vocabulary_size
        self.num_oov_buckets = self.categorical_indices_op.num_oov_buckets

        self.categorical_dropout_op = CategoricalDropout(dropout_rate=self.dropout_rate)

        feature_info_new = copy.deepcopy(feature_info)
        feature_info_new["feature_layer_info"]["args"][self.NUM_BUCKETS] = self.vocabulary_size
        feature_info_new["feature_layer_info"]["args"][self.DEFAULT_VALUE] = 0

        self.embedding_op = CategoricalEmbeddingWithIndices(
            feature_info=feature_info_new, file_io=file_io, **kwargs
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
        categorical_indices = self.categorical_indices_op(inputs, training=training)
        categorical_indices = self.categorical_dropout_op(categorical_indices, training=training)
        embedding = self.embedding_op(categorical_indices, training=training)

        return embedding


class CategoricalIndicatorWithVocabularyFile(BaseFeatureLayerOp):
    """
    Converts a string tensor into a categorical one-hot representation.
    Works by using a vocabulary file to convert the string tensor into categorical indices
    and then converting the categories into one-hot representation.
    """
    LAYER_NAME = "categorical_indicator_with_vocabulary_file"

    VOCABULARY_FILE = "vocabulary_file"
    MAX_LENGTH = "max_length"
    NUM_OOV_BUCKETS = "num_oov_buckets"

    def __init__(self, feature_info: dict, file_io: FileIO, **kwargs):
        """
        Parameters
        ----------
        feature_info : dict
            Dictionary representing the configuration parameters for the specific feature from the FeatureConfig
        file_io : FileIO object
            FileIO handler object for reading and writing

        Notes
        -----
        Args under feature_layer_info:
            vocabulary_file : string
                path to vocabulary CSV file for the input tensor containing the vocabulary to look-up.
                uses the "key" named column as vocabulary of the 1st column if no "key" column present.
            max_length : int
                max number of rows to consider from the vocabulary file.
                if null, considers the entire file vocabulary.
            num_oov_buckets : int, optional
                number of out of vocabulary buckets/slots to be used to
                encode strings into categorical indices. If not specified, the default is 1.

        The vocabulary CSV file must contain two columns - key, id,
        where the key is mapped to one id thereby resulting in a
        many-to-one vocabulary mapping.
        If id field is absent, a unique whole number id is assigned by default
        resulting in a one-to-one mapping
        """
        super().__init__(feature_info=feature_info, file_io=file_io, **kwargs)

        self.categorical_indices_op = CategoricalIndicesFromVocabularyFile(
            feature_info=feature_info, file_io=file_io, **kwargs)
        self.vocabulary_size = self.categorical_indices_op.vocabulary_size
        self.num_oov_buckets = self.categorical_indices_op.num_oov_buckets
        
        self.categorical_identity_fc = feature_column.categorical_column_with_identity(
            CATEGORICAL_VARIABLE, num_buckets=self.vocabulary_size + self.num_oov_buckets
        )
        self.indicator_fc = feature_column.indicator_column(self.categorical_identity_fc)

        self.categorical_one_hot_op = layers.DenseFeatures(
            self.indicator_fc,
            name="{}_one_hot".format(self.feature_name),
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
        #
        ##########################################################################
        #
        # NOTE:
        # Current bug[1] with saving a Keras model when using
        # feature_column.categorical_column_with_vocabulary_list.
        # Tracking the issue currently and should be able to upgrade
        # to current latest stable release 2.2.0 to test.
        #
        # Can not use TF2.1.0 due to issue[2] regarding saving Keras models with
        # custom loss, metric layers
        #
        # Can not use TF2.2.0 due to issues[3, 4] regarding incompatibility of
        # Keras Functional API models and Tensorflow
        #
        # References:
        # [1] https://github.com/tensorflow/tensorflow/issues/31686
        # [2] https://github.com/tensorflow/tensorflow/issues/36954
        # [3] https://github.com/tensorflow/probability/issues/519
        # [4] https://github.com/tensorflow/tensorflow/issues/35138
        #
        # CATEGORICAL_VARIABLE = "categorical_variable"
        # categorical_fc = feature_column.categorical_column_with_vocabulary_list(
        #     CATEGORICAL_VARIABLE,
        #     vocabulary_list=vocabulary_list,
        #     default_value=feature_layer_info["args"].get("default_value", -1),
        #     num_oov_buckets=feature_layer_info["args"].get("num_oov_buckets", 0),
        # )
        #
        # indicator_fc = feature_column.indicator_column(categorical_fc)
        #
        # categorical_one_hot = layers.DenseFeatures(
        #     indicator_fc,
        #     name="{}_one_hot".format(feature_info.get("node_name", feature_info["name"])),
        # )({CATEGORICAL_VARIABLE: feature_tensor})
        # categorical_one_hot = tf.expand_dims(categorical_one_hot, axis=1)
        #
        ##########################################################################
        #
        categorical_indices = self.categorical_indices_op(inputs, training=training)

        categorical_one_hot = self.categorical_one_hot_op(
            {CATEGORICAL_VARIABLE: categorical_indices}, training=training)
        categorical_one_hot = tf.expand_dims(categorical_one_hot, axis=1)

        return categorical_one_hot
