import tensorflow as tf

from ml4ir.base.features.feature_fns.base import BaseFeatureLayerOp
from ml4ir.base.features.feature_fns.utils import get_vocabulary_info
from ml4ir.base.config.keys import VocabularyInfoArgsKey
from ml4ir.base.io.file_io import FileIO


class CategoricalVector(BaseFeatureLayerOp):
    """
    Converts a sequence string tensor into a categorical one-hot or embedding representation.
    Works by using a vocabulary file to convert the string tensor into categorical indices
    and then embedding or one-hot vectorizing the index.
    """
    LAYER_NAME = "sequence_categorical_vector"

    VOCABULARY = "vocabulary"
    NUM_OOV_BUCKETS = "num_oov_buckets"
    OUTPUT_MODE = "output_mode"
    EMBEDDING_OUTPUT_MODE = "embedding"
    ONE_HOT_OUTPUT_MODE = "one_hot"
    EMBEDDING_SIZE = "embedding_size"
    EMBEDDINGS_INITIALIZER = "embeddings_initializer"


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
            vocabulary : string or list of strings
                Either path to vocabulary CSV file for the input tensor containing the vocabulary to look-up.
                uses the "key" named column as vocabulary of the 1st column if no "key" column present.
                Or list of strings to be used as the vocabulary.
            num_oov_buckets : int
                number of out of vocabulary buckets/slots to be used to
                             encode strings into categorical indices
            output_mode : str
                the type of vector representation to compute
                currently supports either embedding or one_hot
            embedding_size : int
                dimension size of categorical embedding
            embedding_initializer: string
                the tensorflow initializer to use for the embedding matrix
        """
        super().__init__(feature_info=feature_info, file_io=file_io, **kwargs)

        self.vocabulary = self.feature_layer_args.get(self.VOCABULARY)
        if isinstance(self.vocabulary, str):
            self.feature_layer_args["vocabulary_file"] = self.vocabulary
            self.vocabulary_keys, _ = get_vocabulary_info({VocabularyInfoArgsKey.VOCABULARY_FILE: self.vocabulary},
                                                          self.file_io,
                                                          self.default_value)
        elif isinstance(self.vocabulary, list):
            self.vocabulary_keys = self.vocabulary
        else:
            raise NotImplementedError("Unsupported value for argument vocabulary")

        self.vocabulary_size = len(self.vocabulary_keys)
        self.num_oov_buckets = self.feature_layer_args.get(self.NUM_OOV_BUCKETS, 1)
        self.output_mode = self.feature_layer_args.get(self.OUTPUT_MODE, self.EMBEDDING_OUTPUT_MODE)

        self.string_lookup = tf.keras.layers.StringLookup(vocabulary=self.vocabulary_keys,
                                                          num_oov_indices=self.num_oov_buckets)

        self.embedding_size = None
        self.embeddings_initializer = None
        self.embedding = None
        if self.output_mode == self.EMBEDDING_OUTPUT_MODE:
            self.embedding_size = self.feature_layer_args[self.EMBEDDING_SIZE]
            self.embeddings_initializer = self.feature_layer_args.get(self.EMBEDDINGS_INITIALIZER, "uniform")
            self.embedding = tf.keras.layers.Embedding(input_dim=self.vocabulary_size + self.num_oov_buckets,
                                                       output_dim=self.embedding_size,
                                                       embeddings_initializer=self.embeddings_initializer)
        if self.output_mode not in [self.EMBEDDING_OUTPUT_MODE, self.ONE_HOT_OUTPUT_MODE]:
            raise NotImplementedError(f"The only available output_mode currently for this layer are -> "
                                      f"{[self.EMBEDDING_OUTPUT_MODE, self.ONE_HOT_OUTPUT_MODE]}")


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
        categorical_indices = self.string_lookup(inputs, training=training)

        vector = None
        if self.output_mode == self.EMBEDDING_OUTPUT_MODE:
            vector = self.embedding(categorical_indices, training=training)
        elif self.output_mode == self.ONE_HOT_OUTPUT_MODE:
            vector = tf.one_hot(categorical_indices, depth=self.vocabulary_size + self.num_oov_buckets)

        return vector