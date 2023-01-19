import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import lookup

from ml4ir.base.io.file_io import FileIO
from ml4ir.base.config.keys import VocabularyInfoArgsKey
from ml4ir.base.features.feature_fns.base import BaseFeatureLayerOp

from typing import Optional


def get_vocabulary_info(feature_layer_args: dict,
                        file_io: FileIO,
                        default_value=None):
    """
    Extract the vocabulary (encoding and values) from the stated vocabulary_file inside feature_info.

    Parameters
    ----------
    feature_layer_args : dict
        Dictionary representing the configuration parameters for the specific feature layer
        from the FeatureConfig
    file_io : FileIO object
        FileIO handler object for reading and writing files

    Returns
    -------
    vocabulary_keys : list
        values of the vocabulary stated in the vocabulary_file.
    vocabulary_ids : list
        corresponding encoding ids (values of the vocabulary_keys).

    Notes
    -----
    Keys under feature_layer_args
        vocabulary_file : str
            path to vocabulary CSV file for the input tensor containing the vocabulary to look-up.
            uses the "key" named column as vocabulary of the 1st column if no "key" column present.
        max_length : int
            max number of rows to consider from the vocabulary file.
            if null, considers the entire file vocabulary.
        default_value : int
            default stated value in the configure used to replace missing data points.
    """
    vocabulary_df = file_io.read_df(feature_layer_args[VocabularyInfoArgsKey.VOCABULARY_FILE])
    if VocabularyInfoArgsKey.KEY in vocabulary_df.columns:
        vocabulary_keys = vocabulary_df[VocabularyInfoArgsKey.KEY]
    else:
        vocabulary_keys = vocabulary_df.iloc[:, 0]
    if VocabularyInfoArgsKey.MAX_LENGTH in feature_layer_args:
        vocabulary_keys = vocabulary_keys[: feature_layer_args[VocabularyInfoArgsKey.MAX_LENGTH]]
    if default_value or VocabularyInfoArgsKey.DEFAULT_VALUE in feature_layer_args:
        default_value = default_value if default_value else feature_layer_args[VocabularyInfoArgsKey.DEFAULT_VALUE]
        vocabulary_keys = vocabulary_keys.fillna(default_value)
    vocabulary_keys = vocabulary_keys.values
    if VocabularyInfoArgsKey.DROPOUT_RATE in feature_layer_args:
        # NOTE: If a dropout_rate is specified, then reserve 0 as the OOV index
        vocabulary_ids = (
            vocabulary_df[VocabularyInfoArgsKey.ID].values
            if VocabularyInfoArgsKey.ID in vocabulary_df
            else list(range(1, len(vocabulary_keys) + 1))
        )
        if 0 in vocabulary_ids:
            raise ValueError(
                "Can not use ID 0 with dropout. Use categorical_embedding_with_vocabulary_file instead."
            )
    else:
        vocabulary_ids = (
            vocabulary_df[VocabularyInfoArgsKey.ID].values
            if VocabularyInfoArgsKey.ID in vocabulary_df
            else list(range(len(vocabulary_keys)))
        )
    return vocabulary_keys, vocabulary_ids


class VocabLookup(layers.Layer):
    """
    The class defines a keras layer wrapper around a tf lookup table using the given vocabulary list.
    Maps each entry of a vocabulary list into categorical indices.

    Attributes
    ----------
    vocabulary_list : list
        List of strings that form the vocabulary set of categorical values
    num_oov_buckets : int
        Number of buckets to be used for out of vocabulary strings
    default_value : int
        Default value to strbe used for OOV values
    feature_name : str
        Name of the input feature tensor
    lookup_table : LookupTable object
        Tensorflow look up table that maps strings to integer indices

    Notes
    -----
    Issue[1] with using LookupTable with keras symbolic tensors; expects eager tensors.

    Ref: https://github.com/tensorflow/tensorflow/issues/38305
    """

    def __init__(
        self,
        vocabulary_keys,
        vocabulary_ids,
        num_oov_buckets: int = None,
        default_value: int = None,
        feature_name="categorical_variable",
    ):
        super(VocabLookup, self).__init__(trainable=False, dtype=tf.int64)
        self.vocabulary_keys = vocabulary_keys
        self.vocabulary_ids = vocabulary_ids
        self.vocabulary_size = len(set(vocabulary_ids))
        self.num_oov_buckets = num_oov_buckets
        self.default_value = default_value
        self.feature_name = feature_name

    def build(self, input_shape):
        """
        Defines a Lookup Table  using a KeyValueTensorInitializer to map the keys to the IDs.
        Allows definition of two types of lookup tables based on whether the user specifies num_oov_buckets or the default_value
        """
        table_init = lookup.KeyValueTensorInitializer(
            keys=self.vocabulary_keys,
            values=self.vocabulary_ids,
            key_dtype=tf.string,
            value_dtype=tf.int64,
        )

        """
        NOTE:
        If num_oov_buckets are specified, use a StaticVocabularyTable
        otherwise, use a StaticHashTable with the specified default_value

        For most cases, StaticVocabularyTable is sufficient. But when we want to use a custom default_value(like in the case of the dropout function), we need to use StaticHashTable.
        """
        if self.num_oov_buckets is not None:
            self.lookup_table = lookup.StaticVocabularyTable(
                initializer=table_init,
                num_oov_buckets=self.num_oov_buckets,
                name="{}_lookup_table".format(self.feature_name),
            )
        elif self.default_value is not None:
            self.lookup_table = lookup.StaticHashTable(
                initializer=table_init,
                default_value=self.default_value,
                name="{}_lookup_table".format(self.feature_name),
            )
        else:
            raise KeyError("You must specify either num_oov_buckets or default_value")
        self.built = True

    def call(self, inputs, training=None):
        """
        Convert string tensors to numeric indices using lookup table

        Parameters
        ----------
        inputs : Tensor object
            String categorical tensor

        Returns
        -------
        Tensor object
            Numeric tensor object with corresponding lookup indices
        """
        return self.lookup_table.lookup(inputs)

    def get_config(self):
        """
        Get tensorflow configuration for the lookup table

        Returns
        -------
        dict
            Configuration dictionary for the lookup table layer
        """
        config = super(VocabLookup, self).get_config()
        config.update(
            {
                "vocabulary_keys": self.vocabulary_keys,
                "vocabulary_ids": self.vocabulary_ids,
                "vocabulary_size": self.vocabulary_size,
                "num_oov_buckets": self.num_oov_buckets,
                "feature_name": self.feature_name,
            }
        )
        return config


class CategoricalDropout(layers.Layer):
    """
    Custom Dropout class for categorical indices

    Examples
    --------
    >>> inputs: [[1, 2, 3], [4, 1, 2]]
    >>> dropout_rate = 0.5

    >>> When training, output: [[0, 0, 3], [0, 1, 2]]
    >>> When testing, output: [[1, 2, 3], [4, 1, 2]]

    Notes
    -----
    At training time, mask indices to 0 at dropout_rate

    This works similar to tf.keras.layers.Dropout without the scaling
    Ref: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
    """

    def __init__(self, dropout_rate, seed=None, **kwargs):
        """
        Parameters
        ----------
        dropout_rate : float
            fraction of units to drop, i.e., set to OOV token 0
        seed : int
            random seed for sampling to mask/drop categorical labels

        Notes
        -----
        We define OOV index to be 0 for this function and when dropout is applied, it converts p% of the values to 0(which is the OOV index). This allows us to train a good average embedding for the OOV token.
        """
        super(CategoricalDropout, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.seed = seed

    def get_config(self):
        """
        Get config for the CategoricalDropout tensorflow layer

        Returns
        -------
        dict
            Configuration dictionary for the tensorflow layer
        """
        config = super(CategoricalDropout, self).get_config()
        config.update({"dropout_rate": self.dropout_rate, "seed": self.seed})
        return config

    def call(self, inputs, training=None):
        """
        Run the CategoricalDropout layer by masking input labels to OOV
        index 0 at `dropout_rate`

        Parameters
        ----------
        input : Tensor object
            int categorical index tensor to be masked
        training : bool
            If the layer should be run as training or not

        Returns
        -------
        Tensor object
            Masked tensor object with values set to 0 at probability of dropout_rate
        """
        if training:
            return tf.math.multiply(
                tf.cast(
                    tf.random.uniform(shape=tf.shape(inputs), seed=self.seed) >= self.dropout_rate,
                    dtype=tf.int64,
                ),
                inputs,
            )
        else:
            return inputs


class CategoricalIndicesFromVocabularyFile(BaseFeatureLayerOp):
    """
    Extract the vocabulary (encoding and values) from the stated vocabulary_file inside feature_info.
    And encode the feature_tensor with the vocabulary.
    """
    LAYER_NAME = "categorical_indices_from_vocabulary_file"

    CATEGORICAL_INDICES = "categorical_indices"
    NUM_OOV_BUCKETS = "num_oov_buckets"
    VOCABULARY_KEYS = "vocabulary_keys"
    DROPOUT_RATE = "dropout_rate"

    def __init__(self, feature_info: dict, file_io: FileIO, **kwargs):
        """
        Initialize the layer to convert string tensor into categorical indices

        Parameters
        ----------
        feature_tensor : Tensor object
            String feature tensor
        feature_info : dict
            Dictionary representing the configuration parameters for the specific feature from the FeatureConfig
        file_io : FileIO object
            FileIO handler object for reading and writing
        """
        super().__init__(feature_info=feature_info, file_io=file_io, **kwargs)

        self.num_oov_buckets = self.feature_layer_args.get(self.NUM_OOV_BUCKETS)

        vocabulary_keys, vocabulary_ids = get_vocabulary_info(self.feature_layer_args,
                                                              self.file_io,
                                                              self.default_value)

        if self.DROPOUT_RATE in self.feature_layer_args:
            default_value = 0  # Default OOV index when using dropout
            if self.num_oov_buckets:
                raise RuntimeError(
                    "Cannot have both dropout_rate and num_oov_buckets set. "
                    "OOV buckets are not supported with dropout"
                )
            self.num_oov_buckets = None
            self.vocabulary_size = len(set(vocabulary_keys)) + 1  # one more for default value
        else:
            default_value = None
            self.num_oov_buckets = self.num_oov_buckets if self.num_oov_buckets else 1
            self.vocabulary_size = len(set(vocabulary_keys))

        self.lookup_table = VocabLookup(
            vocabulary_keys=vocabulary_keys,
            vocabulary_ids=vocabulary_ids,
            num_oov_buckets=self.num_oov_buckets,
            default_value=default_value,
            feature_name=self.feature_name,
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
        categorical_indices = self.lookup_table(inputs, training=training)

        return categorical_indices
