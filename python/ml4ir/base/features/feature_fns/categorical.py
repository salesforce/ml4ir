import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import feature_column
from tensorflow import lookup

import copy

from ml4ir.base.features.feature_fns.sequence import get_bilstm_encoding
from ml4ir.base.io.file_io import FileIO


CATEGORICAL_VARIABLE = "categorical_variable"


def categorical_embedding_with_hash_buckets(feature_tensor, feature_info, file_io: FileIO):
    """
    Converts a string feature tensor into a categorical embedding.
    Works by first converting the string into num_hash_buckets buckets
    each of size hash_bucket_size, then converting each hash bucket into
    a categorical embdding of dimension embedding_size. Finally, these embeddings
    are combined either through mean, sum or concat operations to generate the final
    embedding based on the feature_info.

    Parameters
    ----------
    feature_tensor : Tensor
        String feature tensor
    feature_info : dict
        Dictionary representing the configuration parameters for the specific feature from the FeatureConfig
    file_io : FileIO object
        FileIO handler object for reading and writing

    Returns
    -------
    Tensor object
        categorical embedding for the input feature_tensor

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
    feature_layer_info = feature_info.get("feature_layer_info")
    embeddings_list = list()
    for i in range(feature_layer_info["args"]["num_hash_buckets"]):
        augmented_string = layers.Lambda(lambda x: tf.add(x, str(i)))(feature_tensor)

        hash_bucket = tf.strings.to_hash_bucket_fast(
            augmented_string, num_buckets=feature_layer_info["args"]["hash_bucket_size"]
        )
        embeddings_list.append(
            layers.Embedding(
                input_dim=feature_layer_info["args"]["hash_bucket_size"],
                output_dim=feature_layer_info["args"]["embedding_size"],
                name="categorical_embedding_{}_{}".format(feature_info.get("name"), i),
            )(hash_bucket)
        )

    embedding = None
    if feature_layer_info["args"]["merge_mode"] == "mean":
        embedding = tf.reduce_mean(
            embeddings_list,
            axis=0,
            name="categorical_embedding_{}".format(feature_info.get("name")),
        )
    elif feature_layer_info["args"]["merge_mode"] == "sum":
        embedding = tf.reduce_sum(
            embeddings_list,
            axis=0,
            name="categorical_embedding_{}".format(feature_info.get("name")),
        )
    elif feature_layer_info["args"]["merge_mode"] == "concat":
        embedding = tf.concat(
            embeddings_list,
            axis=-1,
            name="categorical_embedding_{}".format(feature_info.get("name")),
        )
    else:
        raise KeyError(
            "The merge_mode currently supported under categorical_embedding_with_hash_buckets are ['mean', 'sum', 'concat']. merge_mode specified in the feature config: {}".format(
                feature_layer_info["args"]["merge_mode"]
            )
        )

    embedding = tf.expand_dims(embedding, axis=1)

    return embedding


def categorical_embedding_with_indices(feature_tensor, feature_info, file_io: FileIO):
    """
    Converts input integer tensor into categorical embedding.
    Works by converting the categorical indices in the input feature_tensor,
    represented as integer values, into categorical embeddings based on the feature_info.

    Parameters
    ----------
    feature_tensor : Tensor object
        int feature tensor
    feature_info : dict
        Dictionary representing the configuration parameters for the specific feature from the FeatureConfig
    file_io : FileIO object
        FileIO handler object for reading and writing

    Returns
    -------
    Tensor object
        categorical embedding for the input feature_tensor

    Notes
    -----
    Args under feature_layer_info:
        num_buckets : int
            Maximum number of categorical values
        default_value : int
            default value to be assigned to indices out of the num_buckets range
        embedding_size : int
            dimension size of the categorical embedding

    String based categorical features should already be converted into numeric indices
    """
    feature_layer_info = feature_info.get("feature_layer_info")

    categorical_fc = feature_column.categorical_column_with_identity(
        CATEGORICAL_VARIABLE,
        num_buckets=feature_layer_info["args"]["num_buckets"],
        default_value=feature_layer_info["args"].get("default_value", None),
    )
    embedding_fc = feature_column.embedding_column(
        categorical_fc, dimension=feature_layer_info["args"]["embedding_size"], trainable=True
    )

    embedding = layers.DenseFeatures(
        embedding_fc,
        name="{}_embedding".format(feature_info.get("node_name", feature_info["name"])),
    )({CATEGORICAL_VARIABLE: feature_tensor})
    embedding = tf.expand_dims(embedding, axis=1)

    return embedding


def categorical_embedding_to_encoding_bilstm(feature_tensor, feature_info, file_io: FileIO):
    """
    Encode a string tensor into categorical embedding.
    Works by converting the string into a word sequence and then generating a categorical/char embedding for each words
    based on the List of strings that form the vocabulary set of categorical values, defined by the argument
    vocabulary_file.
    The char/byte embeddings are then combined using a biLSTM.

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
    args = feature_info.get("feature_layer_info")["args"]

    (
        categorical_indices,
        vocabulary_size,
        num_oov_buckets,
    ) = categorical_indices_from_vocabulary_file(feature_info, feature_tensor, file_io)

    input_dim = vocabulary_size + num_oov_buckets if num_oov_buckets else vocabulary_size
    categorical_embeddings = layers.Embedding(
        input_dim=input_dim,
        output_dim=args["embedding_size"],
        mask_zero=True,
        input_length=args.get("max_length"),
    )(categorical_indices)

    categorical_embeddings = tf.squeeze(categorical_embeddings, axis=1)
    kernel_initializer = args.get("lstm_kernel_initializer", "glorot_uniform")
    encoding = get_bilstm_encoding(
        embedding=categorical_embeddings,
        lstm_units=int(args["encoding_size"] / 2),
        kernel_initializer=kernel_initializer,
    )
    return encoding


def categorical_embedding_with_vocabulary_file(feature_tensor, feature_info, file_io: FileIO):
    """
    Converts a string tensor into a categorical embedding representation.
    Works by using a vocabulary file to convert the string tensor into categorical indices
    and then converting the categories into embeddings based on the feature_info.

    Parameters
    ----------
    feature_tensor : Tensor object
        String feature tensor
    feature_info : dict
        Dictionary representing the configuration parameters for the specific feature from the FeatureConfig
    file_io : FileIO object
        FileIO handler object for reading and writing

    Returns
    -------
    Tensor object
        Categorical embedding representation of input feature_tensor

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
    (
        categorical_indices,
        vocabulary_size,
        num_oov_buckets,
    ) = categorical_indices_from_vocabulary_file(feature_info, feature_tensor, file_io)
    feature_info_new = copy.deepcopy(feature_info)
    feature_info_new["feature_layer_info"]["args"]["num_buckets"] = (
        vocabulary_size + num_oov_buckets
    )
    feature_info_new["feature_layer_info"]["args"]["default_value"] = vocabulary_size

    embedding = categorical_embedding_with_indices(
        feature_tensor=categorical_indices, feature_info=feature_info_new, file_io=file_io
    )

    return embedding


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


def categorical_embedding_with_vocabulary_file_and_dropout(
    feature_tensor, feature_info, file_io: FileIO
):
    """
    Converts a string tensor into a categorical embedding representation.
    Works by using a vocabulary file to convert the string tensor into categorical indices
    and then converting the categories into embeddings based on the feature_info.
    Also uses a dropout to convert categorical indices to the OOV index of 0 at a rate of dropout_rate

    Parameters
    ----------
    feature_tensor : Tensor object
        String feature tensor
    feature_info : dict
        Dictionary representing the configuration parameters for the specific feature from the FeatureConfig
    file_io : FileIO object
        FileIO handler object for reading and writing

    Returns
    -------
    Tensor object
        Categorical embedding representation of input feature_tensor

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
    feature_layer_info = feature_info.get("feature_layer_info")
    (
        feature_tensor_indices,
        vocabulary_size,
        num_oov_buckets,
    ) = categorical_indices_from_vocabulary_file(feature_info, feature_tensor, file_io)

    # Mask indices to OOV key of 0 at dropout_rate
    feature_tensor_indices = CategoricalDropout(
        dropout_rate=feature_layer_info["args"]["dropout_rate"]
    )(feature_tensor_indices)

    feature_info_new = copy.deepcopy(feature_info)
    feature_info_new["feature_layer_info"]["args"]["num_buckets"] = vocabulary_size
    feature_info_new["feature_layer_info"]["args"]["default_value"] = 0

    embedding = categorical_embedding_with_indices(
        feature_tensor=feature_tensor_indices, feature_info=feature_info_new, file_io=file_io
    )

    return embedding


def categorical_indicator_with_vocabulary_file(feature_tensor, feature_info, file_io: FileIO):
    """
    Converts a string tensor into a categorical one-hot representation.
    Works by using a vocabulary file to convert the string tensor into categorical indices
    and then converting the categories into one-hot representation.

    Parameters
    ----------
    feature_tensor : Tensor object
        String feature tensor
    feature_info : dict
        Dictionary representing the configuration parameters for the specific feature from the FeatureConfig
    file_io : FileIO object
        FileIO handler object for reading and writing

    Returns
    -------
    Tensor object
        Categorical one-hot representation of input feature_tensor

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
    (
        feature_tensor_indices,
        vocabulary_size,
        num_oov_buckets,
    ) = categorical_indices_from_vocabulary_file(feature_info, feature_tensor, file_io)

    categorical_identity_fc = feature_column.categorical_column_with_identity(
        CATEGORICAL_VARIABLE, num_buckets=vocabulary_size + num_oov_buckets
    )
    indicator_fc = feature_column.indicator_column(categorical_identity_fc)

    categorical_one_hot = layers.DenseFeatures(
        indicator_fc,
        name="{}_one_hot".format(feature_info.get("node_name", feature_info["name"])),
    )({CATEGORICAL_VARIABLE: feature_tensor_indices})
    categorical_one_hot = tf.expand_dims(categorical_one_hot, axis=1)

    return categorical_one_hot


def categorical_indices_from_vocabulary_file(feature_info, feature_tensor, file_io):
    """
    Extract the vocabulary (encoding and values) from the stated vocabulary_file inside feature_info.
    And encode the feature_tensor with the vocabulary.

    Parameters
    ----------
    feature_tensor : Tensor object
        String feature tensor
    feature_info : dict
        Dictionary representing the configuration parameters for the specific feature from the FeatureConfig
    file_io : FileIO object
        FileIO handler object for reading and writing

    Returns
    -------
    categorical_indices : Tensor object
        corresponding to encode of the feature_tensor based on the vocabulary.
    num_oov_buckets : int
        applied num_oov_buckets
    vocabulary_keys : list
        values of the vocabulary stated in the vocabulary_file.
    """
    vocabulary_keys, vocabulary_ids = get_vocabulary_info(feature_info, file_io)
    args = feature_info.get("feature_layer_info")["args"]

    if "dropout_rate" in args:
        default_value = 0  # Default OOV index when using dropout
        if args.get("num_oov_buckets"):
            raise RuntimeError(
                "Cannot have both dropout_rate and num_oov_buckets set. "
                "OOV buckets are not supported with dropout"
            )
        num_oov_buckets = None
        vocabulary_size = len(set(vocabulary_keys)) + 1  # one more for default value
    else:
        default_value = None
        num_oov_buckets = args.get("num_oov_buckets", 1)
        vocabulary_size = len(set(vocabulary_keys))

    lookup_table = VocabLookup(
        vocabulary_keys=vocabulary_keys,
        vocabulary_ids=vocabulary_ids,
        num_oov_buckets=num_oov_buckets,
        default_value=default_value,
        feature_name=feature_info.get("node_name", feature_info["name"]),
    )
    categorical_indices = lookup_table(feature_tensor)

    return categorical_indices, vocabulary_size, num_oov_buckets


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

    def call(self, input_text):
        """
        Convert string tensors to numeric indices using lookup table

        Parameters
        ----------
        input_text : Tensor object
            String categorical tensor

        Returns
        -------
        Tensor object
            Numeric tensor object with corresponding lookup indices
        """
        return self.lookup_table.lookup(input_text)

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


def get_vocabulary_info(feature_info, file_io):
    """
    Extract the vocabulary (encoding and values) from the stated vocabulary_file inside feature_info.

    Parameters
    ----------
    feature_info : dict
        Dictionary representing the configuration parameters for the specific feature from the FeatureConfig
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
    Args under `feature_layer_info`
        vocabulary_file : str
            path to vocabulary CSV file for the input tensor containing the vocabulary to look-up.
            uses the "key" named column as vocabulary of the 1st column if no "key" column present.
        max_length : int
            max number of rows to consider from the vocabulary file.
            if null, considers the entire file vocabulary.
        default_value : int
            default stated value in the configure used to replace missing data points.
    """
    args = feature_info.get("feature_layer_info")["args"]
    vocabulary_df = file_io.read_df(args["vocabulary_file"])
    if "key" in vocabulary_df.columns:
        vocabulary_keys = vocabulary_df["key"]
    else:
        vocabulary_keys = vocabulary_df.iloc[:, 0]
    if "max_length" in args:
        vocabulary_keys = vocabulary_keys[: args["max_length"]]
    if "default_value" in feature_info:
        vocabulary_keys = vocabulary_keys.fillna(feature_info["default_value"])
    vocabulary_keys = vocabulary_keys.values
    if "dropout_rate" in args:
        # NOTE: If a dropout_rate is specified, then reserve 0 as the OOV index
        vocabulary_ids = (
            vocabulary_df["id"].values
            if "id" in vocabulary_df
            else list(range(1, len(vocabulary_keys) + 1))
        )
        if 0 in vocabulary_ids:
            raise ValueError(
                "Can not use ID 0 with dropout. Use categorical_embedding_with_vocabulary_file instead."
            )
    else:
        vocabulary_ids = (
            vocabulary_df["id"].values
            if "id" in vocabulary_df
            else list(range(len(vocabulary_keys)))
        )
    return vocabulary_keys, vocabulary_ids
