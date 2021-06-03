import string
import re
import tensorflow as tf

from ml4ir.base.features.feature_fns.categorical import categorical_indicator_with_vocabulary_file
from ml4ir.base.io.file_io import FileIO


class PreprocessingMap:
    """
    Class defining a map of predefined and custom preprocessing functions
    """

    def __init__(self):
        """
        Instantiate a PreprocessingMap object with predefined functions
        """
        self.key_to_fn = {
            preprocess_text.__name__: preprocess_text,
            split_and_pad_string.__name__: split_and_pad_string,
            natural_log.__name__: natural_log,
            convert_label_to_clicks.__name__: convert_label_to_clicks
            # Add more here
        }

    def add_fn(self, key, fn):
        """
        Add custom preprocessing function to the PreprocessingMap

        Parameters
        ----------
        key : str
            Name of the feature preprocessing function
        fn : function
            Function definition for the preprocessing function
        """
        self.key_to_fn[key] = fn

    def add_fns(self, keys_to_fns_dict):
        """
        Add custom preprocessing functions to the PreprocessingMap

        Parameters
        ----------
        keys_to_fns_dict : dict
            Dictionary of preprocessing functions to add to PreprocessingMap
        """
        self.key_to_fn.update(keys_to_fns_dict)

    def get_fns(self):
        """
        Get dictionary of functions in PreprocessingMap

        Returns
        -------
        dict
            Dictionary of preprocessing functions
        """
        return self.key_to_fn

    def get_fn(self, key):
        """
        Get preprocessing function from name

        Parameters
        ----------
        key : str
            Name of preprocessing function to fetch

        Returns
        -------
        function
            Function to preprocess feature corresponding to the key passed
        """
        return self.key_to_fn.get(key)

    def pop_fn(self, key):
        """
        Get preprocessing function from name and remove from PreprocessingMap

        Parameters
        ----------
        key : str
            Name of preprocessing function to fetch

        Returns
        -------
        function
            Function to preprocess feature corresponding to the key passed
        """
        self.key_to_fn.pop(key)


@tf.function
def preprocess_text(
    feature_tensor,
    remove_punctuation=False,
    to_lower=False,
    punctuation=string.punctuation,
    replace_with_whitespace=False,
):
    """
    String preprocessing function that removes punctuation and converts strings to lower case
    based on the arguments.

    Parameters
    feature_tensor : Tensor object
        input feature tensor of type tf.string
    remove_punctuation : bool
        Whether to remove punctuation characters from strings
    to_lower : bool
        Whether to convert string to lower case
    punctuation : str
        Punctuation characters to replace (a single string containing the character to remove
    replace_with_whitespace : bool
        if True punctuation will be replaced by whitespace (i.e. used as separator), note that
        leading and trailing whitespace will also be removed, as well as consecutive whitespaces.

    Returns
    -------
    Tensor object
        Processed string tensor

    Examples
    --------
    Input:
        >>> feature_tensor = "ABCabc123,,,"
        >>> remove_punctuation = True
        >>> to_lower = True
    Output:
        >>> "abcabc123"
    """
    if remove_punctuation:
        replacement = ""
        if replace_with_whitespace:
            replacement = " "
        punctuation_regex = (
            "[" + "".join([re.escape(c) for c in list(punctuation + replacement)]) + "]+"
        )
        feature_tensor = tf.strings.regex_replace(feature_tensor, punctuation_regex, replacement)

        if replace_with_whitespace:
            feature_tensor = tf.strings.strip(feature_tensor)

    if to_lower:
        feature_tensor = tf.strings.lower(feature_tensor)

    return feature_tensor


def get_one_hot_label_vectorizer(feature_info, file_io: FileIO):
    """
    Returns a tf function to convert categorical string labels to a one hot encoding.

    Parameters
    ----------
    feature_info : dict
        Dictionary representing the configuration parameters for the specific feature from the FeatureConfig.
        See categorical_indicator_with_vocabulary_file, here it is used to read a vocabulary file to
        create the one hot encoding.
    file_io: FileIO required to load the vocabulary file.

    Returns
    -------
    function
        Function that converts labels into one hot vectors

    Examples
    --------
    Input:
        >>> feature_tensor = ["abc", "xyz", "abc"]
        >>> vocabulary file
        >>>    abc -> 0
        >>>    xyz -> 1
        >>>    def -> 2
    Output:
        >>> [[1, 0, 0], [0, 1, 0], [1, 0, 0]]
    """
    label_str = tf.keras.Input(shape=(1,), dtype=tf.string)
    label_one_hot = categorical_indicator_with_vocabulary_file(label_str, feature_info, file_io)
    # FIXME: we should avoid use another keras Model here (we are wrapping two Keras models here, which cause issues at
    #  saving time).
    one_hot_vectorizer = tf.keras.Model(inputs=label_str, outputs=label_one_hot)

    @tf.function
    def one_hot_vectorize(feature_tensor):
        """
        Convert label tensor to one hot vector representation

        Parameters
        ----------
        feature_tensor : Tensor object
            input feature tensor of type tf.string

        Returns
        -------
            Tensor with numerical value corresponding to the one hot encoding from this string.
        """
        return tf.squeeze(one_hot_vectorizer(feature_tensor), axis=[0])

    return one_hot_vectorize


@tf.function
def split_and_pad_string(feature_tensor, split_char=",", max_length=20):
    """
    String preprocessing function that splits and pads a sequence based on the max_length.

    Parameters
    ----------
    feature_tensor : Tensor object
        Input feature tensor of type tf.string.
    split_char : str
        String separator to split the string input.
    max_length : int
        max length of the sequence produced after padding.

    Returns
    -------
    Tensor object
        processed float tensor

    Examples
    --------
    Input:
        >>> feature_tensor = "AAA,BBB,CCC"
        >>> split_char = ","
        >>> max_length = 5
    Output:
        >>> ['AAA', 'BBB', 'CCC', '', '']
    """
    tokens = tf.strings.split(feature_tensor, sep=split_char).to_tensor()
    padded_tokens = tf.image.pad_to_bounding_box(
        tf.expand_dims(tokens[:, :max_length], axis=-1),
        offset_height=0,
        offset_width=0,
        target_height=1,
        target_width=max_length,
    )
    padded_tokens = tf.squeeze(padded_tokens, axis=-1)
    return padded_tokens


@tf.function
def natural_log(feature_tensor, shift=1.0):
    """
    Compute the signed log of the feature_tensor

    Parameters
    ----------
    feature_tensor : Tensor object
        input feature tensor of type tf.float32
    shift : int
        floating point shift that is added to the feature tensor element wise before computing natural log
        (used to handle 0 values)

    Examples
    --------
    Input:
        >>> feature_tensor = [10, 0]
        >>> shift = 1
    Output:
        >>> [2.39, 0.]
    """
    return tf.math.log(tf.add(feature_tensor, tf.cast(tf.constant(shift), tf.float32)))

@tf.function
def convert_label_to_clicks(label_vector, dtype):
    """Convert the label vector to binary clicks. Documents with the maximum labels are considered clicked and receive
        label (1). Any other document is considered not clicked and receive label (0)
            Parameters
            ----------
            label_vector : tf tensor
                input label tensor of type label_dtype
            dtype : str
                Data type of the input label_vector


            Returns
            -------
            tf tensor
                converted clicks
    """

    typ = dtype
    if dtype == 'int':
        typ = 'int64'
    maximum = tf.reduce_max(label_vector)
    cond = tf.math.equal(label_vector, maximum)
    clicks = tf.dtypes.cast(cond, typ)
    return clicks

##########################################
# Add any new preprocessing functions here
##########################################
