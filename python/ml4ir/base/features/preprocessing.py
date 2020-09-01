import string
import re
import tensorflow as tf

from ml4ir.base.features.feature_fns.categorical import categorical_indicator_with_vocabulary_file
from ml4ir.base.io.file_io import FileIO


# NOTE: We can eventually make this regex configurable through the FeatureConfig
# Keeping it simple for now
PUNCTUATION_REGEX = "|".join([re.escape(c) for c in list(string.punctuation)])


class PreprocessingMap:
    def __init__(self):
        self.key_to_fn = {
            preprocess_text.__name__: preprocess_text,
            split_and_pad_string.__name__: split_and_pad_string,
            natural_log.__name__: natural_log
            # Add more here
        }

    def add_fn(self, key, fn):
        self.key_to_fn[key] = fn

    def add_fns(self, keys_to_fns_dict):
        self.key_to_fn.update(keys_to_fns_dict)

    def get_fns(self):
        return self.key_to_fn

    def get_fn(self, key):
        return self.key_to_fn.get(key)

    def pop_fn(self, key):
        self.key_to_fn.pop(key)


@tf.function
def preprocess_text(feature_tensor, remove_punctuation=False, to_lower=False):
    """
    String preprocessing function that removes punctuation and converts strings to lower case
    based on the arguments.

    Args:
        feature_tensor: input feature tensor of type tf.string
        remove_punctuation: bool; whether to remove punctuation characters from strings
        to_lower: bool; whether to convert string to lower case

    Returns:
        processed float tensor
    """
    if remove_punctuation:
        feature_tensor = tf.strings.regex_replace(feature_tensor, PUNCTUATION_REGEX, "")
    if to_lower:
        feature_tensor = tf.strings.lower(feature_tensor)

    return feature_tensor


def get_one_hot_label_vectorizer(feature_info, file_io: FileIO):
    """
    Returns a tf function to convert categorical string labels to a one hot encoding.

    Args:
        feature_info: Dictionary representing the configuration parameters for the specific feature from the FeatureConfig.
                      See categorical_indicator_with_vocabulary_file, here it is used to read a vocabulary file to
                      create the one hot encoding.
        file_io: FileIO required to load the vocabulary file.

    Returns:
        processed float tensor
    """
    label_str = tf.keras.Input(shape=(1,), dtype=tf.string)
    label_one_hot = categorical_indicator_with_vocabulary_file(label_str, feature_info, file_io)
    # FIXME: we should avoid use another keras Model here (we are wrapping two Keras models here, which cause issues at
    #  saving time).
    one_hot_vectorizer = tf.keras.Model(inputs=label_str, outputs=label_one_hot)

    @tf.function
    def one_hot_vectorize(feature_tensor):
        """
        Args:
            feature_tensor: input feature tensor of type tf.string.
        Returns:
            numerical value corresponding to the one hot encoding from this string.
        """
        return tf.squeeze(one_hot_vectorizer(feature_tensor), axis=[0])

    return one_hot_vectorize


@tf.function
def split_and_pad_string(feature_tensor, split_char=",", max_length=20):
    """
    String preprocessing function that splits and pads a sequence based on the max_length.

    Args:
        feature_tensor: input feature tensor of type tf.string.
        split_char: string; string separator to split the string input.
        max_length: int; max length of the sequence produced after padding.

    Returns:
        processed float tensor

    Example:
        feature_tensor="AAA,BBB,CCC"
        split_char=","
        max_length=5
        could returns the padded tokens ['AAA', 'BBB', 'CCC', '', '']
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

    Args:
        feature_tensor: input feature tensor of type tf.float32
        shift: floating point shift that is added to the feature tensor element wise before computing natural log
            (used to handle 0 values)
    """
    return tf.math.log(tf.add(feature_tensor, tf.cast(tf.constant(shift), tf.float32)))


##########################################
# Add any new preprocessing functions here
##########################################
