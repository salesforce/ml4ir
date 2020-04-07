import string
import re
import tensorflow as tf
from tensorflow import io


# NOTE: We can eventually make this regex configurable through the FeatureConfig
# Keeping it simple for now
PUNCTUATION_REGEX = "|".join([re.escape(c) for c in list(string.punctuation)])


@tf.function
def preprocess_text(feature_tensor, preprocessing_info):
    """
    Args:
        feature_tensor: input feature tensor of type tf.string
        preprocessing_info: dictionary containing preprocessing information for the feature

    Returns:
        processed float tensor
    """
    if preprocessing_info.get("remove_punctuation", False):
        feature_tensor = tf.strings.regex_replace(feature_tensor, PUNCTUATION_REGEX, "")
    if preprocessing_info.get("to_lower", False):
        feature_tensor = tf.strings.lower(feature_tensor)

    # Convert string to bytes
    feature_tensor = io.decode_raw(
        feature_tensor, out_type=tf.uint8, fixed_length=preprocessing_info["max_length"],
    )

    return tf.cast(feature_tensor, tf.float32)


##########################################
# Add any new preprocessing functions here
##########################################
