import string
import re
import tensorflow as tf


# NOTE: We can eventually make this regex configurable through the FeatureConfig
# Keeping it simple for now
PUNCTUATION_REGEX = "|".join([re.escape(c) for c in list(string.punctuation)])


class PreprocessingMap:
    def __init__(self):
        self.key_to_fn = {
            preprocess_text.__name__: preprocess_text
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
    Args:
        feature_tensor: input feature tensor of type tf.string
        remove_pun

    Returns:
        processed float tensor
    """
    if remove_punctuation:
        feature_tensor = tf.strings.regex_replace(feature_tensor, PUNCTUATION_REGEX, "")
    if to_lower:
        feature_tensor = tf.strings.lower(feature_tensor)

    return feature_tensor


##########################################
# Add any new preprocessing functions here
##########################################
