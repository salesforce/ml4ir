from tensorflow import train
import tensorflow as tf
import pandas as pd

from ml4ir.base.config.keys import SequenceExampleTypeKey


def _bytes_feature(values):
    """Returns a bytes_list from a string / byte."""
    values = [value.encode("utf-8") for value in values]
    return train.Feature(bytes_list=train.BytesList(value=values))


def _float_feature(values):
    """Returns a float_list from a float / double."""
    return train.Feature(float_list=train.FloatList(value=values))


def _int64_feature(values):
    """Returns an int64_list from a bool / enum / int / uint."""
    return train.Feature(int64_list=train.Int64List(value=values))


def _get_feature_fn(dtype):
    """Returns appropriate feature function based on datatype"""
    if dtype == tf.string:
        return _bytes_feature
    elif dtype == tf.float32:
        return _float_feature
    elif dtype == tf.int64:
        return _int64_feature
    else:
        raise Exception("Feature dtype {} not supported".format(dtype))


def get_example_proto(row, features):
    """
    Get an Example protobuf from a pandas dataframe row

    Parameters
    ----------
    row : pandas DataFrame row
        pandas dataframe row to be converted to an example proto
    features : dict
        dictionary containing configuration for all features

    Returns
    -------
    `Example` protobuffer object
        Example object loaded from the specified row
    """

    features_dict = dict()
    for feature_info in features:
        feature_name = feature_info["name"]
        feature_fn = _get_feature_fn(feature_info["dtype"])
        # FIXME
        # When applying functions with axis=1, pandas performs upcasting,
        # so if we have a mix of floats/ints, converts everything to float
        # that breaks this part of the code. Example:
        # https://stackoverflow.com/questions/47143631/
        # how-do-i-preserve-datatype-when-using-apply-row-wise-in-pandas-dataframe
        if feature_name not in row:
            raise Exception(
                "Could not find column {} in record: {}".format(feature_name, str(row))
            )
        features_dict[feature_name] = feature_fn(
            [row[feature_name]]
            if not pd.isna(row[feature_name])
            else [feature_info["default_value"]]
        )

    return train.Example(features=train.Features(feature=features_dict))


def get_sequence_example_proto(group, context_features, sequence_features):
    """
    Get a SequenceExample protobuf from a dataframe group

    Parameters
    ----------
    group : pandas dataframe group
    context_features : dict
        dictionary containing the configuration for all the context features
    sequence_features : dict
        dictionary containing the configuration for all the sequence features

    Returns
    -------
    `SequenceExample` object
        SequenceExample object loaded the dataframe group
    """
    sequence_features_dict = dict()
    context_features_dict = dict()

    for feature_info in context_features:
        feature_name = feature_info["name"]
        feature_fn = _get_feature_fn(feature_info["dtype"])
        context_features_dict[feature_name] = feature_fn([group[feature_name].tolist()[0]])

    for feature_info in sequence_features:
        feature_name = feature_info["name"]
        feature_fn = _get_feature_fn(feature_info["dtype"])
        if feature_info["tfrecord_type"] == SequenceExampleTypeKey.SEQUENCE:
            sequence_features_dict[feature_name] = train.FeatureList(
                feature=[feature_fn(group[feature_name].tolist())]
            )

    return train.SequenceExample(
        context=train.Features(feature=context_features_dict),
        feature_lists=train.FeatureLists(feature_list=sequence_features_dict),
    )
