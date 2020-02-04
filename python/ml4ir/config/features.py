import ml4ir.io.file_io as file_io
import json

from ml4ir.config.keys import FeatureTypeKey
from tensorflow.keras import Input
from typing import List, Dict

"""
Feature config format
{
    <feature_name> : {
        "type" : <"numeric" or "string" or "categorical" or "label" | str>,
        "trainable" : <true or false | boolean>,
        "node_name" : <str>,
        "max_length" : <int> (used for padding if list),
        "embedding" : {
            "size" : <int>,
            "vocab_file" : <str>
            "vocab_list" : [
                <str>,
                <str>,
                ...
                <str>
            ],
            "use_chars" : <true or false | boolean>,
            "use_bytes" : <true or false | boolean>
        },
        "tfrecord_info" : {
            "type" : <"sequence" or "context" | str>,
            "dtype" : <"float" or "int" or "bytes" | str>,
            "query_key" : <true or false | boolean>
        }
    }
    ...
}
"""


class Features:
    """
    Class to store features to be used for the ranking model
    """

    def __init__(self, feature_config: dict):
        self.feature_config = feature_config

        self.record_features: List[str] = self.get_feature_names(trainable=True)
        self.metadata_features: List[str] = self.get_feature_names(trainable=False)
        # TODO: Define query level features and clean up tfrecord data format
        self.query_features = None
        self.query_key = self.get_query_key()
        # TODO: this loop is silly - the label should be trackable without looping over the whole dict.
        for feature, feature_info in self.feature_config.items():
            if feature_info["type"] == "label":
                self.label = feature
        if len(self.record_features) == 0:
            raise Exception("No trainable features specified in the feature config")
        if not self.label:
            raise Exception("No label field defined in the feature config")

    def get_feature_names(self, trainable=None, include_label=False, dtype=None) -> List[str]:
        """
        Get feature names based on specified args

        Args:
            trainable: boolean value specifying whether to get only
                            trainable features
            include_label: boolean value specifying whether to include
                           label field
            dtype: str value specifying filter on specific feature dtype

        Returns:
            list of feature names based on the args specified (possibly empty)
        """

        def remove(feature_info: dict) -> bool:
            return (
                (not include_label)
                and (feature_info["type"] == "label")
                or ((trainable is not None) and (feature_info["trainable"] != trainable))
                or (dtype and feature_info["type"] != dtype)
            )

        features_list = [
            feature
            for feature, feature_info in self.feature_config.items()
            if not remove(feature_info)
        ]

        return sorted(list(set(features_list)))

    def get_X(self) -> List[str]:
        """
        Returns:
            list of X feature names
        """
        return self.record_features + self.metadata_features

    def get_y(self) -> str:
        """
        Returns:
            label field
        """
        return self.label

    def get_dict(self) -> dict:
        """
        TODO: remove this method, and expose feature_config as a property, or rename this get_feature_config()
        Returns:
            feature config as a dictionary
        """
        return self.feature_config

    def define_inputs(self, max_num_records: int) -> Dict[str, Input]:
        """
        Define the input layer for the tensorflow model

        Args:
            - max_num_records: Maximum number of records per query in the training data

        Returns:
            Dictionary of tensorflow graph input nodes
        """

        def get_shape(feature_info: dict):
            if feature_info["type"] == FeatureTypeKey.NUMERIC:
                return (max_num_records,)
            elif feature_info["type"] == FeatureTypeKey.STRING:
                return (max_num_records, feature_info["max_length"])
            elif feature_info["type"] == FeatureTypeKey.CATEGORICAL:
                raise NotImplementedError
            elif feature_info["type"] == FeatureTypeKey.LABEL:
                return (max_num_records,)

        shapes = {feature: get_shape(self.get_dict()[feature]) for feature in self.get_X()}

        return {
            feature: Input(shape=shape, name=self.get_dict()[feature].get("node_name", feature))
            for feature, shape in shapes.items()
        }

    def get_query_key(self):
        """
        TODO: we re-loop over the entire feature config each time this method is called, looking for the query_key.
        Returns:
            list of query key features
        """
        query_key = list()
        for feature, feature_info in self.feature_config.items():
            if "tfrecord_info" in feature_info:
                if "query_key" in feature_info["tfrecord_info"]:
                    if feature_info["tfrecord_info"]["query_key"]:
                        query_key.append(feature)
        return query_key

    # FIXME
    # By updating the feature_config hashmap with this mask, we break tfrecord_reader:
    # tfrecord_info = feature_info['tfrecord_info'] will throw a KeyException
    # this kind of bug is exactly why we don't use mutable state.
    # The mask info should be kept in another structure in the Features object, not in this dict.
    def add_mask(self):
        """Add mask feature"""
        if "mask" not in self.metadata_features:
            self.metadata_features.append("mask")
        self.feature_config["mask"] = {"type": "numeric", "trainable": False, "node_name": "mask"}


def parse_config(feature_config) -> Features:
    if feature_config.endswith(".json"):
        feature_config = file_io.read_json(feature_config)
    else:
        feature_config = json.loads(feature_config)

    return Features(feature_config)
