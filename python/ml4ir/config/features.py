import ml4ir.io.file_io as file_io
import yaml

from ml4ir.config.keys import FeatureTypeKey, TFRecordTypeKey
from tensorflow.keras import Input
from typing import List, Dict, Optional
from logging import Logger

"""
Feature config YAML format

query_key:  # Unique query ID field
    name: <str> # name of the feature in the input data
    node_name: <str> # tf graph node name
    trainable: <bool> # if the feature is a trainable tf element
    dtype: <Supported tensorflow data type | str>
    log_at_inference: <boolean | default: false> # if feature should be logged to file in inference mode
    is_group_metric_key: <boolean | default: false> # if feature should be used a groupby key to compute metrics
    feature_layer_info:
        type: <str> # some supported/predefined feature layer type; eg: embedding categorical
        shape: <list[int]>
        # following keys are not supported yet
        embedding_size: <int> # Embedding size for categorical/string features
        embedding_type: <categorical or char or string | str>
        ...
    preprocessing_info:
        max_length: <int> # Max length of string features
        to_lower: <bool> # Whether to convert string to lower case
        remove_punctuation: <bool> # Whether to remove punctuations from string
    serving_info:
        name: <str> # name of input feature at serving time
        preprocessing_type: <str> # Any predefined feature preprocessing step to apply
                                     # Defaults to just padding filling null values
        required: <bool> # Whether the feature is mandatory at serving time
    tfrecord_type: <context or sequence | str>
rank:  # Field representing the initial rank of the record in a query
    ...
label:  # Binary click label
    ...
features:
    - name: feature_0
        ...
    - name: feature_1
        ...
    ...

"""


class FeatureConfigKey:
    QUERY_KEY = "query_key"
    RANK = "rank"
    LABEL = "label"
    FEATURES = "features"


class FeatureConfig:
    """
    Class to store features to be used for the ranking model
    """

    def __init__(self, features_dict, logger: Optional[Logger] = None):
        self.all_features = list()

        try:
            self.query_key = features_dict.get(FeatureConfigKey.QUERY_KEY)
            self.all_features.append(self.query_key)
        except KeyError:
            self.query_key = None
            if logger:
                logger.warning("'query_key' key not found in the feature_config specified")

        try:
            self.rank = features_dict.get(FeatureConfigKey.RANK)
            self.all_features.append(self.rank)
        except KeyError:
            self.rank = None
            if logger:
                logger.warning("'rank' key not found in the feature_config specified")

        try:
            self.label = features_dict.get(FeatureConfigKey.LABEL)
            self.all_features.append(self.label)
        except KeyError:
            raise KeyError("'label' key not found in the feature_config specified")

        try:
            self.features: List = features_dict.get(FeatureConfigKey.FEATURES)
            self.all_features.extend(self.features)
        except KeyError:
            raise KeyError("'features' key not found in the feature_config specified")

        # Features that can be used for training the model
        self.ranking_features = list()
        # Features that provide additional information about the query+records
        self.metadata_features = list()
        # Features that contain information at the query level common to all records
        self.context_features = list()
        # Features that contain information at the record level
        self.sequence_features = list()
        # Features to log at inference time
        self.features_to_log = list()
        # Features to be used as keys for computing group metrics
        self.group_metrics_keys = list()
        for feature_info in self.all_features:
            if feature_info.get("trainable", True):
                self.ranking_features.append(feature_info)
            else:
                self.metadata_features.append(feature_info)

            if feature_info.get("tfrecord_type") == TFRecordTypeKey.CONTEXT:
                self.context_features.append(feature_info)
            elif feature_info.get("tfrecord_type") == TFRecordTypeKey.SEQUENCE:
                self.sequence_features.append(feature_info)
            else:
                raise TypeError(
                    "tfrecord_type should be either context or sequence "
                    "but is {} for {}".format(
                        feature_info.get("tfrecord_type"), feature_info.get("name")
                    )
                )

            if feature_info.get("log_at_inference", False):
                self.features_to_log.append(feature_info)

            if feature_info.get("is_group_metric_key", False):
                self.group_metrics_keys.append(feature_info)

        if len(self.ranking_features) == 0:
            raise Exception("No trainable features specified in the feature config")

        self.log_initialization(logger)
        self.mask = self.generate_mask()
        self.all_features.append(self.get_mask())

    def log_initialization(self, logger):
        if logger:
            logger.info("Feature config loaded successfully")
            logger.info(
                "Trainable Features : \n{}".format("\n".join(self.get_ranking_features("name")))
            )
            logger.info("Click label : {}".format(self.get_label("name")))
            logger.info(
                "Metadata Features : \n{}".format("\n".join(self.get_metadata_features("name")))
            )
            logger.info(
                "Context Features : \n{}".format("\n".join(self.get_ranking_features("name")))
            )
            logger.info(
                "Sequence Features : \n{}".format("\n".join(self.get_ranking_features("name")))
            )

    def _get_key_or_dict(self, dict_, key: str = None):
        """Helper method to return dictionary or fetch a value"""
        if key:
            if key == "node_name":
                return dict_.get("node_name", dict_["name"])
            else:
                return dict_.get(key)
        else:
            return dict_

    def _get_list_of_keys_or_dicts(self, list_of_dicts, key: str = None):
        """Helper method to get respective dictionaries from list"""
        if key:
            return [self._get_key_or_dict(f, key) for f in list_of_dicts]
        else:
            return list_of_dicts

    def get_query_key(self, key: str = None):
        """
        Getter method for query_key in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict
        """
        return self._get_key_or_dict(self.query_key, key=key)

    def get_label(self, key: str = None):
        """
        Getter method for label in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict
        """
        return self._get_key_or_dict(self.label, key=key)

    def get_rank(self, key: str = None):
        """
        Getter method for rank in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict
        """
        return self._get_key_or_dict(self.rank, key=key)

    def get_mask(self, key: str = None):
        """
        Getter method for mask in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict
        """
        return self._get_key_or_dict(self.mask, key=key)

    def get_all_features(self, key: str = None, include_label: bool = True):
        """
        Getter method for all_features in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict
        """
        all_features = self._get_list_of_keys_or_dicts(self.all_features, key=key)
        if include_label:
            return all_features
        else:
            if key:
                return [f for f in all_features if f != self.get_label(key)]
            else:
                return [fdict for fdict in all_features if fdict["name"] != self.get_label("name")]

    def get_ranking_features(self, key: str = None):
        """
        Getter method for ranking_features in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict
        """
        return self._get_list_of_keys_or_dicts(self.ranking_features, key=key)

    def get_metadata_features(self, key: str = None):
        """
        Getter method for metadata_features in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict
        """
        return self._get_list_of_keys_or_dicts(self.metadata_features, key=key)

    def get_context_features(self, key: str = None):
        """
        Getter method for context_features in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict
        """
        return self._get_list_of_keys_or_dicts(self.context_features, key=key)

    def get_sequence_features(self, key: str = None):
        """
        Getter method for sequence_features in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict
        """
        return self._get_list_of_keys_or_dicts(self.sequence_features, key=key)

    def get_features_to_log(self, key: str = None):
        """
        Getter method for features_to_log in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict
        """
        return self._get_list_of_keys_or_dicts(self.features_to_log, key=key)

    def get_group_metrics_keys(self, key: str = None):
        """
        Getter method for group_metrics_keys in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict
        """
        return self._get_list_of_keys_or_dicts(self.group_metrics_keys, key=key)

    def define_inputs(self, max_num_records: int) -> Dict[str, Input]:
        """
        Define the input layer for the tensorflow model

        Args:
            - max_num_records: Maximum number of records per query in the training data

        Returns:
            Dictionary of tensorflow graph input nodes
        """

        def get_shape(feature_info: dict):
            feature_layer_info = feature_info["feature_layer_info"]
            preprocessing_info = feature_info.get("preprocessing_info", {})
            if feature_info["tfrecord_type"] == TFRecordTypeKey.CONTEXT:
                num_records = 1
            else:
                num_records = max_num_records
            if feature_layer_info["type"] == FeatureTypeKey.NUMERIC:
                return (num_records,)
            elif feature_layer_info["type"] == FeatureTypeKey.STRING:
                return (num_records, preprocessing_info["max_length"])
            elif feature_layer_info["type"] == FeatureTypeKey.CATEGORICAL:
                raise NotImplementedError

        inputs: Dict[str, Input] = dict()
        for feature_info in self.get_all_features(include_label=False):
            """
                NOTE: We currently do NOT define label as an input node in the model
                We could do this in the future, to help define more complex loss functions
            """
            node_name = feature_info.get("node_name", feature_info["name"])
            shape = get_shape(feature_info)
            inputs[node_name] = Input(shape=shape, name=node_name)

        return inputs

    def generate_mask(self):
        """Add mask information used to flag padded records"""
        return {
            "name": "mask",
            "trainable": False,
            "dtype": "float",
            "feature_layer_info": {"type": FeatureTypeKey.NUMERIC, "shape": None},
            "serving_info": {"name": "mask", "required": False},
            "tfrecord_type": TFRecordTypeKey.SEQUENCE,
        }


def parse_config(feature_config, logger: Optional[Logger] = None) -> FeatureConfig:
    if feature_config.endswith(".yaml"):
        feature_config = file_io.read_yaml(feature_config)
        if logger:
            logger.info("Reading feature config from YAML file : {}".format(feature_config))
    else:
        feature_config = yaml.safe_load(feature_config)
        if logger:
            logger.info("Reading feature config from YAML string : {}".format(feature_config))

    return FeatureConfig(feature_config, logger=logger)
