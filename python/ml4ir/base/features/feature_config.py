import pandas as pd
import json
from tensorflow.keras import Input
from logging import Logger
import tensorflow as tf

from ml4ir.base.data.tfrecord_helper import get_sequence_example_proto
from ml4ir.base.config.keys import FeatureTypeKey, TFRecordTypeKey, SequenceExampleTypeKey

from typing import List, Dict, Optional

"""
Feature config YAML format

query_key:  # Unique query ID field
    name: <str> # name of the feature in the input data
    node_name: <str> # tf graph node name
    trainable: <bool> # if the feature is a trainable tf element
    dtype: <Supported tensorflow data type | str>
    log_at_inference: <boolean | default: false> # if feature should be logged to file in inference mode
    # if feature should be used a groupby key to compute metrics
    is_group_metric_key: <boolean | default: false>
    is_secondary_label: <boolean | default: false>
    feature_layer_info:
        shape: <list[int]>
        fn: function name to be used for feature engineering; can be pre-defined or passed in
        args: arguments for the fn specified
    preprocessing_info:
        - fn: function name to be used for preprocessing; can be pre-defined or passed in
          args: arguments for the fn specified
        - ...
    serving_info:
        name: <str> # name of input feature at serving time
        preprocessing_type: <str> # Any predefined feature preprocessing step to apply
                                     # Defaults to just padding filling null values
        required: <bool> # Whether the feature is mandatory at serving time
    default_value: default value for the feature
    tfrecord_type: <context or sequence | str>
rank:  # Field representing the initial rank of the record in a query
    ...
label:  # Binary label
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
    LABEL = "label"
    FEATURES = "features"
    RANK = "rank"


class FeatureConfig:
    """
    Class to store features to be used for the Relevance models
    """

    def __init__(self, features_dict, logger: Optional[Logger] = None):
        self.all_features: List[Optional[Dict]] = list()
        self.query_key: Optional[Dict] = None
        self.label: Optional[Dict] = None
        self.features: List[Optional[Dict]] = list()

        # Features that can be used for training the model
        self.train_features: List[Dict] = list()

        # Features that provide additional information about the query+records
        self.metadata_features: List[Dict] = list()

        # Features to log at inference time
        self.features_to_log: List[Dict] = list()

        # Features to be used as keys for computing group metrics
        # NOTE: Implementation is open-ended
        self.group_metrics_keys: List[Dict] = list()

        # Features to be used as secondary labels for computing secondary metrics
        # NOTE: Implementation is open-ended
        self.secondary_labels: List[Dict] = list()

        self.extract_features(features_dict, logger)

        self.define_features()

        self.log_initialization(logger)

        if len(self.train_features) == 0:
            raise Exception("No trainable features specified in the feature config")

    @staticmethod
    def get_instance(feature_config_dict: dict, tfrecord_type: str, logger: Logger):
        logger.debug(json.dumps(feature_config_dict, indent=4))
        if tfrecord_type == TFRecordTypeKey.EXAMPLE:
            return ExampleFeatureConfig(feature_config_dict, logger=logger)
        else:
            return SequenceExampleFeatureConfig(feature_config_dict, logger=logger)

    def extract_features(self, features_dict, logger: Optional[Logger] = None):

        try:
            self.query_key = features_dict.get(FeatureConfigKey.QUERY_KEY)
            self.all_features.append(self.query_key)
        except KeyError:
            self.query_key = None
            if logger:
                logger.warning(
                    "'%s' key not found in the feature_config specified"
                    % FeatureConfigKey.QUERY_KEY
                )

        try:
            self.label = features_dict.get(FeatureConfigKey.LABEL)
            self.all_features.append(self.label)
        except KeyError:
            raise KeyError(
                "'%s' key not found in the feature_config specified" % FeatureConfigKey.LABEL
            )

        try:
            self.features = features_dict.get(FeatureConfigKey.FEATURES)
            self.all_features.extend(self.features)
        except KeyError:
            raise KeyError(
                "'%s' key not found in the feature_config specified" % FeatureConfigKey.FEATURES
            )

    def define_features(self):
        for feature_info in self.all_features:
            if feature_info.get("trainable", True):
                self.train_features.append(feature_info)
            else:
                self.metadata_features.append(feature_info)

            if feature_info.get("log_at_inference", False):
                self.features_to_log.append(feature_info)

            if feature_info.get("is_group_metric_key", False):
                self.group_metrics_keys.append(feature_info)

            if feature_info.get("is_secondary_label", False):
                self.secondary_labels.append(feature_info)

    def log_initialization(self, logger):
        if logger:
            logger.info("Feature config loaded successfully")
            logger.info(
                "Trainable Features : \n{}".format("\n".join(self.get_train_features("name")))
            )
            logger.info("Label : {}".format(self.get_label("name")))
            logger.info(
                "Metadata Features : \n{}".format("\n".join(self.get_metadata_features("name")))
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

    def get_train_features(self, key: str = None):
        """
        Getter method for train_features in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict
        """
        return self._get_list_of_keys_or_dicts(self.train_features, key=key)

    def get_metadata_features(self, key: str = None):
        """
        Getter method for metadata_features in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict
        """
        return self._get_list_of_keys_or_dicts(self.metadata_features, key=key)

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

    def get_secondary_labels(self, key: str = None):
        """
        Getter method for secondary_labels in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict
        """
        return self._get_list_of_keys_or_dicts(self.secondary_labels, key=key)

    def get_dtype(self, feature_info: dict):
        return feature_info["dtype"]

    def get_default_value(self, feature_info):
        if feature_info["dtype"] == tf.float32:
            return feature_info["serving_info"].get("default_value", 0.0)
        elif feature_info["dtype"] == tf.int64:
            return feature_info["serving_info"].get("default_value", 0)
        elif feature_info["dtype"] == tf.string:
            return feature_info["serving_info"].get("default_value", "")
        else:
            raise Exception("Unknown dtype {}".format(feature_info["dtype"]))

    def define_inputs(self) -> Dict[str, Input]:
        """
        Define the input layer for the tensorflow model

        Returns:
            Dictionary of tensorflow graph input nodes
        """

        def get_shape(feature_info: dict):
            return feature_info.get("shape", (1,))

        inputs: Dict[str, Input] = dict()
        for feature_info in self.get_all_features(include_label=False):
            """
                NOTE: We currently do NOT define label as an input node in the model
                We could do this in the future, to help define more complex loss functions
            """
            node_name = feature_info.get("node_name", feature_info["name"])
            inputs[node_name] = Input(
                shape=get_shape(feature_info), name=node_name, dtype=self.get_dtype(feature_info)
            )

        return inputs

    def create_dummy_protobuf(self, num_records=1, required_only=False):
        """Create an Example protobuffer with dummy values"""
        """
        TODO: The method should also be able to create a SequenceExample from
        an input dictionary of feature values
        """
        raise NotImplementedError

    def get_hyperparameter_dict(self):
        """
        Create hyperparameter configs to track model metadata for best model selection
        Unwraps the feature config for each of the features to add
        preprocessing_info and feature_layer_info as key value pairs
        that can be tracked across the experiment. This can be used to
        identify the values that were set for the different feature layers
        in a given experiment. Will be used during best model selection and
        Hyper Parameter Optimization.
        """
        config = dict()

        config["num_trainable_features"] = len(self.get_train_features())

        for feature_info in self.get_train_features():
            feature_name = feature_info.get("node_name", feature_info["name"])

            # Track preprocessing arguments
            if "preprocessing_info" in feature_info:
                for preprocessing_info in feature_info["preprocessing_info"]:
                    config.update(
                        {
                            "{}_{}_{}".format(feature_name, preprocessing_info["fn"], k): v
                            for k, v in preprocessing_info["args"].items()
                        }
                    )

            # Track feature layer arguments
            if "feature_layer_info" in feature_info:
                feature_layer_info = feature_info["feature_layer_info"]
                if "args" in feature_layer_info:
                    config.update(
                        {
                            "{}_{}".format(feature_name, k): v
                            for k, v in feature_layer_info["args"].items()
                        }
                    )

        return config


class ExampleFeatureConfig(FeatureConfig):
    """Feature config overrides for data containing Example protos"""

    def create_dummy_protobuf(self, num_records=1, required_only=False):
        """Create a SequenceExample protobuffer with dummy values"""
        raise NotImplementedError


class SequenceExampleFeatureConfig(FeatureConfig):
    """Feature config overrides for data containing SequenceExample protos"""

    def __init__(self, features_dict, logger):
        self.rank = None
        # Features that contain information at the query level common to all records
        self.context_features = list()
        # Features that contain information at the record level
        self.sequence_features = list()

        super(SequenceExampleFeatureConfig, self).__init__(features_dict, logger)

        self.mask = self.generate_mask()
        self.all_features.append(self.get_mask())

    def extract_features(self, features_dict, logger: Optional[Logger] = None):
        super().extract_features(features_dict, logger)
        try:
            self.rank = features_dict.get(FeatureConfigKey.RANK)
            self.all_features.append(self.rank)
        except KeyError:
            self.rank = None
            if logger:
                logger.warning("'rank' key not found in the feature_config specified")

    def define_features(self):
        for feature_info in self.all_features:
            if feature_info.get("trainable", True):
                self.train_features.append(feature_info)
            else:
                self.metadata_features.append(feature_info)

            if feature_info.get("tfrecord_type") == SequenceExampleTypeKey.CONTEXT:
                self.context_features.append(feature_info)
            elif feature_info.get("tfrecord_type") == SequenceExampleTypeKey.SEQUENCE:
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

            if feature_info.get("is_secondary_label", False):
                self.secondary_labels.append(feature_info)

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

    def log_initialization(self, logger):
        if logger:
            logger.info("Feature config loaded successfully")
            logger.info(
                "Trainable Features : \n{}".format("\n".join(self.get_train_features("name")))
            )
            logger.info("Label : {}".format(self.get_label("name")))
            logger.info(
                "Metadata Features : \n{}".format("\n".join(self.get_metadata_features("name")))
            )
            logger.info(
                "Context Features : \n{}".format("\n".join(self.get_context_features("name")))
            )
            logger.info(
                "Sequence Features : \n{}".format("\n".join(self.get_sequence_features("name")))
            )

    def generate_mask(self):
        """
        Add mask information used to flag padded records.
        In order to create a batch of sequence examples from n TFRecords,
        we need to make sure that they all have the same number of sequences.
        To do this, we pad sequence records to a fixed max_sequence_size.
        Now, we do not want to use these additional padded sequence records
        to compute metrics and losses. Hence we maintain a boolean mask to
        tell ml4ir the sequence records that were originally present.

        In this method, we add the feature_info for the above mask feature as it
        is not implicitly present in the data.
        """
        return {
            "name": "mask",
            "node_name": "mask",
            "trainable": False,
            "dtype": self.get_rank("dtype"),
            "feature_layer_info": {"type": FeatureTypeKey.NUMERIC, "shape": None},
            "serving_info": {"name": "mask", "required": False},
            "tfrecord_type": SequenceExampleTypeKey.SEQUENCE,
        }

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

    def define_inputs(self) -> Dict[str, Input]:
        """
        Define the input layer for the tensorflow model

        Returns:
            Dictionary of tensorflow graph input nodes
        """

        def get_shape(feature_info: dict):
            # Setting size to None for sequence features as the num_records is variable
            if feature_info["tfrecord_type"] == SequenceExampleTypeKey.SEQUENCE:
                return feature_info.get("shape", (None,))
            else:
                return feature_info.get("shape", (1,))

        inputs: Dict[str, Input] = dict()
        for feature_info in self.get_all_features(include_label=False):
            """
                NOTE: We currently do NOT define label as an input node in the model
                We could do this in the future, to help define more complex loss functions
            """
            node_name = feature_info.get("node_name", feature_info["name"])
            inputs[node_name] = Input(
                shape=get_shape(feature_info), name=node_name, dtype=self.get_dtype(feature_info)
            )

        return inputs

    def create_dummy_protobuf(self, num_records=1, required_only=False):
        """Create a SequenceExample protobuffer with dummy values"""
        """
        TODO: The method should also be able to create a SequenceExample from
        an input dictionary of feature values
        """
        context_features = [
            f
            for f in self.get_context_features()
            if ((not required_only) or (f["serving_info"].get("required", False)))
        ]
        sequence_features = [
            f
            for f in self.get_sequence_features()
            if ((not required_only) or (f["serving_info"].get("required", False)))
        ]

        dummy_query = dict()
        for feature_info in self.get_all_features():
            dummy_value = self.get_default_value(feature_info)
            feature_node_name = feature_info.get("node_name", feature_info["name"])
            dummy_query[feature_node_name] = [dummy_value] * num_records

        dummy_query_group = pd.DataFrame(dummy_query).groupby(
            self.get_context_features("node_name")
        )

        return dummy_query_group.apply(
            lambda g: get_sequence_example_proto(
                group=g, context_features=context_features, sequence_features=sequence_features
            )
        ).values[0]
