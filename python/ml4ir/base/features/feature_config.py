import pandas as pd
import json
from tensorflow.keras import Input
from logging import Logger
import tensorflow as tf

from ml4ir.base.data.tfrecord_helper import get_sequence_example_proto
from ml4ir.base.config.keys import (
    FeatureTypeKey,
    TFRecordTypeKey,
    SequenceExampleTypeKey,
)

from typing import List, Dict, Optional


class FeatureConfigKey:
    QUERY_KEY = "query_key"
    LABEL = "label"
    FEATURES = "features"
    RANK = "rank"


class FeatureConfig:
    """
    Class that defines the features and their configurations used for
    training, evaluating and serving a RelevanceModel on ml4ir.

    Attributes
    ----------
    features_dict : dict
        Dictionary of features containing the configuration for every feature
        in the model. This dictionary is used to define the FeatureConfig
        object.
    logger : `Logging` object
        Logging handler to log progress messages
    query_key : dict
        Dictionary containing the feature configuration for the unique data point
        ID, query key
    label : dict
        Dictionary containing the feature configuration for the label field
        for training and evaluating the model
    mask : dict
        Dictionary containing the feature configuration for the computed mask
        field which is used to identify padded values
    features : list of dict
        List of dictionaries containing configurations for all the features
        excluding query_key and label
    all_features : list of dict
        List of dictionaries containing configurations for all the features
        including query_key and label
    train_features : list of dict
        List of dictionaries containing configurations for all the features
        which are used for training, identified by `trainable=False`
    metadata_features : list of dict
        List of dictionaries containing configurations for all the features which
        are NOT used for training, identified by `trainable=False`. These can be
        used for computing custom losses and metrics.
    features_to_log : list of dict
        List of dictionaries containing configurations for all the features which
        will be logged when running `model.predict()`, identified using
        `log_at_inference=True`
    group_metrics_keys : list of dict
        List of dictionaries containing configurations for all the features
        which will be used to compute groupwise metrics
    secondary_labels : list of dict
        List of dictionaries containing configurations for all the features
        which will be used as secondary labels to compute secondary metrics.
        The implementation of the secondary metrics and the usage of the secondary
        labels is up to the users of ml4ir

    Notes
    -----
    Abstract class that is overriden by ExampleFeatureConfig and
    SequenceExampleFeatureConfig for the respective TFRecord types
    """

    def __init__(self, features_dict, logger: Optional[Logger] = None):
        """
        Constructor to instantiate a FeatureConfig object

        Parameters
        ----------
        features_dict : dict
            Dictionary containing the feature configuration for each of the
            model features
        logger : `Logging` object, optional
            Logging object handler for logging progress messages
        """
        self.features_dict = features_dict
        self.logger = logger

        self.initialize_features()
        self.extract_features()
        self.log_initialization()

        if len(self.train_features) == 0:
            raise Exception("No trainable features specified in the feature config")

    def initialize_features(self):
        """
        Initialize the feature attributes with empty lists accordingly
        """
        self.all_features: List[Optional[Dict]] = list()
        self.query_key: Optional[Dict] = None
        self.label: Optional[Dict] = None
        self.mask: Optional[Dict] = None
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

    @staticmethod
    def get_instance(feature_config_dict: dict, tfrecord_type: str, logger: Logger):
        """
        Factory method to get `FeatureConfig` object from a dictionary of
        feature configurations based on the TFRecord type

        Parameters
        ----------
        feature_config_dict : dict
            Dictionary containing the feature definitions for all the features
            for the model
        tfrecord_type : {"example", "sequence_example"}
            Type of the TFRecord message type used for the ml4ir RelevanceModel
        logger : `Logging` object
            Logging object handler to log status and progress messages

        Returns
        -------
        `FeatureConfig` object
            ExampleFeatureConfig or SequenceExampleFeatureConfig object computed
            from the feature configuration dictionary
        """
        logger.debug(json.dumps(feature_config_dict, indent=4))
        if tfrecord_type == TFRecordTypeKey.EXAMPLE:
            return ExampleFeatureConfig(feature_config_dict, logger=logger)
        else:
            return SequenceExampleFeatureConfig(feature_config_dict, logger=logger)

    def extract_features(self):
        """
        Extract the features from the input feature config dictionary
        and assign to relevant FeatureConfig attributes
        """
        for mandatory_key in [FeatureConfigKey.LABEL, FeatureConfigKey.FEATURES]:
            if mandatory_key not in self.features_dict:
                raise KeyError(
                    "'%s' key not found in the feature_config specified" % mandatory_key
                )
        if FeatureConfigKey.QUERY_KEY not in self.features_dict:
            if self.logger:
                self.logger.warning(
                    "'%s' key not found in the feature_config specified"
                    % FeatureConfigKey.QUERY_KEY
                )
            self.query_key = None
        else:
            self.query_key = self.features_dict.get(FeatureConfigKey.QUERY_KEY)
            self.all_features.append(self.query_key)

        self.label = self.features_dict.get(FeatureConfigKey.LABEL)
        self.all_features.append(self.label)

        self.features = self.features_dict.get(FeatureConfigKey.FEATURES)
        self.all_features.extend(self.features)

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

    def log_initialization(self):
        """
        Log initial state of FeatureConfig object after extracting
        all the attributes
        """
        if self.logger:
            self.logger.info("Feature config loaded successfully")
            self.logger.info(
                "Trainable Features : \n{}".format("\n".join(self.get_train_features("name")))
            )
            self.logger.info("Label : {}".format(self.get_label("name")))
            self.logger.info(
                "Metadata Features : \n{}".format("\n".join(self.get_metadata_features("name")))
            )

    def _get_key_or_dict(self, dict_, key: str = None):
        """
        Wrapper method to return config dictionary or fetch a configuration value
        from the Featureconfig.

        Parameters
        ----------
        dict_ : dict
            Dictionary to fetch the value from
        key : str, optional
            Key to select the value for from the dictionary

        Returns
        -------
        str or bool or dict
            Dictionary value if key is passed, otherwise return input dictionary
        """
        if key:
            if key == "node_name":
                return dict_.get("node_name", dict_["name"])
            else:
                return dict_.get(key)
        else:
            return dict_

    def _get_list_of_keys_or_dicts(self, list_of_dicts, key: str = None):
        """
        Wrapper method to return list of config dictionary or fetch a
        configuration value from the Featureconfig.

        Parameters
        ----------
        list_of_dicts : list of dict
            Dictionary to fetch the value from
        key : str, optional
            Key to select the value for from the dictionary

        Returns
        -------
        list of str or bool or dict
            List of dictionary value if key is passed, otherwise return input dictionary
        """
        if key:
            return [self._get_key_or_dict(f, key) for f in list_of_dicts]
        else:
            return list_of_dicts

    def get_query_key(self, key: str = None):
        """
        Getter method for query_key in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict

        Parameters
        ----------
        key : str
            Value from the query_key feature configuration to be fetched

        Returns
        -------
        str or int or bool or dict
            Query key value or entire config dictionary based on if the key is passed
        """
        return self._get_key_or_dict(self.query_key, key=key)

    def get_label(self, key: str = None):
        """
        Getter method for label in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict

        Parameters
        ----------
        key : str
            Value from the label feature configuration to be fetched

        Returns
        -------
        str or int or bool or dict
            Label value or entire config dictionary based on if the key is passed
        """
        return self._get_key_or_dict(self.label, key=key)

    def get_mask(self, key: str = None):
        """
        Getter method for mask in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict

        Parameters
        ----------
        key : str
            Value from the mask feature configuration to be fetched

        Returns
        -------
        str or int or bool or dict
            Label value or entire config dictionary based on if the key is passed
        """
        return self._get_key_or_dict(self.mask, key=key)

    def get_feature(self, name: str):
        """
        Getter method for feature in FeatureConfig object

        Parameters
        ----------
        name : str
            Name of the feature to fetch

        Returns
        -------
        dict
            Feature config dictionary for the name of the feature passed
        """
        for feature_info in self.get_all_features():
            if feature_info["name"] == name:
                return feature_info

        raise KeyError("No feature named {} in FeatureConfig".format(name))

    def set_feature(self, name: str, new_feature_info: dict):
        """
        Setter method to set the feature_info of a feature in the FeatureConfig
        as specified by the name argument

        Parameters
        ----------
        name : str
            name of feature whose feature_info is to be updated
        new_feature_info : dict
            dictionary used to set the feature_info for the
            feature with specified name
        """
        feature_found = False
        for feature_info in self.features_dict["features"]:
            if feature_info["name"] == name:
                feature_found = True
                self.features_dict["features"].remove(feature_info)
                self.features_dict["features"].append(new_feature_info)

        if feature_found:
            self.initialize_features()
            self.extract_features()
        else:
            raise KeyError("No feature named {} in FeatureConfig".format(name))

    def get_all_features(
        self, key: str = None, include_label: bool = True, include_mask: bool = True
    ):
        """
        Getter method for all_features in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict

        Parameters
        ----------
        key : str, optional
            Name of the configuration key to be fetched.
            If None, then entire dictionary for the feature is returned
        include_label : bool, optional
            Include label in list of features returned
        include_mask : bool, optional
            Include mask in the list of features returned.
            Only applicable with SequenceExampleFeatureConfig currently

        Returns
        -------
        list
            Lift of feature configuration dictionaries or values for
            all features in FeatureConfig
        """
        all_features = self._get_list_of_keys_or_dicts(self.all_features, key=key)

        # Populate features to skip
        features_to_skip = []
        if not include_label:
            features_to_skip.append(self.get_label(key=key))
        if not include_mask:
            features_to_skip.append(self.get_mask(key=key))

        return [f for f in all_features if f not in features_to_skip]

    def get_train_features(self, key: str = None):
        """
        Getter method for train_features in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict

        Parameters
        ----------
        key : str, optional
            Name of the configuration key to be fetched.
            If None, then entire dictionary for the feature is returned

        Returns
        -------
        list
            Lift of feature configuration dictionaries or values for
            trainable features in FeatureConfig
        """
        return self._get_list_of_keys_or_dicts(self.train_features, key=key)

    def get_metadata_features(self, key: str = None):
        """
        Getter method for metadata_features in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict

        Parameters
        ----------
        key : str, optional
            Name of the configuration key to be fetched.
            If None, then entire dictionary for the feature is returned

        Returns
        -------
        list
            Lift of feature configuration dictionaries or values for
            metadata features in FeatureConfig
        """
        return self._get_list_of_keys_or_dicts(self.metadata_features, key=key)

    def get_features_to_log(self, key: str = None):
        """
        Getter method for features_to_log in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict

        Parameters
        ----------
        key : str, optional
            Name of the configuration key to be fetched.
            If None, then entire dictionary for the feature is returned

        Returns
        -------
        list
            Lift of feature configuration dictionaries or values for
            features to be logged at inference
        """
        return self._get_list_of_keys_or_dicts(self.features_to_log, key=key)

    def get_group_metrics_keys(self, key: str = None):
        """
        Getter method for group_metrics_keys in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict

        Parameters
        ----------
        key : str, optional
            Name of the configuration key to be fetched.
            If None, then entire dictionary for the feature is returned

        Returns
        -------
        list
            Lift of feature configuration dictionaries or values for
            features used to compute groupwise metrics
        """
        return self._get_list_of_keys_or_dicts(self.group_metrics_keys, key=key)

    def get_secondary_labels(self, key: str = None):
        """
        Getter method for secondary_labels in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict

        Parameters
        ----------
        key : str, optional
            Name of the configuration key to be fetched.
            If None, then entire dictionary for the feature is returned

        Returns
        -------
        list
            Lift of feature configuration dictionaries or values for
            features to be used as secondary labels
        """
        return self._get_list_of_keys_or_dicts(self.secondary_labels, key=key)

    def get_dtype(self, feature_info: dict):
        """
        Retrieve data type of a feature

        Parameters
        ----------
        feature_info : dict
            Dictionary containing configuration for the feature

        Returns
        -------
        str
            Data type of the feature
        """
        return feature_info["dtype"]

    def get_default_value(self, feature_info):
        """
        Retrieve default value of a feature

        Parameters
        ----------
        feature_info : dict
            Dictionary containing configuration for the feature

        Returns
        -------
        str or int or float
            Default value of the feature
        """
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
        Define the keras input placeholder tensors for the tensorflow model

        Returns
        -------
        dict
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
                shape=get_shape(feature_info), name=node_name, dtype=self.get_dtype(feature_info),
            )

        return inputs

    def create_dummy_protobuf(self, num_records=1, required_only=False):
        """
        Generate a dummy TFRecord protobuffer with dummy values

        Parameters
        ----------
        num_records : int
            Number of records or sequence features per TFRecord message to fetch
        required_only : bool
            Whether to fetch on fields with `required_only=True`

        Returns
        -------
        protobuffer object
            Example or SequenceExample object with dummy values generated
            from the FeatureConfig
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

        Returns
        -------
        dict
            Flattened dictionary of important configuration keys and values
            that can be used for tracking the experiment run
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
    """
    Class that defines the features and their configurations used for
    training, evaluating and serving a RelevanceModel on ml4ir for
    Example data

    Attributes
    ----------
    features_dict : dict
        Dictionary of features containing the configuration for every feature
        in the model. This dictionary is used to define the FeatureConfig
        object.
    logger : `Logging` object
        Logging handler to log progress messages
    query_key : dict
        Dictionary containing the feature configuration for the unique data point
        ID, query key
    label : dict
        Dictionary containing the feature configuration for the label field
        for training and evaluating the model
    features : list of dict
        List of dictionaries containing configurations for all the features
        excluding query_key and label
    all_features : list of dict
        List of dictionaries containing configurations for all the features
        including query_key and label
    train_features : list of dict
        List of dictionaries containing configurations for all the features
        which are used for training, identified by `trainable=False`
    metadata_features : list of dict
        List of dictionaries containing configurations for all the features which
        are NOT used for training, identified by `trainable=False`. These can be
        used for computing custom losses and metrics.
    features_to_log : list of dict
        List of dictionaries containing configurations for all the features which
        will be logged when running `model.predict()`, identified using
        `log_at_inference=True`
    group_metrics_keys : list of dict
        List of dictionaries containing configurations for all the features
        which will be used to compute groupwise metrics
    secondary_labels : list of dict
        List of dictionaries containing configurations for all the features
        which will be used as secondary labels to compute secondary metrics.
        The implementation of the secondary metrics and the usage of the secondary
        labels is up to the users of ml4ir
    """

    def create_dummy_protobuf(self, num_records=1, required_only=False):
        """Create a SequenceExample protobuffer with dummy values"""
        raise NotImplementedError


class SequenceExampleFeatureConfig(FeatureConfig):
    """
    Class that defines the features and their configurations used for
    training, evaluating and serving a RelevanceModel on ml4ir for
    SequenceExample data

    Attributes
    ----------
    features_dict : dict
        Dictionary of features containing the configuration for every feature
        in the model. This dictionary is used to define the FeatureConfig
        object.
    logger : `Logging` object
        Logging handler to log progress messages
    query_key : dict
        Dictionary containing the feature configuration for the unique data point
        ID, query key
    label : dict
        Dictionary containing the feature configuration for the label field
        for training and evaluating the model
    rank : dict
        Dictionary containing the feature configuration for the rank field
        for training and evaluating the model. `rank` is used to assign an
        ordering to the sequences in the SequenceExample
    mask : dict
        Dictionary containing the feature configuration for the mask field
        for training and evaluating the model. `mask` is used to identify
        which sequence features are padded. A value of 1 represents an
        existing sequence feature and 0 represents a padded sequence feature.
    features : list of dict
        List of dictionaries containing configurations for all the features
        excluding query_key and label
    all_features : list of dict
        List of dictionaries containing configurations for all the features
        including query_key and label
    context_features : list of dict
        List of dictionaries containing configurations for all the features
        which represent the features common to the entire sequence in a
        protobuf message
    sequence_features : list of dict
        List of dictionaries containing configurations for all the features
        which represent the feature unique to a sequence
    train_features : list of dict
        List of dictionaries containing configurations for all the features
        which are used for training, identified by `trainable=False`
    metadata_features : list of dict
        List of dictionaries containing configurations for all the features which
        are NOT used for training, identified by `trainable=False`. These can be
        used for computing custom losses and metrics.
    features_to_log : list of dict
        List of dictionaries containing configurations for all the features which
        will be logged when running `model.predict()`, identified using
        `log_at_inference=True`
    group_metrics_keys : list of dict
        List of dictionaries containing configurations for all the features
        which will be used to compute groupwise metrics
    secondary_labels : list of dict
        List of dictionaries containing configurations for all the features
        which will be used as secondary labels to compute secondary metrics.
        The implementation of the secondary metrics and the usage of the secondary
        labels is up to the users of ml4ir
    """

    def __init__(self, features_dict, logger):
        """
        Constructor to instantiate a FeatureConfig object

        Parameters
        ----------
        features_dict : dict
            Dictionary containing the feature configuration for each of the
            model features
        logger : `Logging` object, optional
            Logging object handler for logging progress messages
        """
        super(SequenceExampleFeatureConfig, self).__init__(features_dict, logger)

    def initialize_features(self):
        """
        Initialize the feature attributes with empty lists accordingly
        """
        super().initialize_features()

        # Feature to capture the rank of the records for a query
        self.rank = None
        # Feature to track padded records
        self.mask = None
        # Features that contain information at the query level common to all records
        self.context_features = list()
        # Features that contain information at the record level
        self.sequence_features = list()

    def extract_features(self):
        """
        Extract the features from the input feature config dictionary
        and assign to relevant FeatureConfig attributes
        """
        super().extract_features()
        try:
            self.rank = self.features_dict.get(FeatureConfigKey.RANK)
            self.all_features.append(self.rank)
        except KeyError:
            self.rank = None
            if self.logger:
                self.logger.warning("'rank' key not found in the feature_config specified")

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

        self.mask = self.generate_mask()
        self.all_features.append(self.get_mask())

    def get_context_features(self, key: str = None):
        """
        Getter method for context_features in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict

        Parameters
        ----------
        key : str, optional
            Name of the configuration key to be fetched.
            If None, then entire dictionary for the feature is returned

        Returns
        -------
        list
            Lift of feature configuration dictionaries or values for
            context features common to all sequence
        """
        return self._get_list_of_keys_or_dicts(self.context_features, key=key)

    def get_sequence_features(self, key: str = None):
        """
        Getter method for sequence_features in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict

        Parameters
        ----------
        key : str, optional
            Name of the configuration key to be fetched.
            If None, then entire dictionary for the feature is returned

        Returns
        -------
        list
            Lift of feature configuration dictionaries or values for
            sequence features unique to each sequence
        """
        return self._get_list_of_keys_or_dicts(self.sequence_features, key=key)

    def log_initialization(self):
        """
        Log initial state of FeatureConfig object after extracting
        all the attributes
        """
        if self.logger:
            self.logger.info("Feature config loaded successfully")
            self.logger.info(
                "Trainable Features : \n{}".format("\n".join(self.get_train_features("name")))
            )
            self.logger.info("Label : {}".format(self.get_label("name")))
            self.logger.info(
                "Metadata Features : \n{}".format("\n".join(self.get_metadata_features("name")))
            )
            self.logger.info(
                "Context Features : \n{}".format("\n".join(self.get_context_features("name")))
            )
            self.logger.info(
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

        Returns
        -------
        dict
            Dictionary configuration for the mask field that captures
            which sequence have been masked in a SequenceExample message
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

        Parameters
        ----------
        key : str
            Value from the rank feature configuration to be fetched

        Returns
        -------
        str or int or bool or dict
            Rank value or entire config dictionary based on if the key is passed
        """
        return self._get_key_or_dict(self.rank, key=key)

    def get_mask(self, key: str = None):
        """
        Getter method for mask in FeatureConfig object
        Can additionally be used to only fetch a particular value from the dict

        Parameters
        ----------
        key : str
            Value from the mask feature configuration to be fetched

        Returns
        -------
        str or int or bool or dict
            Mask value or entire config dictionary based on if the key is passed
        """
        return self._get_key_or_dict(self.mask, key=key)

    def define_inputs(self) -> Dict[str, Input]:
        """
        Define the keras input placeholder tensors for the tensorflow model

        Returns
        -------
        dict
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
                shape=get_shape(feature_info), name=node_name, dtype=self.get_dtype(feature_info),
            )

        return inputs

    def create_dummy_protobuf(self, num_records=1, required_only=False):
        """
        Generate a dummy TFRecord protobuffer with dummy values

        Parameters
        ----------
        num_records : int
            Number of records or sequence features per TFRecord message to fetch
        required_only : bool
            Whether to fetch on fields with `required_only=True`

        Returns
        -------
        protobuffer object
            Example or SequenceExample object with dummy values generated
            from the FeatureConfig
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
                group=g, context_features=context_features, sequence_features=sequence_features,
            )
        ).values[0]
