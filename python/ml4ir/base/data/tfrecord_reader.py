import tensorflow as tf
from tensorflow import io
from tensorflow import data
from tensorflow import sparse
from tensorflow import image
from logging import Logger

from ml4ir.base.io.file_io import FileIO
from ml4ir.base.features.preprocessing import PreprocessingMap
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.config.keys import SequenceExampleTypeKey, TFRecordTypeKey

from typing import Optional


class TFRecordParser(object):
    """
    Base class for parsing TFRecord examples. This class consolidates the
    parsing and feature extraction pipeline for both Example and SequenceExample
    protobuf messages
    """

    def __init__(
        self,
        feature_config: FeatureConfig,
        preprocessing_map: PreprocessingMap,
        required_fields_only: Optional[bool] = False,
    ):
        """
        Constructor method for instantiating a TFRecordParser object

        Parameters
        ----------
        feature_config : `FeatureConfig`
            FeatureConfig object defining context and sequence feature information
        preprocessing_map : `PreprocessingMap` object
            Object mapping preprocessing feature function names to their definitons
        required_fields_only : bool, optional
            Whether to only use required fields from the feature_config
        """
        self.feature_config = feature_config
        self.preprocessing_map = preprocessing_map
        self.required_fields_only = required_fields_only
        self.features_spec = self.get_features_spec()

    def get_features_spec(self):
        """
        Define the features spec from the feature_config.
        The features spec will be used to parse the serialized TFRecord

        Returns
        -------
        dict
            feature specification dictionary that can be used to parse TFRecords

        Notes
        -----
        For SequenceExample messages, this method returns a pair of dictionaries,
        one each for context and sequence features.
        """
        raise NotImplementedError

    def extract_features_from_proto(self, proto):
        """
        Parse the serialized proto string to extract features

        Parameters
        ----------
        proto: tf.Tensor
            A scalar string tensor that is the serialized form of a TFRecord object

        Returns
        -------
        dict of Tensors
            Dictionary of features extracted from the proto as per the features_spec

        Notes
        -----
        For SequenceExample proto messages, this function returns two dictionaries,
        one for context and another for sequence feature tensors.
        For Example proto messages, this function returns a single dictionary of feature tensors.
        """
        raise NotImplementedError

    def get_default_tensor(self, feature_info, sequence_size=0):
        """
        Get the default tensor for a given feature configuration

        Parameters
        ----------
        feature_info: dict
            Feature configuration information for the feature as specified in the feature_config
        sequence_size: int, optional
            Number of elements in the sequence of a SequenceExample

        Returns
        -------
        tf.Tensor
            Tensor object that can be used as a default tensor if the expected feature
            is missing from the TFRecord
        """
        raise NotImplementedError

    def get_feature(self, feature_info, extracted_features, sequence_size=0):
        """
        Fetch the feature from the feature dictionary of extracted features

        Parameters
        ----------
        feature_info: dict
            Feature configuration information for the feature as specified in the feature_config
        extracted_features: dict
            Dictionary of feature tensors extracted by parsing the serialized TFRecord
        sequence_size: int, optional
            Number of elements in the sequence of a SequenceExample

        Returns
        -------
        tf.Tensor
            Feature tensor that is obtained from the extracted features for the given
            feature_info
        """
        raise NotImplementedError

    def generate_and_add_mask(self, extracted_features, features_dict):
        """
        Create a mask to identify padded values

        Parameters
        ----------
        extracted_features: dict
            Dictionary of tensors extracted from the serialized TFRecord
        features_dict: dict
            Dictionary of tensors that will be used for model training/serving
            as inputs to the model

        Returns
        -------
        features_dict: dict
            Dictionary of tensors that will be used for model training/serving updated
            with the mask tensor if applicable
        sequence_size: int
            Number of elements in the sequence of the TFRecord
        """
        raise NotImplementedError

    def pad_feature(self, feature_tensor, feature_info):
        """
        Pad the feature to the `max_sequence_size` in order to create
        uniform data batches for training
        Parameters
        ----------
        feature_tensor: tf.Tensor
            Feature tensor to be padded
        feature_info: dict
            Feature configuration information for the feature as specified in the feature_config
        Returns
        -------
        tf.Tensor
            Feature tensor padded to the `max_sequence_size`
        """

        raise NotImplementedError

    def preprocess_feature(self, feature_tensor, feature_info):
        """
        Preprocess feature based on the feature configuration

        Parameters
        ----------
        feature_tensor: `tf.Tensor`
            input feature tensor to be preprocessed
        feature_info: dict
            Feature configuration information for the feature as specified in the feature_config

        Returns
        -------
        `tf.Tensor`
            preprocessed tensor object

        Notes
        -----
        Only preprocessing functions part of the `preprocessing_map`
        can be used in this function for preprocessing at data loading

        Pass custom preprocessing functions while instantiating the
        RelevanceDataset object with `preprocessing_keys_to_fns` argument
        """
        preprocessing_info = feature_info.get("preprocessing_info")

        if preprocessing_info:
            for preprocessing_step in preprocessing_info:
                preprocessing_fn = self.preprocessing_map.get_fn(preprocessing_step["fn"])
                if preprocessing_fn:
                    feature_tensor = preprocessing_fn(
                        feature_tensor, **preprocessing_step.get("args", {})
                    )

        return feature_tensor

    def get_parse_fn(self) -> tf.function:
        """
        Define a parsing function that will be used to load the TFRecordDataset
        and create input features for the model.

        Returns
        -------
        `tf.function`
            Parsing function that takes in a serialized TFRecord protobuf
            message and extracts a dictionary of feature tensors

        Notes
        -----
        This function will also be used with the TFRecord serving signature in the
        saved model.
        """

        @tf.function
        def _parse_fn(proto):
            """
            Parse a serialized TFRecord proto message

            Parameters
            ----------
            proto: tf.Tensor
                Scalar string tensor that needs to be parsed to extract features

            Returns
            -------
            features: dict
                Dictionary of feature tensors
            labels: tf.Tensor
                Label feature tensor used for training
            """
            # Parse the proto message to extract feature tensors
            extracted_features = self.extract_features_from_proto(proto)
            features_dict = dict()

            # Create a mask tensor and add to the features dictionary
            features_dict, sequence_size = self.generate_and_add_mask(
                extracted_features, features_dict
            )

            # Process all features, including label to construct the feature tensor dictionary
            for feature_info in self.feature_config.get_all_features(include_mask=False):
                feature_node_name = feature_info.get("node_name", feature_info["name"])

                # Fetch the feature corresponding to the feature_info from the extracted features
                feature_tensor = self.get_feature(feature_info, extracted_features, sequence_size)

                # Pad the extracted feature to the max_sequence_size for training
                feature_tensor = self.pad_feature(feature_tensor, feature_info)

                # Preprocess the extracted feature using the specification from
                # the FeatureConfig and functions from PreprocessingMap
                feature_tensor = self.preprocess_feature(feature_tensor, feature_info)

                # Add the processed feature tensor to the features dictionary
                features_dict[feature_node_name] = feature_tensor

            # Extract the label feature to return separately
            labels = features_dict.pop(self.feature_config.get_label(key="name"))

            # return X and y which can be used with fit(), predict() and evaluate()
            return features_dict, labels

        return _parse_fn


class TFRecordExampleParser(TFRecordParser):
    """
    Class for parsing Example TFRecord protobuf messages
    """

    def get_features_spec(self):
        """
        Define the features spec from the feature_config.
        This will be used to parse the serialized TFRecord

        Returns
        -------
        dict
            feature specification dictionary that can be used to parse TFRecords
        """
        features_spec = dict()

        for feature_info in self.feature_config.get_all_features():
            serving_info = feature_info["serving_info"]
            if not self.required_fields_only or serving_info.get(
                "required", feature_info["trainable"]
            ):
                feature_name = feature_info["name"]
                dtype = feature_info["dtype"]
                default_value = self.feature_config.get_default_value(feature_info)
                features_spec[feature_name] = io.FixedLenFeature([], dtype, default_value=default_value)

        return features_spec

    def extract_features_from_proto(self, serialized):
        """
        Parse the serialized proto string to extract features

        Parameters
        ----------
        proto: tf.Tensor
            A scalar string tensor that is the serialized form of a TFRecord object

        Returns
        -------
        dict of Tensors
            Dictionary of features extracted from the proto as per the features_spec
        """
        return io.parse_single_example(serialized=serialized, features=self.features_spec)

    def get_default_tensor(self, feature_info, sequence_size=0):
        """
        Get the default tensor for a given feature configuration

        Parameters
        ----------
        feature_info: dict
            Feature configuration information for the feature as specified in the feature_config
        sequence_size: int, optional
            Number of elements in the sequence of a SequenceExample

        Returns
        -------
        tf.Tensor
            Tensor object that can be used as a default tensor if the expected feature
            is missing from the TFRecord
        """
        return tf.constant(
            value=self.feature_config.get_default_value(feature_info), dtype=feature_info["dtype"],
        )

    def get_feature(self, feature_info, extracted_features, sequence_size=0):
        """
        Fetch the feature from the feature dictionary of extracted features

        Parameters
        ----------
        feature_info: dict
            Feature configuration information for the feature as specified in the feature_config
        extracted_features: dict
            Dictionary of feature tensors extracted by parsing the serialized TFRecord
        sequence_size: int, optional
            Number of elements in the sequence of a SequenceExample

        Returns
        -------
        tf.Tensor
            Feature tensor that is obtained from the extracted features for the given
            feature_info
        """
        default_tensor = self.get_default_tensor(feature_info, sequence_size)

        feature_tensor = extracted_features.get(feature_info["name"], default_tensor)
        
        # Adjust shape
        feature_tensor = tf.expand_dims(feature_tensor, axis=0)

        return feature_tensor

    def generate_and_add_mask(self, extracted_features, features_dict):
        """
        Create a mask to identify padded values

        Parameters
        ----------
        extracted_features: dict
            Dictionary of tensors extracted from the serialized TFRecord
        features_dict: dict
            Dictionary of tensors that will be used for model training/serving
            as inputs to the model

        Returns
        -------
        features_dict: dict
            Dictionary of tensors that will be used for model training/serving updated
            with the mask tensor if applicable
        sequence_size: int
            Number of elements in the sequence of the TFRecord
        """
        return features_dict, tf.constant(0)

    def pad_feature(self, feature_tensor, feature_info):
        """
        Pad the feature to the `max_sequence_size` in order to create
        uniform data batches for training
        Parameters
        ----------
        feature_tensor: tf.Tensor
            Feature tensor to be padded
        feature_info: dict
            Feature configuration information for the feature as specified in the feature_config
        Returns
        -------
        tf.Tensor
            Feature tensor padded to the `max_sequence_size`
        """
        return feature_tensor


class TFRecordSequenceExampleParser(TFRecordParser):
    def __init__(
        self,
        feature_config: FeatureConfig,
        preprocessing_map: PreprocessingMap,
        required_fields_only: Optional[bool] = False,
        pad_sequence: Optional[bool] = True,
        max_sequence_size: Optional[int] = 25,
    ):
        """
        Constructor method for instantiating a TFRecordParser object

        Parameters
        ----------
        feature_config : `FeatureConfig`
            FeatureConfig object defining context and sequence feature information
        preprocessing_map : `PreprocessingMap` object
            Object mapping preprocessing feature function names to their definitons
        required_fields_only : bool, optional
            Whether to only use required fields from the feature_config
        pad_sequence: bool, optional
            Whether to pad sequence
        max_sequence_size: int, optional
            Maximum number of sequence per query. Used for padding
        """
        self.pad_sequence = pad_sequence
        self.max_sequence_size = max_sequence_size
        super(TFRecordSequenceExampleParser, self).__init__(
            feature_config=feature_config,
            preprocessing_map=preprocessing_map,
            required_fields_only=required_fields_only,
        )

    def get_features_spec(self):
        """
        Define the features spec from the feature_config.
        This will be used to parse the serialized TFRecord

        Returns
        -------
        dict
            Feature specification dictionary that can be used to parse
            Context features from the serialized SequenceExample
        dict
            Feature specification dictionary that can be used to parse
            Sequence features (or feature lists) from the serialized SequenceExample
        """

        context_features_spec = dict()
        sequence_features_spec = dict()

        for feature_info in self.feature_config.get_all_features():
            if feature_info.get("name") == self.feature_config.get_mask("name"):
                continue
            serving_info = feature_info["serving_info"]
            if not self.required_fields_only or serving_info.get("required", feature_info["trainable"]):
                feature_name = feature_info["name"]
                dtype = feature_info["dtype"]
                default_value = self.feature_config.get_default_value(
                    feature_info)
                if feature_info["tfrecord_type"] == SequenceExampleTypeKey.CONTEXT:
                    context_features_spec[feature_name] = io.FixedLenFeature(
                        [], dtype, default_value=default_value
                    )
                elif feature_info["tfrecord_type"] == SequenceExampleTypeKey.SEQUENCE:
                    sequence_features_spec[feature_name] = io.VarLenFeature(
                        dtype=dtype)
                else:
                    raise KeyError("Invalid SequenceExample type: {}".format(
                        feature_info["tfrecord_type"]))

        return context_features_spec, sequence_features_spec

    def extract_features_from_proto(self, serialized):
        """
        Parse the serialized proto string to extract features

        Parameters
        ----------
        proto: tf.Tensor
            A scalar string tensor that is the serialized form of a TFRecord object

        Returns
        -------
        dict of Tensors
            Dictionary of context feature tensors extracted from the proto
            as per the `features_spec`
        dict of Tensors
            Dictionary of sequence feature tensors extracted from the proto
            as per the `features_spec`
        """
        return io.parse_single_sequence_example(
            serialized=serialized,
            context_features=self.features_spec[0],
            sequence_features=self.features_spec[1],
        )

    def get_default_tensor(self, feature_info, sequence_size):
        """
        Get the default tensor for a given feature configuration
        Parameters
        ----------
        feature_info: dict
            Feature configuration information for the feature as specified in the feature_config
        sequence_size: int, optional
            Number of elements in the sequence of a SequenceExample
        Returns
        -------
        tf.Tensor
            Tensor object that can be used as a default tensor if the expected feature
            is missing from the TFRecord
        """
        if feature_info.get("tfrecord_type", SequenceExampleTypeKey.CONTEXT) == SequenceExampleTypeKey.CONTEXT:
            return tf.constant(
                value=self.feature_config.get_default_value(feature_info), dtype=feature_info["dtype"],
            )
        else:
            return tf.fill(
                value=tf.constant(
                    value=self.feature_config.get_default_value(feature_info),
                    dtype=feature_info["dtype"],
                ),
                dims=[sequence_size],
            )

    def get_feature(self, feature_info, extracted_features, sequence_size):
        """
        Fetch the feature from the feature dictionary of extracted features
        Parameters
        ----------
        feature_info: dict
            Feature configuration information for the feature as specified in the feature_config
        extracted_features: dict
            Dictionary of feature tensors extracted by parsing the serialized TFRecord
        sequence_size: int, optional
            Number of elements in the sequence of a SequenceExample
        Returns
        -------
        tf.Tensor
            Feature tensor that is obtained from the extracted features for the given
            feature_info
        """
        extracted_context_features, extracted_sequence_features = extracted_features

        default_tensor = self.get_default_tensor(feature_info, sequence_size)

        if feature_info["tfrecord_type"] == SequenceExampleTypeKey.CONTEXT:
            feature_tensor = extracted_context_features.get(
                feature_info["name"], default_tensor)
            # Adjust shape
            feature_tensor = tf.expand_dims(feature_tensor, axis=0)
        else:
            feature_tensor = extracted_sequence_features.get(
                feature_info["name"], default_tensor)
            if isinstance(feature_tensor, sparse.SparseTensor):
                feature_tensor = sparse.reset_shape(feature_tensor)
                feature_tensor = sparse.to_dense(feature_tensor)
                feature_tensor = tf.squeeze(feature_tensor, axis=0)

        return feature_tensor

    def generate_and_add_mask(self, extracted_features, features_dict):
        """
        Create a mask to identify padded values

        Parameters
        ----------
        extracted_features: dict
            Dictionary of tensors extracted from the serialized TFRecord
        features_dict: dict
            Dictionary of tensors that will be used for model training/serving
            as inputs to the model

        Returns
        -------
        features_dict: dict
            Dictionary of tensors that will be used for model training/serving updated
            with the mask tensor if applicable
        sequence_size: int
            Number of elements in the sequence of the TFRecord
        """
        context_features, sequence_features = extracted_features
        if (
            self.required_fields_only
            and not self.feature_config.get_rank("serving_info")["required"]
        ):
            """
            Define dummy mask if the rank field is not a required field for serving
            NOTE:
            This masks all max_sequence_size as 1 as there is no real way to know
            the number of sequence in the query. There is no predefined required field,
            and hence we would need to do a full pass of all features to find the record shape.
            This approach might be unstable if different features have different shapes.
            Hence we just mask all sequence
            """
            mask = tf.constant(
                value=1,
                shape=[self.max_sequence_size],
                dtype=self.feature_config.get_rank("dtype"),
            )
            sequence_size = tf.constant(self.max_sequence_size, dtype=tf.int64)
        else:
            # Typically used at training time, to pad/clip to a fixed number of sequence per query

            # Use rank as a reference tensor to infer shape/sequence_size in query
            reference_tensor = sequence_features.get(self.feature_config.get_rank(key="node_name"))

            # Add mask for identifying padded sequence
            mask = tf.ones_like(sparse.to_dense(sparse.reset_shape(reference_tensor)))

            if self.pad_sequence:
                mask = tf.squeeze(mask, axis=0)

                def crop_fn():
                    # NOTE: We currently ignore these cases as there is no clear
                    # way to select max_sequence_size from all the sequence features
                    tf.print("\n[WARN] Bad query found. Number of sequence : ", tf.shape(mask)[0])
                    return mask

                mask = tf.cond(
                    tf.shape(mask)[0] <= self.max_sequence_size,
                    # Pad if there are missing sequence
                    lambda: tf.pad(
                        mask, [[0, self.max_sequence_size - tf.shape(mask)[0]]]),
                    # Crop if there are extra sequence
                    crop_fn,
                )
                sequence_size = tf.constant(self.max_sequence_size, dtype=tf.int64)
            else:
                mask = tf.squeeze(mask, axis=0)
                sequence_size = tf.cast(tf.reduce_sum(mask), tf.int64)

        # Check validity of mask
        tf.debugging.assert_greater(sequence_size, tf.constant(0, dtype=tf.int64))

        # Update features dictionary with the computed mask tensor
        features_dict["mask"] = mask

        return features_dict, sequence_size

    def pad_feature(self, feature_tensor, feature_info):
        """
        Pad the feature to the `max_sequence_size` in order to create
        uniform data batches for training
        Parameters
        ----------
        feature_tensor: tf.Tensor
            Feature tensor to be padded
        feature_info: dict
            Feature configuration information for the feature as specified in the feature_config
        Returns
        -------
        tf.Tensor
            Feature tensor padded to the `max_sequence_size`
        """
        if self.pad_sequence and feature_info["tfrecord_type"] == SequenceExampleTypeKey.SEQUENCE:
            pad_len = self.max_sequence_size - tf.shape(feature_tensor)[0]
            feature_tensor = tf.pad(feature_tensor, [[0, pad_len]])

        return feature_tensor


def get_parse_fn(
    tfrecord_type: str,
    feature_config: FeatureConfig,
    preprocessing_keys_to_fns: dict,
    max_sequence_size: int = 0,
    required_fields_only: bool = False,
    pad_sequence: bool = True,
) -> tf.function:
    """
    Create a parsing function to extract features from serialized TFRecord data
    using the definition from the FeatureConfig

    Parameters
    ----------
    tfrecord_type: {"example", "sequence_example"}
        Type of TFRecord data to be loaded into a dataset
    feature_config: `FeatureConfig` object
        FeatureConfig object defining the features to be extracted
    preprocessing_keys_to_fns: dict of(str, function), optional
        dictionary of function names mapped to function definitions
        that can now be used for preprocessing while loading the
        TFRecordDataset to create the RelevanceDataset object
    max_sequence_size: int
        Maximum number of sequence per query. Used for padding
    required_fields_only: bool, optional
        Whether to only use required fields from the feature_config
    pad_sequence: bool
        Whether to pad sequence

    Returns
    -------
    `tf.function`
        Parsing function that takes in a serialized SequenceExample or Example message
        and extracts a dictionary of feature tensors
    """
    # Define preprocessing functions
    preprocessing_map = PreprocessingMap()
    preprocessing_map.add_fns(preprocessing_keys_to_fns)

    # Generate parsing function
    if tfrecord_type == TFRecordTypeKey.EXAMPLE:
        parser: TFRecordParser = TFRecordExampleParser(
            feature_config=feature_config,
            preprocessing_map=preprocessing_map,
            required_fields_only=required_fields_only,
        )
    elif tfrecord_type == TFRecordTypeKey.SEQUENCE_EXAMPLE:
        parser = TFRecordSequenceExampleParser(
            feature_config=feature_config,
            preprocessing_map=preprocessing_map,
            max_sequence_size=max_sequence_size,
            required_fields_only=required_fields_only,
            pad_sequence=pad_sequence,
        )
    else:
        raise KeyError("Invalid TFRecord type specified: {}".format(tfrecord_type))

    return parser.get_parse_fn()


def read(
    data_dir: str,
    feature_config: FeatureConfig,
    tfrecord_type: str,
    file_io: FileIO,
    max_sequence_size: int = 0,
    batch_size: int = 0,
    preprocessing_keys_to_fns: dict = {},
    parse_tfrecord: bool = True,
    use_part_files: bool = False,
    logger: Logger = None,
    **kwargs
) -> data.TFRecordDataset:
    """
    Extract features by reading and parsing TFRecord data
    and converting into a TFRecordDataset using the FeatureConfig

    Parameters
    ----------
    data_dir: str
        path to the directory containing train, validation and test data
    feature_config: `FeatureConfig` object
        FeatureConfig object that defines the features to be loaded in the dataset
        and the preprocessing functions to be applied to each of them
    tfrecord_type: {"example", "sequence_example"}
        Type of the TFRecord protobuf message to be used for TFRecordDataset
    file_io: `FileIO` object
        file I/O handler objects for reading and writing data
    max_sequence_size: int, optional
        maximum number of sequence to be used with a single SequenceExample proto message
        The data will be appropriately padded or clipped to fit the max value specified
    batch_size: int, optional
        size of each data batch
    preprocessing_keys_to_fns: dict of(str, function), optional
        dictionary of function names mapped to function definitions
        that can now be used for preprocessing while loading the
        TFRecordDataset to create the RelevanceDataset object
    use_part_files: bool, optional
        load dataset from part files checked using "part-" prefix
    parse_tfrecord: bool, optional
        parse the TFRecord string from the dataset;
        returns strings as is otherwise
    logger: `Logger`, optional
        logging handler for status messages

    Returns
    -------
    `TFRecordDataset`
        TFRecordDataset loaded from the `data_dir` specified using the FeatureConfig
    """
    parse_fn = get_parse_fn(
        feature_config=feature_config,
        tfrecord_type=tfrecord_type,
        preprocessing_keys_to_fns=preprocessing_keys_to_fns,
        max_sequence_size=max_sequence_size,
    )

    # Get all tfrecord files in directory
    tfrecord_files = file_io.get_files_in_directory(
        data_dir,
        extension="" if use_part_files else ".tfrecord",
        prefix="part-" if use_part_files else "",
    )

    # Parse the protobuf data to create a TFRecordDataset
    dataset = data.TFRecordDataset(tfrecord_files)

    if parse_tfrecord:
        # Parallel calls set to AUTOTUNE: improved training performance by 40% with a classification model
        dataset = (
            dataset.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # .apply(data.experimental.ignore_errors())
        )

    # Create BatchedDataSet
    if batch_size:
        dataset = dataset.batch(batch_size, drop_remainder=False)

    if logger:
        logger.info(
            "Created TFRecordDataset from SequenceExample protobufs from {} files : {}".format(
                len(tfrecord_files), str(tfrecord_files)[:50]
            )
        )

    # We apply prefetch as it improved train/test/validation throughput by 30% in some real model training.
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset
