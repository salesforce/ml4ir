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

"""
This module contains helper methods for reading and writing
data in the train.SequenceExample protobuf format
"""


def preprocess_feature(feature_tensor, feature_info, preprocessing_map):
    # Preprocess features
    preprocessing_info = feature_info.get("preprocessing_info", [])

    if preprocessing_info:
        for preprocessing_step in preprocessing_info:
            preprocessing_fn = preprocessing_map.get_fn(preprocessing_step["fn"])
            if preprocessing_fn:
                feature_tensor = preprocessing_fn(
                    feature_tensor, **preprocessing_step.get("args", {})
                )
    return feature_tensor


def make_example_parse_fn(
    feature_config: FeatureConfig,
    preprocessing_map: PreprocessingMap,
    required_fields_only: bool = False,
) -> tf.function:
    """
    Create a parse function using the Example features spec

    Args:
        feature_config: FeatureConfig object defining context and sequence features
        max_sequence_size: Maximum number of sequence per query. Used for padding.
        required_fields_only: Whether to only use required fields from the feature_config
        pad_sequence: Whether to pad sequence
    """

    features_spec = dict()

    for feature_info in feature_config.get_all_features():
        serving_info = feature_info["serving_info"]
        if not required_fields_only or serving_info.get("required", feature_info["trainable"]):
            feature_name = feature_info["name"]
            dtype = feature_info["dtype"]
            default_value = feature_config.get_default_value(feature_info)
            features_spec[feature_name] = io.FixedLenFeature(
                [], dtype, default_value=default_value
            )
    print(features_spec)

    @tf.function
    def _parse_example_fn(example_proto):
        """
        Parse the input `tf.Example` proto using the features_spec

        Args:
            example_proto: tfrecord Example protobuf data

        Returns:
            features: parsed features extracted from the protobuf
            labels: parsed label extracted from the protobuf
        """
        features = io.parse_single_example(serialized=example_proto, features=features_spec)

        features_dict = dict()

        # Process all features, including label.
        for feature_info in feature_config.get_all_features():
            feature_node_name = feature_info.get("node_name", feature_info["name"])

            default_tensor = tf.constant(
                value=feature_config.get_default_value(feature_info), dtype=feature_info["dtype"],
            )
            feature_tensor = features.get(feature_info["name"], default_tensor)

            feature_tensor = tf.expand_dims(feature_tensor, axis=0)

            feature_tensor = preprocess_feature(feature_tensor, feature_info, preprocessing_map)

            features_dict[feature_node_name] = feature_tensor

        labels = features_dict.pop(feature_config.get_label(key="name"))

        return features_dict, labels

    return _parse_example_fn


def make_sequence_example_parse_fn(
    feature_config: FeatureConfig,
    preprocessing_map: PreprocessingMap,
    max_sequence_size: int = 25,
    required_fields_only: bool = False,
    pad_sequence: bool = True,
) -> tf.function:
    """
    Create a parse function using the context and sequence features spec

    Args:
        feature_config: FeatureConfig object defining context and sequence features
        max_sequence_size: Maximum number of sequence per query. Used for padding.
        required_fields_only: Whether to only use required fields from the feature_config
        pad_sequence: Whether to pad sequence
    """

    context_features_spec = dict()
    sequence_features_spec = dict()

    for feature_info in feature_config.get_all_features():
        serving_info = feature_info["serving_info"]
        if not required_fields_only or serving_info.get("required", feature_info["trainable"]):
            feature_name = feature_info["name"]
            dtype = feature_info["dtype"]
            default_value = feature_config.get_default_value(feature_info)
            if feature_info["tfrecord_type"] == SequenceExampleTypeKey.CONTEXT:
                context_features_spec[feature_name] = io.FixedLenFeature(
                    [], dtype, default_value=default_value
                )
            elif feature_info["tfrecord_type"] == SequenceExampleTypeKey.SEQUENCE:
                sequence_features_spec[feature_name] = io.VarLenFeature(dtype=dtype)

    @tf.function
    def _parse_sequence_example_fn(sequence_example_proto):
        """
        Parse the input `tf.SequenceExample` proto using the features_spec

        Args:
            sequence_example_proto: tfrecord SequenceExample protobuf data

        Returns:
            features: parsed features extracted from the protobuf
            labels: parsed label extracted from the protobuf
        """
        context_features, sequence_features = io.parse_single_sequence_example(
            serialized=sequence_example_proto,
            context_features=context_features_spec,
            sequence_features=sequence_features_spec,
        )

        features_dict = dict()

        # Handle context features
        for feature_info in feature_config.get_context_features():
            feature_node_name = feature_info.get("node_name", feature_info["name"])

            default_tensor = tf.constant(
                value=feature_config.get_default_value(feature_info), dtype=feature_info["dtype"],
            )
            feature_tensor = context_features.get(feature_info["name"], default_tensor)

            feature_tensor = tf.expand_dims(feature_tensor, axis=0)

            # Preprocess features
            feature_tensor = preprocess_feature(feature_tensor, feature_info, preprocessing_map)

            features_dict[feature_node_name] = feature_tensor

        # Define mask to identify padded sequence
        if required_fields_only and not feature_config.get_rank("serving_info")["required"]:
            """
            Define dummy mask if the rank field is not a required field for serving

            NOTE:
            This masks all max_sequence_size as 1 as there is no real way to know
            the number of sequence in the query. There is no predefined required field,
            and hence we would need to do a full pass of all features to find the record shape.
            This approach might be unstable if different features have different shapes.

            Hence we just mask all sequence
            """
            features_dict["mask"] = tf.constant(
                value=1, shape=[max_sequence_size], dtype=feature_config.get_rank("dtype")
            )
            sequence_size = tf.constant(max_sequence_size, dtype=tf.int64)
        else:
            # Typically used at training time, to pad/clip to a fixed number of sequence per query

            # Use rank as a reference tensor to infer shape/sequence_size in query
            reference_tensor = sequence_features.get(feature_config.get_rank(key="node_name"))

            # Add mask for identifying padded sequence
            mask = tf.ones_like(sparse.to_dense(sparse.reset_shape(reference_tensor)))
            sequence_size = tf.cast(tf.reduce_sum(mask), tf.int64)

            if pad_sequence:
                mask = tf.expand_dims(mask, axis=-1)

                def crop_fn():
                    tf.print("\n[WARN] Bad query found. Number of sequence : ", tf.shape(mask)[1])
                    return image.crop_to_bounding_box(
                        mask,
                        offset_height=0,
                        offset_width=0,
                        target_height=1,
                        target_width=max_sequence_size,
                    )

                mask = tf.cond(
                    tf.shape(mask)[1] <= max_sequence_size,
                    # Pad if there are missing sequence
                    lambda: image.pad_to_bounding_box(
                        mask,
                        offset_height=0,
                        offset_width=0,
                        target_height=1,
                        target_width=max_sequence_size,
                    ),
                    # Crop if there are extra sequence
                    crop_fn,
                )
                mask = tf.squeeze(mask)
            else:
                mask = tf.squeeze(mask, axis=0)

            # Check validity of mask
            tf.debugging.assert_greater(sequence_size, tf.constant(0, dtype=tf.int64))

            features_dict["mask"] = mask
            sequence_size = max_sequence_size if pad_sequence else sequence_size

        # Pad sequence features to max_sequence_size
        for feature_info in feature_config.get_sequence_features():
            feature_node_name = feature_info.get("node_name", feature_info["name"])

            default_tensor = tf.fill(
                value=tf.constant(
                    value=feature_config.get_default_value(feature_info),
                    dtype=feature_info["dtype"],
                ),
                dims=[max_sequence_size if pad_sequence else sequence_size],
            )
            feature_tensor = sequence_features.get(feature_info["name"], default_tensor)

            if isinstance(feature_tensor, sparse.SparseTensor):
                feature_tensor = sparse.reset_shape(
                    feature_tensor,
                    new_shape=[1, max_sequence_size if pad_sequence else sequence_size],
                )
                feature_tensor = sparse.to_dense(feature_tensor)
                feature_tensor = tf.squeeze(feature_tensor, axis=0)

            # Preprocess features
            feature_tensor = preprocess_feature(feature_tensor, feature_info, preprocessing_map)

            features_dict[feature_node_name] = feature_tensor

        labels = features_dict.pop(feature_config.get_label(key="name"))

        if not required_fields_only:
            # Check if label is one-hot and correctly masked
            tf.debugging.assert_equal(tf.cast(tf.reduce_sum(labels), tf.float32), tf.constant(1.0))

        return features_dict, labels

    return _parse_sequence_example_fn


def get_parse_fn(
    tfrecord_type: str,
    feature_config: FeatureConfig,
    preprocessing_keys_to_fns: dict,
    max_sequence_size: int = 0,
    required_fields_only: bool = False,
    pad_sequence: bool = True,
):
    # Define preprocessing functions
    preprocessing_map = PreprocessingMap()
    preprocessing_map.add_fns(preprocessing_keys_to_fns)

    # Generate parsing function
    if tfrecord_type == TFRecordTypeKey.EXAMPLE:
        parse_fn = make_example_parse_fn(
            feature_config=feature_config,
            preprocessing_map=preprocessing_map,
            required_fields_only=required_fields_only,
        )
    elif tfrecord_type == TFRecordTypeKey.SEQUENCE_EXAMPLE:
        parse_fn = make_sequence_example_parse_fn(
            feature_config=feature_config,
            preprocessing_map=preprocessing_map,
            max_sequence_size=max_sequence_size,
            required_fields_only=required_fields_only,
            pad_sequence=pad_sequence,
        )
    else:
        raise KeyError("Invalid TFRecord type specified: {}".format(tfrecord_type))

    return parse_fn


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
    - reads tfrecord data from an input directory
    - selects relevant features
    - creates X and y data

    Args:
        data_dir: Path to directory containing csv files to read
        feature_config: ml4ir.config.features.Features object extracted from the feature config
        tfrecord_type: either example or sequence_example
        batch_size: int value specifying the size of the batch
        preprocessing_keys_to_fns: dictionary mapping preprocessing keys in the feature_config to functions
        parse_tfrecord: whether to parse SequenceExamples into features
        logger: logging object

    Returns:
        tensorflow dataset
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
        dataset = dataset.map(parse_fn).apply(data.experimental.ignore_errors())

    # Create BatchedDataSet
    if batch_size:
        dataset = dataset.batch(batch_size, drop_remainder=True)

    if logger:
        logger.info(
            "Created TFRecordDataset from SequenceExample protobufs from {} files : {}".format(
                len(tfrecord_files), str(tfrecord_files)[:50]
            )
        )

    return dataset
