import tensorflow as tf
from tensorflow import io
from tensorflow import data
from tensorflow import sparse
from tensorflow import image
from logging import Logger
from ml4ir.io import file_io
from ml4ir.config.features import FeatureConfig
from ml4ir.config.keys import TFRecordTypeKey, FeatureTypeKey

from typing import Union, Optional

"""
This module contains helper methods for reading and writing
data in the train.SequenceExample protobuf format
"""


def make_parse_fn(feature_config: FeatureConfig, max_num_records: int = 25) -> tf.function:
    """Create a parse function using the context and sequence features spec"""

    context_features_spec = dict()
    sequence_features_spec = dict()

    for feature_info in feature_config.get_all_features():
        feature_name = feature_info["name"]
        dtype = tf.float32
        default_value: Optional[Union[float, str]] = None
        if feature_info["dtype"] == "float":
            dtype = tf.float32
            default_value = 0.0
        elif feature_info["dtype"] == "int":
            dtype = tf.int64
            default_value = 0
        elif feature_info["dtype"] == "bytes":
            dtype = tf.string
            default_value = ""
        else:
            raise Exception("Unknown dtype {} for {}".format(feature_info["dtype"], feature_name))
        if feature_info["tfrecord_type"] == TFRecordTypeKey.CONTEXT:
            context_features_spec[feature_name] = io.FixedLenFeature(
                [], dtype, default_value=default_value
            )
        elif feature_info["tfrecord_type"] == TFRecordTypeKey.SEQUENCE:
            sequence_features_spec[feature_name] = io.VarLenFeature(dtype=dtype)

    @tf.function
    def _parse_sequence_example_fn(sequence_example_proto):
        """
        Parse the input `tf.Example` proto using the features_spec

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

        # Explode context features into all records
        for feature_info in feature_config.get_context_features():
            feature_name = feature_info["name"]
            feature_layer_info = feature_info.get("feature_layer_info")

            feature_tensor = context_features.get(feature_name)

            feature_tensor = tf.expand_dims(feature_tensor, axis=0)
            feature_tensor = tf.tile(feature_tensor, multiples=[max_num_records])

            # If feature is a string, then decode into numbers
            if feature_layer_info["type"] == FeatureTypeKey.STRING:
                feature_tensor = io.decode_raw(
                    feature_tensor,
                    out_type=tf.uint8,
                    fixed_length=feature_layer_info["max_length"],
                )
                feature_tensor = tf.cast(feature_tensor, tf.float32)

            features_dict[feature_name] = feature_tensor

        # Pad sequence features to max_num_records
        for feature_info in feature_config.get_sequence_features():
            feature_name = feature_info["name"]
            feature_layer_info = feature_info["feature_layer_info"]

            feature_tensor = sequence_features.get(feature_name)

            if isinstance(feature_tensor, sparse.SparseTensor):
                if feature_name == feature_config.get_rank(key="name"):
                    # Add mask for identifying padded records
                    mask = tf.ones_like(sparse.to_dense(sparse.reset_shape(feature_tensor)))
                    mask = tf.expand_dims(mask, axis=2)

                    def crop_fn():
                        tf.print(
                            "\n[WARN] Bad query found. Number of records : ", tf.shape(mask)[1]
                        )
                        return image.crop_to_bounding_box(
                            mask,
                            offset_height=0,
                            offset_width=0,
                            target_height=1,
                            target_width=max_num_records,
                        )

                    mask = tf.cond(
                        tf.shape(mask)[1] < max_num_records,
                        # Pad if there are missing records
                        lambda: image.pad_to_bounding_box(
                            mask,
                            offset_height=0,
                            offset_width=0,
                            target_height=1,
                            target_width=max_num_records,
                        ),
                        # Crop if there are extra records
                        crop_fn,
                    )
                    mask = tf.squeeze(mask)

                    # Check validity of mask
                    tf.debugging.assert_greater(
                        tf.cast(tf.reduce_sum(mask), tf.float32), tf.constant(0.0)
                    )

                    features_dict["mask"] = mask

                feature_tensor = sparse.reset_shape(feature_tensor, new_shape=[1, max_num_records])
                feature_tensor = sparse.to_dense(feature_tensor)
                feature_tensor = tf.squeeze(feature_tensor)

                # If feature is a string, then decode into numbers
                if feature_layer_info["type"] == FeatureTypeKey.STRING:
                    feature_tensor = io.decode_raw(
                        feature_tensor,
                        out_type=tf.uint8,
                        fixed_length=feature_layer_info["max_length"],
                    )
                    feature_tensor = tf.cast(feature_tensor, tf.float32)
            else:
                #
                # Handle dense tensors
                #
                # if len(t.shape) == 1:
                #     feature_tensor = tf.expand_dims(t, axis=0)
                # if len(t.shape) == 2:
                #     feature_tensor = tf.pad(t, paddings=[[0, 0], [0, max_num_records]])
                #     feature_tensor = tf.squeeze(t)
                # else:
                #     raise Exception('Invalid input : {}'.format(feat))
                raise ValueError("Invalid input : {}".format(feature_name))

            features_dict[feature_name] = feature_tensor

        labels = features_dict.pop(feature_config.get_label(key="name"))

        # Check if label is one-hot and correctly masked
        tf.debugging.assert_equal(tf.cast(tf.reduce_sum(labels), tf.float32), tf.constant(1.0))

        return features_dict, labels

    return _parse_sequence_example_fn


def read(
    data_dir: str,
    feature_config: FeatureConfig,
    max_num_records: int = 25,
    batch_size: int = 128,
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
        batch_size: int value specifying the size of the batch
        parse_tfrecord: whether to parse SequenceExamples into features
        logger: logging object

    Returns:
        tensorflow dataset
    """
    # Generate parsing function
    parse_sequence_example_fn = make_parse_fn(
        feature_config=feature_config, max_num_records=max_num_records
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
        dataset = dataset.map(parse_sequence_example_fn).apply(data.experimental.ignore_errors())
    dataset = dataset.batch(batch_size, drop_remainder=True)

    if logger:
        logger.info(
            "Created TFRecordDataset from SequenceExample protobufs from {} files : {}".format(
                len(tfrecord_files), str(tfrecord_files)[:50]
            )
        )

    return dataset
