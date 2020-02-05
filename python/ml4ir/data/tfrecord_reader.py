import tensorflow as tf
from tensorflow import io
from tensorflow import data
from tensorflow import sparse
from tensorflow import image
from logging import Logger
from ml4ir.io import file_io
from ml4ir.config.features import Features
from ml4ir.config.keys import TFRecordTypeKey, FeatureTypeKey

from typing import Union, Optional

"""
This module contains helper methods for reading and writing
data in the train.SequenceExample protobuf format
"""


def make_parse_fn(feature_config: Features, max_num_records: int = 25) -> tf.function:
    """Create a parse function using the context and sequence features spec"""

    context_features_spec = dict()
    sequence_features_spec = dict()

    for feature, feature_info in feature_config.get_dict().items():
        # FIXME(ashish) - without this next guard we break if there are masks.
        if "node_name" in feature_info and feature_info["node_name"] == "mask":
            continue
        tfrecord_info = feature_info["tfrecord_info"]
        dtype = tf.float32
        default_value: Optional[Union[float, str]] = None
        if tfrecord_info["dtype"] == "float":
            dtype = tf.float32
            default_value = 0.0
        elif tfrecord_info["dtype"] == "int":
            dtype = tf.int64
            default_value = 0
        elif tfrecord_info["dtype"] == "bytes":
            dtype = tf.string
            default_value = ""
        else:
            raise Exception("Unknown dtype {} for {}".format(tfrecord_info["dtype"], feature))
        if tfrecord_info["type"] == TFRecordTypeKey.CONTEXT:
            context_features_spec[feature] = io.FixedLenFeature(
                [], dtype, default_value=default_value
            )
        elif tfrecord_info["type"] == TFRecordTypeKey.SEQUENCE:
            sequence_features_spec[feature] = io.VarLenFeature(dtype=dtype)

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
        context, examples = io.parse_single_sequence_example(
            serialized=sequence_example_proto,
            context_features=context_features_spec,
            sequence_features=sequence_features_spec,
        )

        features_dict = dict()

        # Explode context features into all records
        for feat, t in context.items():
            t = tf.expand_dims(t, axis=0)
            t = tf.tile(t, multiples=[max_num_records])

            # If feature is a string, then decode into numbers
            if feature_config.get_dict()[feat]["type"] == FeatureTypeKey.STRING:
                t = io.decode_raw(
                    t,
                    out_type=tf.uint8,
                    fixed_length=feature_config.get_dict()[feat]["max_length"],
                )
                t = tf.cast(t, tf.float32)

            features_dict[feat] = t

        # Pad sequence features to max_num_records
        for feat, t in examples.items():
            if isinstance(t, sparse.SparseTensor):
                if feat == "pos":
                    # Add mask for identifying padded records
                    mask = tf.ones_like(sparse.to_dense(sparse.reset_shape(t)))
                    mask = tf.expand_dims(mask, axis=2)
                    mask = image.pad_to_bounding_box(
                        mask,
                        offset_height=0,
                        offset_width=0,
                        target_height=1,
                        target_width=max_num_records,
                    )
                    features_dict["mask"] = tf.squeeze(mask)

                t = sparse.reset_shape(t, new_shape=[1, max_num_records])
                t = sparse.to_dense(t)
                t = tf.squeeze(t)

                # If feature is a string, then decode into numbers
                if feature_config.get_dict()[feat]["type"] == FeatureTypeKey.STRING:
                    t = io.decode_raw(
                        t,
                        out_type=tf.uint8,
                        fixed_length=feature_config.get_dict()[feat]["max_length"],
                    )
                    t = tf.cast(t, tf.float32)
            else:
                #
                # Handle dense tensors
                #
                # if len(t.shape) == 1:
                #     t = tf.expand_dims(t, axis=0)
                # if len(t.shape) == 2:
                #     t = tf.pad(t, paddings=[[0, 0], [0, max_num_records]])
                #     t = tf.squeeze(t)
                # else:
                #     raise Exception('Invalid input : {}'.format(feat))
                raise ValueError("Invalid input : {}".format(feat))

            features_dict[feat] = t

        labels = features_dict.pop(feature_config.label)
        return features_dict, labels

    return _parse_sequence_example_fn


def read(
    data_dir: str,
    features: Features,
    max_num_records: int = 25,
    batch_size: int = 128,
    parse_tfrecord: bool = True,
    logger: Logger = None,
    **kwargs
) -> tf.data.TFRecordDataset:
    """
    - reads tfrecord data from an input directory
    - selects relevant features
    - creates X and y data

    Args:
        data_dir: Path to directory containing csv files to read
        features: ml4ir.config.features.Features object extracted from the feature config
        batch_size: int value specifying the size of the batch
        parse_tfrecord: whether to parse SequenceExamples into features
        logger: logging object

    Returns:
        tensorflow dataset
    """
    # Generate parsing function
    parse_sequence_example_fn = make_parse_fn(
        feature_config=features, max_num_records=max_num_records
    )

    # Get all tfrecord files in directory
    tfrecord_files = file_io.get_files_in_directory(data_dir, extension=".tfrecord")

    # Parse the protobuf data to create a TFRecordDataset
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    if parse_tfrecord:
        dataset = dataset.map(parse_sequence_example_fn)
    dataset = dataset.batch(batch_size)

    if logger:
        logger.info(
            "Created TFRecordDataset from SequenceExample protobufs from {} files : {}".format(
                len(tfrecord_files), str(tfrecord_files)[:50]
            )
        )

    return dataset
