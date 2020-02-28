from tensorflow import train
from tensorflow import io
from ml4ir.io import file_io
from typing import List
from logging import Logger
from ml4ir.config.features import FeatureConfig, parse_config
from ml4ir.config.keys import TFRecordTypeKey
from argparse import ArgumentParser
import glob
import os
import sys
from ml4ir.io.logging_utils import setup_logging


"""
This module contains helper methods for writing
data in the train.SequenceExample protobuf format

To use it as a standalone script, refer to the argument spec
at the bottom

Syntax:
python ml4ir/data/tfrecord_writer.py \
--csv_dir <PATH_TO_CSV_DIR> \
--tfrecord_dir <PATH_TO_OUTPUT_DIR> \
--feature_config <PATH_TO_FEATURE_CONFIG> \
--convert_single_files <True/False>


Example:
python ml4ir/data/tfrecord_writer.py \
--csv_dir `pwd`/python/ml4ir/tests/data/csv/train \
--tfrecord_dir `pwd`/python/ml4ir/tests/data/tfrecord/train \
--feature_config `pwd`/python/ml4ir/tests/data/csv/feature_config.json \
--convert_single_files True

"""


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
    if dtype == "bytes":
        return _bytes_feature
    elif dtype == "float":
        return _float_feature
    elif dtype == "int":
        return _int64_feature
    else:
        raise Exception("Feature dtype {} not supported".format(dtype))


def _get_sequence_example_proto(group, feature_config: FeatureConfig):
    """
    Get a sequence example protobuf from a dataframe group

    Args:
        - group: pandas dataframe group
    """
    sequence_features_dict = dict()
    context_features_dict = dict()

    for feature_info in feature_config.get_context_features():
        feature_name = feature_info["name"]
        feature_fn = _get_feature_fn(feature_info["dtype"])
        context_features_dict[feature_name] = feature_fn([group[feature_name].tolist()[0]])

    for feature_info in feature_config.get_sequence_features():
        feature_name = feature_info["name"]
        feature_fn = _get_feature_fn(feature_info["dtype"])
        if feature_info["tfrecord_type"] == TFRecordTypeKey.SEQUENCE:
            sequence_features_dict[feature_name] = train.FeatureList(
                feature=[feature_fn(group[feature_name].tolist())]
            )

    sequence_example_proto = train.SequenceExample(
        context=train.Features(feature=context_features_dict),
        feature_lists=train.FeatureLists(feature_list=sequence_features_dict),
    )

    return sequence_example_proto


def write(
    csv_files: List[str], tfrecord_file: str, feature_config: FeatureConfig, logger: Logger = None
):
    """
    Converts data from CSV files into tfrecord data.
    Output data protobuf format -> train.SequenceExample

    Args:
        csv_files: list of csv file paths to read data from
        tfrecord_file: tfrecord file path to write the output
        feature_config: str path to JSON feature config or str JSON feature config
        logger: logging object

    NOTE: This method should be moved out of ml4ir and into the preprocessing pipeline
    """

    # Read CSV data into a pandas dataframe
    df = file_io.read_df_list(csv_files, log=logger)

    # Group pandas dataframe on query_id/query key and
    # convert each group to a single sequence example proto
    if logger:
        logger.info("Writing SequenceExample protobufs to : {}".format(tfrecord_file))
    with io.TFRecordWriter(tfrecord_file) as tf_writer:
        context_feature_names = feature_config.get_context_features(key="name")
        sequence_example_protos = df.groupby(context_feature_names).apply(
            lambda g: _get_sequence_example_proto(group=g, feature_config=feature_config)
        )
        for sequence_example_proto in sequence_example_protos:
            tf_writer.write(sequence_example_proto.SerializeToString())
    tf_writer.close()


def main(argv):
    """Convert CSV files into tfrecord SequenceExample files"""

    # Define script arguments
    parser = ArgumentParser(description="Process arguments for ml4ir ranking pipeline.")

    parser.add_argument(
        "--csv_dir", type=str, default=None, help="Path to the data directory containing CSV files"
    )
    parser.add_argument(
        "--csv_file", type=str, default=None, help="Path to the CSV file to convert"
    )
    parser.add_argument(
        "--tfrecord_dir",
        type=str,
        default=None,
        help="Path to the output directory to write TFRecord files",
    )
    parser.add_argument(
        "--tfrecord_file",
        type=str,
        default=None,
        help="Path to the output file to write TFRecord data",
    )
    parser.add_argument(
        "--feature_config",
        type=str,
        default=None,
        help="Path to feature config JSON file or feature config JSON string",
    )
    parser.add_argument(
        "--convert_single_files",
        type=bool,
        default=False,
        help="Whether to convert each CSV file individually"
        "All occurences of a query key should be within a single file",
    )
    args = parser.parse_args(argv)

    # Get all CSV files to be converted
    if args.csv_dir:
        csv_files: List[str] = glob.glob(os.path.join(args.csv_dir, "*.csv"))
    else:
        csv_files: List[str] = [args.csv_file]

    feature_config: FeatureConfig = parse_config(args.feature_config)

    # Setup logging
    logger: Logger = setup_logging()

    # Convert to TFRecord SequenceExample protobufs and save
    file_count = 0
    if args.convert_single_files:
        # Convert each CSV file individually - better performance
        for csv_file in csv_files:
            if args.tfrecord_dir:
                tfrecord_file: str = os.path.join(
                    args.tfrecord_dir, "file_{}.tfrecord".format(file_count)
                )
            else:
                tfrecord_file: str = args.tfrecord_file

            write(
                csv_files=[csv_file],
                tfrecord_file=tfrecord_file,
                feature_config=feature_config,
                logger=logger,
            )

            file_count += 1
    else:
        # Convert all CSV files at once - expensive groupby operation
        if args.tfrecord_dir:
            tfrecord_file: str = os.path.join(
                args.tfrecord_dir, "file_{}.tfrecord".format(file_count)
            )
        else:
            tfrecord_file: str = args.tfrecord_file

        write(
            csv_files=csv_files,
            tfrecord_file=tfrecord_file,
            feature_config=feature_config,
            logger=logger,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
