from tensorflow import io
from typing import List
from logging import Logger
from argparse import ArgumentParser
import glob
import os
import sys
from pandas import DataFrame

from ml4ir.base.io import file_io
from ml4ir.base.config.keys import TFRecordTypeKey
from ml4ir.base.features.feature_config import FeatureConfig, parse_config
from ml4ir.base.io.logging_utils import setup_logging
from ml4ir.base.data.tfrecord_helper import get_sequence_example_proto, get_example_proto


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


def write_from_files(
    csv_files: List[str],
    tfrecord_file: str,
    feature_config: FeatureConfig,
    tfrecord_type: str,
    logger: Logger = None,
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

    write_from_df(df, tfrecord_file, feature_config, tfrecord_type, logger)


def write_from_df(
    df: DataFrame,
    tfrecord_file: str,
    feature_config: FeatureConfig,
    tfrecord_type: str,
    logger: Logger = None,
):
    """
    Converts data from CSV files into tfrecord data.
    Output data protobuf format -> train.SequenceExample

    Args:
        df: pandas DataFrame
        tfrecord_file: tfrecord file path to write the output
        feature_config: str path to JSON feature config or str JSON feature config
        logger: logging object

    NOTE: This method should be moved out of ml4ir and into the preprocessing pipeline
    """

    if logger:
        logger.info("Writing SequenceExample protobufs to : {}".format(tfrecord_file))
    with io.TFRecordWriter(tfrecord_file) as tf_writer:
        if tfrecord_type == TFRecordTypeKey.EXAMPLE:
            protos = df.apply(
                lambda row: get_example_proto(row=row, features=feature_config.get_all_features()),
                axis=1,
            )
        elif tfrecord_type == TFRecordTypeKey.SEQUENCE_EXAMPLE:
            # Group pandas dataframe on query_id/query key and
            # convert each group to a single sequence example proto
            context_feature_names = feature_config.get_context_features(key="name")
            protos = df.groupby(context_feature_names).apply(
                lambda g: get_sequence_example_proto(
                    group=g,
                    context_features=feature_config.get_context_features(),
                    sequence_features=feature_config.get_sequence_features(),
                )
            )

        # Write to disk
        for proto in protos:
            tf_writer.write(proto.SerializeToString())

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

            write_from_files(
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

        write_from_files(
            csv_files=csv_files,
            tfrecord_file=tfrecord_file,
            feature_config=feature_config,
            logger=logger,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
