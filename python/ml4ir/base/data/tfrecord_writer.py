"""
Writes data in Example or SequenceExample protobuf (tfrecords) format.

To use it as a standalone script, refer to the argument spec
at the bottom

Notes
-----

Setting ``--keep-single-files`` writes one tfrecord file
for each CSV file (better performance). If not set,
joins everything to a single tfrecord file.

Examples
--------

Syntax to convert a single or several CSVs:

>>> python ml4ir/base/data/tfrecord_writer.py \\
... sequence_example|example \\
... --csv-files <SPACE_SEPARATED_PATHS_TO_CSV_FILES> \\
... --out-dir <PATH_TO_OUTPUT_DIR> \\
... --feature_config <PATH_TO_YAML_FEATURE_CONFIG> \\
... --keep-single-files

or to convert all CSV files in a dir

>>> python ml4ir/base/data/tfrecord_writer.py \\
... sequence_example|example \\
... --csv-dir <DIR_WITH_CSVs> \\
... --out-dir <PATH_TO_OUTPUT_DIR> \\
... --feature_config <PATH_TO_YAML_FEATURE_CONFIG> \\
... --keep-single-files

Usage example:

>>> python ml4ir/base/data/tfrecord_writer.py \\
... sequence_example \\
... --csv-files /tmp/d.csv /tmp/d2.csv \\
... --out-dir /tmp \\
... --feature-config /tmp/fconfig.yaml \\
... --keep-single-files
 """

from tensorflow import io
from typing import List
from logging import Logger
from argparse import ArgumentParser
import os
from pandas import DataFrame

from ml4ir.base.io.file_io import FileIO
from ml4ir.base.io.local_io import LocalIO
from ml4ir.base.config.keys import TFRecordTypeKey
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.io.logging_utils import setup_logging
from ml4ir.base.data.tfrecord_helper import get_sequence_example_proto, get_example_proto

MODES = {"example": TFRecordTypeKey.EXAMPLE, "sequence_example": TFRecordTypeKey.SEQUENCE_EXAMPLE}


def write_from_files(
    csv_files: List[str],
    tfrecord_file: str,
    feature_config: FeatureConfig,
    tfrecord_type: str,
    file_io: FileIO,
    logger: Logger = None,
):
    """
    Converts data from CSV files into tfrecord files

    Parameters
    ----------
    csv_files : list of str
        list of csv file paths to read data from
    tfrecord_file : str
        tfrecord file path to write the output
    feature_config : `FeatureConfig`
        FeatureConfig object that defines the features to be loaded in the dataset
        and the preprocessing functions to be applied to each of them
    tfrecord_type : {"example", "sequence_example"}
        Type of the TFRecord protobuf message to be used for TFRecordDataset
    logger : `Logger`, optional
        logging handler for status messages
    """

    # Read CSV data into a pandas dataframe
    df = file_io.read_df_list(csv_files)
    write_from_df(df, tfrecord_file, feature_config, tfrecord_type, logger)


def write_from_df(
    df: DataFrame,
    tfrecord_file: str,
    feature_config: FeatureConfig,
    tfrecord_type: str,
    logger: Logger = None,
):
    """
    Converts data from CSV files into tfrecord files

    Parameters
    df : `pd.DataFrame`
        pandas DataFrame to be converted to TFRecordDataset
    tfrecord_file : str
        tfrecord file path to write the output
    feature_config : `FeatureConfig`
        FeatureConfig object that defines the features to be loaded in the dataset
        and the preprocessing functions to be applied to each of them
    tfrecord_type : {"example", "sequence_example"}
        Type of the TFRecord protobuf message to be used for TFRecordDataset
    logger : `Logger`, optional
        logging handler for status messages
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
        else:
            raise Exception(
                "You have entered {} as tfrecords write mode. "
                "We only support {} and {}.".format(
                    tfrecord_type, TFRecordTypeKey.EXAMPLE, TFRecordTypeKey.SEQUENCE_EXAMPLE
                )
            )
        # Write to disk
        for proto in protos:
            tf_writer.write(proto.SerializeToString())


def main(args):
    """Convert CSV files into tfrecord Example/SequenceExample files"""
    # Setup logging
    logger: Logger = setup_logging()
    file_io = LocalIO(logger)

    # Get all CSV files to be converted, depending on user's arguments
    if args.csv_dir:
        csv_files: List[str] = file_io.get_files_in_directory(
            indir=args.csv_dir, extension="*.csv"
        )
    else:
        csv_files: List[str] = args.csv_files

    # Load feat config

    feature_config: FeatureConfig = FeatureConfig.get_instance(
        tfrecord_type=MODES[args.tfmode],
        feature_config_dict=file_io.read_yaml(args.feature_config),
        logger=logger,
    )

    # Convert to TFRecord SequenceExample protobufs and save
    if args.keep_single_files:
        # Convert each CSV file individually - better performance
        for csv_file in csv_files:
            tfrecord_file: str = os.path.basename(csv_file).replace(".csv", "")
            tfrecord_file: str = os.path.join(args.out_dir, "{}.tfrecord".format(tfrecord_file))
            write_from_files(
                csv_files=[csv_file],
                tfrecord_file=tfrecord_file,
                feature_config=feature_config,
                logger=logger,
                tfrecord_type=MODES[args.tfmode],
            )

    else:
        # Convert all CSV files at once - expensive groupby operation
        tfrecord_file: str = os.path.join(args.out_dir, "combined.tfrecord")
        write_from_files(
            csv_files=csv_files,
            tfrecord_file=tfrecord_file,
            feature_config=feature_config,
            logger=logger,
            tfrecord_type=MODES[args.tfmode],
            file_io=file_io,
        )


def define_arguments():
    first_doc_line = __doc__.strip().split("\n")[0]
    parser = ArgumentParser(description=first_doc_line)
    parser.add_argument(
        "tfmode",
        choices=MODES,  # Choices with a dict, shows the keys
        help="select between `sequence` and `example` to write tf.Example or tf.SequenceExample",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--csv-dir", type=str, help="Directory with CSV files; every .csv file will be converted."
    )
    group.add_argument(
        "--csv-files",
        type=str,
        nargs="+",
        help="A single or more (space separated) CSV files to be converted.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Output directory for TFRecord files. Base filenames are maintained and .tfrecord is added as extension.",
    )
    parser.add_argument(
        "--feature-config",
        type=str,
        default=None,
        help="Path to feature config YAML file or feature config YAML string",
    )
    parser.add_argument(
        "--keep-single-files",
        action="store_true",
        help="When passed, converts CSV files individually. "
        "Results are written to out-dir replacing the filename's extension with .tfrecord."
        "If not set, a single combined.tfrecord is created."
        "All occurrences of a query key should be within a single file",
    )
    return parser


if __name__ == "__main__":
    my_parser = define_arguments()
    args = my_parser.parse_args()
    main(args)
