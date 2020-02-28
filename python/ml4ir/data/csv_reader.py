import glob
from ml4ir.io import file_io
import os
import tensorflow as tf
from ml4ir.config.features import FeatureConfig
from ml4ir.data import tfrecord_reader, tfrecord_writer
from typing import List

# Constants
TFRECORD_FILE = "file_0.tfrecord"


def read(
    data_dir: str,
    feature_config: FeatureConfig,
    tfrecord_dir: str,
    batch_size: int = 128,
    max_num_records: int = 25,
    parse_tfrecord: bool = True,
    logger=None,
    **kwargs
) -> tf.data.TFRecordDataset:
    """
    - reads csv-formatted data from an input directory
    - selects relevant features
    - creates Dataset X and y

    Current execution plan:
        1. Load CSVs as pandas dataframes
        2. Convert each query into tf.train.SequenceExample protobufs
        3. Write the protobufs into a .tfrecord file
        4. Load .tfrecord file into a TFRecordDataset and parse the protobufs

    Args:
        - data_dir: Path to directory containing csv files to read
        - feature_config: ml4ir.config.features.FeatureConfig object extracted from the feature config
        - tfrecord_dir: Path to directory where the serialized .tfrecord files will be stored
        - batch_size: int value specifying the size of the batch
        - max_num_records: int value specifying max number of records per query
        - logger: logging object

    Returns:
        tensorflow TFRecordDataset
    """
    csv_files: List[str] = glob.glob(os.path.join(data_dir, "*.csv"))

    # Create a directory for storing tfrecord files
    file_io.make_directory(tfrecord_dir, clear_dir=True)

    # Write tfrecord files
    tfrecord_writer.write(
        csv_files=csv_files,
        tfrecord_file=os.path.join(tfrecord_dir, TFRECORD_FILE),
        feature_config=feature_config,
        logger=logger,
    )

    dataset = tfrecord_reader.read(
        data_dir=tfrecord_dir,
        feature_config=feature_config,
        max_num_records=max_num_records,
        batch_size=batch_size,
        parse_tfrecord=parse_tfrecord,
        logger=logger,
    )

    return dataset
