import os
import pandas as pd
import tensorflow as tf

from ml4ir.base.io.file_io import FileIO
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.data import tfrecord_reader, tfrecord_writer, ranklib_helper

from typing import List

TFRECORD_FILE = "file_0.tfrecord"

def read(
    data_dir: str,
    feature_config: FeatureConfig,
    tfrecord_type: str,
    tfrecord_dir: str,
    file_io: FileIO,
    batch_size: int = 128,
    preprocessing_keys_to_fns: dict = {},
    use_part_files: bool = False,
    max_sequence_size: int = 25,
    parse_tfrecord: bool = True,
    logger=None,
    keep_additional_info: bool = False,
    non_zero_features_only: bool = True,
    **kwargs
) -> tf.data.TFRecordDataset:
    """
    - reads ranklib-formatted data from an input directory
    - selects relevant features
    - creates Dataset X and y

    Current execution plan:
        1. Convert ranklib to a dataframe
        2. Convert each query into tf.train.SequenceExample protobufs
        3. Write the protobufs into a .tfrecord file
        4. Load .tfrecord file into a TFRecordDataset and parse the protobufs

        Parameters
        ----------
        data_dir: str
            Path to directory containing csv files to read
        feature_config: ml4ir.config.features.FeatureConfig object
            FeatureConfig object extracted from the feature config
        tfrecord_dir: str
            Path to directory where the serialized .tfrecord files will be stored
        batch_size: int
            Value specifying the size of the batch
        use_part_files: bool
            Value specifying whether to look for part files
        max_sequence_size: int
            Value specifying max number of records per query
        logger: logging object
            logging object
        keep_additional_info: int
            Option to keep additional info (All info after the "#") 1 to keep, 0 to ignore
        non_zero_features_only: int
            Only non zero features are stored. 1 for yes, 0 otherwise

        Returns
        -------
        tensorflow TFRecordDataset
            Processed dataset
    """
    ranklib_files: List[str] = file_io.get_files_in_directory(
        data_dir,
        extension="" if use_part_files else ".txt",
        prefix="part-" if use_part_files else "",
    )

    gl_2_clicks = False

    # Create a directory for storing tfrecord files
    file_io.make_directory(tfrecord_dir, clear_dir=True)

    #Convert input ranklib file to dataframe
    df = pd.concat(
        [
            ranklib_helper.convert(f, keep_additional_info, gl_2_clicks, non_zero_features_only,
                                   feature_config.get_query_key()['name'], feature_config.get_label()['name'])
            for f in ranklib_files
        ])

    #Write tfrecord files
    tfrecord_writer.write_from_df(df=df,
                                  tfrecord_file=os.path.join(tfrecord_dir, TFRECORD_FILE),
                                  feature_config=feature_config,
                                  tfrecord_type=tfrecord_type,
                                  logger=logger)


    dataset = tfrecord_reader.read(
        data_dir=tfrecord_dir,
        feature_config=feature_config,
        tfrecord_type=tfrecord_type,
        max_sequence_size=max_sequence_size,
        batch_size=batch_size,
        preprocessing_keys_to_fns=preprocessing_keys_to_fns,
        parse_tfrecord=parse_tfrecord,
        file_io=file_io,
        logger=logger,
    )

    return dataset