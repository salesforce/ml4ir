import glob
import os
from typing import Optional
from logging import Logger
import tensorflow as tf

from ml4ir.base.config.keys import DataFormatKey, DataSplitKey
from ml4ir.base.data import csv_reader
from ml4ir.base.data import tfrecord_reader
from ml4ir.base.data import ranklib_reader
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.io.file_io import FileIO


class RelevanceDataset:
    """class to create/load TFRecordDataset for train, validation and test"""

    def __init__(
        self,
        data_dir: str,
        data_format: str,
        feature_config: FeatureConfig,
        tfrecord_type: str,
        file_io: FileIO,
        max_sequence_size: int = 0,
        batch_size: int = 128,
        preprocessing_keys_to_fns: dict = {},
        train_pcent_split: float = 0.8,
        val_pcent_split: float = -1,
        test_pcent_split: float = -1,
        use_part_files: bool = False,
        parse_tfrecord: bool = True,
        logger: Optional[Logger] = None,
        keep_additional_info: int = 0,
        non_zero_features_only: int = 0,
    ):
        """
        Constructor method to instantiate a RelevanceDataset object
        Loads and creates the TFRecordDataset for train, validation and test splits

        Parameters
        ----------
        data_dir : str
            path to the directory containing train, validation and test data
        data_format : {"tfrecord", "csv", "libsvm"}
            type of data files to be converted into TFRecords and loaded as a TFRecordDataset
        feature_config : `FeatureConfig` object
            FeatureConfig object that defines the features to be loaded in the dataset
            and the preprocessing functions to be applied to each of them
        tfrecord_type : {"example", "sequence_example"}
            Type of the TFRecord protobuf message to be used for TFRecordDataset
        file_io : `FileIO` object
            file I/O handler objects for reading and writing data
        max_sequence_size : int, optional
            maximum number of sequence to be used with a single SequenceExample proto message
            The data will be appropriately padded or clipped to fit the max value specified
        batch_size : int, optional
            size of each data batch
        preprocessing_keys_to_fns : dict of (str, function), optional
            dictionary of function names mapped to function definitions
            that can now be used for preprocessing while loading the
            TFRecordDataset to create the RelevanceDataset object
        train_pcent_split : float, optional
            ratio of overall data to be used as training set
        val_pcent_split : float, optional
            ratio of overall data to be used as validation set
        test_pcent_split : float, optional
            ratio of overall data to be used as test set
        use_part_files : bool, optional
            load dataset from part files checked using "part-" prefix
        parse_tfrecord : bool, optional
            parse the TFRecord string from the dataset;
            returns strings as is otherwise
        logger : `Logger`, optional
            logging handler for status messages

        Notes
        -----
        * Currently supports CSV, TFRecord and Libsvm data formats
        * Does not support automatically splitting train, validation and test
        * `data_dir` should contain `train`, `validation` and `test` directories with files within them
        """
        self.feature_config = feature_config
        self.max_sequence_size = max_sequence_size
        self.logger = logger
        self.data_dir = data_dir
        self.data_format: str = data_format
        self.tfrecord_type = tfrecord_type
        self.batch_size: int = batch_size
        self.preprocessing_keys_to_fns = preprocessing_keys_to_fns
        self.file_io = file_io

        self.train_pcent_split: float = train_pcent_split
        self.val_pcent_split: float = val_pcent_split
        self.test_pcent_split: float = test_pcent_split
        self.use_part_files: bool = use_part_files

        self.keep_additional_info = keep_additional_info
        self.non_zero_features_only = non_zero_features_only

        self.train: Optional[tf.data.TFRecordDataset] = None
        self.validation: Optional[tf.data.TFRecordDataset] = None
        self.test: Optional[tf.data.TFRecordDataset] = None
        self.create_dataset(parse_tfrecord)

    def create_dataset(self, parse_tfrecord=True):
        """
        Loads and creates train, validation and test datasets

        Parameters
        ----------
        parse_tfrecord : bool
            parse the TFRecord string from the dataset;
            returns strings as is otherwise
        """
        to_split = len(glob.glob(os.path.join(self.data_dir, DataSplitKey.TEST))) == 0

        if self.data_format == DataFormatKey.CSV:
            data_reader = csv_reader
        elif self.data_format == DataFormatKey.TFRECORD:
            data_reader = tfrecord_reader
        elif self.data_format == DataFormatKey.RANKLIB:
            data_reader = ranklib_reader
        else:
            raise NotImplementedError(
                "Unsupported data format: {}. We currenty support {} and {}.".format(
                    self.data_format, DataFormatKey.CSV, DataFormatKey.TFRECORD
                )
            )

        if to_split:
            """
            If the data is stored as
            data_dir
            │
            ├── data_file
            ├── data_file
            ├── ...
            └── data_file
            """
            raise NotImplementedError

        else:
            """
            If the data is stored as
            data_dir
            │
            ├── train
            │   ├── data_file
            │   ├── data_file
            │   ├── ...
            │   └── data_file
            ├── validation
            │   ├── data_file
            │   ├── data_file
            │   ├── ...
            │   └── data_file
            └── test
                ├── data_file
                ├── data_file
                ├── ...
                └── data_file

            We also apply prefetch(tf.data.experimental.AUTOTUNE)
            as it improved train/test/validation throughput
            by 30% in some real model training.
            """
            self.train = data_reader.read(
                data_dir=os.path.join(self.data_dir, DataSplitKey.TRAIN),
                feature_config=self.feature_config,
                tfrecord_type=self.tfrecord_type,
                tfrecord_dir=os.path.join(self.data_dir, "tfrecord", DataSplitKey.TRAIN),
                max_sequence_size=self.max_sequence_size,
                batch_size=self.batch_size,
                preprocessing_keys_to_fns=self.preprocessing_keys_to_fns,
                use_part_files=self.use_part_files,
                parse_tfrecord=parse_tfrecord,
                file_io=self.file_io,
                logger=self.logger,
                keep_additional_info=self.keep_additional_info,
                non_zero_features_only=self.non_zero_features_only,
            )
            self.validation = data_reader.read(
                data_dir=os.path.join(self.data_dir, DataSplitKey.VALIDATION),
                feature_config=self.feature_config,
                tfrecord_type=self.tfrecord_type,
                tfrecord_dir=os.path.join(self.data_dir, "tfrecord", DataSplitKey.VALIDATION),
                max_sequence_size=self.max_sequence_size,
                batch_size=self.batch_size,
                preprocessing_keys_to_fns=self.preprocessing_keys_to_fns,
                use_part_files=self.use_part_files,
                parse_tfrecord=parse_tfrecord,
                file_io=self.file_io,
                logger=self.logger,
                keep_additional_info=self.keep_additional_info,
                non_zero_features_only=self.non_zero_features_only,
            )
            self.test = data_reader.read(
                data_dir=os.path.join(self.data_dir, DataSplitKey.TEST),
                feature_config=self.feature_config,
                tfrecord_type=self.tfrecord_type,
                tfrecord_dir=os.path.join(self.data_dir, "tfrecord", DataSplitKey.TEST),
                max_sequence_size=self.max_sequence_size,
                batch_size=self.batch_size,
                preprocessing_keys_to_fns=self.preprocessing_keys_to_fns,
                use_part_files=self.use_part_files,
                parse_tfrecord=parse_tfrecord,
                file_io=self.file_io,
                logger=self.logger,
                keep_additional_info=self.keep_additional_info,
                non_zero_features_only=self.non_zero_features_only,
            )

    def balance_classes(self):
        """
        Balance class labels in the train dataset
        """
        raise NotImplementedError

    def train_val_test_split(self):
        """
        Split the dataset into train, validation and test
        """
        raise NotImplementedError
