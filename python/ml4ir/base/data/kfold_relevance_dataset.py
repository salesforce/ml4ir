from typing import Optional
from logging import Logger
import tensorflow as tf

from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.io.file_io import FileIO


class KfoldRelevanceDataset(RelevanceDataset):
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
            num_folds: int = 3,
            include_testset_in_kfold: bool = False,
            read_data_sets: bool = False
    ):
        self.include_testset_in_kfold = include_testset_in_kfold
        self.num_folds = num_folds

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
        if read_data_sets:
            self.create_dataset(parse_tfrecord)

    def merge_datasets(self):
        """
        Concat the datasets (training, validation, test) together

        Returns
        -------
        all_data: Tensorflow Dataset
            The final concatenated dataset.
        """

        if self.include_testset_in_kfold:
            all_data = self.train.concatenate(self.validation).concatenate(
                self.test)
        else:
            all_data = self.train.concatenate(self.validation)

        # un-batch and shuffle all queries
        all_data = all_data.unbatch()
        # shuffling before using the shard method gives unexpected results. Should be avoided
        # all_data = all_data.shuffle(batch_size * 2)
        return all_data

    def create_folds(self, fold_id, merged_data, relevance_dataset):
        """
        Create training, validation and test set according to the passed fold id.

        Parameters
        ----------
        fold_id : int
            current fold number
        merged_data: Tensorflow Dataset
            the dataset used to create folds
        relevance_dataset: RelevanceDataset object
            Used to access the test set to setup folds
        """
        test = None
        training_idx = list(range(self.num_folds))
        if self.include_testset_in_kfold:
            validation = merged_data.shard(self.num_folds, fold_id)
            test_idx = fold_id + 1
            if fold_id + 1 >= self.num_folds:
                test_idx = 0
            test = merged_data.shard(self.num_folds, test_idx)
            training_idx.remove(test_idx)
        else:
            validation = merged_data.shard(self.num_folds, fold_id)
        training_idx.remove(fold_id)
        train = None
        for f_id in training_idx:
            if not train:
                train = merged_data.shard(self.num_folds, f_id)
            else:
                train = train.concatenate(merged_data.shard(self.num_folds, f_id))

        # batchify training, validation and test sets.
        validation = validation.batch(self.batch_size, drop_remainder=False)
        train = train.batch(self.batch_size, drop_remainder=False)
        # We apply prefetch as it improved train/test/validation throughput by 30% in some real model training.
        self.train = train.prefetch(tf.data.experimental.AUTOTUNE)
        self.validation = validation.prefetch(tf.data.experimental.AUTOTUNE)
        if self.include_testset_in_kfold:
            test = test.batch(self.batch_size, drop_remainder=False)
            self.test = test.prefetch(tf.data.experimental.AUTOTUNE)
        else:
            self.test = relevance_dataset.test
