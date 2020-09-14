import socket
import ast
import tensorflow as tf
import numpy as np
import json
import random
import traceback
import os
import sys
import time
from argparse import Namespace
from logging import Logger
import wandb

from ml4ir.base.config.parse_args import get_args
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.io import logging_utils
from ml4ir.base.io.local_io import LocalIO
from ml4ir.base.io.spark_io import SparkIO
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.base.model.relevance_model import RelevanceModel
from ml4ir.base.config.keys import OptimizerKey
from ml4ir.base.config.keys import DataFormatKey
from ml4ir.base.config.keys import ExecutionModeKey
from ml4ir.base.config.keys import TFRecordTypeKey
from ml4ir.base.config.keys import DefaultDirectoryKey
from ml4ir.base.config.keys import FileHandlerKey

from typing import List


class RelevancePipeline(object):
    def __init__(self, args: Namespace):
        self.args = args

        # Generate Run ID
        if len(self.args.run_id) > 0:
            self.run_id: str = self.args.run_id
        else:
            self.run_id = "-".join([socket.gethostname(), time.strftime("%Y%m%d-%H%M%S")])
        self.start_time = time.time()

        # Setup directories
        self.local_io = LocalIO()
        self.models_dir_hdfs = None
        self.logs_dir_hdfs = None
        self.data_dir_hdfs = None
        if self.args.file_handler == FileHandlerKey.SPARK:
            self.models_dir = os.path.join(self.args.models_dir, self.run_id)
            self.logs_dir = os.path.join(self.args.logs_dir, self.run_id)
            self.data_dir = self.args.data_dir

            self.models_dir_local = os.path.join(DefaultDirectoryKey.MODELS, self.run_id)
            self.logs_dir_local = os.path.join(DefaultDirectoryKey.LOGS, self.run_id)
            self.data_dir_local = os.path.join(
                DefaultDirectoryKey.TEMP_DATA, os.path.basename(self.data_dir)
            )
        else:
            self.models_dir_local = os.path.join(self.args.models_dir, self.run_id)
            self.logs_dir_local = os.path.join(self.args.logs_dir, self.run_id)
            self.data_dir_local = self.args.data_dir

        # Setup logging
        self.local_io.make_directory(self.logs_dir_local, clear_dir=True)
        self.logger: Logger = self.setup_logging()
        self.logger.info("Logging initialized. Saving logs to : {}".format(self.logs_dir_local))
        self.logger.info("Run ID: {}".format(self.run_id))
        self.logger.debug("CLI args: \n{}".format(json.dumps(vars(self.args), indent=4)))
        self.local_io.set_logger(self.logger)
        self.local_io.make_directory(self.models_dir_local, clear_dir=False)
        self.model_file = self.args.model_file

        # Set the file handlers and respective setup
        if self.args.file_handler == FileHandlerKey.LOCAL:
            self.file_io = self.local_io
        elif self.args.file_handler == FileHandlerKey.SPARK:
            self.file_io = SparkIO(self.logger)

            # Copy data dir from HDFS to local file system
            self.local_io.make_directory(dir_path=DefaultDirectoryKey.TEMP_DATA, clear_dir=True)
            self.file_io.copy_from_hdfs(self.data_dir, DefaultDirectoryKey.TEMP_DATA)

            # Copy model_file if present from HDFS to local file system
            if self.model_file:
                self.local_io.make_directory(
                    dir_path=DefaultDirectoryKey.TEMP_MODELS, clear_dir=True
                )
                self.file_io.copy_from_hdfs(self.model_file, DefaultDirectoryKey.TEMP_MODELS)
                self.model_file = os.path.join(
                    DefaultDirectoryKey.TEMP_MODELS, os.path.basename(self.model_file)
                )

        # Read/Parse model config YAML
        self.model_config_file = self.args.model_config

        # Setup other arguments
        self.loss_key: str = self.args.loss_key
        self.optimizer_key: str = self.args.optimizer_key
        if self.args.metrics_keys[0] == "[":
            self.metrics_keys: List[str] = ast.literal_eval(self.args.metrics_keys)
        else:
            self.metrics_keys = [self.args.metrics_keys]
        self.data_format: str = self.args.data_format
        self.tfrecord_type: str = self.args.tfrecord_type

        # Validate args
        self.validate_args()

        # Set random seeds
        self.set_seeds()

        # Load and parse feature config
        self.feature_config: FeatureConfig = FeatureConfig.get_instance(
            feature_config_dict=self.file_io.read_yaml(self.args.feature_config),
            tfrecord_type=self.tfrecord_type,
            logger=self.logger,
        )

        # Setup experiment tracking configuration
        self.setup_experiment_tracking_config()

        # Finished initialization
        self.logger.info("Relevance Pipeline successfully initialized!")

    def setup_logging(self) -> Logger:
        # Remove status file from any previous job at the start of the current job
        for status_file in ["_SUCCESS", "_FAILURE"]:
            self.local_io.rm_file(os.path.join(self.logs_dir_local, status_file))

        return logging_utils.setup_logging(
            reset=True,
            file_name=os.path.join(self.logs_dir_local, "output_log.csv"),
            log_to_file=True,
        )

    def set_seeds(self, reset_graph=True):
        # for repeatability
        if reset_graph:
            tf.keras.backend.clear_session()
            self.logger.info("Tensorflow default graph has been reset")
        np.random.seed(self.args.random_state)
        tf.random.set_seed(self.args.random_state)
        random.seed(self.args.random_state)

    def validate_args(self):
        unset_arguments = {key: value for (key, value) in vars(self.args).items() if value is None}

        if len(unset_arguments) > 0:
            raise Exception(
                "Unset arguments (check usage): \n{}".format(
                    json.dumps(unset_arguments).replace(",", "\n")
                )
            )

        if self.optimizer_key not in OptimizerKey.get_all_keys():
            raise Exception(
                "Optimizer specified [{}] is not one of : {}".format(
                    self.optimizer_key, OptimizerKey.get_all_keys()
                )
            )

        if self.data_format not in DataFormatKey.get_all_keys():
            raise Exception(
                "Data format[{}] is not one of : {}".format(
                    self.data_format, DataFormatKey.get_all_keys()
                )
            )

        if self.tfrecord_type not in TFRecordTypeKey.get_all_keys():
            raise Exception(
                "TFRecord type [{}] is not one of : {}".format(
                    self.data_format, TFRecordTypeKey.get_all_keys()
                )
            )

        if self.args.file_handler not in FileHandlerKey.get_all_keys():
            raise Exception(
                "FileHandler [{}] is not one of : {}".format(
                    self.args.file_handler, FileHandlerKey.get_all_keys()
                )
            )

        return self

    def setup_experiment_tracking_config(self):
        if self.args.track_experiment:
            config = dict()

            # Add command line script arguments
            config.update(vars(self.args))

            # Add feature config information
            config.update(self.feature_config.get_hyperparameter_dict())

            """
            Set the following environment variables to run wandb in offline mode without server
            Ref: https://docs.wandb.com/library/init#save-logs-offline
            """
            os.environ["WANDB_MODE"] = "dryrun"

            # Setup wandb
            self.local_io.make_directory(os.path.join(self.logs_dir_local, "wandb"))
            wandb.init(
                project="ml4ir",
                name=self.run_id,
                notes=self.args.run_notes,
                group=self.args.run_group,
                config=config,
                dir=os.path.join(self.logs_dir_local, "wandb"),
            )

            self.logger.info("Setup weights and biases config")

    def finish(self):
        # Delete temp data directories
        if self.data_format == DataFormatKey.CSV:
            self.local_io.rm_dir(os.path.join(self.data_dir_local, "tfrecord"))
        self.local_io.rm_dir(DefaultDirectoryKey.TEMP_DATA)
        self.local_io.rm_dir(DefaultDirectoryKey.TEMP_MODELS)

        if self.args.file_handler == FileHandlerKey.SPARK:
            # Copy logs and models to HDFS
            self.file_io.copy_to_hdfs(self.models_dir_local, self.models_dir, overwrite=True)
            self.file_io.copy_to_hdfs(self.logs_dir_local, self.logs_dir, overwrite=True)

        e = int(time.time() - self.start_time)
        self.logger.info(
            "Done! Elapsed time: {:02d}:{:02d}:{:02d}".format(e // 3600, (e % 3600 // 60), e % 60)
        )

        return self

    def get_relevance_dataset(self, preprocessing_keys_to_fns={}) -> RelevanceDataset:
        """
        Creates RelevanceDataset

        NOTE: Override this method to create custom dataset objects
        """
        # Prepare Dataset
        relevance_dataset = RelevanceDataset(
            data_dir=self.data_dir_local,
            data_format=self.data_format,
            feature_config=self.feature_config,
            tfrecord_type=self.tfrecord_type,
            max_sequence_size=self.args.max_sequence_size,
            batch_size=self.args.batch_size,
            preprocessing_keys_to_fns=preprocessing_keys_to_fns,
            train_pcent_split=self.args.train_pcent_split,
            val_pcent_split=self.args.val_pcent_split,
            test_pcent_split=self.args.test_pcent_split,
            use_part_files=self.args.use_part_files,
            parse_tfrecord=True,
            file_io=self.local_io,
            logger=self.logger,
        )

        return relevance_dataset

    def get_relevance_model(self, feature_layer_keys_to_fns={}) -> RelevanceModel:
        """
        Creates RelevanceModel

        NOTE: Override this method to create custom loss, scorer, model objects
        """
        raise NotImplementedError

    def run(self):
        try:
            job_status = ("_SUCCESS", "")

            # Build dataset
            relevance_dataset = self.get_relevance_dataset()
            self.logger.info("Relevance Dataset created")

            # Build model
            relevance_model = self.get_relevance_model()
            self.logger.info("Relevance Model created")

            if self.args.execution_mode in {
                ExecutionModeKey.TRAIN_INFERENCE_EVALUATE,
                ExecutionModeKey.TRAIN_EVALUATE,
                ExecutionModeKey.TRAIN_INFERENCE,
                ExecutionModeKey.TRAIN_ONLY,
            }:
                # Train
                relevance_model.fit(
                    dataset=relevance_dataset,
                    num_epochs=self.args.num_epochs,
                    models_dir=self.models_dir_local,
                    logs_dir=self.logs_dir_local,
                    logging_frequency=self.args.logging_frequency,
                    monitor_metric=self.args.monitor_metric,
                    monitor_mode=self.args.monitor_mode,
                    patience=self.args.early_stopping_patience,
                    track_experiment=self.args.track_experiment,
                )

            if self.args.execution_mode in {
                ExecutionModeKey.TRAIN_INFERENCE_EVALUATE,
                ExecutionModeKey.TRAIN_EVALUATE,
                ExecutionModeKey.EVALUATE_ONLY,
                ExecutionModeKey.INFERENCE_EVALUATE,
                ExecutionModeKey.INFERENCE_EVALUATE_RESAVE,
                ExecutionModeKey.EVALUATE_RESAVE,
            }:
                # Evaluate
                relevance_model.evaluate(
                    test_dataset=relevance_dataset.test,
                    inference_signature=self.args.inference_signature,
                    logging_frequency=self.args.logging_frequency,
                    group_metrics_min_queries=self.args.group_metrics_min_queries,
                    logs_dir=self.logs_dir_local,
                    track_experiment=self.args.track_experiment,
                )

            if self.args.execution_mode in {
                ExecutionModeKey.TRAIN_INFERENCE_EVALUATE,
                ExecutionModeKey.TRAIN_INFERENCE,
                ExecutionModeKey.INFERENCE_EVALUATE,
                ExecutionModeKey.INFERENCE_ONLY,
                ExecutionModeKey.INFERENCE_EVALUATE_RESAVE,
                ExecutionModeKey.INFERENCE_RESAVE,
            }:
                # Predict relevance scores
                relevance_model.predict(
                    test_dataset=relevance_dataset.test,
                    inference_signature=self.args.inference_signature,
                    additional_features={},
                    logs_dir=self.logs_dir_local,
                    logging_frequency=self.args.logging_frequency,
                )

            # Save model
            # NOTE: Model will be saved with the latest serving signatures
            if self.args.execution_mode in {
                ExecutionModeKey.TRAIN_INFERENCE_EVALUATE,
                ExecutionModeKey.TRAIN_EVALUATE,
                ExecutionModeKey.TRAIN_INFERENCE,
                ExecutionModeKey.TRAIN_ONLY,
                ExecutionModeKey.INFERENCE_EVALUATE_RESAVE,
                ExecutionModeKey.EVALUATE_RESAVE,
                ExecutionModeKey.INFERENCE_RESAVE,
                ExecutionModeKey.RESAVE_ONLY,
            }:
                # Save model
                relevance_model.save(
                    models_dir=self.models_dir_local,
                    preprocessing_keys_to_fns={},
                    postprocessing_fn=None,
                    required_fields_only=not self.args.use_all_fields_at_inference,
                    pad_sequence=self.args.pad_sequence_at_inference,
                )

            # Finish
            self.finish()

        except Exception as e:
            self.logger.error("!!! Error Training Model: !!!\n{}".format(str(e)))
            traceback.print_exc()
            job_status = ("_FAILURE", "{}\n{}".format(str(e), traceback.format_exc()))

        # Write job status to file
        with open(os.path.join(self.logs_dir_local, job_status[0]), "w") as f:
            f.write(job_status[1])


def main(argv):
    # Define args
    args = get_args(argv)

    # Initialize Relevance Pipeline and run in train/inference mode
    rp = RelevancePipeline(args=args)
    rp.run()


if __name__ == "__main__":
    main(sys.argv[1:])
