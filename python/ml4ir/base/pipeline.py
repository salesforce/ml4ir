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

from ml4ir.base.config.parse_args import get_args
from ml4ir.base.features.feature_config import parse_config, FeatureConfig
from ml4ir.base.io import logging_utils
from ml4ir.base.io import file_io
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.base.model.relevance_model import RelevanceModel
from ml4ir.base.config.keys import OptimizerKey
from ml4ir.base.config.keys import DataFormatKey
from ml4ir.base.config.keys import ExecutionModeKey
from ml4ir.base.config.keys import TFRecordTypeKey

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

        self.logs_dir: str = os.path.join(self.args.logs_dir, self.run_id)

        # Setup logging
        file_io.make_directory(self.logs_dir, clear_dir=True, log=None)
        self.logger: Logger = self.setup_logging()
        self.logger.info("Logging initialized. Saving logs to : {}".format(self.logs_dir))
        self.logger.info("Run ID: {}".format(self.run_id))
        self.logger.info("CLI args: \n{}".format(json.dumps(vars(self.args)).replace(",", "\n")))

        # Setup directories
        self.models_dir: str = os.path.join(self.args.models_dir, self.run_id)
        self.data_dir: str = self.args.data_dir
        file_io.make_directory(self.models_dir, clear_dir=False, log=self.logger)

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
        self.feature_config: FeatureConfig = parse_config(
            tfrecord_type=self.tfrecord_type,
            feature_config=self.args.feature_config,
            logger=self.logger,
        )
        self.logger.info("Feature config parsed and loaded")

        # Finished initialization
        self.logger.info("Relevance Pipeline successfully initialized!")

    def setup_logging(self) -> Logger:
        # Remove status file from any previous job at the start of the current job
        for status_file in ["_SUCCESS", "_FAILURE"]:
            if os.path.exists(os.path.join(self.logs_dir, status_file)):
                os.remove(os.path.join(self.logs_dir, status_file))

        outfile: str = os.path.join(self.logs_dir, "output_log.csv")

        return logging_utils.setup_logging(reset=True, file_name=outfile, log_to_file=True)

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

        return self

    def finish(self):
        # Delete temp directories
        if self.data_format == DataFormatKey.CSV:
            file_io.rm_dir(os.path.join(self.data_dir, "tfrecord"))

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
            data_dir=self.data_dir,
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
                    models_dir=self.models_dir,
                    logs_dir=self.logs_dir,
                    logging_frequency=self.args.logging_frequency,
                    monitor_metric=self.args.monitor_metric,
                    monitor_mode=self.args.monitor_mode,
                    patience=self.args.early_stopping_patience,
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
                    logs_dir=self.logs_dir,
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
                    logs_dir=self.logs_dir,
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
                    models_dir=self.models_dir,
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
        with open(os.path.join(self.logs_dir, job_status[0]), "w") as f:
            f.write(job_status[1])


def main(argv):
    # Define args
    args = get_args(argv)

    # Initialize Relevance Pipeline and run in train/inference mode
    rp = RelevancePipeline(args=args)
    rp.run()


if __name__ == "__main__":
    main(sys.argv[1:])
