# type: ignore
# TODO: Fix typing

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
import yaml
from argparse import Namespace
from logging import Logger
from ml4ir.config.parse_args import get_args
from ml4ir.features.feature_config import parse_config, FeatureConfig
from ml4ir.io import logging_utils
from ml4ir.io import file_io
from ml4ir.data.ranking_dataset import RankingDataset
from ml4ir.model.ranking_model import RankingModel

from ml4ir.config.keys import LossKey
from ml4ir.config.keys import ScoringKey
from ml4ir.config.keys import MetricKey
from ml4ir.config.keys import OptimizerKey
from ml4ir.config.keys import DataFormatKey
from ml4ir.config.keys import ExecutionModeKey

from typing import List


class RankingPipeline(object):
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
        self.model_config = self._read_model_config(self.args.model_config)

        # Setup other arguments
        self.loss: str = self.args.loss
        self.scoring: str = self.args.scoring
        self.optimizer: str = self.args.optimizer
        if self.args.metrics[0] == "[":
            self.metrics: List[str] = ast.literal_eval(self.args.metrics)
        else:
            self.metrics = [self.args.metrics]
        self.data_format: str = self.args.data_format

        # Validate args
        self.validate_args()

        # Set random seeds
        self.set_seeds()

        # Load and parse feature config
        self.feature_config: FeatureConfig = parse_config(
            self.args.feature_config, logger=self.logger
        )
        self.logger.info("Feature config parsed and loaded")

        # Finished initialization
        self.logger.info("Ranking Pipeline successfully initialized!")

    def _read_model_config(self, model_config_str):
        if model_config_str.endswith(".yaml"):
            model_config = file_io.read_yaml(model_config_str)
            self.logger.info(
                "Reading model config from YAML file : {} \n{}".format(
                    model_config, model_config_str
                )
            )
        else:
            model_config = yaml.safe_load(model_config_str)
            self.logger.info("Reading model config from YAML string : \n{}".format(model_config))
        return model_config

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

        if self.loss not in LossKey.get_all_keys():
            raise Exception(
                "Loss specified [{}] is not one of : {}".format(self.loss, LossKey.get_all_keys())
            )
        if self.scoring not in ScoringKey.get_all_keys():
            raise Exception(
                "Scoring method specified [{}] is not one of : {}".format(
                    self.scoring, ScoringKey.get_all_keys()
                )
            )
        if self.optimizer not in OptimizerKey.get_all_keys():
            raise Exception(
                "Optimizer specified [{}] is not one of : {}".format(
                    self.optimizer, OptimizerKey.get_all_keys()
                )
            )
        for metric in self.metrics:
            if metric not in MetricKey.get_all_keys():
                raise Exception(
                    "Metric specified [{}] is not one of : {}".format(
                        metric, MetricKey.get_all_keys()
                    )
                )
        if self.data_format not in DataFormatKey.get_all_keys():
            raise Exception(
                "Data format[{}] is not one of : {}".format(
                    self.data_format, DataFormatKey.get_all_keys()
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

    def run(self):
        try:
            job_status = ("_SUCCESS", "")

            # Prepare Dataset
            ranking_dataset = RankingDataset(
                data_dir=self.data_dir,
                data_format=self.data_format,
                feature_config=self.feature_config,
                max_num_records=self.args.max_num_records,
                loss_key=self.loss,
                scoring_key=self.scoring,
                batch_size=self.args.batch_size,
                train_pcent_split=self.args.train_pcent_split,
                val_pcent_split=self.args.val_pcent_split,
                test_pcent_split=self.args.test_pcent_split,
                use_part_files=self.args.use_part_files,
                logger=self.logger,
            )
            self.logger.info("Ranking Dataset created")

            # Build model
            ranking_model = RankingModel(
                model_config=self.model_config,
                loss_key=self.loss,
                scoring_key=self.scoring,
                metrics_keys=self.metrics,
                optimizer_key=self.optimizer,
                feature_config=self.feature_config,
                max_num_records=self.args.max_num_records,
                model_file=self.args.model_file,
                learning_rate=self.args.learning_rate,
                learning_rate_decay=self.args.learning_rate_decay,
                learning_rate_decay_steps=self.args.learning_rate_decay_steps,
                gradient_clip_value=self.args.gradient_clip_value,
                compute_intermediate_stats=self.args.compute_intermediate_stats,
                compile_keras_model=self.args.compile_keras_model,
                logger=self.logger,
            )
            self.logger.info("Ranking Model created")

            if self.args.execution_mode in {
                ExecutionModeKey.TRAIN_INFERENCE_EVALUATE,
                ExecutionModeKey.TRAIN_EVALUATE,
                ExecutionModeKey.TRAIN_INFERENCE,
                ExecutionModeKey.TRAIN_ONLY,
            }:
                # Train
                ranking_model.fit(
                    dataset=ranking_dataset,
                    num_epochs=self.args.num_epochs,
                    models_dir=self.models_dir,
                    logs_dir=self.logs_dir,
                    logging_frequency=self.args.logging_frequency,
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
                ranking_model.evaluate(
                    test_dataset=ranking_dataset.test,
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
                # Predict ranking scores
                ranking_model.predict(
                    test_dataset=ranking_dataset.test,
                    inference_signature=self.args.inference_signature,
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
                ranking_model.save(
                    models_dir=self.models_dir, pad_records=self.args.pad_records_at_inference
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

    # Initialize Ranker and run in train/inference mode
    rp = RankingPipeline(args=args)
    rp.run()


if __name__ == "__main__":
    main(sys.argv[1:])
