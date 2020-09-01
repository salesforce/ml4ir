import os
from logging import Logger
import tensorflow as tf
from tensorflow.keras import callbacks, Input, Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow import data
from tensorflow.keras import metrics as kmetrics
from wandb.keras import WandbCallback
import pandas as pd

from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.io.file_io import FileIO
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.base.model.losses.loss_base import RelevanceLossBase
from ml4ir.base.model.metrics.metrics_impl import get_metrics_impl
from ml4ir.base.model.scoring.scoring_model import ScorerBase, RelevanceScorer
from ml4ir.base.model.scoring.interaction_model import InteractionModel, UnivariateInteractionModel
from ml4ir.base.model.serving import define_serving_signatures
from ml4ir.base.model.scoring.prediction_helper import get_predict_fn

from typing import Dict, Optional, List, Union, Type


class RelevanceModelConstants:

    MODEL_PREDICTIONS_CSV_FILE = "model_predictions.csv"
    GROUP_METRICS_CSV_FILE = "group_metrics.csv"
    CHECKPOINT_FNAME = "checkpoint.tf"


class RelevanceModel:
    def __init__(
        self,
        feature_config: FeatureConfig,
        tfrecord_type: str,
        file_io: FileIO,
        scorer: Optional[ScorerBase] = None,
        metrics: List[Union[Type[kmetrics.Metric], str]] = [],
        optimizer: Optional[Optimizer] = None,
        model_file: Optional[str] = None,
        compile_keras_model: bool = False,
        output_name: str = "score",
        logger=None,
    ):
        """Use this constructor to define a custom scorer"""
        self.feature_config: FeatureConfig = feature_config
        self.logger: Logger = logger
        self.output_name = output_name
        self.scorer = scorer
        self.tfrecord_type = tfrecord_type
        self.file_io = file_io

        if scorer:
            self.max_sequence_size = scorer.interaction_model.max_sequence_size
        else:
            self.max_sequence_size = 0

        # Load/Build Model
        if model_file and not compile_keras_model:
            """
            If a model file is specified, load it without compiling into a keras model

            NOTE:
            This will allow the model to be only used for inference and
            cannot be used for retraining.
            """
            self.model: Model = self.load(model_file)
            self.is_compiled = False
        else:

            """
            Specify inputs to the model

            Individual input nodes are defined for each feature
            Each data point represents features for all records in a single query
            """
            inputs: Dict[str, Input] = feature_config.define_inputs()
            scores, train_features, metadata_features = scorer(inputs)

            # Create model with functional Keras API
            self.model = Model(inputs=inputs, outputs={self.output_name: scores})

            # Get loss fn
            loss_fn = scorer.loss.get_loss_fn(**metadata_features)

            # Get metric objects
            metrics_impl: List[Union[str, kmetrics.Metric]] = get_metrics_impl(
                metrics=metrics, feature_config=feature_config, metadata_features=metadata_features
            )

            # Compile model
            """
            NOTE:
            Related Github issue: https://github.com/tensorflow/probability/issues/519
            """
            self.model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=metrics_impl,
                experimental_run_tf_function=False,
            )

            # Write model summary to logs
            model_summary = list()
            self.model.summary(print_fn=lambda x: model_summary.append(x))
            self.logger.info("\n".join(model_summary))

            if model_file:
                """
                If model file is specified, load the weights from the SavedModel

                NOTE:
                The architecture, loss and metrics of self.model need to
                be the same as the loaded SavedModel
                """
                self.load_weights(model_file)

            self.is_compiled = True

    @classmethod
    def from_relevance_scorer(
        cls,
        interaction_model: InteractionModel,
        model_config: dict,
        feature_config: FeatureConfig,
        loss: RelevanceLossBase,
        metrics: List[Union[kmetrics.Metric, str]],
        optimizer: Optimizer,
        tfrecord_type: str,
        file_io: FileIO,
        model_file: Optional[str] = None,
        compile_keras_model: bool = False,
        output_name: str = "score",
        logger=None,
    ):
        """Use this as constructor to define a custom InteractionModel with RelevanceScorer"""
        assert isinstance(interaction_model, InteractionModel)
        assert isinstance(loss, RelevanceLossBase)

        scorer: ScorerBase = RelevanceScorer(
            model_config=model_config,
            interaction_model=interaction_model,
            loss=loss,
            output_name=output_name,
        )

        return cls(
            scorer=scorer,
            feature_config=feature_config,
            metrics=metrics,
            optimizer=optimizer,
            tfrecord_type=tfrecord_type,
            model_file=model_file,
            compile_keras_model=compile_keras_model,
            output_name=output_name,
            file_io=file_io,
            logger=logger,
        )

    @classmethod
    def from_univariate_interaction_model(
        cls,
        model_config,
        feature_config: FeatureConfig,
        tfrecord_type: str,
        loss: RelevanceLossBase,
        metrics: List[Union[kmetrics.Metric, str]],
        optimizer: Optimizer,
        feature_layer_keys_to_fns: dict = {},
        model_file: Optional[str] = None,
        compile_keras_model: bool = False,
        output_name: str = "score",
        max_sequence_size: int = 0,
        file_io: FileIO = None,
        logger=None,
    ):
        """Use this as constructor to use UnivariateInteractionModel and RelevanceScorer"""

        interaction_model: InteractionModel = UnivariateInteractionModel(
            feature_config=feature_config,
            feature_layer_keys_to_fns=feature_layer_keys_to_fns,
            tfrecord_type=tfrecord_type,
            max_sequence_size=max_sequence_size,
        )

        return cls.from_relevance_scorer(
            interaction_model=interaction_model,
            model_config=model_config,
            feature_config=feature_config,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            tfrecord_type=tfrecord_type,
            model_file=model_file,
            compile_keras_model=compile_keras_model,
            output_name=output_name,
            file_io=file_io,
            logger=logger,
        )

    def fit(
        self,
        dataset: RelevanceDataset,
        num_epochs: int,
        models_dir: str,
        logs_dir: Optional[str] = None,
        logging_frequency: int = 25,
        monitor_metric: str = "",
        monitor_mode: str = "",
        patience=2,
        track_experiment=False,
    ):
        """
        Trains model for defined number of epochs

        Args:
            dataset: an instance of RankingDataset
            num_epochs: int value specifying number of epochs to train for
            models_dir: directory to save model checkpoints
            logs_dir: directory to save model logs
            logging_frequency: every #batches to log results
            monitor_metric: name of the metric to monitor for early stopping, checkpointing
            monitor_mode: whether to maximize or minimize the monitoring metric
            patience: early stopping patience
            track_experiment: save results of model training and validation
        """
        if not monitor_metric.startswith("val_"):
            monitor_metric = "val_{}".format(monitor_metric)
        callbacks_list: list = self._build_callback_hooks(
            models_dir=models_dir,
            logs_dir=logs_dir,
            is_training=True,
            logging_frequency=logging_frequency,
            monitor_mode=monitor_mode,
            monitor_metric=monitor_metric,
            patience=patience,
            track_experiment=track_experiment,
        )
        if self.is_compiled:
            self.model.fit(
                x=dataset.train,
                validation_data=dataset.validation,
                epochs=num_epochs,
                verbose=True,
                callbacks=callbacks_list,
            )
        else:
            raise NotImplementedError(
                "The model could not be trained. Check if the model was compiled correctly. Training loaded SavedModel is not currently supported."
            )

    def predict(
        self,
        test_dataset: data.TFRecordDataset,
        inference_signature: str = "serving_default",
        additional_features: dict = {},
        logs_dir: Optional[str] = None,
        logging_frequency: int = 25,
    ):
        """
        Predict the labels for the trained model

        Args:
            test_dataset: an instance of tf.data.dataset
            inference_signature: If using a SavedModel for prediction, specify the inference signature
            logging_frequency: integer representing how often(in batches) to log status

        Returns:
            ranking scores or new ranks for each record in a query
        """
        if logs_dir:
            outfile = os.path.join(logs_dir, RelevanceModelConstants.MODEL_PREDICTIONS_CSV_FILE)
            # Delete file if it exists
            self.file_io.rm_file(outfile)

        _predict_fn = get_predict_fn(
            model=self.model,
            tfrecord_type=self.tfrecord_type,
            feature_config=self.feature_config,
            inference_signature=inference_signature,
            is_compiled=self.is_compiled,
            output_name=self.output_name,
            features_to_return=self.feature_config.get_features_to_log(),
            additional_features=additional_features,
            max_sequence_size=self.max_sequence_size,
        )

        predictions_df_list = list()
        batch_count = 0
        for predictions_dict in test_dataset.map(_predict_fn).take(-1):
            predictions_df = pd.DataFrame(predictions_dict)
            if logs_dir:
                if os.path.isfile(outfile):
                    predictions_df.to_csv(outfile, mode="a", header=False, index=False)
                else:
                    # If writing first time, write headers to CSV file
                    predictions_df.to_csv(outfile, mode="w", header=True, index=False)
            else:
                predictions_df_list.append(predictions_df)

            batch_count += 1
            if batch_count % logging_frequency == 0:
                self.logger.info("Finished predicting scores for {} batches".format(batch_count))

        predictions_df = None
        if logs_dir:
            self.logger.info("Model predictions written to -> {}".format(outfile))
        else:
            predictions_df = pd.concat(predictions_df_list)

        return predictions_df

    def evaluate(
        self,
        test_dataset: data.TFRecordDataset,
        inference_signature: str = None,
        additional_features: dict = {},
        group_metrics_min_queries: int = 50,
        logs_dir: Optional[str] = None,
        logging_frequency: int = 25,
    ):
        """
        Evaluate the ranking model

        Args:
            test_dataset: an instance of tf.data.dataset
            inference_signature: If using a SavedModel for prediction, specify the inference signature
            logging_frequency: integer representing how often(in batches) to log status
            metric_group_keys: list of fields to compute group based metrics on
            save_to_file: set to True to save predictions to file like self.predict()

        Returns:
            metrics and groupwise metrics as pandas DataFrames

        NOTE:
        Only if the keras model is compiled, you can directly do a model.evaluate()

        Override this method to implement your own evaluation metrics.
        """
        if self.is_compiled:
            return self.model.evaluate(test_dataset)
        else:
            raise NotImplementedError

    def save(
        self,
        models_dir: str,
        preprocessing_keys_to_fns={},
        postprocessing_fn=None,
        required_fields_only: bool = True,
        pad_sequence: bool = False,
    ):
        """
        Save tf.keras model to models_dir

        Args:
            models_dir: path to directory to save the model
        """

        model_file = os.path.join(models_dir, "final")

        # Save model with default signature
        self.model.save(filepath=os.path.join(model_file, "default"))

        """
        Save model with custom signatures

        Currently supported
        - signature to read TFRecord SequenceExample inputs
        """
        self.model.save(
            filepath=os.path.join(model_file, "tfrecord"),
            signatures=define_serving_signatures(
                model=self.model,
                tfrecord_type=self.tfrecord_type,
                feature_config=self.feature_config,
                preprocessing_keys_to_fns=preprocessing_keys_to_fns,
                postprocessing_fn=postprocessing_fn,
                required_fields_only=required_fields_only,
                pad_sequence=pad_sequence,
                max_sequence_size=self.max_sequence_size,
            ),
        )
        self.logger.info("Final model saved to : {}".format(model_file))

    def load(self, model_file: str) -> Model:
        """
        Loads model from the SavedModel file specified

        Args:
            model_file: path to file with saved tf keras model

        Returns:
            loaded tf keras model
        """

        """
        NOTE:
        There is currently a bug in Keras Model with saving/loading
        models with custom losses and metrics.

        Therefore, we are currently loading the SavedModel with compile=False
        The saved model signatures can be used for inference at serving time

        NOTE: Retraining currently not supported!
        Would require compiling the model with the right loss and optimizer states

        Ref:
        https://github.com/keras-team/keras/issues/5916
        https://github.com/tensorflow/tensorflow/issues/32348
        https://github.com/keras-team/keras/issues/3977

        """
        model = tf.keras.models.load_model(model_file, compile=False)

        self.logger.info("Successfully loaded SavedModel from {}".format(model_file))
        self.logger.warning("Retraining is not yet supported. Model is loaded with compile=False")

        return model

    def load_weights(self, model_file: str):
        # Load saved model with compile=False
        loaded_model = self.load(model_file)

        # Set weights of Keras model from the loaded model weights
        self.model.set_weights(loaded_model.get_weights())
        self.logger.info("Weights have been set from SavedModel. RankingModel can now be trained.")

    def _build_callback_hooks(
        self,
        models_dir: str,
        logs_dir: Optional[str] = None,
        is_training=True,
        logging_frequency=25,
        monitor_metric: str = "",
        monitor_mode: str = "",
        patience=2,
        track_experiment=False,
    ):
        """
        Build callback hooks for the training loop

        Returns:
            callbacks_list: list of callbacks
        """
        callbacks_list: list = list()

        if is_training:
            # Model checkpoint
            if models_dir and monitor_metric:
                checkpoints_path = os.path.join(
                    models_dir, RelevanceModelConstants.CHECKPOINT_FNAME
                )
                cp_callback = callbacks.ModelCheckpoint(
                    filepath=checkpoints_path,
                    save_weights_only=False,
                    verbose=1,
                    save_best_only=True,
                    mode=monitor_mode,
                    monitor=monitor_metric,
                )
                callbacks_list.append(cp_callback)

            # Early Stopping
            if monitor_metric:
                early_stopping_callback = callbacks.EarlyStopping(
                    monitor=monitor_metric,
                    mode=monitor_mode,
                    patience=patience,
                    verbose=1,
                    restore_best_weights=True,
                )
                callbacks_list.append(early_stopping_callback)

        # TensorBoard
        if logs_dir:
            tensorboard_callback = callbacks.TensorBoard(
                log_dir=logs_dir, histogram_freq=1, update_freq=5
            )
            callbacks_list.append(tensorboard_callback)

        # Debugging/Logging
        logger = self.logger

        class DebuggingCallback(callbacks.Callback):
            def __init__(self, patience=0):
                super(DebuggingCallback, self).__init__()

                self.epoch = 0

            def on_train_batch_end(self, batch, logs=None):
                if batch % logging_frequency == 0:
                    logger.info("[epoch: {} | batch: {}] {}".format(self.epoch, batch, logs))

            def on_epoch_end(self, epoch, logs=None):
                logger.info("End of Epoch {}".format(self.epoch))
                logger.info(logs)

            def on_epoch_begin(self, epoch, logs=None):
                self.epoch = epoch + 1
                logger.info("Starting Epoch : {}".format(self.epoch))
                logger.info(logs)

            def on_train_begin(self, logs):
                logger.info("Training Model")

            def on_test_begin(self, logs):
                logger.info("Evaluating Model")

            def on_predict_begin(self, logs):
                logger.info("Predicting scores using model")

            def on_train_end(self, logs):
                logger.info("Completed training model")
                logger.info(logs)

            def on_test_end(self, logs):
                logger.info("Completed evaluating model")
                logger.info(logs)

            def on_predict_end(self, logs):
                logger.info("Completed Predicting scores using model")
                logger.info(logs)

        callbacks_list.append(DebuggingCallback())

        # Add Weights and Biases experiment tracking callback
        if track_experiment:
            callbacks_list.append(WandbCallback())

        # Add more here

        return callbacks_list
