import os
from logging import Logger
import tensorflow as tf
from tensorflow.keras import callbacks, Input, Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow import saved_model
from tensorflow import data
from tensorflow.keras import metrics as kmetrics
from tensorflow.keras import backend as K

from ml4ir.features.feature_config import FeatureConfig
from ml4ir.features.feature_layer import define_feature_layer
from ml4ir.config.keys import FeatureTypeKey, ScoringKey
from ml4ir.config.keys import LossTypeKey
from ml4ir.model.optimizer import get_optimizer
from ml4ir.model.losses.loss_base import RankingLossBase
from ml4ir.model.losses import loss_factory
from ml4ir.model.metrics import metric_factory, metrics_helper
from ml4ir.model.scoring import scoring_factory
from ml4ir.model.serving import define_serving_signatures
from ml4ir.io import file_io
import pandas as pd
import numpy as np

from typing import Dict, Optional, List, Type


# Constants
MODEL_PREDICTIONS_CSV_FILE = "model_predictions.csv"
GROUP_METRICS_CSV_FILE = "group_metrics.csv"
CHECKPOINT_FNAME = "checkpoint.tf"
NEW_RANK_FIELD = "new_rank"
NEW_SCORE_FIELD = "new_score"


class RankingModel:
    def __init__(
        self,
        model_config: dict,
        loss_key: str,
        scoring_key: str,
        metrics_keys: List[str],
        optimizer_key: str,
        feature_config: FeatureConfig,
        max_num_records: int,
        model_file: str,
        learning_rate: float,
        learning_rate_decay: float,
        learning_rate_decay_steps: int,
        gradient_clip_value: float,
        compute_intermediate_stats: bool,
        compile_keras_model: bool,
        logger=None,
    ):
        self.model_config: dict = model_config
        self.feature_config: FeatureConfig = feature_config
        self.scoring_key: str = scoring_key
        self.logger: Logger = logger
        self.max_num_records = max_num_records

        # Load/Build Model
        if model_file and not compile_keras_model:
            """
            If a model file is specified, load it without compiling into a keras model

            NOTE:
            This will allow the model to be only used for inference and
            cannot be used for retraining.
            """
            self.model: Model = self.load_model(model_file)
            self.is_compiled = False
        else:
            # Define optimizer
            optimizer: Optimizer = get_optimizer(
                optimizer_key=optimizer_key,
                learning_rate=learning_rate,
                learning_rate_decay=learning_rate_decay,
                learning_rate_decay_steps=learning_rate_decay_steps,
                gradient_clip_value=gradient_clip_value,
            )

            # Define loss function
            loss: RankingLossBase = loss_factory.get_loss(
                loss_key=loss_key, scoring_key=scoring_key
            )

            # Define metrics
            metrics: List[Type[kmetrics.Metric]] = [
                metric_factory.get_metric(metric_key=key) for key in metrics_keys
            ]

            """
            Specify inputs to the model

            Individual input nodes are defined for each feature
            Each data point represents features for all records in a single query
            """
            inputs: Dict[str, Input] = feature_config.define_inputs()
            self.model = self.build_model(inputs, optimizer, loss, metrics)

            if model_file:
                """
                If model file is specified, load the weights from the SavedModel

                NOTE:
                The architecture, loss and metrics of self.model need to
                be the same as the loaded SavedModel
                """
                self.load_weights(model_file)

            self.is_compiled = True

    def build_model(
        self,
        inputs: Dict[str, Input],
        optimizer: Optimizer,
        loss: RankingLossBase,
        metrics: List[Type[kmetrics.Metric]],
    ) -> Model:
        """
        Builds model by assembling the different ranking components:
        - inputs
        - architecture
        - scoring
        - loss
        - metrics

        Returns:
            compiled tf keras model
        """

        """
        Generate dense features from input feature layer

        ranking_features is a dense feature layer that is obtained by passing the
        input feature nodes through their respective feature columns and then
        combining the outputs into a dense numeric layer
        Shape - [batch_size, record_features_size]

        metadata_features is a key value map of numeric features obtained by
        passing metadata(non-trainable) input features through a feature column and
        an individual dense layer.
        Shape(of single metadata_feature) - [batch_size, metadata_feature_size]
        """
        ranking_features, metadata_features = define_feature_layer(
            feature_config=self.feature_config, max_num_records=self.max_num_records
        )(inputs)

        """
        Transform data appropriately for the specified scoring and loss

        Shape of features depends on the type of loss and scoring function
        being used

        - pointwise loss + pointwise scoring
            [batch_size, max_num_records, record_features_size]
        - listwise loss + pointwise scoring
            [batch_size, max_num_records, record_features_size]
        - pointwise loss + groupwise scoring
            [batch_size, num_groups, group_size, record_features_size]
        """
        ranking_features, metadata_features = self._transform_features(
            ranking_features, metadata_features, loss
        )

        """
        Define scoring function

        The shape of scores depends on the type of loss+scoring function being used

        - pointwise loss + pointwise scoring
            [batch_size, 1]
        - listwise loss + pointwise scoring
            [batch_size, max_num_records]
        """
        scores = self._add_scoring_fn(ranking_features, loss)

        """
        Generate output predictions

        Same shape as scores, but with an additional activation layer
        """
        predictions = self._add_predictions(scores, loss=loss, mask=metadata_features.get("mask"))

        # Create model with functional Keras API
        model = Model(inputs=inputs, outputs={"ranking_scores": predictions})

        # Get loss and metrics
        loss_fn = loss.get_loss_fn(mask=metadata_features["mask"])
        metric_fns: List[kmetrics.Metric] = list()
        rank_feature_name = self.feature_config.get_rank("node_name")
        for metric in metrics:
            metric_fns.extend(
                metric_factory.get_metric_impl(
                    metric,
                    rank=metadata_features[rank_feature_name],
                    mask=metadata_features["mask"],
                )
            )

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metric_fns,
            experimental_run_tf_function=False,
        )

        # Write model summary to logs
        model_summary = list()
        model.summary(print_fn=lambda x: model_summary.append(x))
        self.logger.info("\n".join(model_summary))

        return model

    def fit(self, dataset, num_epochs, models_dir, logs_dir=None, logging_frequency=25):
        """
        Trains model for defined number of epochs

        Args:
            dataset: an instance of RankingDataset
            num_epochs: int value specifying number of epochs to train for
            models_dir: directory to save model checkpoints
            logs_dir: directory to save model logs
            logging_frequency: every #batches to log results
        """
        callbacks_list: list = self._build_callback_hooks(
            models_dir=models_dir,
            logs_dir=logs_dir,
            is_training=True,
            logging_frequency=logging_frequency,
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
        inference_signature: str = None,
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
            outfile = os.path.join(logs_dir, MODEL_PREDICTIONS_CSV_FILE)
            # Delete file if it exists
            file_io.rm_file(outfile)

        _predict_fn = self._get_predict_fn(
            inference_signature=inference_signature,
            features_to_return=self.feature_config.get_features_to_log(),
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
            self.logger.info("Model Predictions: ")
            predictions_df = pd.concat(predictions_df_list)
            self.logger.info(predictions_df)

        return predictions_df

    def evaluate(
        self,
        test_dataset: data.TFRecordDataset,
        inference_signature: str = None,
        logging_frequency: int = 25,
        group_metrics_min_queries: int = 50,
        logs_dir: Optional[str] = None,
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
        """
        group_metrics_keys = self.feature_config.get_group_metrics_keys()
        evaluation_features = group_metrics_keys + [
            self.feature_config.get_query_key(),
            self.feature_config.get_label(),
            self.feature_config.get_rank(),
        ]

        _predict_fn = self._get_predict_fn(
            inference_signature=inference_signature, features_to_return=evaluation_features
        )

        batch_count = 0
        df_grouped_stats = pd.DataFrame()
        for predictions_dict in test_dataset.map(_predict_fn).take(-1):
            predictions_df = pd.DataFrame(predictions_dict)

            df_batch_grouped_stats = metrics_helper.get_grouped_stats(
                df=predictions_df,
                query_key_col=self.feature_config.get_query_key("node_name"),
                label_col=self.feature_config.get_label("node_name"),
                old_rank_col=self.feature_config.get_rank("node_name"),
                new_rank_col=NEW_RANK_FIELD,
                group_keys=self.feature_config.get_group_metrics_keys("node_name"),
            )
            df_grouped_stats = df_grouped_stats.add(df_batch_grouped_stats, fill_value=0.0)
            batch_count += 1
            if batch_count % logging_frequency == 0:
                self.logger.info("Finished evaluating {} batches".format(batch_count))

        # Compute overall metrics
        df_overall_metrics = metrics_helper.summarize_grouped_stats(df_grouped_stats)
        self.logger.info("Overall Metrics: \n{}".format(df_overall_metrics))

        df_group_metrics = None
        df_group_metrics_summary = None
        if group_metrics_keys:
            # Filter groups by min_query_count
            df_grouped_stats = df_grouped_stats[
                df_grouped_stats["query_count"] >= group_metrics_min_queries
            ]

            # Compute group metrics
            df_group_metrics = df_grouped_stats.apply(
                metrics_helper.summarize_grouped_stats, axis=1
            )
            if logs_dir:
                file_io.write_df(
                    df_group_metrics, outfile=os.path.join(logs_dir, GROUP_METRICS_CSV_FILE)
                )

            # Compute group metrics summary
            df_group_metrics_summary = df_group_metrics.describe()
            self.logger.info(
                "Computing group metrics using keys: {}".format(
                    self.feature_config.get_group_metrics_keys("node_name")
                )
            )
            self.logger.info("Groupwise Metrics: \n{}".format(df_group_metrics_summary.T))

        return df_overall_metrics, df_group_metrics

    def _get_predict_fn(self, inference_signature, features_to_return):
        if self.is_compiled:
            infer = self.model
        else:
            # If SavedModel was loaded without compilation
            infer = self.model.signatures[inference_signature]

        # Get features to log
        features_to_log = [f.get("node_name", f["name"]) for f in features_to_return] + [
            "new_score",
            "new_rank",
        ]

        @tf.function
        def _flatten_records(x):
            """Collapse first two dimensions -> [batch_size, max_num_records]"""
            return tf.reshape(x, tf.concat([[-1], tf.shape(x)[2:]], axis=0))

        @tf.function
        def _filter_records(x, mask):
            """Filter records that were padded in each query"""
            return tf.squeeze(
                tf.gather_nd(
                    x,
                    tf.where(tf.not_equal(tf.cast(mask, tf.int64), tf.constant(0, dtype="int64"))),
                )
            )

        @tf.function
        def _predict_score(features, label):
            # features = {k: tf.cast(v, tf.float32) for k, v in features.items()}
            if self.is_compiled:
                scores = infer(features)["ranking_scores"]
            else:
                scores = infer(**features)["ranking_scores"]

            # Set scores of padded records to 0
            scores = tf.where(tf.equal(features["mask"], 0), tf.constant(-np.inf), scores)
            new_rank = tf.add(
                tf.argsort(
                    tf.argsort(scores, axis=-1, direction="DESCENDING", stable=True), stable=True
                ),
                tf.constant(1),
            )

            predictions_dict = dict()
            mask = _flatten_records(features["mask"])
            for feature_name in features_to_log:
                if feature_name == self.feature_config.get_label(key="node_name"):
                    feat_ = label
                elif feature_name == NEW_SCORE_FIELD:
                    feat_ = scores
                elif feature_name == NEW_RANK_FIELD:
                    feat_ = new_rank
                else:
                    if feature_name in features:
                        feat_ = features[feature_name]
                    else:
                        raise KeyError(
                            "{} was not found in input training data".format(feature_name)
                        )

                # Explode context features to each record for logging
                # NOTE: This assumes that the record dimension is on axis 1, like previously
                feat_ = tf.cond(
                    tf.equal(tf.shape(feat_)[1], tf.constant(1)),
                    true_fn=lambda: K.repeat_elements(feat_, rep=self.max_num_records, axis=1),
                    false_fn=lambda: feat_,
                )

                # Collapse from one query per data point to one record per data point
                # and remove padded dummy records
                feat_ = _filter_records(_flatten_records(feat_), mask)

                predictions_dict[feature_name] = feat_

            return predictions_dict

        return _predict_score

    def save(self, models_dir: str, pad_records: bool):
        """
        Save tf.keras model to models_dir

        Args:
            models_dir: path to directory to save the model
        """

        model_file = os.path.join(models_dir, "final")

        # Save model with default signature
        saved_model.save(self.model, export_dir=os.path.join(model_file, "default"))

        """
        Save model with custom signatures

        Currently supported
        - signature to read TFRecord SequenceExample inputs
        """
        saved_model.save(
            self.model,
            export_dir=os.path.join(model_file, "tfrecord"),
            signatures=define_serving_signatures(
                self.model, self.feature_config, pad_records, self.max_num_records
            ),
        )
        self.logger.info("Final model saved to : {}".format(model_file))

    def load_model(self, model_file: str) -> Model:
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

        #
        # Converting the low level AutoTrackable model into a Keras Model
        # using tensorflow hub KerasLayer as a wrapper
        #
        # def build_model(loaded_model):
        #     x = self.feature_config.define_inputs()
        #     # Wrap what's loaded to a KerasLayer
        #     keras_layer = hub.KerasLayer(loaded_model, trainable=True)(x)
        #     model = tf.keras.Model(x, keras_layer)
        #     return model

        # loaded_model = tf.saved_model.load(model_file)
        # model = build_model(loaded_model)

        model = tf.keras.models.load_model(model_file, compile=False)

        self.logger.info("Successfully loaded SavedModel from {}".format(model_file))
        self.logger.warning("Retraining is not yet supported. Model is loaded with compile=False")

        return model

    def load_weights(self, model_file: str):
        # Load saved model with compile=False
        loaded_model = self.load_model(model_file)

        # Set weights of Keras model from the loaded model weights
        self.model.set_weights(loaded_model.get_weights())
        self.logger.info("Weights have been set from SavedModel. RankingModel can now be trained.")

    def _build_callback_hooks(
        self, models_dir: str, logs_dir: str, is_training=True, logging_frequency=25
    ):
        """
        Build callback hooks for the training loop

        Returns:
            callbacks_list: list of callbacks
        """
        callbacks_list: list = list()

        if is_training:
            # Model checkpoint
            if models_dir:
                checkpoints_path = os.path.join(models_dir, CHECKPOINT_FNAME)
                cp_callback = callbacks.ModelCheckpoint(
                    filepath=checkpoints_path,
                    save_weights_only=False,
                    verbose=1,
                    save_best_only=True,
                    mode="max",
                    monitor="val_new_MRR",
                )
                callbacks_list.append(cp_callback)

            # Early Stopping
            early_stopping_callback = callbacks.EarlyStopping(
                monitor="val_new_MRR", mode="max", patience=2, verbose=1, restore_best_weights=True
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

        # Add more here

        return callbacks_list

    def _transform_features(self, ranking_features, metadata_features, loss: RankingLossBase):
        """
        Transform the features as necessary for different
        scoring and loss types
        """
        if self.scoring_key == ScoringKey.POINTWISE:
            if loss.loss_type == LossTypeKey.POINTWISE:
                """
                If using pointwise scoring and pointwise loss, return as is.
                """
                return ranking_features, metadata_features
            elif loss.loss_type == LossTypeKey.PAIRWISE:
                #
                # TODO
                #
                raise NotImplementedError
            elif loss.loss_type == LossTypeKey.LISTWISE:
                #
                # TODO
                #
                return ranking_features, metadata_features
        elif self.scoring_key == ScoringKey.PAIRWISE:
            if loss.loss_type == LossTypeKey.POINTWISE:
                """
                If using pointwise scoring and pointwise loss, return as is.
                """
                return ranking_features, metadata_features
            elif loss.loss_type == LossTypeKey.PAIRWISE:
                #
                # TODO
                #
                raise NotImplementedError
            elif loss.loss_type == LossTypeKey.LISTWISE:
                #
                # TODO
                #
                raise NotImplementedError

        elif self.scoring_key == ScoringKey.GROUPWISE:
            if loss.loss_type == LossTypeKey.POINTWISE:
                """
                If using pointwise scoring and pointwise loss, return as is.
                """
                raise NotImplementedError
            elif loss.loss_type == LossTypeKey.PAIRWISE:
                #
                # TODO
                #
                raise NotImplementedError
            elif loss.loss_type == LossTypeKey.LISTWISE:
                #
                # TODO
                #
                raise NotImplementedError

    def _add_scoring_fn(self, ranking_features, loss: RankingLossBase):
        """
        Define the model architecture based on the type of scoring function selected
        """
        scoring_fn = scoring_factory.get_scoring_fn(
            scoring_key=self.scoring_key, model_config=self.model_config, loss_type=loss.loss_type,
        )

        scores = scoring_fn(ranking_features)

        return scores

    def _add_predictions(self, logits, loss, mask):
        """
        Convert ranking scores into probabilities

        Args:
            logits: scores from the final model layer
            loss: RankingLoss object
            mask: mask tensor for handling padded records
        """
        activation_fn = loss.get_final_activation_op()

        return activation_fn(logits, mask)
