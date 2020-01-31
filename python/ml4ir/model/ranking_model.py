import os
from logging import Logger
from typing import List
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import callbacks, layers, Input, Model
from tensorflow.keras.optimizers import Optimizer
from tensorflow import saved_model
from tensorflow import TensorSpec, TensorArray
from tensorflow import data

from ml4ir.config.features import Features
from ml4ir.config.keys import FeatureTypeKey, ScoringKey
from ml4ir.config.keys import LossTypeKey, ServingSignatureKey
from ml4ir.model.optimizer import get_optimizer
from ml4ir.model.losses.loss_base import RankingLossBase
from ml4ir.model.losses import loss_factory
from ml4ir.model.metrics import metric_factory
from ml4ir.model.scoring import scoring_factory
from ml4ir.data.tfrecord_reader import make_parse_fn
import pandas as pd
import numpy as np

from typing import Dict


class RankingModel:
    def __init__(
        self,
        architecture_key: str,
        loss_key: str,
        scoring_key: str,
        metrics_keys: List[str],
        optimizer_key: str,
        features: Features,
        max_num_records: int,
        model_file: str,
        learning_rate: float,
        learning_rate_decay: float,
        compute_intermediate_stats: bool,
        logger=None,
    ):
        self.architecture_key: str = architecture_key
        self.features: Features = features
        self.scoring_key: str = scoring_key
        self.logger: Logger = logger
        self.max_num_records = max_num_records

        # Define optimizer
        optimizer: Optimizer = get_optimizer(
            optimizer_key=optimizer_key,
            learning_rate=learning_rate,
            learning_rate_decay=learning_rate_decay,
        )

        # Define loss function
        loss: RankingLossBase = loss_factory.get_loss(loss_key=loss_key, scoring_key=scoring_key)

        # Define metrics
        metrics: List[str] = [metric_factory.get_metric(metric_key=key) for key in metrics_keys]

        # Load/Build Model
        if model_file:
            """
            NOTE: Retraining not supported. Currently loading SavedModel
                  as a low level AutoTrackable object for inference
            """
            self.model: Model = self.load_model(model_file, optimizer, loss, metrics)
            self.is_compiled = False
        else:
            inputs: Dict[str, Input] = features.define_inputs(max_num_records)
            self.model = self.build_model(inputs, optimizer, loss, metrics)
            self.is_compiled = True

    def build_model(
        self,
        inputs: Dict[str, Input],
        optimizer: Optimizer,
        loss: RankingLossBase,
        metrics: List[str],
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
        Specify inputs to the model

        Define the individual input nodes are defined for each feature
        Each data sample represents features for a single record
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
        ranking_features, metadata_features = self._add_feature_layer(inputs)

        """
        Transform data appropriately for the specified scoring and loss

        Shape of features depends on the type of loss and scoring function
        being used

        - pointwise loss + pointwise scoring
            [batch_size, record_features_size]
        - listwise loss + pointwise scoring
            [batch_size, max_num_records, record_features_size]
        - pointwise loss + groupwise scoring
            [batch_size, group_size, record_features_size]
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
        predictions = self._add_predictions(scores)

        # Create model with functional Keras API
        model = Model(inputs=inputs, outputs={"ranking_scores": predictions})

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=loss.get_loss_fn(mask=metadata_features["mask"]),
            metrics=metrics,
            experimental_run_tf_function=False,
        )

        self.logger.info(model.summary())

        return model

    def fit(self, dataset, num_epochs, models_dir):
        """
        Trains model for defined number of epochs

        Args:
            dataset: an instance of RankingDataset
            num_epochs: int value specifying number of epochs to train for
            models_dir: directory to save model checkpoints
        """
        callbacks_list: list = self._build_callback_hooks(models_dir=models_dir)

        self.logger.info("Training model...")
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
        rerank: bool = False,
    ):
        """
        Predict the labels for the trained model

        Args:
            test_dataset: an instance of tf.data.dataset
            inference_signature: If using a SavedModel for prediction, specify the inference signature
            rerank: boolean specifying if new ranks should be returned

        Returns:
            ranking scores or new ranks for each record in a query
        """
        self.logger.info("Predicting scores on test set...")
        if self.is_compiled:
            infer = self.model
        else:
            # If SavedModel was loaded without compilation
            infer = self.model.signatures[inference_signature]

        @tf.function
        def _predict_score(features, label):
            features = {k: tf.cast(v, tf.float32) for k, v in features.items()}
            if self.is_compiled:
                scores = infer(features)["ranking_scores"]
            else:
                scores = infer(**features)["ranking_scores"]

            # Set scores of padded records to 0
            scores = tf.where(tf.equal(features["mask"], 0), tf.constant(-np.inf), scores)
            return scores

        scores_list = list()
        for scores in test_dataset.map(_predict_score).take(-1):
            scores_list.append(scores.numpy())

        ranking_scores = np.vstack(scores_list)
        self.logger.info("Ranking Scores: ")
        self.logger.info(pd.DataFrame(ranking_scores))
        return ranking_scores

    def evaluate(self, test_dataset):
        """
        Predict the labels/ranks and compute metrics for the test set

        Args:
            test_dataset: an instance of tf.data.dataset

        Returns:
            evaluated metrics specified on the dataset
        """
        self.logger.info("Evaluating model on test set...")
        if self.is_compiled:
            return self.model.evaluate(test_dataset)
        else:
            raise NotImplementedError(
                "The model could not be evaluated. Check if the model was compiled correctly. Training a SavedModel is not currently supported."
            )

    def save(self, models_dir: str):
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
            signatures=self._build_saved_model_signatures(),
        )
        self.logger.info("Final model saved to : {}".format(model_file))

    def load_model(
        self, model_file: str, optimizer: Optimizer, loss: RankingLossBase, metrics: List[str]
    ) -> Model:
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
        #     x = self.features.define_inputs()
        #     # Wrap what's loaded to a KerasLayer
        #     keras_layer = hub.KerasLayer(loaded_model, trainable=True)(x)
        #     model = tf.keras.Model(x, keras_layer)
        #     return model

        # loaded_model = tf.saved_model.load(model_file)
        # model = build_model(loaded_model)

        model = tf.keras.models.load_model(model_file, compile=False)

        self.logger.info("Successfully loaded SavedModel from {}".format(model_file))
        self.logger.warning("Retraining is not supported. Model is loaded with compile=False")

        return model

    def _build_saved_model_signatures(self):
        """
        Add signatures to the tf keras savedmodel
        """

        # Default signature
        # TODO: Define input_signature
        # @tf.function(input_signature=[])
        # def _serve_default(**features):
        #     features_dict = {k: tf.cast(v, tf.float32) for k, v in features.items()}
        #     # Run the model to get predictions
        #     predictions = self.model(inputs=features_dict)

        #     # Mask the padded records
        #     for key, value in predictions.items():
        #         predictions[key] = tf.where(
        #             tf.equal(features_dict['mask'], 0),
        #             tf.constant(-np.inf),
        #             predictions[key])

        #     return predictions

        # TFRecord Signature
        # Define a parsing function for tfrecord protos
        inputs = self.features.get_X()
        self.features.feature_config.pop("mask")
        tfrecord_parse_fn = make_parse_fn(
            feature_config=self.features, max_num_records=self.max_num_records
        )

        # Define a serving signature for tfrecord
        @tf.function(input_signature=[TensorSpec(shape=[None], dtype=tf.string)])
        def _serve_tfrecord(sequence_example_protos):
            input_size = tf.shape(sequence_example_protos)[0]
            features_dict = {
                feature: TensorArray(dtype=tf.float32, size=input_size) for feature in inputs
            }

            # Define loop index
            i = tf.constant(0)

            # Define loop condition
            def loop_condition(i, sequence_example_protos, features_dict):
                return tf.less(i, input_size)

            # Define loop body
            def loop_body(i, sequence_example_protos, features_dict):
                """
                TODO: Modify parse_fn from
                parse_single_sequence_example -> parse_sequence_example
                to handle a batch of TFRecord proto
                """
                features, labels = tfrecord_parse_fn(sequence_example_protos[i])
                for feature, feature_val in features.items():
                    features_dict[feature] = features_dict[feature].write(
                        i, tf.cast(feature_val, tf.float32)
                    )

                i += 1

                return i, sequence_example_protos, features_dict

            # Parse all SequenceExample protos to get features
            _, _, features_dict = tf.while_loop(
                cond=loop_condition,
                body=loop_body,
                loop_vars=[i, sequence_example_protos, features_dict],
            )

            # Convert TensorArray to tensor
            features_dict = {k: v.stack() for k, v in features_dict.items()}

            # Run the model to get predictions
            predictions = self.model(inputs=features_dict)

            # Mask the padded records
            for key, value in predictions.items():
                predictions[key] = tf.where(
                    tf.equal(features_dict["mask"], 0), tf.constant(-np.inf), predictions[key]
                )

            return predictions

        return {
            # ServingSignatureKey.DEFAULT: _serve_default,
            ServingSignatureKey.TFRECORD: _serve_tfrecord
        }

    def _build_callback_hooks(self, models_dir: str):
        """
        Build callback hooks for the training loop

        Returns:
            callbacks_list: list of callbacks
        """
        callbacks_list: list = list()

        # Model checkpoint
        checkpoints_path = os.path.join(models_dir, "checkpoints")
        cp_callback = callbacks.ModelCheckpoint(
            filepath=checkpoints_path, save_weights_only=False, verbose=1
        )
        callbacks_list.append(cp_callback)

        # Add more here

        return callbacks_list

    def _add_feature_layer(self, inputs):
        """
        Add feature layer by processing the inputs
        NOTE: Embeddings or any other in-graph preprocessing goes here
        """
        # Add mask to features object
        # TODO: if we always call this - why not always, in the Features _init_?
        self.features.add_mask()

        ranking_features = list()
        metadata_features = dict()

        def _get_dense_feature(inputs, feature, shape=(1,)):
            """
            Convert an input into a dense numeric feature

            NOTE: Can remove this in the future and
                  pass inputs[feature] directly
            """
            feature_col = feature_column.numeric_column(feature, shape=shape)
            dense_feature = layers.DenseFeatures(feature_col)(inputs)
            return dense_feature

        for feature, feature_info in self.features.get_dict().items():
            feature_node_name = feature_info.get("node_name", feature)

            if feature_info["type"] == FeatureTypeKey.NUMERIC:
                dense_feature = _get_dense_feature(
                    inputs, feature_node_name, shape=(self.max_num_records, 1)
                )
                if feature_info["trainable"]:
                    ranking_features.append(tf.cast(dense_feature, tf.float32))
                else:
                    metadata_features[feature] = tf.cast(dense_feature, tf.float32)
            elif feature_info["type"] == FeatureTypeKey.STRING:
                # TODO: Add embedding layer here
                pass
            elif feature_info["type"] == FeatureTypeKey.CATEGORICAL:
                # TODO: Add embedding layer with vocabulary here
                raise NotImplementedError
            elif feature_info["type"] == FeatureTypeKey.LABEL:
                # NOTE: Can implement in the future if necessary
                pass
            else:
                raise Exception(
                    "Unknown feature type {} for feature : {}".format(
                        feature_info["type"], feature
                    )
                )

        """
        Reshape ranking features to create features of shape
        [batch, max_num_records, num_features]
        """
        ranking_features = tf.stack(ranking_features, axis=1)
        ranking_features = tf.transpose(ranking_features, perm=[0, 2, 1])

        return ranking_features, metadata_features

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
                raise NotImplementedError
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
            scoring_key=self.scoring_key,
            architecture_key=self.architecture_key,
            loss_type=loss.loss_type,
        )

        scores = scoring_fn(ranking_features)

        return scores

    def _add_predictions(self, scores):
        """
        Convert ranking scores into probabilities

        NOTE: Depends on the loss being used
        Could be moved to the loss function in the future
        """
        #
        # TODO : Move to separate module
        #
        predictions = layers.Activation("sigmoid", name="ranking_scores")(scores)

        return predictions
