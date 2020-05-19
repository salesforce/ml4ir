# type: ignore
# TODO: Fix typing

from ml4ir.model.relevance_model import RelevanceModel

import tensorflow as tf
from tensorflow import data

from typing import Optional


class RankingConstants:
    NEW_RANK = "new_rank"


class RankingModel(RelevanceModel):
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
        additional_features[RankingConstants.NEW_RANK] = lambda features, label, scores: tf.add(
            tf.argsort(
                tf.argsort(scores, axis=-1, direction="DESCENDING", stable=True), stable=True
            ),
            tf.constant(1),
        )

        return super().predict(
            test_dataset=test_dataset,
            inference_signature=inference_signature,
            additional_features=additional_features,
            logs_dir=logs_dir,
            logging_frequency=logging_frequency,
        )

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
        pass

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

        def mask_padded_records(predictions, features_dict):
            for key, value in predictions.items():
                predictions[key] = tf.where(
                    tf.equal(features_dict["mask"], 0), tf.constant(0.0), predictions[key]
                )

            return predictions

        super().save(
            models_dir=models_dir,
            preprocessing_keys_to_fns=preprocessing_keys_to_fns,
            postprocessing_fn=mask_padded_records,
            required_fields_only=required_fields_only,
            pad_sequence=pad_sequence,
        )
