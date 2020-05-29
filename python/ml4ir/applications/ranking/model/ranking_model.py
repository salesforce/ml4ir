import tensorflow as tf
from tensorflow import data
import os
import pandas as pd

from ml4ir.base.model.relevance_model import RelevanceModel
from ml4ir.base.io import file_io
from ml4ir.base.model.scoring.prediction_helper import get_predict_fn
from ml4ir.base.model.relevance_model import RelevanceModelConstants
from ml4ir.applications.ranking.model.scoring import prediction_helper
from ml4ir.applications.ranking.model.metrics import metrics_helper

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
        additional_features[RankingConstants.NEW_RANK] = prediction_helper.convert_score_to_rank

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
        """
        group_metrics_keys = self.feature_config.get_group_metrics_keys()
        evaluation_features = group_metrics_keys + [
            self.feature_config.get_query_key(),
            self.feature_config.get_label(),
            self.feature_config.get_rank(),
        ]
        additional_features[RankingConstants.NEW_RANK] = prediction_helper.convert_score_to_rank

        _predict_fn = get_predict_fn(
            model=self.model,
            tfrecord_type=self.tfrecord_type,
            feature_config=self.feature_config,
            inference_signature=inference_signature,
            is_compiled=self.is_compiled,
            output_name=self.output_name,
            features_to_return=evaluation_features,
            additional_features=additional_features,
            max_sequence_size=self.max_sequence_size,
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
                new_rank_col=RankingConstants.NEW_RANK,
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
                    df_group_metrics,
                    outfile=os.path.join(logs_dir, RelevanceModelConstants.GROUP_METRICS_CSV_FILE),
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
