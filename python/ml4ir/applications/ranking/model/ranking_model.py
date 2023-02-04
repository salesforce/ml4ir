import os
import sys
import tensorflow as tf
from tensorflow import data
import pandas as pd
import numpy as np
from typing import Optional

from ml4ir.base.model.relevance_model import RelevanceModel
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.base.model.scoring.prediction_helper import get_predict_fn
from ml4ir.base.model.relevance_model import RelevanceModelConstants
from ml4ir.base.model.architectures.dnn import DNNLayerKey
from ml4ir.applications.ranking.model.scoring import prediction_helper
from ml4ir.applications.ranking.model.metrics.helpers import metrics_helper
from ml4ir.applications.ranking.config.keys import PositionalBiasHandler
from ml4ir.applications.ranking.t_test import perform_click_rank_dist_paired_t_test, compute_stats_from_stream, t_test_log_results


pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)


class RankingConstants:
    NEW_RANK = "new_rank"
    TTEST_PVALUE_THRESHOLD = 0.1


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
        Predict the scores on the test dataset using the trained model

        Parameters
        ----------
        test_dataset : `Dataset` object
            `Dataset` object for which predictions are to be made
        inference_signature : str, optional
            If using a SavedModel for prediction, specify the inference signature to be used for computing scores
        additional_features : dict, optional
            Dictionary containing new feature name and function definition to
            compute them. Use this to compute additional features from the scores.
            For example, converting ranking scores for each document into ranks for
            the query
        logs_dir : str, optional
            Path to directory to save logs
        logging_frequency : int
            Value representing how often(in batches) to log status

        Returns
        -------
        `pd.DataFrame`
            pandas DataFrame containing the predictions on the test dataset
            made with the `RelevanceModel`
        """
        additional_features[RankingConstants.NEW_RANK] = prediction_helper.convert_score_to_rank

        return super().predict(
            test_dataset=test_dataset,
            inference_signature=inference_signature,
            additional_features=additional_features,
            logs_dir=logs_dir,
            logging_frequency=logging_frequency,
        )


    def run_ttest(self, mean, variance, n, ttest_pvalue_threshold):
        """
        Compute the paired t-test statistic and its p-value given mean, standard deviation and sample count
        Parameters
        ----------
        mean: float
            The mean of the rank differences for the entire dataset
        variance: float
            The variance of the rank differences for the entire dataset
        n: int
            The number of samples in the entire dataset
        ttest_pvalue_threshold: float
            P-value threshold for student t-test

        Returns
        -------
        t_test_metrics_dict: Dictionary
            A dictionary with the t-test metrics recorded.
        """
        t_test_stat, pvalue = perform_click_rank_dist_paired_t_test(mean, variance, n)
        t_test_log_results(t_test_stat, pvalue, ttest_pvalue_threshold, self.logger)

        # Log t-test statistic and p-value
        t_test_metrics_dict = {'Rank distribution t-test statistic': t_test_stat,
                               'Rank distribution t-test pvalue': pvalue,
                               'Rank distribution t-test difference is statistically significant': pvalue < ttest_pvalue_threshold}
        return t_test_metrics_dict

    def evaluate(
        self,
        test_dataset: data.TFRecordDataset,
        inference_signature: str = None,
        additional_features: dict = {},
        group_metrics_min_queries: int = 50,
        logs_dir: Optional[str] = None,
        logging_frequency: int = 25,
        compute_intermediate_stats: bool = True,
    ):
        """
        Evaluate the RelevanceModel

        Parameters
        ----------
        test_dataset: an instance of tf.data.dataset
        inference_signature : str, optional
            If using a SavedModel for prediction, specify the inference signature to be used for computing scores
        additional_features : dict, optional
            Dictionary containing new feature name and function definition to
            compute them. Use this to compute additional features from the scores.
            For example, converting ranking scores for each document into ranks for
            the query
        group_metrics_min_queries : int, optional
            Minimum count threshold per group to be considered for computing
            groupwise metrics
        logs_dir : str, optional
            Path to directory to save logs
        logging_frequency : int
            Value representing how often(in batches) to log status
        compute_intermediate_stats : bool
            [Currently ignored] Determines if group metrics and other intermediate stats on the test set should be computed

        Returns
        -------
        df_overall_metrics : `pd.DataFrame` object
            `pd.DataFrame` containing overall metrics
        df_groupwise_metrics : `pd.DataFrame` object
            `pd.DataFrame` containing groupwise metrics if
            group_metric_keys are defined in the FeatureConfig
        metrics_dict : dict
            metrics as a dictionary of metric names mapping to values

        Notes
        -----
        You can directly do a `model.evaluate()` only if the keras model is compiled

        Override this method to implement your own evaluation metrics.
        """
        metrics_dict = dict()
        group_metrics_keys = self.feature_config.get_group_metrics_keys()
        evaluation_features = (
            group_metrics_keys
            + [
                self.feature_config.get_query_key(),
                self.feature_config.get_label(),
                self.feature_config.get_rank(),
            ]
        )

        # Add aux_label if present
        aux_label = self.feature_config.get_aux_label()
        if aux_label and (
                aux_label.get("node_name") or (
                aux_label["name"] not in self.feature_config.get_group_metrics_keys("node_name"))):
            evaluation_features += [aux_label]

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
        # defining variables to compute running mean and variance for t-test computations
        agg_count, agg_mean, agg_M2 = 0, 0, 0
        for predictions_dict in test_dataset.map(_predict_fn).take(-1):
            predictions_df = pd.DataFrame(predictions_dict)

            # Accumulating statistics for t-test calculation using 1/rank
            clicked_records = predictions_df[predictions_df[self.feature_config.get_label("node_name")] == 1.0]
            diff = (1/clicked_records[RankingConstants.NEW_RANK] - 1/clicked_records[
                self.feature_config.get_rank("node_name")]).to_list()
            agg_count, agg_mean, agg_M2 = compute_stats_from_stream(diff, agg_count, agg_mean, agg_M2)

            df_batch_grouped_stats = metrics_helper.get_grouped_stats(
                df=predictions_df,
                query_key_col=self.feature_config.get_query_key("node_name"),
                label_col=self.feature_config.get_label("node_name"),
                old_rank_col=self.feature_config.get_rank("node_name"),
                new_rank_col=RankingConstants.NEW_RANK,
                group_keys=list(set(self.feature_config.get_group_metrics_keys(
                    "node_name"))),
                aux_label=self.feature_config.get_aux_label("node_name"),
            )
            if df_grouped_stats.empty:
                df_grouped_stats = df_batch_grouped_stats
            else:
                df_grouped_stats = df_grouped_stats.add(
                    df_batch_grouped_stats, fill_value=0.0)
            batch_count += 1
            if batch_count % logging_frequency == 0:
                self.logger.info(
                    "Finished evaluating {} batches".format(batch_count))

        # performing click rank distribution t-test
        t_test_metrics_dict = self.run_ttest(agg_mean, (agg_M2 / (agg_count - 1)), agg_count, RankingConstants.TTEST_PVALUE_THRESHOLD)
        metrics_dict.update(t_test_metrics_dict)

        # Compute overall metrics
        df_overall_metrics = metrics_helper.summarize_grouped_stats(
            df_grouped_stats)
        self.logger.info("Overall Metrics: \n{}".format(df_overall_metrics))

        # Log metrics to weights and biases
        metrics_dict.update(
            {"test_{}".format(k): v for k,
             v in df_overall_metrics.to_dict().items()}
        )

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
                self.file_io.write_df(
                    df_group_metrics,
                    outfile=os.path.join(
                        logs_dir, RelevanceModelConstants.GROUP_METRICS_CSV_FILE),
                )

            # Compute group metrics summary
            df_group_metrics_summary = df_group_metrics.describe()
            self.logger.info(
                "Computing group metrics using keys: {}".format(
                    self.feature_config.get_group_metrics_keys("node_name")
                )
            )
            self.logger.info("Groupwise Metrics: \n{}".format(
                df_group_metrics_summary.T))

            # Log metrics to weights and biases
            metrics_dict.update(
                {
                    "test_group_mean_{}".format(k): v
                    for k, v in df_group_metrics_summary.T["mean"].to_dict().items()
                }
            )

        return df_overall_metrics, df_group_metrics, metrics_dict

    def save(
        self,
        models_dir: str,
        preprocessing_keys_to_fns={},
        postprocessing_fn=None,
        required_fields_only: bool = True,
        pad_sequence: bool = False,
        dataset: Optional[RelevanceDataset] = None,
        experiment_details: Optional[dict] = None
    ):
        """
        Save the RelevanceModel as a tensorflow SavedModel to the `models_dir`
        Additionally, sets the score for the padded records to 0

        There are two different serving signatures currently used to save the model
            `default`: default keras model without any pre/post processing wrapper
            `tfrecord`: serving signature that allows keras model to be served using TFRecord proto messages.
                      Allows definition of custom pre/post processing logic

        Additionally, each model layer is also saved as a separate numpy zipped
        array to enable transfer learning with other ml4ir models.

        Parameters
        ----------
        models_dir : str
            path to directory to save the model
        preprocessing_keys_to_fns : dict
            dictionary mapping function names to tf.functions that should be saved in the preprocessing step of the tfrecord serving signature
        postprocessing_fn: function
            custom tensorflow compatible postprocessing function to be used at serving time.
            Saved as part of the postprocessing layer of the tfrecord serving signature
        required_fields_only: bool
            boolean value defining if only required fields
            need to be added to the tfrecord parsing function at serving time
        pad_sequence: bool, optional
            Value defining if sequences should be padded for SequenceExample proto inputs at serving time.
            Set this to False if you want to not handle padded scores.
        dataset : `RelevanceDataset` object
            RelevanceDataset object that can optionally be passed to be used by downstream jobs
            that want to save the data along with the model.
            Note that this feature is currently unimplemented and is upto the users to override
            and customize.
        experiment_details: dict
            Dictionary containing metadata and results about the current experiment

        Notes
        -----
        All the functions passed under `preprocessing_keys_to_fns` here must be
        serializable tensor graph operations
        """

        def mask_padded_records(predictions, features_dict):
            for key, value in predictions.items():
                predictions[key] = tf.where(
                    tf.equal(features_dict["mask"], 0), tf.constant(
                        0.0), predictions[key]
                )

            return predictions

        super().save(
            models_dir=models_dir,
            preprocessing_keys_to_fns=preprocessing_keys_to_fns,
            postprocessing_fn=mask_padded_records,
            required_fields_only=required_fields_only,
            pad_sequence=pad_sequence,
        )

        # Logging positional biases
        for layer in self.model.layers:
            if layer.name == PositionalBiasHandler.FIXED_ADDITIVE_POSITIONAL_BIAS:
                positional_bias_coefficients = pd.DataFrame(
                    [{'rank': i + 1, 'positional_bias': layer.get_weights()[0][i][0]}
                     for i in range(len(layer.get_weights()[0]))]
                )
                positional_biases = positional_bias_coefficients['positional_bias']
                softmax_positional_biases = np.exp(positional_biases) / np.sum(np.exp(positional_biases), axis=0)
                positional_bias_coefficients['softmax_positional_bias'] = softmax_positional_biases
                self.file_io.write_df(
                    positional_bias_coefficients,
                    outfile=os.path.join(models_dir, "positional_biases.csv"),
                    index=False
                )


class LinearRankingModel(RankingModel):
    """
    Subclass of the RankingModel object customized for training and saving a
    simple linear ranking model. Current implementation overrides the save model
    functionality to map input feature nodes into corresponding weights/coefficients.
    """

    def save(
        self,
        models_dir: str,
        preprocessing_keys_to_fns={},
        postprocessing_fn=None,
        required_fields_only: bool = True,
        pad_sequence: bool = False,
        dataset: Optional[RelevanceDataset] = None,
        experiment_details: Optional[dict] = None
    ):
        """
        Save the RelevanceModel as a tensorflow SavedModel to the `models_dir`
        Additionally, sets the score for the padded records to 0

        There are two different serving signatures currently used to save the model
            `default`: default keras model without any pre/post processing wrapper
            `tfrecord`: serving signature that allows keras model to be served using TFRecord proto messages.
                      Allows definition of custom pre/post processing logic

        Parameters
        ----------
        models_dir : str
            path to directory to save the model
        preprocessing_keys_to_fns : dict
            dictionary mapping function names to tf.functions that should be saved in the preprocessing step of the tfrecord serving signature
        postprocessing_fn: function
            custom tensorflow compatible postprocessing function to be used at serving time.
            Saved as part of the postprocessing layer of the tfrecord serving signature
        required_fields_only: bool
            boolean value defining if only required fields
            need to be added to the tfrecord parsing function at serving time
        pad_sequence: bool, optional
            Value defining if sequences should be padded for SequenceExample proto inputs at serving time.
            Set this to False if you want to not handle padded scores.
        dataset : `RelevanceDataset` object
            RelevanceDataset object that can optionally be passed to be used by downstream jobs
            that want to save the data along with the model.
            Note that this feature is currently unimplemented and is upto the users to override
            and customize.
        experiment_details: dict
            Dictionary containing metadata and results about the current experiment

        Notes
        -----
        - All the functions passed under `preprocessing_keys_to_fns` here must be
        serializable tensor graph operations
        - The LinearRankingModel.save() method specifically saves the weights
        of the dense layer as a CSV file where the feature names are mapped to
        the weights as key-value pairs.
        """

        # Save the linear model coefficients as a CSV
        dnn_model = self.model.get_layer(DNNLayerKey.MODEL_NAME)
        dense_layer = None
        for layer in dnn_model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                dense_layer = layer
                assert dense_layer.units == 1
        if not dense_layer:
            raise KeyError("No dense layer found in the model. Can not save the linear ranking model coefficients.")

        linear_model_coefficients = pd.DataFrame(
            list(zip(
                dnn_model.train_features,
                tf.squeeze(dense_layer.get_weights()[0]).numpy())
            ),
            columns=["feature", "weight"])
        # Adding log for bias value
        if len(dense_layer.get_weights()) > 1:
            bias_val = dense_layer.get_weights()[1][0]
            linear_model_coefficients.loc[len(linear_model_coefficients.index)] = ['bias', bias_val]
        self.logger.info("Linear Model Coefficients:\n{}".format(
            linear_model_coefficients.to_csv(index=False)))
        self.file_io.write_df(
            linear_model_coefficients,
            outfile=os.path.join(models_dir, "coefficients.csv"),
            index=False
        )

        # Call super save method to persist the SavedModel files
        super().save(
            models_dir=models_dir,
            preprocessing_keys_to_fns=preprocessing_keys_to_fns,
            postprocessing_fn=postprocessing_fn,
            required_fields_only=required_fields_only,
            pad_sequence=pad_sequence,
        )

    def calibrate(self, **kwargs):
        raise NotImplementedError
