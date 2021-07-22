import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import data
from ml4ir.base.model.relevance_model import RelevanceModel
from ml4ir.base.model.relevance_model import RelevanceModelConstants


class ClassificationModel(RelevanceModel):
    """Inherits from a Relevance model
    to create a classification model to
    create classification evaluate and predict
    methods."""

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
        Evaluate the Classification Model
        To avoid evaluating once on the the whole test
        dataset, we calculate the metrics incrementally using
        the batch size the user defined. Currently, we have a hacky
        way to get the batch size since it is not part of the
        `relevance_model` or `classification_model`.

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
            Determines if group metrics and other intermediate stats on the test set should be computed

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
        """

        if not self.is_compiled:
            return NotImplementedError
        group_metrics_keys = self.feature_config.get_group_metrics_keys()
        metrics_dict = self.model.evaluate(test_dataset)
        metrics_dict = dict(zip(self.model.metrics_names, metrics_dict))
        if compute_intermediate_stats:
            self.logger.info("Computing grouped metrics.")
            self.logger.warning("Warning: currently, group-wise metric computation "
                                "collects the test data and predictions "
                                "in memory in order to perform the groupBy operations. "
                                "With large test datasets, it can lead in OOM issues.")
            predictions = self.predict(test_dataset,
                                       inference_signature=inference_signature,
                                       additional_features=additional_features,
                                       logs_dir=logs_dir,
                                       logging_frequency=logging_frequency)
            global_metrics = []  # group_name, metric, value
            grouped_metrics = []
            # instead of calculating measure with a single update_state (can result in a call with
            # thousands of examples at once, we use the same batch size used during training.
            # Helps prevent OOM issues.
            batch_size = test_dataset._input_dataset._batch_size.numpy()  # Hacky way to get batch_size
            # Letting metrics in the outer loop to avoid tracing
            for metric in self.model.metrics:
                global_metrics.append(
                    self.calculate_metric_on_batch(metric, predictions, batch_size))
                self.logger.info(f"Global metric {metric.name} completed."
                                 f" Score: {global_metrics[-1]['value']}")

                for group_ in group_metrics_keys:  # Calculate metrics for group metrics
                    for name, group in predictions.groupby(group_['name']):
                        self.logger.info(f"Per feature metric {metric.name}."
                                         f" Feature: {group_['name']}, value: {name}")
                        if group.shape[0] >= group_metrics_min_queries:
                            grouped_metrics.append(self.calculate_metric_on_batch(metric,
                                                                                  group,
                                                                                  batch_size,
                                                                                  group_['name'],
                                                                                  name))
            global_metrics = pd.DataFrame(global_metrics)
            grouped_metrics = pd.DataFrame(grouped_metrics).sort_values(by='size')
            if logs_dir:
                self.file_io.write_df(
                    grouped_metrics,
                    outfile=os.path.join(logs_dir, RelevanceModelConstants.GROUP_METRICS_CSV_FILE),
                    index=False
                )
                self.file_io.write_df(
                    global_metrics,
                    outfile=os.path.join(logs_dir, RelevanceModelConstants.METRICS_CSV_FILE),
                    index=False
                )
                self.logger.info(f"Evaluation Results written at: {logs_dir}")
            return global_metrics, grouped_metrics, metrics_dict
        if logs_dir:
            global_metrics = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=["value"])
            global_metrics = global_metrics.reset_index().rename(columns={'index': 'metric'})
            self.file_io.write_df(
                global_metrics,
                outfile=os.path.join(logs_dir, RelevanceModelConstants.METRICS_CSV_FILE),
                index=False
            )
            self.logger.info(f"Evaluation Results written at: {logs_dir}")
        return None, None, metrics_dict

    @staticmethod
    def get_chunks_from_df(dataframe, size):
        """
        Given a pd.DataFrame, creates batches of size `size`.
        Instead of having a metric applied on big data once, this
        ensures we are using batches of examples during test too.
        """
        for pos in range(0, len(dataframe), size):
            yield dataframe.iloc[pos:pos + size]

    def calculate_metric_on_batch(self, metric, predictions, batch_size, group_name=None, group_key=None):
        """
        Given a metric and a dataframe with `predictions`, it iterates the dataframe in
        using `batch_size` and updates the metric. Once the dataframe is fully iterated,
        the score of the metric is returned as a dictionary containing
        {"group_name": group_name,
         "group_key": group_key,
         "metric": metric.name,
         "value": metric.result().numpy(),
         "size": predictions.shape[0]
        }
        When group_name and group_key are not provided, they are exluded from the dict.

        Parameters
        ----------
        metric : an instance of tf.keras.metrics
        predictions : pd.DataFrame
        batch_size: int
        group_name : str, optional
            The name to be used in the returned dictionary
        group_key : str, optional
            A key to be used in the returned dictionary

        Returns
        -------
        A dictionary with the metric scores for the given data.
        """
        label_name = self.feature_config.get_label()['name']
        output_name = self.output_name
        metric.reset_states()
        for chunk in self.get_chunks_from_df(predictions, batch_size):
            metric.update_state(tf.constant(chunk[label_name].values.tolist(), dtype=tf.float32),
                                tf.constant(chunk[output_name].values.tolist(), dtype=tf.float32))
        if group_name:
            return {"group_name": group_name,
                    "group_key": group_key,
                    "metric": metric.name,
                    "value": metric.result().numpy(),
                    "size": predictions.shape[0]}
        return {"metric": metric.name,
                "value": metric.result().numpy(),
                "size": predictions.shape[0]}

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
        if logs_dir:
            outfile = os.path.join(logs_dir, RelevanceModelConstants.MODEL_PREDICTIONS_CSV_FILE)
            # Delete file if it exists
            self.file_io.rm_file(outfile)
        predictions_df = self._create_prediction_dataframe(logging_frequency,
                                                           test_dataset)
        predictions_ = np.squeeze(self.model.predict(test_dataset))
        # Below, avoid doing predictions.tolist() as it explodes the memory
        # tolist() will create a list of lists, which consumes more memory
        # than a list on numpy arrays
        predictions_df[self.output_name] = [x for x in predictions_]
        if logs_dir:
            np.set_printoptions(formatter={'all':lambda x: str(x.decode('utf-8')) if isinstance(x, bytes) else str(x)},
                                linewidth=sys.maxsize, threshold=sys.maxsize)  # write the full vector in the csv not ...
            for col in predictions_df.columns:
                if isinstance(predictions_df[col].values[0], bytes):
                    predictions_df[col] = predictions_df[col].str.decode('utf8')
            predictions_df.to_csv(outfile, mode="w", header=True, index=False)
            self.logger.info(f"Model predictions written to: {outfile}")
        return predictions_df

    def _create_prediction_dataframe(self, logging_frequency, test_dataset):
        """
        Iterates through the test data and collects for each
        data point the information to be logged.
        """
        predictions = {}  # keys are features we logs, values lists
        # containing the values of this feature for each batch
        label_name = self.feature_config.get_label()["name"]
        features_to_log = [f.get("node_name", f["name"]) for f in
                           self.feature_config.get_features_to_log()] + [label_name]
        features_to_log = set(features_to_log)
        for batch_count, (x, y) in enumerate(test_dataset.take(-1)):  # returns (x, y) tuples
            if batch_count:
                for key in x.keys():
                    if key in features_to_log:  # only log if necessary
                        # Here I am appending, np.vstack or other numpy tricks will be slow
                        # as they create new arrays (a np.array require contiguous memory)
                        predictions[key].append(np.squeeze(x[key].numpy()))
                predictions[label_name].append(np.squeeze(y.numpy()))
            else:  # we initialize the {key: list()} in the 1st batch
                for key in x.keys():
                    if key in features_to_log:
                        predictions[key] = [np.squeeze(x[key].numpy())]
                predictions[label_name] = [np.squeeze(y.numpy())]
            if batch_count % logging_frequency == 0:
                self.logger.info(f"Finished evaluating {batch_count} batches")
        # This is a memory bottleneck; we bring everything in memory
        for key, val in predictions.items():
            # we want to create np.arrays that contain the full data here.
            # to do this, if something is 2-dim we use np.vstack
            # for 1-dim data we use np.hstack for the intended result
            predictions[key] = np.hstack(val) if len(predictions[key][0].shape) == 1 else np.vstack(
                val)
        predictions_df = pd.DataFrame({key: val if len(val.shape) == 1 else [inner for inner in val]
                                       for key, val in predictions.items()})
        return predictions_df
