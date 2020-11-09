import os
import numpy as np
import pandas as pd
from tensorflow import data
from ml4ir.base.model.relevance_model import RelevanceModelConstants
from ml4ir.base.model.relevance_model import RelevanceModel
from typing import Optional


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
    ):
        """
        Evaluate the Classification Model

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
        predictions = self.predict(
            test_dataset,
            inference_signature=inference_signature,
            additional_features=additional_features,
            logs_dir=None,
            logging_frequency=logging_frequency,
        )

        label_name = self.feature_config.get_label()["name"]
        output_name = self.output_name
        global_metrics = []  # group_name, metric, value
        grouped_metrics = []
        for metric in self.model.metrics:
            metric.reset_states()
            metric.update_state(
                predictions[label_name].values.tolist(), predictions[output_name].values.tolist()
            )
            global_metrics.append({"metric": metric.name, "value": metric.result().numpy()})
            for group_ in group_metrics_keys:
                for name, group in predictions.groupby(group_["name"]):
                    if group.shape[0] >= group_metrics_min_queries:
                        metric.reset_states()
                        metric.update_state(
                            group[label_name].values.tolist(), group[output_name].values.tolist()
                        )
                        grouped_metrics.append(
                            {
                                "group_name": group_["name"],
                                "group_key": name,
                                "metric": metric.name,
                                "value": metric.result().numpy(),
                                "size": group.shape[0],
                            }
                        )
        global_metrics = pd.DataFrame(global_metrics)
        grouped_metrics = pd.DataFrame(grouped_metrics).sort_values(by="size")
        if logs_dir:
            self.file_io.write_df(
                grouped_metrics,
                outfile=os.path.join(logs_dir, RelevanceModelConstants.GROUP_METRICS_CSV_FILE),
                index=False,
            )
            self.file_io.write_df(
                global_metrics,
                outfile=os.path.join(logs_dir, RelevanceModelConstants.METRICS_CSV_FILE),
                index=False,
            )
            self.logger.info(f"Evaluation Results written at: {logs_dir}")
        return global_metrics, grouped_metrics, metrics_dict

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

        predictions_df_list = list()
        for (x, y) in test_dataset.take(-1):  # returns (x, y) tuples
            batch_predictions = pd.DataFrame(
                {key: np.squeeze(x[key].numpy()).tolist() for key in x.keys()}
            )
            batch_predictions[self.feature_config.get_label()["name"]] = np.squeeze(
                y.numpy()
            ).tolist()
            predictions_df_list.append(batch_predictions)
        predictions_df = pd.concat(predictions_df_list)
        predictions_df[self.output_name] = np.squeeze(self.model.predict(test_dataset)).tolist()
        features_to_log = [
            f.get("node_name", f["name"]) for f in self.feature_config.get_features_to_log()
        ] + [self.output_name]
        predictions_df = predictions_df[features_to_log]
        if logs_dir:
            predictions_df.to_csv(outfile, mode="w", header=True, index=False)
            self.logger.info("Model predictions written to -> {}".format(outfile))

        return predictions_df
