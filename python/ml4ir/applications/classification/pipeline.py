import os
import sys
import ast
from argparse import Namespace
import pandas as pd
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Optimizer
from tensorflow import data
from ml4ir.base.model.relevance_model import RelevanceModelConstants
from ml4ir.applications.classification.config.parse_args import get_args
from ml4ir.applications.classification.model.losses import categorical_cross_entropy
from ml4ir.applications.classification.model.metrics import metrics_factory
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.base.features.preprocessing import get_one_hot_label_vectorizer
from ml4ir.base.model.losses.loss_base import RelevanceLossBase
from ml4ir.base.model.optimizers.optimizer import get_optimizer
from ml4ir.base.model.relevance_model import RelevanceModel
from ml4ir.base.model.scoring.scoring_model import ScorerBase, RelevanceScorer
from ml4ir.base.model.scoring.interaction_model import InteractionModel, UnivariateInteractionModel
from ml4ir.base.pipeline import RelevancePipeline

from typing import Union, List, Type, Optional


class ClassificationPipeline(RelevancePipeline):
    """Base class that defines a pipeline to train, evaluate and save
    a RelevanceModel for classification using ml4ir"""

    def __init__(self, args: Namespace):
        """
        Constructor to create a RelevancePipeline object to train, evaluate
        and save a model on ml4ir.
        This method sets up data, logs, models directories, file handlers used.
        The method also loads and sets up the FeatureConfig for the model training
        pipeline

        Parameters
        ----------
        args: argparse Namespace
            arguments to be used with the pipeline.
            Typically, passed from command line arguments
        """
        self.loss_key = args.loss_key
        super().__init__(args)

    def get_relevance_model(self, feature_layer_keys_to_fns={}) -> RelevanceModel:
        """
        Creates a RelevanceModel that can be used for training and evaluating

        Parameters
        ----------
        feature_layer_keys_to_fns : dict of (str, function)
            dictionary of function names mapped to tensorflow compatible
            function definitions that can now be used in the InteractionModel
            as a feature function to transform input features

        Returns
        -------
        `RelevanceModel`
            RelevanceModel that can be used for training and evaluating
            a classification model

        Notes
        -----
        Override this method to create custom loss, scorer, model objects
        """

        # Define interaction model
        interaction_model: InteractionModel = UnivariateInteractionModel(
            feature_config=self.feature_config,
            feature_layer_keys_to_fns=feature_layer_keys_to_fns,
            tfrecord_type=self.tfrecord_type,
            file_io=self.file_io,
        )

        # Define loss object from loss key
        loss: RelevanceLossBase = categorical_cross_entropy.get_loss(loss_key=self.loss_key)

        # Define scorer
        scorer: ScorerBase = RelevanceScorer.from_model_config_file(
            model_config_file=self.model_config_file,
            feature_config=self.feature_config,
            interaction_model=interaction_model,
            loss=loss,
            output_name=self.args.output_name,
            logger=self.logger,
            file_io=self.file_io,
        )

        # Define metrics objects from metrics keys
        metrics: List[Union[Type[Metric], str]] = [
            metrics_factory.get_metric(metric_key=metric_key) for metric_key in self.metrics_keys
        ]

        # Define optimizer
        optimizer: Optimizer = get_optimizer(
            model_config_file=self.model_config_file, file_io=self.file_io,
        )

        # Combine the above to define a RelevanceModel
        relevance_model: RelevanceModel = RelevanceModel(
            feature_config=self.feature_config,
            scorer=scorer,
            metrics=metrics,
            optimizer=optimizer,
            tfrecord_type=self.tfrecord_type,
            model_file=self.args.model_file,
            initialize_layers_dict=ast.literal_eval(self.args.initialize_layers_dict),
            freeze_layers_list=ast.literal_eval(self.args.freeze_layers_list),
            compile_keras_model=self.args.compile_keras_model,
            output_name=self.args.output_name,
            file_io=self.local_io,
            logger=self.logger,
        )
        return relevance_model

    def get_relevance_dataset(
        self, parse_tfrecord=True, preprocessing_keys_to_fns={}
    ) -> RelevanceDataset:
        """
        Create RelevanceDataset object by loading train, test data as tensorflow datasets
        Defines a preprocessing feature function to one hot vectorize
        classification labels

        Parameters
        ----------
        preprocessing_keys_to_fns : dict of (str, function)
            dictionary of function names mapped to function definitions
            that can now be used for preprocessing while loading the
            TFRecordDataset to create the RelevanceDataset object

        Returns
        -------
        `RelevanceDataset` object
            RelevanceDataset object that can be used for training and evaluating
            the model

        Notes
        -----
        Override this method to create custom dataset objects
        """
        # Adding one_hot_vectorizer needed for classification
        preprocessing_keys_to_fns = {
            "one_hot_vectorize_label": get_one_hot_label_vectorizer(
                self.feature_config.get_label(), self.file_io
            )
        }

        # Prepare Dataset
        relevance_dataset = RelevanceDataset(
            data_dir=self.data_dir_local,
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
            parse_tfrecord=parse_tfrecord,
            file_io=self.local_io,
            logger=self.logger,
        )

        return relevance_dataset

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
        relevance_model = self.get_relevance_model()
        if not relevance_model.is_compiled:
            return NotImplementedError

        group_metrics_keys = self.feature_config.get_group_metrics_keys()

        metrics_dict = relevance_model.model.evaluate(test_dataset)
        metrics_dict = dict(zip(relevance_model.model.metrics_names, metrics_dict))

        predictions = relevance_model.predict(test_dataset,
                                              inference_signature=inference_signature,
                                              additional_features=additional_features,
                                              logs_dir=None,  # Return pd.DataFrame of predictions
                                              logging_frequency=logging_frequency)
        # Need to convert predictions from tuples of tf.Tensor to lists of int/float
        # so that we are compatible with tf.keras.metrics
        label_name = self.feature_config.get_label()['name']
        output_name = relevance_model.output_name
        predictions[label_name] = predictions[label_name].apply(lambda l: [item.numpy() for item in l])
        predictions[output_name] = predictions[output_name].apply(lambda l: [item.numpy() for item in l])

        global_metrics = []  # group_name, metric, value
        grouped_metrics = []
        for metric in relevance_model.model.metrics:
            metric.reset_states()
            metric.update_state(predictions[label_name].values.tolist(), predictions[output_name].values.tolist())
            global_metrics.append({"metric": metric.name,
                                   "value": metric.result().numpy()})
            for group_ in group_metrics_keys:
                for name, group in predictions.groupby(group_['name']):
                    if group.shape[0] >= group_metrics_min_queries:
                        metric.reset_states()
                        metric.update_state(group[label_name].values.tolist(),
                                            group[output_name].values.tolist())
                        grouped_metrics.append({"group_name": group_['name'],
                                                "group_key": name,
                                                "metric": metric.name,
                                                "value": metric.result().numpy(),
                                                "size": group.shape[0]})
        global_metrics = pd.DataFrame(global_metrics)
        grouped_metrics = pd.DataFrame(grouped_metrics)
        if logs_dir:
            self.file_io.write_df(
                grouped_metrics,
                outfile=os.path.join(logs_dir, RelevanceModelConstants.GROUP_METRICS_CSV_FILE),
                )
            self.file_io.write_df(
                global_metrics,
                outfile=os.path.join(logs_dir, RelevanceModelConstants.METRICS_CSV_FILE)
            )
        return global_metrics, grouped_metrics, metrics_dict


def main(argv):
    # Define args
    args: Namespace = get_args(argv)

    # Initialize Relevance Pipeline and run in train/inference mode
    rp = ClassificationPipeline(args=args)
    rp.run()


if __name__ == "__main__":
    main(sys.argv[1:])
