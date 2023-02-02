import sys
from argparse import Namespace
from typing import Union, List

from ml4ir.applications.classification.config.parse_args import get_args
from ml4ir.applications.classification.model.classification_model import ClassificationModel
from ml4ir.applications.classification.model.losses import categorical_cross_entropy
from ml4ir.applications.classification.model.metrics import metrics_factory
from ml4ir.base.data.kfold_relevance_dataset import KfoldRelevanceDataset
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.base.features.preprocessing import get_one_hot_label_vectorizer
from ml4ir.base.pipeline import RelevancePipeline
from tensorflow.keras.metrics import Metric


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

    def get_relevance_model_cls(self):
        """
        Fetch the class of the RelevanceModel to be used for the ml4ir pipeline

        Returns
        -------
        RelevanceModel class
        """
        return ClassificationModel

    def get_loss(self):
        """
        Get the primary loss function to be used with the RelevanceModel

        Returns
        -------
        RelevanceLossBase object
        """
        return categorical_cross_entropy.get_loss(loss_key=self.loss_key,
                                                  output_name=self.args.output_name)

    @staticmethod
    def get_metrics(metrics_keys: List[str]) -> List[Union[Metric, str]]:
        """
        Get the list of keras metrics to be used with the RelevanceModel

        Parameters
        ----------
        metrics_keys: List of str
            List of strings indicating the metrics to instantiate and retrieve

        Returns
        -------
        list of keras Metric objects
        """
        return [
            metrics_factory.get_metric(metric_key=metric_key) for metric_key in metrics_keys
        ]

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

    def get_kfold_relevance_dataset(
            self,
            num_folds,
            include_testset_in_kfold,
            read_data_sets=False,
            parse_tfrecord=True,
            preprocessing_keys_to_fns={}
    ) -> KfoldRelevanceDataset:
        """
        Create KfoldRelevanceDataset object by loading train, test data as tensorflow datasets
        Defines a preprocessing feature function to one hot vectorize
        classification labels

        Parameters
        ----------
        num_folds: int
            Number of folds in kfold CV
        include_testset_in_kfold: bool
            Whether to include testset in the folds
        read_data_sets: bool
            Whether to read datasets from disk
        preprocessing_keys_to_fns : dict of (str, function)
            dictionary of function names mapped to function definitions
            that can now be used for preprocessing while loading the
            TFRecordDataset to create the KfoldRelevanceDataset object

        Returns
        -------
        `KfoldRelevanceDataset` object
            KfoldRelevanceDataset object that can be used for training and evaluating
            the model in a kfold cross validation mode.

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
        relevance_dataset = KfoldRelevanceDataset(
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
            parse_tfrecord=True,
            file_io=self.local_io,
            logger=self.logger,
            non_zero_features_only=self.non_zero_features_only,
            keep_additional_info=self.keep_additional_info,
            num_folds=num_folds,
            include_testset_in_kfold=include_testset_in_kfold,
            read_data_sets=read_data_sets
        )

        return relevance_dataset

    def create_pipeline_for_kfold(self, args):
        """
        Create a ClassificationPipeline object used in running kfold cross validation.
        """
        return ClassificationPipeline(args=args)

    def run_kfold_analysis(self, base_logs_dir, base_run_id, num_folds, metrics):
        # TODO: Implement the kfold CV analysis for classification
        self.logger.warning("Kfold analysis is not implemented for Classification pipeline")


def main(argv):
    # Define args
    args: Namespace = get_args(argv)

    # Initialize Relevance Pipeline and run in train/inference mode
    rp = ClassificationPipeline(args=args)
    rp.run()


if __name__ == "__main__":
    main(sys.argv[1:])