import sys
import ast
from argparse import Namespace
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Optimizer
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
from ml4ir.applications.classification.model.classification_model import ClassificationModel
from typing import Union, List, Type


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
        scorer: ScorerBase = RelevanceScorer(
            feature_config=self.feature_config,
            model_config=self.model_config,
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
        optimizer: Optimizer = get_optimizer(model_config=self.model_config)

        # Combine the above to define a RelevanceModel
        relevance_model: RelevanceModel = ClassificationModel(
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


def main(argv):
    # Define args
    args: Namespace = get_args(argv)

    # Initialize Relevance Pipeline and run in train/inference mode
    rp = ClassificationPipeline(args=args)
    rp.run()


if __name__ == "__main__":
    main(sys.argv[1:])
