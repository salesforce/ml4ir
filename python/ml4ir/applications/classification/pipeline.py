import sys
import ast
from argparse import Namespace

from tensorflow.keras.metrics import Metric, Precision
from tensorflow.keras.optimizers import Optimizer

from ml4ir.applications.classification.config.parse_args import get_args
from ml4ir.applications.classification.model.losses import categorical_cross_entropy
from ml4ir.applications.ranking.model.metrics import metric_factory
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.base.features.preprocessing import get_one_hot_label_vectorizer
from ml4ir.base.model.losses.loss_base import RelevanceLossBase
from ml4ir.base.model.optimizer import get_optimizer
from ml4ir.base.model.relevance_model import RelevanceModel
from ml4ir.base.model.scoring.scoring_model import ScorerBase, RelevanceScorer
from ml4ir.base.model.scoring.interaction_model import InteractionModel, UnivariateInteractionModel
from ml4ir.base.pipeline import RelevancePipeline

from typing import Union, List, Type


class ClassificationPipeline(RelevancePipeline):
    """
    Pipeline defining the classification models.
    """

    def __init__(self, args: Namespace):
        self.loss_key = args.loss_key
        super().__init__(args)

    def get_relevance_model(self, feature_layer_keys_to_fns={}) -> RelevanceModel:
        """
        Creates RelevanceModel suited for classification use-case.

        NOTE: Override this method to create custom loss, scorer, model objects.
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
            metric_factory.get_metric(metric_key=metric_key) for metric_key in self.metrics_keys
        ]

        # Define optimizer
        optimizer: Optimizer = get_optimizer(
            optimizer_key=self.optimizer_key,
            learning_rate=self.args.learning_rate,
            learning_rate_decay=self.args.learning_rate_decay,
            learning_rate_decay_steps=self.args.learning_rate_decay_steps,
            gradient_clip_value=self.args.gradient_clip_value,
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
        Creates RelevanceDataset

        NOTE: Override this method to create custom dataset objects
        """
        # Adding one_hot_vectorizer needed for classification.
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
