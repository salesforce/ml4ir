import sys
import ast
from argparse import Namespace
from tensorflow.keras.metrics import Metric
from tensorflow.keras.optimizers import Optimizer

from ml4ir.base.pipeline import RelevancePipeline
from ml4ir.base.config.keys import ArchitectureKey
from ml4ir.base.model.relevance_model import RelevanceModel
from ml4ir.base.model.losses.loss_base import RelevanceLossBase
from ml4ir.base.model.scoring.scoring_model import ScorerBase, RelevanceScorer
from ml4ir.base.model.scoring.interaction_model import InteractionModel, UnivariateInteractionModel
from ml4ir.base.model.optimizers.optimizer import get_optimizer
from ml4ir.applications.ranking.model.ranking_model import RankingModel, LinearRankingModel
from ml4ir.applications.ranking.config.keys import LossKey
from ml4ir.applications.ranking.config.keys import MetricKey
from ml4ir.applications.ranking.config.keys import ScoringTypeKey
from ml4ir.applications.ranking.model.losses import loss_factory
from ml4ir.applications.ranking.model.metrics import metric_factory
from ml4ir.applications.ranking.config.parse_args import get_args

from typing import Union, List, Type


class RankingPipeline(RelevancePipeline):
    """Base class that defines a pipeline to train, evaluate and save
    a RankingModel using ml4ir"""

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
        self.scoring_type = args.scoring_type
        self.loss_type = args.loss_type

        super().__init__(args)

    def get_relevance_model(self, feature_layer_keys_to_fns={}) -> RelevanceModel:
        """
        Creates a RankingModel that can be used for training and evaluating

        Parameters
        ----------
        feature_layer_keys_to_fns : dict of (str, function)
            dictionary of function names mapped to tensorflow compatible
            function definitions that can now be used in the InteractionModel
            as a feature function to transform input features

        Returns
        -------
        `RankingModel`
            RankingModel that can be used for training and evaluating
            a ranking model

        Notes
        -----
        Override this method to create custom loss, scorer, model objects
        """

        # Define interaction model
        interaction_model: InteractionModel = UnivariateInteractionModel(
            feature_config=self.feature_config,
            feature_layer_keys_to_fns=feature_layer_keys_to_fns,
            tfrecord_type=self.tfrecord_type,
            max_sequence_size=self.args.max_sequence_size,
            file_io=self.file_io,
        )

        # Define loss object from loss key
        loss: RelevanceLossBase = loss_factory.get_loss(
            loss_key=self.loss_key, scoring_type=self.scoring_type
        )

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
            metric_factory.get_metric(metric_key=metric_key) for metric_key in self.metrics_keys
        ]

        optimizer: Optimizer = get_optimizer(model_config=self.model_config)

        # Combine the above to define a RelevanceModel
        if self.model_config["architecture_key"] == ArchitectureKey.LINEAR:
            RankingModelClass = LinearRankingModel
        else:
            RankingModelClass = RankingModel
        relevance_model: RelevanceModel = RankingModelClass(
            feature_config=self.feature_config,
            tfrecord_type=self.tfrecord_type,
            scorer=scorer,
            metrics=metrics,
            optimizer=optimizer,
            model_file=self.model_file,
            initialize_layers_dict=ast.literal_eval(self.args.initialize_layers_dict),
            freeze_layers_list=ast.literal_eval(self.args.freeze_layers_list),
            compile_keras_model=self.args.compile_keras_model,
            output_name=self.args.output_name,
            file_io=self.local_io,
            logger=self.logger,
        )

        return relevance_model

    def validate_args(self):
        """
        Validate the arguments to be used with RelevancePipeline
        """
        super().validate_args()

        if self.loss_key not in LossKey.get_all_keys():
            raise Exception(
                "Loss specified [{}] is not one of : {}".format(
                    self.loss_key, LossKey.get_all_keys()
                )
            )

        for metric_key in self.metrics_keys:
            if metric_key not in MetricKey.get_all_keys():
                raise Exception(
                    "Metric specified [{}] is not one of : {}".format(
                        metric_key, MetricKey.get_all_keys()
                    )
                )

        if self.scoring_type not in ScoringTypeKey.get_all_keys():
            raise Exception(
                "Scoring type specified [{}] is not one of : {}".format(
                    self.scoring_type, ScoringTypeKey.get_all_keys()
                )
            )


def main(argv):
    # Define args
    args: Namespace = get_args(argv)

    # Initialize Relevance Pipeline and run in train/inference mode
    rp = RankingPipeline(args=args)
    rp.run()


if __name__ == "__main__":
    main(sys.argv[1:])
