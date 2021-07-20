import sys
import ast
import os
import pathlib
import pandas as pd
from scipy import stats
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


def kfold_analysis(original_logs_dir, run_id, num_folds, logger):
    """
    Aggregate results of the k-fold runs and perform t-test on the results.
    """
    rows = []
    pvalue_threshold = 0.1
    metrics = ['train_old_MRR', 'train_new_MRR', 'val_old_MRR', 'val_new_MRR', 'test_old_MRR', 'test_new_MRR']
    for i in range(num_folds):
        row = {"fold":i}
        logs_dir = pathlib.Path(original_logs_dir) / run_id / "fold_{}".format(i) / run_id
        with open(logs_dir/"_SUCCESS", 'r') as f:
            for line in f:
                parts = line.split(',')
                if parts[0] in metrics:
                    row[parts[0]] = float(parts[1])
        rows.append(row)
    results = pd.DataFrame(rows)
    logger.info(results)
    # calculate statistical paired t-test
    for dataset in ['train', 'val', 'test']:
        t_test_stat, pvalue = stats.ttest_rel(results["{}_old_MRR".format(dataset)], results["{}_new_MRR".format(dataset)])
        logger.info("{}_t_test_stat={}".format(dataset, t_test_stat))
        logger.info("{}_pvalue={} --> (statistically significant={})".format(dataset, pvalue, pvalue < pvalue_threshold))


def kfold_cross_validation_run(args):
    """
    Performs K-fold cross validation
    """
    rp = RankingPipeline(args=args)
    rp.logger.info("K-fold Cross Validation mode starting with k={}".format(args.kfold))
    rp.logger.info("Reading datasets ...")
    relevance_dataset = rp.get_relevance_dataset()
    rp.logger.info("Relevance Dataset created")
    if args.include_testset_in_kfold:
        all_data = relevance_dataset.train.concatenate(relevance_dataset.validation).concatenate(relevance_dataset.test)
    else:
        all_data = relevance_dataset.train.concatenate(relevance_dataset.validation)
    all_data = all_data.unbatch()
    all_data = all_data.shuffle(args.batch_size * 2)
    folds = args.kfold
    original_logs_dir = str(rp.args.logs_dir)
    original_models_dir = str(rp.args.models_dir)
    original_run_id = rp.run_id
    rp.logger.info("Starting K-fold Cross Validation with k={}".format(args.kfold))
    rp.logger.info("Include testset in the folds={}".format(str(args.include_testset_in_kfold)))
    for i in range(folds):  # indexes (folds)
        rp.logger.info("fold={}".format(i))
        logs_dir = pathlib.Path(original_logs_dir) / rp.args.run_id / "fold_{}".format(i)
        models_dir = pathlib.Path(original_models_dir) / rp.args.run_id / "fold_{}".format(i)
        args.logs_dir = pathlib.Path(logs_dir).as_posix()
        args.models_dir = pathlib.Path(models_dir).as_posix()
        training_idx = list(range(folds))
        if args.include_testset_in_kfold:
            validation = all_data.shard(folds, i)
            test_idx = i+1
            if i+1 >= folds:
                test_idx = 0
            test = all_data.shard(folds, test_idx)
            training_idx.remove(test_idx)
        else:
            validation = all_data.shard(folds, i)
        training_idx.remove(i)
        training = None
        for j in training_idx:
            if not training:
                training = all_data.shard(folds, j)
            else:
                training = training.concatenate(all_data.shard(folds, j))

        relevance_dataset.validation = validation.batch(args.batch_size, drop_remainder=False)
        relevance_dataset.train = training.batch(args.batch_size, drop_remainder=False)

        pipeline = RankingPipeline(args=args)
        pipeline.run(relevance_dataset=relevance_dataset)
    rp.local_io.rm_dir(os.path.join(rp.data_dir_local, "tfrecord"))
    kfold_analysis(original_logs_dir, original_run_id, folds, rp.logger)


def main(argv):
    # Define args
    args: Namespace = get_args(argv)
    if args.kfold > 1:
        # Run ml4ir with kfold cross validation
        kfold_cross_validation_run(args=args)
    else:
        # Initialize Relevance Pipeline and run in train/inference mode
        rp = RankingPipeline(args=args)
        rp.run()


if __name__ == "__main__":
    main(sys.argv[1:])
