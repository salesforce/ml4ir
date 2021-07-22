import ast
from argparse import ArgumentParser, Namespace, Action
from typing import List
from ml4ir.base.config.keys import OptimizerKey, DataFormatKey, TFRecordTypeKey, ExecutionModeKey, ServingSignatureKey, FileHandlerKey
from ml4ir.applications.ranking.config.keys import LossKey as RankingLoss, MetricKey as RankingMetricKey
from ml4ir.applications.classification.config.keys import LossKey as ClassificationLoss, MetricKey as ClassificationMetricKey


class CustomFeatureDictUpdater(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        """
        Add a value to a custom argument dictionary.
        If dictionary does not exist a new one is created.
        If dictionary already exists, it is updated with new key-value pair.

        Can be used to collect all custom arguments for feature_config
        by specifying as feature_config.custom_arg 128. This will update the
        dictionary for feature_config custom args with the new key "custom_arg"
        and corresponding value "128".

        Parameters
        ----------
        parser: ArgumentParser
            ArgumentParser to use this action with
        namespace: Namespace
            The namespace object that will be updated with the parsed value
        values: str
            Value to be added to dictionary
        option_string: str
            Name of command line argument the action is used with

        Returns
        -------
        dict
            Dictionary with the updated key value pair
        """
        old_custom_args = getattr(namespace, self.dest)
        new_custom_args = {".".join(option_string.split(".")[1:]): values}
        if isinstance(old_custom_args, dict):
            new_custom_args.update(old_custom_args)
        setattr(namespace, self.dest, new_custom_args)


class RelevanceArgParser(ArgumentParser):
    """Defines the parser for the command line arguments for RelevancePipeline"""

    def __init__(self, allow_abbrev=False, **kwargs):
        super().__init__(allow_abbrev=allow_abbrev, **kwargs)
        self.define_args()
        self.set_default_args()


    def define_args(self):
        self.add_argument(
            "--data_dir",
            type=str,
            help="Path to the data directory to be used for training and inference. "
            "Can optionally include train/ val/ and test/ subdirectories. "
            "If subdirectories are not present, data will be split based on train_pcent_split",
        )

        self.add_argument(
            "--data_format",
            type=str,
            choices=DataFormatKey.get_all_keys(),
            default="tfrecord",
            help="Format of the data to be used. "
            "Supported Data Formats  are specified in ml4ir/base/config/keys.py",
        )

        self.add_argument(
            "--tfrecord_type",
            type=str,
            choices=TFRecordTypeKey.get_all_keys(),
            default="example",
            help="TFRecord type of the data to be used. "
            "Supported TFRecord type are specified in ml4ir/base/config/keys.py",
        )

        self.add_argument(
            "--feature_config",
            type=str,
            help="Path to YAML file or YAML string with feature metadata for training.",
        )

        self.add_argument(
            "--model_file",
            type=str,
            default=None,
            required=False,
            help="Path to a pretrained model to load for either resuming training or for running in"
            "inference mode.",
        )

        self.add_argument(
            "--model_config",
            type=str,
            default="ml4ir/base/config/default_model_config.yaml",
            help="Path to the Model config YAML used to build the model architecture.",
        )

        self.add_argument(
            "--loss_key",
            type=str,
            choices=RankingLoss.get_all_keys() + ClassificationLoss.get_all_keys(),
            help="Loss to optimize."
        )

        self.add_argument(
            "--metrics_keys",
            type=str,
            nargs="+",
            default=None,
            choices=RankingMetricKey.get_all_keys() + ClassificationMetricKey.get_all_keys(),
            help="A space separated list of metrics to compute.",
        )

        self.add_argument(
            "--monitor_metric",
            type=str,
            default=None,
            choices=RankingMetricKey.get_all_keys() + ClassificationMetricKey.get_all_keys(),
            help="Metric name to use for monitoring training loop in callbacks.",
        )

        self.add_argument(
            "--monitor_mode",
            type=str,
            default=None,
            help="Metric mode to use for monitoring training loop in callbacks",
        )

        self.add_argument(
            "--num_epochs",
            type=int,
            default=5,
            help="Max number of training epochs(or full pass over the data)",
        )

        self.add_argument(
            "--batch_size",
            type=int,
            default=128,
            help="Number of data samples to use per batch.",
        )

        self.add_argument(
            "--compute_intermediate_stats",
            type=ast.literal_eval,
            default=True,
            help="Whether to compute intermediate stats on test set (mrr, acr, etc) (slow)",
        )

        self.add_argument(
            "--execution_mode",
            type=str,
            choices=ExecutionModeKey.get_all_keys(),
            default="train_inference_evaluate",
            help="Execution mode for the pipeline.",
        )

        self.add_argument(
            "--random_state",
            type=int,
            default=123,
            help="Initialize the seed to control randomness for replication",
        )

        self.add_argument(
            "--run_id",
            type=str,
            default="",
            help="Unique string identifier for the current training run. "
                 "Used to identify logs and models directories. "
                 "Autogenerated if not specified.",
        )

        self.add_argument(
            "--run_group",
            type=str,
            default="general",
            help="Unique string identifier to group multiple model training runs."
                 " Allows for defining a meta grouping to filter different model "
                 "training runs for best model selection as a post step.",
        )

        self.add_argument(
            "--run_notes",
            type=str,
            default="",
            help="Notes for the current training run. Use this argument "
                 "to add short description of the model training run that "
                 "helps in identifying the run later.",
        )

        self.add_argument(
            "--models_dir",
            type=str,
            default="models/",
            help="Path to save the model. Will be expanded to models_dir/run_id",
        )

        self.add_argument(
            "--logs_dir",
            type=str,
            default="logs/",
            help="Path to save the training/inference logs. "
                 "Will be expanded to logs_dir/run_id",
        )

        self.add_argument(
            "--checkpoint_model",
            type=ast.literal_eval,
            default=True,
            help="Whether to save model checkpoints at the end of each epoch. Recommended - set to True",
        )

        self.add_argument(
            "--train_pcent_split",
            type=float,
            default=0.8,
            help="Percentage of all data to be used for training. The remaining is used for validation and "
            "testing. Remaining data is split in half if val_pcent_split or test_pcent_split are not "
            "specified.",
        )

        self.add_argument(
            "--val_pcent_split",
            type=float,
            default=-1,
            help="Percentage of all data to be used for testing.",
        )

        self.add_argument(
            "--test_pcent_split",
            type=float,
            default=-1,
            help="Percentage of all data to be used for testing.",
        )

        self.add_argument(
            "--max_sequence_size",
            type=int,
            default=0,
            help="Maximum number of elements per sequence feature.",
        )

        self.add_argument(
            "--inference_signature",
            type=str,
            choices=ServingSignatureKey.get_all_keys(),
            default="serving_default",
            help="SavedModel signature to be used for inference",
        )

        self.add_argument(
            "--use_part_files",
            type=ast.literal_eval,
            default=False,
            help="Whether to look for part files while loading data",
        )

        self.add_argument(
            "--logging_frequency",
            type=int,
            default=25,
            help="How often to log results to log file. Int representing number of batches.",
        )

        self.add_argument(
            "--group_metrics_min_queries",
            type=int,
            default=None,
            help="Minimum number of queries per group to be used to computed groupwise metrics.",
        )

        self.add_argument(
            "--compile_keras_model",
            type=ast.literal_eval,
            default=False,
            help="Whether to compile a loaded SavedModel into a Keras model. "
            "NOTE: This requires that the SavedModel's architecture, loss, metrics, etc are the same as the RankingModel"
            "If that is not the case, then you can still use a SavedModel from a model_file for inference/evaluation only",
        )

        self.add_argument(
            "--use_all_fields_at_inference",
            type=ast.literal_eval,
            default=False,
            help="Whether to require all fields in the serving signature of the SavedModel."
                 " If set to False, only requires fields with required_only=True",
        )

        self.add_argument(
            "--pad_sequence_at_inference",
            type=ast.literal_eval,
            default=False,
            help="Whether to pad sequence at inference time. "
                 "Used to define the TFRecord serving signature in the SavedModel",
        )

        self.add_argument(
            "--output_name",
            type=str,
            default="relevance_score",
            help="Name of the output node of the model",
        )

        self.add_argument(
            "--early_stopping_patience",
            type=int,
            default=2,
            help="How many epochs to wait before early stopping on metric degradation",
        )

        self.add_argument(
            "--file_handler",
            type=str,
            default="local",
            choices=FileHandlerKey.get_all_keys(),
            help="String specifying the file handler to be used.",
        )

        self.add_argument(
            "--initialize_layers_dict",
            type=str,
            default="{}",
            help="Dictionary of pretrained layers to be loaded."
            "The key is the name of the layer to be assigned the pretrained weights."
            "The value is the path to the pretrained weights.",
        )

        self.add_argument(
            "--freeze_layers_list",
            type=str,
            default="[]",
            help="List of layer names that are to be frozen instead of training."
            "Usually coupled with initialize_layers_dict to load pretrained weights and freeze them",
        )

        self.add_argument(
            "--non_zero_features_only",
            type=ast.literal_eval,
            default=False,
            help="[Ranklib format only] Only non zero features are stored.",
        )

        self.add_argument(
            "--keep_additional_info",
            type=ast.literal_eval,
            default=False,
            help="[Ranklib format only] Option to keep additional info "
                 "(All info after the '#' in the format [key = val]).",
        )

    def set_default_args(self):
        pass

    def parse_args(self, args: List[str]) -> Namespace:
        """
        Parse command line arguments passed as a list of strings.
        Additionally, handles dynamic arguments for feature_config
        and model_config with prefixes feature_config. and model_config.
        respectively

        Parameters
        ----------
        args: list of str
            List of command line args to be parsed

        Returns
        -------
        Namespace
            Namespace object obtained by parsing the input
        """
        dynamic_args = self.parse_known_args(args)[1]

        for i in range(int(len(dynamic_args) / 2)):
            key = dynamic_args[i * 2]
            if key.split(".")[0] not in {"--feature_config", "--model_config"}:
                raise KeyError(
                    "Dynamic arguments currently supported must have the prefix feature_config. or model_config., but found: {}".format(key))
            dest = "{}_custom".format(key.split(".")[0]).replace("--", "")
            self.add_argument(key,
                              dest=dest,
                              action=CustomFeatureDictUpdater)

        return super(RelevanceArgParser, self).parse_args(args)


def get_args(args: List[str]) -> Namespace:
    return RelevanceArgParser().parse_args(args)
