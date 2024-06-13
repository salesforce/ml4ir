from ml4ir.base.model.scoring.scoring_model import RelevanceScorer
from ml4ir.base.model.scoring.monte_carlo_scorer import MonteCarloScorer
from ml4ir.base.config.keys import MonteCarloInferenceKey


def get_scorer(model_config: dict,
               feature_config: FeatureConfig,
               interaction_model: InteractionModel,
               loss: RelevanceLossBase,
               file_io: FileIO,
               aux_loss: Optional[RelevanceLossBase] = None,
               aux_loss_weight: float = 0.0,
               aux_metrics: Optional[List[Union[Metric, str]]] = None,
               output_name: str = "score",
               logger: Optional[Logger] = None,
               logs_dir: Optional[str] = "",
               **kwargs
               ):
    """
    Factory method for creating a RelevanceScorer object

    Parameters
    ----------
    model_config : dict
        Dictionary defining the model layer configuration
    feature_config : `FeatureConfig` object
        FeatureConfig object defining the features and their configurations
    interaction_model : `InteractionModel` object
        InteractionModel that defines the feature transformation layers
        on the input model features
    loss : `RelevanceLossBase` object
        Relevance loss object that defines the final activation layer
        and the loss function for the model
    file_io : `FileIO` object
        FileIO object that handles read and write
    aux_loss : `RelevanceLossBase` object
        Auxiliary loss to be used in conjunction with the primary loss
    aux_loss_weight: float
        Floating point number in [0, 1] to indicate the proportion of the auxiliary loss
        in the total final loss value computed using a linear combination
        total loss = (1 - aux_loss_weight) * loss + aux_loss_weight * aux_loss
    aux_metrics: List of keras.metrics.Metric
        Keras metric list to be computed on the aux label
    eval_config: Dict
        Dictionary specifying the evaluation specifications
    output_name : str, optional
        Name of the output that captures the score computed by the model
    logger : Logger, optional
        Logging handler
    logs_dir : str, optional
        Path to the logging directory

    Notes
    -----
    logs_dir : Used to point model architectures to local logging directory,
        primarily for saving visualizations.
    """

    if (MonteCarloInferenceKey.MONTE_CARLO_INFERENCE_TRIALS in model_config and
            (model_config[MonteCarloInferenceKey.MONTE_CARLO_INFERENCE_TRIALS].get(MonteCarloInferenceKey.NUM_TEST_TRIALS, 0) or
            model_config[MonteCarloInferenceKey.MONTE_CARLO_INFERENCE_TRIALS].get(MonteCarloInferenceKey.NUM_TRAINING_TRIALS, 0))):
              
        scorer = MonteCarloScorer(
            feature_config=feature_config,
            model_config=model_config,
            interaction_model=interaction_model,
            loss=loss,
            aux_loss=aux_loss,
            aux_loss_weight=aux_loss_weight,
            aux_metrics=aux_metrics,
            output_name=output_name,
            logger=logger,
            file_io=file_io,
            logs_dir=logs_dir
        )
    else:
        scorer = RelevanceScorer(
            feature_config=feature_config,
            model_config=model_config,
            interaction_model=interaction_model,
            loss=loss,
            aux_loss=aux_loss,
            aux_loss_weight=aux_loss_weight,
            aux_metrics=aux_metrics,
            output_name=output_name,
            logger=logger,
            file_io=file_io,
            logs_dir=logs_dir
        )

    return scorer
