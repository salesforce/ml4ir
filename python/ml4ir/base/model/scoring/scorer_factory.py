from ml4ir.base.model.scoring.scoring_model import RelevanceScorer
from ml4ir.base.model.scoring.monte_carlo_scorer import MonteCarloScorer
from ml4ir.base.config.keys import MonteCarloInferenceKey


def get_scorer(model_config, feature_config, interaction_model, loss, aux_loss, aux_loss_weight,
               aux_metrics, output_name, logger, file_io, logs_dir):
    """
    Parameters:

        model_config: dict
            the model configuration.
        feature_config: dict
            the feature configuration.
        interaction_model: object
            the interaction model.
        loss: object
            the loss function.
        aux_loss: object
            the auxiliary loss function.
        aux_metrics: list
            the auxiliary metrics.
        output_name: str
            The name of the output node
        logger: object
            the logger.
        file_io: object
            the file I/O.
        logs_dir: str
            the local logs directory.

    Returns:
        scorer: object, a scorer based on the provided configurations and parameters.
    """

    if (MonteCarloInferenceKey.MONTE_CARLO_INFERENCE_TRIALS in model_config and
            model_config[MonteCarloInferenceKey.MONTE_CARLO_INFERENCE_TRIALS].get(MonteCarloInferenceKey.NUM_TRIALS,
                                                                                  0)):
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
