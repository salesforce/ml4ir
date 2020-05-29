from ml4ir.base.config.keys import ScoringKey
from ml4ir.base.model.scoring.pointwise_scoring import PointwiseScoring
from ml4ir.base.model.scoring.pairwise_scoring import PairwiseScoring
from ml4ir.base.model.scoring.groupwise_scoring import GroupwiseScoring


def get_scoring_fn(scoring_key, model_config, loss_type):
    """
    Returns the scoring function based on the params
    """

    if scoring_key == ScoringKey.POINTWISE:
        scoring = PointwiseScoring(
            scoring_key=scoring_key, model_config=model_config, loss_type=loss_type
        )
    elif scoring_key == ScoringKey.PAIRWISE:
        scoring = PairwiseScoring(
            scoring_key=scoring_key, model_config=model_config, loss_type=loss_type
        )
    elif scoring_key == ScoringKey.GROUPWISE:
        scoring = GroupwiseScoring(
            scoring_key=scoring_key, model_config=model_config, loss_type=loss_type
        )
    else:
        raise NotImplementedError

    return scoring.get_scoring_fn()
