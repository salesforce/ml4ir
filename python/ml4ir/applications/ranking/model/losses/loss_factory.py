from ml4ir.base.model.losses.loss_base import RelevanceLossBase
import ml4ir.applications.ranking.model.losses.pointwise_losses as pointwise_losses
import ml4ir.applications.ranking.model.losses.pairwise_losses as pairwise_losses
import ml4ir.applications.ranking.model.losses.listwise_losses as listwise_losses
from ml4ir.applications.ranking.config.keys import LossKey


def get_loss(loss_key, scoring_type) -> RelevanceLossBase:
    """
    Factor method to get Loss function object

    Parameters
    ----------
    loss_key : str
        Name of the loss function as specified by LossKey
    scoring_type : str
        Type of scoring function - pointwise, pairwise, groupwise

    Returns
    -------
    RelevanceLossBase
        RelevanceLossBase object that applies the final activation layer
        and computes the loss function from the model score
    """
    if loss_key == LossKey.SIGMOID_CROSS_ENTROPY:
        return pointwise_losses.SigmoidCrossEntropy(loss_key=loss_key, scoring_type=scoring_type)
    elif loss_key == LossKey.RANK_ONE_LISTNET:
        return listwise_losses.RankOneListNet(loss_key=loss_key, scoring_type=scoring_type)
    else:
        raise NotImplementedError
