from ml4ir.base.model.losses.loss_base import RelevanceLossBase
import ml4ir.applications.ranking.model.losses.pointwise_losses as pointwise_losses
import ml4ir.applications.ranking.model.losses.pairwise_losses as pairwise_losses
import ml4ir.applications.ranking.model.losses.listwise_losses as listwise_losses
from ml4ir.applications.ranking.config.keys import LossKey


def get_loss(loss_key, scoring_type) -> RelevanceLossBase:
    if loss_key == LossKey.SIGMOID_CROSS_ENTROPY:
        return pointwise_losses.SigmoidCrossEntropy(loss_key=loss_key, scoring_type=scoring_type)
    elif loss_key == LossKey.RANK_ONE_LISTNET:
        return listwise_losses.RankOneListNet(loss_key=loss_key, scoring_type=scoring_type)
    else:
        raise NotImplementedError
