# type: ignore
# TODO: Fix typing

from ml4ir.model.losses.loss_base import RankingLossBase
import ml4ir.model.losses.pointwise_losses as pointwise_losses
import ml4ir.model.losses.pairwise_losses as pairwise_losses
import ml4ir.model.losses.listwise_losses as listwise_losses
from ml4ir.config.keys import LossKey


def get_loss(loss_key, scoring_key) -> RankingLossBase:
    if loss_key == LossKey.SIGMOID_CROSS_ENTROPY:
        return pointwise_losses.SigmoidCrossEntropy(loss_key=loss_key, scoring_key=scoring_key)
    elif loss_key == LossKey.RANK_ONE_LISTNET:
        return listwise_losses.RankOneListNet(loss_key=loss_key, scoring_key=scoring_key)
    else:
        raise NotImplementedError
