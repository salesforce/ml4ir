from ml4ir.applications.ranking.config.keys import LossTypeKey
from ml4ir.base.model.losses.loss_base import RelevanceLossBase


class RankingLossBase(RelevanceLossBase):
    def __init__(self, loss_type, loss_key, scoring_type):
        self.loss_type = None
        self.loss_key = loss_key
        self.scoring_type = scoring_type


class PointwiseLossBase(RankingLossBase):
    def __init__(self, loss_key, scoring_type):
        super(PointwiseLossBase, self).__init__(
            loss_type=LossTypeKey.POINTWISE, loss_key=loss_key, scoring_type=scoring_type
        )


class PairwiseLossBase(RankingLossBase):
    def __init__(self, loss_key, scoring_type):
        super(PairwiseLossBase, self).__init__(
            loss_type=LossTypeKey.PAIRWISE, loss_key=loss_key, scoring_type=scoring_type
        )


class ListwiseLossBase(RankingLossBase):
    def __init__(self, loss_key, scoring_type):
        super(ListwiseLossBase, self).__init__(
            loss_type=LossTypeKey.LISTWISE, loss_key=loss_key, scoring_type=scoring_type
        )
