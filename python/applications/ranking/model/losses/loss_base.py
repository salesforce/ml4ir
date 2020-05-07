from ml4ir.config.keys import LossTypeKey
from ml4ir.model.losses.loss_base import RelevanceLossBase


class PointwiseLossBase(RelevanceLossBase):
    def __init__(self):
        super(PointwiseLossBase, self).__init__()
        self.loss_type = LossTypeKey.POINTWISE


class PairwiseLossBase(RelevanceLossBase):
    def __init__(self):
        super(PairwiseLossBase, self).__init__()
        self.loss_type = LossTypeKey.PAIRWISE


class ListwiseLossBase(RelevanceLossBase):
    def __init__(self):
        super(ListwiseLossBase, self).__init__()
        self.loss_type = LossTypeKey.LISTWISE
