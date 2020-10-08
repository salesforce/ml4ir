from ml4ir.applications.ranking.config.keys import LossTypeKey
from ml4ir.base.model.losses.loss_base import RelevanceLossBase


class RankingLossBase(RelevanceLossBase):
    """
    Abstract class that defines a Ranking loss function
    """

    def __init__(self, loss_type, loss_key, scoring_type):
        """
        Instantiate a RankingLossBase object

        Parameters
        ----------
        loss_type : str
            Type of the loss function - pointwise, pairwise, listwise
        loss_key : str
            Name of the loss function used
        scoring_type : str
            Type of scoring function - pointwise, pairwise, groupwise
        """
        self.loss_type = None
        self.loss_key = loss_key
        self.scoring_type = scoring_type


class PointwiseLossBase(RankingLossBase):
    """
    Abstract class that defines a pointwise ranking loss function
    """

    def __init__(self, loss_key, scoring_type):
        """
        Instantiate a PointwiseLossBase object

        Parameters
        ----------
        loss_key : str
            Name of the loss function used
        scoring_type : str
            Type of scoring function - pointwise, pairwise, groupwise
        """
        super(PointwiseLossBase, self).__init__(
            loss_type=LossTypeKey.POINTWISE, loss_key=loss_key, scoring_type=scoring_type
        )


class PairwiseLossBase(RankingLossBase):
    """
    Abstract class that defines a pairwise ranking loss function
    """

    def __init__(self, loss_key, scoring_type):
        """
        Instantiate a PairwiseLossBase object

        Parameters
        ----------
        loss_key : str
            Name of the loss function used
        scoring_type : str
            Type of scoring function - pointwise, pairwise, groupwise
        """
        super(PairwiseLossBase, self).__init__(
            loss_type=LossTypeKey.PAIRWISE, loss_key=loss_key, scoring_type=scoring_type
        )


class ListwiseLossBase(RankingLossBase):
    """
    Abstract class that defines a listwise ranking loss function
    """

    def __init__(self, loss_key, scoring_type):
        """
        Instantiate a ListwiseLossBase object

        Parameters
        ----------
        loss_key : str
            Name of the loss function used
        scoring_type : str
            Type of scoring function - pointwise, pairwise, groupwise
        """
        super(ListwiseLossBase, self).__init__(
            loss_type=LossTypeKey.LISTWISE, loss_key=loss_key, scoring_type=scoring_type
        )
