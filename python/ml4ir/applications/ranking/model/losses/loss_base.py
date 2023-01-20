from ml4ir.applications.ranking.config.keys import LossTypeKey
from ml4ir.base.model.losses.loss_base import RelevanceLossBase


class RankingLossBase(RelevanceLossBase):
    """
    Abstract class that defines a Ranking loss function
    """

    def __init__(self,
                 loss_type: str,
                 loss_key: str,
                 scoring_type: str,
                 output_name: str,
                 **kwargs):
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
        output_name: str
            Name of the output node after final activation op
        """
        super().__init__(**kwargs)

        self.loss_type = None
        self.loss_key = loss_key
        self.scoring_type = scoring_type
        self.output_name = output_name

    def get_config(self):
        """Return layer config that is used while serialization"""
        config = super().get_config()
        config.update({
            "loss_type": self.loss_type,
            "loss_key": self.loss_key,
            "scoring_type": self.scoring_type
        })
        return config


class PointwiseLossBase(RankingLossBase):
    """
    Abstract class that defines a pointwise ranking loss function
    """

    def __init__(self,
                 loss_key: str,
                 scoring_type: str,
                 output_name: str = "score",
                 **kwargs):
        """
        Instantiate a PointwiseLossBase object

        Parameters
        ----------
        loss_key : str
            Name of the loss function used
        scoring_type : str
            Type of scoring function - pointwise, pairwise, groupwise
        output_name: str
            Name of the output node after final activation op
        """
        super(PointwiseLossBase, self).__init__(
            loss_type=LossTypeKey.POINTWISE,
            loss_key=loss_key,
            scoring_type=scoring_type,
            output_name=output_name,
            **kwargs
        )


class PairwiseLossBase(RankingLossBase):
    """
    Abstract class that defines a pairwise ranking loss function
    """

    def __init__(self,
                 loss_key: str,
                 scoring_type: str,
                 output_name: str = "score",
                 **kwargs):
        """
        Instantiate a PairwiseLossBase object

        Parameters
        ----------
        loss_key : str
            Name of the loss function used
        scoring_type : str
            Type of scoring function - pointwise, pairwise, groupwise
        output_name: str
            Name of the output node after final activation op
        """
        super(PairwiseLossBase, self).__init__(
            loss_type=LossTypeKey.PAIRWISE,
            loss_key=loss_key,
            scoring_type=scoring_type,
            output_name=output_name,
            **kwargs
        )


class ListwiseLossBase(RankingLossBase):
    """
    Abstract class that defines a listwise ranking loss function
    """

    def __init__(self,
                 loss_key: str,
                 scoring_type: str,
                 output_name: str = "score",
                 **kwargs):
        """
        Instantiate a ListwiseLossBase object

        Parameters
        ----------
        loss_key : str
            Name of the loss function used
        scoring_type : str
            Type of scoring function - pointwise, pairwise, groupwise
        output_name: str
            Name of the output node after final activation op
        """
        super(ListwiseLossBase, self).__init__(
            loss_type=LossTypeKey.LISTWISE,
            loss_key=loss_key,
            scoring_type=scoring_type,
            output_name=output_name,
            **kwargs
        )