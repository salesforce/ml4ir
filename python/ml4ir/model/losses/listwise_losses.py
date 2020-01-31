from ml4ir.model.losses.loss_base import ListwiseLossBase


class RankOneListNet(ListwiseLossBase):
    def _make_loss_fn(self, **kwargs):
        """
        Define a rank 1 ListNet loss
        Additionally can pass in record positions to handle positional bias

        """
        raise NotImplementedError
