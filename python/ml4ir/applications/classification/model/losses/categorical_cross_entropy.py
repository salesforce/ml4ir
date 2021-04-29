# Stating loss functions related to classification use-case.
from tensorflow.keras import layers
from tensorflow.keras import losses

from ml4ir.base.model.losses.loss_base import RelevanceLossBase
from ml4ir.applications.ranking.config.keys import LossKey


def get_loss(loss_key) -> RelevanceLossBase:
    """
    Factory to get relevance loss related to classification use-case.

    Parameters
    ----------
    loss_key : str
        LossKey name

    Returns
    -------
    RelevanceLossBase
        Corresponding loss object
    """
    if loss_key == LossKey.CATEGORICAL_CROSS_ENTROPY:
        return CategoricalCrossEntropy()
    else:
        raise NotImplementedError


class CategoricalCrossEntropy(RelevanceLossBase):
    def get_loss_fn(self, **kwargs):
        """
        Define a categorical cross entropy loss

        Returns
        -------
        function
            Categorical cross entropy loss function
        """
        cce = losses.CategoricalCrossentropy(reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)

        def _loss_fn(y_true, y_pred):
            return cce(y_true, y_pred)

        return _loss_fn

    def get_final_activation_op(self, output_name):
        """
        Define softmax activation function

        Parameters
        ----------
        output_name : str
            Name of the output to use for final activation layer

        Returns
        -------
        function
            Softmax activation function
        """
        return lambda logits, mask: layers.Activation("softmax", name=output_name)(logits)
