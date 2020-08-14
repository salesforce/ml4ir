# Stating loss functions related to classification use-case.
from tensorflow.keras import layers
from tensorflow.keras import losses

from ml4ir.base.model.losses.loss_base import RelevanceLossBase
from ml4ir.applications.ranking.config.keys import LossKey


def get_loss(loss_key) -> losses.Loss:
    """
    Factory to get relevance loss related to classification use-case.
    :param loss_key: LossKey.
    :return: RelevanceLossBase: corresponding loss.
    """
    if loss_key == LossKey.CATEGORICAL_CROSS_ENTROPY:
        return CategoricalCrossEntropy()
    else:
        raise NotImplementedError


class CategoricalCrossEntropy(losses.CategoricalCrossentropy, RelevanceLossBase):
    def __init__(self, reduction=losses.Reduction.SUM_OVER_BATCH_SIZE, **kwargs):
        """
        Define a softmax cross entropy loss
        """
        super().__init__(reduction=reduction)

    def __call__(self, y_true, y_pred, features):
        return super().__call__(y_true, y_pred)

    def get_final_activation_op(self, output_name):
        return lambda logits, mask: layers.Activation("softmax", name=output_name)(logits)
