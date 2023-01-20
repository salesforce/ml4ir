# Stating loss functions related to classification use-case.
from tensorflow.keras import layers
from tensorflow.keras import losses

from ml4ir.base.config.keys import FeatureTypeKey
from ml4ir.base.model.losses.loss_base import RelevanceLossBase
from ml4ir.applications.ranking.config.keys import LossKey


def get_loss(loss_key, output_name) -> RelevanceLossBase:
    """
    Factory to get relevance loss related to classification use-case.

    Parameters
    ----------
    loss_key : str
        LossKey name
    output_name: str
        Name of the output node after final activation op

    Returns
    -------
    RelevanceLossBase
        Corresponding loss object
    """
    if loss_key == LossKey.CATEGORICAL_CROSS_ENTROPY:
        return CategoricalCrossEntropy(output_name=output_name)
    else:
        raise NotImplementedError


class CategoricalCrossEntropy(RelevanceLossBase):

    def __init__(self, output_name, **kwargs):
        """
        Initialize categorical cross entropy loss

        Parameters
        ----------
        output_name: str
            Name of the output node after final activation op
        """
        super().__init__(**kwargs)

        self.output_name = output_name
        self.final_activation_fn = layers.Activation("softmax", name=self.output_name)

        self.loss_fn = losses.CategoricalCrossentropy(reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)

    def call(self, inputs, y_true, y_pred, training=None):
        """
        Define a categorical cross entropy loss
        
        Parameters
        ----------
        inputs: dict of dict of tensors
            Dictionary of input feature tensors
        y_true: tensor
            True labels
        y_pred: tensor
            Predicted scores
        training: boolean
            Boolean indicating whether the layer is being used in training mode

        Returns
        -------
        function
            Categorical cross entropy loss
        """
        return self.loss_fn(y_true, y_pred)

    def final_activation_op(self, inputs, training=None):
        """
        Get softmax activated scores on logits

        Parameters
        ----------
        inputs: dict of dict of tensors
            Dictionary of input feature tensors

        Returns
        -------
        tensor
            Softmax activated scores
        """
        return self.final_activation_fn(inputs[FeatureTypeKey.LOGITS])

    def get_config(self):
        """Return layer config that is used while serialization"""
        config = super().get_config()
        config.update({
            "loss_fn": "categorical_cross_entropy",
            "output_name": self.output_name
        })
        return config
