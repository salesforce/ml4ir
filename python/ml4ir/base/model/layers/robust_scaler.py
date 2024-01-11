import tensorflow as tf
from tensorflow.keras import layers


class RobustScalerLayer(layers.Layer):
    """
    The RobustScaler is designed to be robust to outliers by using the median and the interquartile range (IQR)
    for scaling. This process centers the data around the median and scales it based on the spread of the middle
    50% of the data, making it robust to outliers.

    Final formulation of robust scaler = X - p25 / (p75 - p25)
    """

    def __init__(self,
                 name="robust_scaler",
                 p25: float = 0.0,
                 p75: float = 0.0,
                 **kwargs):
        """
        Parameters
        ----------
        name: str
            Layer name
        p25 : float
            The value of the 25th percentile of the feature
        p75: float
            The value of the 75th percentile of the feature
        kwargs:
            Additional key-value args that will be used for configuring the layer
        """
        self.p25 = p25
        self.p75 = p75
        super().__init__(name=name, **kwargs)

    def call(self, inputs, training=None):
        """
        Defines the forward pass for the layer on the inputs tensor

        Parameters
        ----------
        inputs: tensor
            Input tensor on which the feature transforms are applied
        training: boolean
            Boolean flag indicating if the layer is being used in training mode or not

        Returns
        -------
        tf.Tensor
            Resulting tensor after the forward pass through the feature transform layer
        """
        return tf.math.divide_no_nan(tf.subtract(inputs, self.p25), (self.p75-self.p25))
