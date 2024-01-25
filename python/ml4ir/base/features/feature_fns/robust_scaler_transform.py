import tensorflow as tf

from ml4ir.base.model.layers.robust_scaler import RobustScalerLayer
from ml4ir.base.features.feature_fns.base import BaseFeatureLayerOp
from ml4ir.base.io.file_io import FileIO


class RobustScaler(BaseFeatureLayerOp):
    """
    The RobustScaler is designed to be robust to outliers by using the median and the interquartile range (IQR)
    for scaling. This process centers the data around the median and scales it based on the spread of the middle
    50% of the data, making it robust to outliers.

    Final formulation of robust scaler = X - p25 / (p75 - p25)
    """
    LAYER_NAME = "robust_scaler"

    P25 = "p25"
    P75 = "p75"

    def __init__(self, feature_info: dict, file_io: FileIO, **kwargs):
        """
        Initialize layer to define a robust scaler layer

        Parameters
        ----------
        feature_info : dict
            Dictionary representing the configuration parameters for the specific feature from the FeatureConfig
        file_io : FileIO object
            FileIO handler object for reading and writing

        Notes
        -----
        Args under feature_info["feature_layer_info"]:
            q25 : float
                The value of the 25th percentile of the feature
            q75: float
                The value of the 75th percentile of the feature
        """
        super().__init__(feature_info=feature_info, file_io=file_io, **kwargs)

        self.robust_scaler_op = RobustScalerLayer(
            q25=self.feature_layer_args.get(self.P25, 0.),
            q75=self.feature_layer_args.get(self.P75, 0.))

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
        robust_scaler_normalized = self.robust_scaler_op(inputs, training)

        return tf.expand_dims(robust_scaler_normalized, axis=-1)
