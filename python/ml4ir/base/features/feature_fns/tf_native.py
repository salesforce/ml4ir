import tensorflow as tf

from ml4ir.base.model.layers.tf_native import TFNativeLayer
from ml4ir.base.features.feature_fns.base import BaseFeatureLayerOp
from ml4ir.base.io.file_io import FileIO


class TFNativeOpLayer(BaseFeatureLayerOp):
    """
    Run a series of tensorflow native operations on the input feature tensor.
    The functions will be applied in the order they are specified.
    """
    LAYER_NAME = "tf_native_op"
    OPS = "ops"

    def __init__(self, feature_info: dict, file_io: FileIO, **kwargs):
        """
        Initialize the feature layer

        Parameters
        ----------
        feature_info : dict
            Dictionary representing the feature_config for the input feature
        file_io : FileIO object
            FileIO handler object for reading and writing

        Notes
        -----
        Args under feature_layer_info:
            ops: list of dict
                List of function specifications with associated arguments

                Arguments under ops:
                    fn : str
                        Tensorflow native function name. Should start with tf.
                        Example: tf.math.log or tf.clip_by_value
                    args : dict
                        Keyword arguments to be passed to the tensorflow function
        """
        super().__init__(feature_info=feature_info, file_io=file_io, **kwargs)
        self.tf_ops = self.feature_layer_args.get(self.OPS, {})
        self.layer = TFNativeLayer(self.tf_ops)

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
        feature_tensor = self.layer(inputs, training)
        # Adjusting the shape to the default feature fns for concatenating in the next step
        feature_tensor = tf.expand_dims(feature_tensor, axis=-1,
                                        name="{}_tf_native_op".format(self.feature_name))
        return feature_tensor