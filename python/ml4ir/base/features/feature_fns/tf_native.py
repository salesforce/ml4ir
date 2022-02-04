import tensorflow as tf
from tensorflow.keras import layers

from ml4ir.base.io.file_io import FileIO
from ml4ir.base.features.feature_fns.base import BaseFeatureLayerOp


class TFNativeOpLayer(BaseFeatureLayerOp):
    """
    Run a series of tensorflow native operations on the input feature tensor.
    The functions will be applied in the order they are specified.
    """
    LAYER_NAME = "tf_native_op"

    ARGS = "args"
    OPS = "ops"
    FN = "fn"

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

        self.tf_ops = self.feature_layer_args.get(self.ARGS, {}).get(self.OPS, {})

    def call(self, inputs, training=None):
        """
        TODO: Add docs

        Returns
        -------
        Tensor object
            Modified feature tensor after applying all the specified ops
        """

        if not self.tf_ops:
            return inputs

        for tf_op in self.tf_ops:
            try:
                fn_, fn_args = eval(tf_op[self.FN]), tf_op.get(self.ARGS, {})
            except AttributeError as e:
                raise KeyError(
                    "Invalid fn specified for tf_native_op : {}\n{}".format(tf_op[self.FN], e))

            try:
                feature_tensor = fn_(inputs, **fn_args)
            except Exception as e:
                raise Exception("Error while applying {} to {} feature:\n{}".format(
                    tf_op[self.FN], self.feature_name, e))

        # Adjusting the shape to the default feature fns for concatenating in the next step
        feature_tensor = tf.expand_dims(feature_tensor, axis=-1,
                                        name="{}_tf_native_op".format(self.feature_name))

        return feature_tensor
