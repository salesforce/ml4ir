import tensorflow as tf
from tensorflow.keras import layers


class TFNativeLayer(layers.Layer):
    """
    Run a series of tensorflow native operations on the input feature tensor.
    The functions will be applied in the order they are specified.
    """
    ARGS = "args"
    FN = "fn"

    def __init__(self, ops, **kwargs):
        """
        Initialize the feature layer

        Parameters
        ----------
        ops: list of dict
            List of function specifications with associated arguments

            Arguments under ops:
                fn : str
                    Tensorflow native function name. Should start with tf.
                    Example: tf.math.log or tf.clip_by_value
                args : dict
                    Keyword arguments to be passed to the tensorflow function
        """
        super().__init__(**kwargs)
        self.tf_ops = ops

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
        if not self.tf_ops:
            return inputs

        input_tensor = inputs
        for tf_op in self.tf_ops:
            try:
                fn_, fn_args = eval(tf_op[self.FN]), tf_op.get(self.ARGS, {})
            except AttributeError as e:
                raise KeyError(
                    "Invalid fn specified for tf_native_op : {}\n{}".format(tf_op[self.FN], e))

            try:
                input_tensor = fn_(input_tensor, **fn_args)
            except Exception as e:
                raise Exception("Error while applying {} to {} feature:\n{}".format(
                    tf_op[self.FN], self.feature_name, e))

        return input_tensor