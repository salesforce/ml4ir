import tensorflow as tf
from tensorflow.keras import layers


class QueryNormalization(layers.Layer):
    """
    Zscore normalization of the feature dimension in a batch along the query axis
    """

    def __init__(self,
                 name="query_norm",
                 requires_mask: bool = False,
                 **kwargs):
        """
        Parameters
        ----------
        name: str
            Layer name
        requires_mask: bool
            Indicates if the layer requires a mask to be passed to it during forward pass
            NOTE: This needs to be True to use the layer
        kwargs:
            Additional key-value args that will be used for configuring the layer
        """
        super().__init__(name=name, **kwargs)
        self.requires_mask = requires_mask
        assert self.requires_mask, "requires_mask needs to be set to True to use QueryNormalization"

    def call(self, inputs, mask=None, training=None):
        """
        Invoke the query normalization layer for the input feature tensor

        Parameters
        ----------
        inputs: Tensor object
            Input ranking feature tensor
            Shape: [batch_size, sequence_len, num_features]
        mask: Tensor object
            Mask to be used to identify records to ignore in query
            Shape: [batch_size, sequence_len]
        training: bool
            If the layer should be run as training or not

        Returns
        -------
        Tensor object
            Normed input feature tensor
            Shape: [batch_size, sequence_len, encoding_size]
        """
        # NOTE: Tensorflow math ops are not consistent on saving and reloading if they are float32.
        #       Both float16 and float64 give consistent results.
        inputs = tf.cast(inputs, tf.float64)
        mask = tf.expand_dims(tf.cast(mask, tf.float64), axis=-1)

        inputs = tf.multiply(inputs, mask)

        count = tf.math.reduce_sum(mask, axis=1)
        mean = tf.math.divide(tf.math.reduce_sum(inputs, axis=1), count)
        mean = tf.multiply(tf.expand_dims(mean, axis=1), mask)

        var = tf.math.divide(tf.math.reduce_sum(tf.math.pow(tf.math.subtract(inputs, mean), 2), axis=1), count)
        std = tf.expand_dims(tf.math.sqrt(var), axis=1)

        zscore = tf.math.divide_no_nan(tf.math.subtract(inputs, mean), std)

        return tf.cast(zscore, tf.float32)
