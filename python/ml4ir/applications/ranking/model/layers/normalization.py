import tensorflow as tf
from tensorflow.keras import layers


class QueryNormalization(layers.Layer):
    """
    Zscore normalization of the feature dimension in a batch along the query axis
    """
    DELTA = tf.constant(1e-15)

    def __init__(self,
                 name="query_norm",
                 requires_mask: bool = True,
                 **kwargs):
        """
        Parameters
        ----------
        name: str
            Layer name
        requires_mask: bool
            Indicates if the layer requires a mask to be passed to it during forward pass
        kwargs:
            Additional key-value args that will be used for configuring the layer
        """
        super().__init__(name=name, **kwargs)
        self.requires_mask = requires_mask

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
        mask = tf.cast(mask, inputs.dtype)[:, :, tf.newaxis]

        inputs = tf.multiply(inputs, mask)

        count = tf.math.reduce_sum(mask, axis=1)
        mean = tf.math.reduce_sum(inputs, axis=1) / count
        mean = tf.multiply(mean[:, tf.newaxis, :], mask)

        var = tf.math.reduce_sum(tf.math.pow((inputs - mean), 2), axis=1) / count
        std = tf.math.sqrt(var)[:, tf.newaxis, :] + QueryNormalization.DELTA

        zscore = (inputs - mean) / std

        return zscore
