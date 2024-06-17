import tensorflow as tf
from tensorflow.keras import layers


class RecordFeatureMask(layers.Layer):
    """
    Mask the record's features of a query batch at the given rate

    Example
    -------
    x = np.ones((2, 5, 4))
        array([[[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]],

               [[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]]])
    record_feature_mask = RecordFeatureMask(0.5)
    record_feature_mask(x)
        array([[[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [0., 0., 0., 0.],
                [1., 1., 1., 1.]],

               [[0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]]])
    """

    def __init__(self,
                 name="record_feature_mask",
                 mask_rate: float = 0.2,
                 mask_at_inference: bool = False,
                 requires_mask: bool = False,
                 **kwargs):
        """
        Parameters
        ----------
        name: str
            Layer name
        mask_rate: float
            Rate at which the records in the query batch need to be masked to 0s
        mask_at_inference: boolean
            Whether to also mask at inference
            Useful for testing performance at inference time, but should be set to False when training a model to deploy
        requires_mask: bool
            Indicates if the layer requires a mask to be passed to it during forward pass
        kwargs:
            Additional key-value args that will be used for configuring the layer
        """
        super().__init__(name=name, **kwargs)

        self.mask_rate = mask_rate
        self.mask_at_inference = mask_at_inference
        self.requires_mask = requires_mask

        # Define the probability of picking labels 0 and 1 using the mask_rate
        self.log_odds = tf.math.log([[self.mask_rate, (1. - self.mask_rate)]])

    def call(self, inputs, mask=None, training=None):
        """
        Invoke the record feature mask layer for the input feature tensor

        Parameters
        ----------
        inputs: Tensor object
            Input ranking feature tensor
            Shape: [batch_size, sequence_len, feature_dim]
        mask: Tensor object
            Mask to be used to identify records to ignore in query (unused in this layer)
            Shape: [batch_size, sequence_len]
        training: bool
            If the layer should be run as training or not

        Returns
        -------
        Tensor object
            Feature tensor where mask_rate of records' features have been masked out (or set to 0.)
            Shape: [batch_size, sequence_len, feature_dim]
        """
        if self.mask_at_inference or training:
            batch_size = tf.shape(inputs)[0]
            record_dim = tf.shape(inputs)[1]

            batch_log_odds = tf.tile(self.log_odds, [batch_size, 1])
            mask = tf.random.categorical(logits=batch_log_odds,
                                         num_samples=record_dim)
            mask = tf.cast(mask, inputs.dtype)[:, :, tf.newaxis]

            return tf.math.multiply(inputs, mask)
        else:
            return inputs


class QueryFeatureMask(layers.Layer):
    """
    Mask the features of all records per query at the given rate

    Example
    -------
    x = np.ones((2, 5, 4))
        array([[[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]],

               [[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]]])
    query_feature_mask = QueryFeatureMask(0.5)
    query_feature_mask(x)
        array([[[1., 1., 0., 1.],
                [1., 1., 0., 1.],
                [1., 1., 0., 1.],
                [1., 1., 0., 1.],
                [1., 1., 0., 1.]],

               [[0., 1., 1., 1.],
                [0., 1., 1., 1.],
                [0., 1., 1., 1.],
                [0., 1., 1., 1.],
                [0., 1., 1., 1.]]])
    """

    def __init__(self,
                 name="query_feature_mask",
                 mask_rate: float = 0.2,
                 mask_at_inference: bool = False,
                 requires_mask: bool = False,
                 **kwargs):
        """
        Parameters
        ----------
        name: str
            Layer name
        mask_rate: float
            Rate at which the records in the query batch need to be masked to 0s
        mask_at_inference: boolean
            Whether to also mask at inference
            Useful for testing performance at inference time, but should be set to False when training a model to deploy
        requires_mask: bool
            Indicates if the layer requires a mask to be passed to it during forward pass
        kwargs:
            Additional key-value args that will be used for configuring the layer
        """
        super().__init__(name=name, **kwargs)

        self.mask_rate = mask_rate
        self.mask_at_inference = mask_at_inference
        self.requires_mask = requires_mask

    def call(self, inputs, mask=None, training=None):
        """
        Invoke the query feature mask layer for the input feature tensor

        Parameters
        ----------
        inputs: Tensor object
            Input ranking feature tensor
            Shape: [batch_size, sequence_len, feature_dim]
        mask: Tensor object
            Mask to be used to identify records to ignore in query (unused in this layer)
            Shape: [batch_size, sequence_len]
        training: bool
            If the layer should be run as training or not

        Returns
        -------
        Tensor object
            Feature tensor where mask_rate of records' features have been masked out (or set to 0.)
            Shape: [batch_size, sequence_len, feature_dim]
        """
        batch_size, sequence_len, feature_dim = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        if self.mask_at_inference or training:
            mask = tf.cast(tf.math.greater(tf.random.uniform([batch_size, feature_dim]), self.mask_rate), dtype=tf.float32)
            mask = tf.expand_dims(mask, axis=1)
            mask = tf.tile(mask, [1, sequence_len, 1])
            return tf.multiply(inputs, mask)
        else:
            return inputs


