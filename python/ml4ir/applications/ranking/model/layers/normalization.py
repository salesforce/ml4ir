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

        normed_inputs = self.normalize(inputs, mean, std)

        return tf.cast(normed_inputs, tf.float32)

    @staticmethod
    def normalize(inputs, mean, std):
        """
        Normalize the inputs using the mean and standard deviation

        Parameters
        ----------
        inputs: Tensor
            Input query tensor to be normalized
        mean: Tensor
            Tensor of query means
        std: Tensor
            Tensor of query standard deviations

        Returns
        -------
        Tensor
            Normalized input tensor
        """
        return tf.math.divide_no_nan(tf.math.subtract(inputs, mean), std)


class EstimatedMinMaxNormalization(layers.Layer):
    """
    Min Max Normalization of individual query features,
    where the min and max for a query are estimated as 3 standard deviations from the mean.

    Reference -> https://medium.com/plain-simple-software/distribution-based-score-fusion-dbsf-a-new-approach-to-vector-search-ranking-f87c37488b18
    """

    def __init__(self,
                 name="emm_norm",
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
        assert self.requires_mask, "requires_mask needs to be set to True"

    @staticmethod
    def normalize(inputs, mean, std):
        """
        Normalize the inputs using the mean and standard deviation

        Parameters
        ----------
        inputs: Tensor
            Input query tensor to be normalized
        mean: Tensor
            Tensor of query means
        std: Tensor
            Tensor of query standard deviations

        Returns
        -------
        Tensor
            Normalized input tensor
        """
        # Min Max scale by [mean - 3 * std, mean + 3 * std]
        estimated_min = mean - (3 * std)
        estimated_max = mean + (3 * std)

        return tf.math.divide_no_nan(tf.math.subtract(inputs, estimated_min),
                                     (estimated_max - estimated_min))


class TheoreticalMinMaxNormalization(layers.Layer):
    """
    Min Max Normalization of individual query features,
    where the theoretical min is used instead of the minimum.

    Reference -> An Analysis of Fusion Functions for Hybrid Retrieval
                 https://arxiv.org/abs/2210.11934
    """

    def __init__(self,
                 name="tmm_norm",
                 theoretical_min: float = 0.0,
                 **kwargs):
        """
        Parameters
        ----------
        name: str
            Layer name
        theoretical_min : float
            Theoretical minimum to use for the query's record features
            Default value of 0. is used if not specified.
        kwargs:
            Additional key-value args that will be used for configuring the layer
        """
        self.theoretical_min = theoretical_min
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
        # Replace values lower than theoretical minimum with theoretical minimum
        inputs = tf.clip_by_value(inputs,
                                  clip_value_min=self.theoretical_min,
                                  clip_value_max=tf.reduce_max(inputs))

        # Compute max values for each query
        query_max = tf.expand_dims(tf.reduce_max(inputs, axis=1), axis=1)

        # Min max normalization
        normed_inputs = tf.math.divide_no_nan(
            tf.math.subtract(inputs, self.theoretical_min),
            tf.math.subtract(query_max, self.theoretical_min))

        return normed_inputs


class Reciprocal(layers.Layer):
    """
    Converts a tensor of scores into reciprocal scores.
    Can optionally add a constant or variable k to decay

    Final formulation of reciprocal  = 1 / (k + score)
    """

    def __init__(self,
                 name="reciprocal",
                 k: float = 0.0,
                 k_trainable: bool = False,
                 scale_to_one: bool = False,
                 **kwargs):
        """
        Parameters
        ----------
        name: str
            Layer name
        k : float
            Constant value to be added to the rank before reciprocal
        k_trainable: bool
            If k should be a learnable variable; will be initialized with value of k
        scale_to_one: bool
            If true, the values are scaled to (0, 1]
            by multiplying the reciprocal ranks by k + 1
        kwargs:
            Additional key-value args that will be used for configuring the layer
        """
        self.k_trainable = k_trainable
        self.k = tf.Variable(
            initial_value=float(k),
            trainable=self.k_trainable
        )
        self.scale_to_one = scale_to_one
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
        # Add k to the scores
        scores = tf.add(inputs, self.k)

        # Reciprocal of the scores
        reciprocals = tf.math.divide_no_nan(1.0, scores)

        # Scale the reciprocals from (0, 1]
        if self.scale_to_one:
            reciprocals = tf.math.multiply_no_nan(reciprocals, (self.k + 1.))

        return reciprocals