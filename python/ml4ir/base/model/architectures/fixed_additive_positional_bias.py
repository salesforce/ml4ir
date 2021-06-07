import tensorflow as tf
from tensorflow.keras import layers


class FixedAdditivePositionalBias(layers.Layer):
    """
    Implementing positional bias handling by learning a fixed additive weights
    """
    def __init__(self, max_ranks):
        super(FixedAdditivePositionalBias, self).__init__()
        self.dense = layers.Dense(1,
                     name="fixed_additive_positional_bias_layer",
                     activation=None,
                     use_bias=False)
        self.max_ranks = max_ranks

    def convert_to_one_hot(self, inputs, training=False):
        """
        Convert the iuputs to one hot vectors.

        Parameters
        ----------
        input : Tensor object
            rank index tensor
        max_ranks: int
            the maximum number of documents per query
        training : bool
            If the layer should be run as training or not

        Returns
        -------
        Tensor object
            A one-hot vector representation of the input rank index. The one-hot entry would
            be masked in case of non training mode.
        """
        features = tf.one_hot(tf.cast(tf.subtract(inputs, 1), dtype=tf.int64), depth=self.max_ranks, dtype=tf.dtypes.float32)
        if not training:
            features = tf.multiply(features, 0.0)
        return features

    def call(self, inputs, training=False):
        """
        Invoking the positional bias handling

        Parameters
        ----------
        input : Tensor object
            rank index tensor
        max_ranks: int
            the maximum number of documents per query
        training : bool
            If the layer should be run as training or not

        Returns
        -------
        Tensor object
            positional biases resulting from a feedforwrd to the converted one hot tensor through a dense layer
        """
        features = self.convert_to_one_hot(inputs, training)
        return self.dense(features)
