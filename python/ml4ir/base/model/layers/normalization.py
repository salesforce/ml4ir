import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


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
                 **kwargs):
        """
        Parameters
        ----------
        name: str
            Layer name
        k : float
            Constant value to be added to the score before reciprocal
        k_trainable: bool
            If k should be a learnable variable; will be initialized with value of k
        kwargs:
            Additional key-value args that will be used for configuring the layer
        """

        super().__init__(name=name, **kwargs)
        self.k_trainable = k_trainable
        self.k = self.add_weight(
            name="reciprocal_k",
            shape=(),
            initializer=tf.constant_initializer(float(k)),
            trainable=self.k_trainable
        )

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

        return reciprocals


class ArcTanNormalization(layers.Layer):
    """
    Converts a tensor of scores into [0, 1] range using the below formula

    arctan(x * k) * 2 / pi

    Reference ->
    - https://www.mdpi.com/2227-7390/10/8/1335
    - https://www.linkedin.com/pulse/guidebook-state-of-the-art-embeddings-information-aapo-tanskanen-pc3mf
    """

    def __init__(self,
                 name="arctan_norm",
                 k: float = 1.0,
                 k_trainable: bool = False,
                 **kwargs):
        """
        Parameters
        ----------
        name: str
            Layer name
        k : float
            Constant value to be multiplied to the rank before reciprocal
        kwargs:
            Additional key-value args that will be used for configuring the layer
        """
        super().__init__(name=name, **kwargs)
        self.k_trainable = k_trainable
        self.k = tf.Variable(
            initial_value=float(k),
            trainable=self.k_trainable,
            name="arctan_k"
        )
        self.scale_constant = tf.constant(2. / np.pi)

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
        # Multiply k to the scores
        scores = tf.multiply(inputs, self.k)

        # Get tan inverse (or arctan)
        arctan_scores = tf.math.atan(scores)

        # Scale to [0, 1] range
        normed_scores = tf.math.multiply_no_nan(arctan_scores, self.scale_constant)

        return normed_scores
