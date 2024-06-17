import tensorflow as tf
from tensorflow.keras import layers


class ReciprocalRankLayer(layers.Layer):
    """
    Converts a tensor of scores into reciprocal ranks.
    Can optionally add a constant or variable k to the rank

    Final formulation of reciprocal rank = 1 / (k + rank)
    """

    def __init__(self,
                 name="reciprocal_rank",
                 k: float = 0.0,
                 k_trainable: bool = False,
                 ignore_zero_score: bool = True,
                 scale_range_to_one: bool = False,
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
        ignore_zero_score: bool
            Use zero reciprocal rank for score value of 0.0
            When using the layer with negative scores, make sure to set this arg to False
        scale_range_to_one: bool
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
        self.ignore_zero_score = ignore_zero_score
        self.scale_range_to_one = scale_range_to_one
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
        # Compute ranks from scores by arg sorting twice
        ranks = tf.add(
            tf.argsort(
                tf.argsort(inputs,
                           axis=1,
                           direction="DESCENDING",
                           stable=True),
                axis=1,
                stable=True),
            # Note -> We add 1 to ensure ranks start from 1 instead of 0
            tf.constant(1))

        # Add k to the ranks
        ranks = tf.cast(ranks, tf.float32)
        ranks = tf.add(ranks, self.k)

        # Reciprocal of the ranks and update the ranks for 0.0 scores to 0.0
        reciprocal_ranks = tf.math.divide_no_nan(1.0, ranks)
        if self.ignore_zero_score:
            reciprocal_ranks = tf.where(
                inputs == tf.constant(0.0),
                tf.constant(0.0),
                reciprocal_ranks)

        # Scale the reciprocal rank range from (0, 1]
        # NOTE: Highest rank of 1 will be set to max scaled reciprocal rank of 1.
        if self.scale_range_to_one:
            reciprocal_ranks = tf.math.multiply_no_nan(reciprocal_ranks, self.k + 1.)

        return reciprocal_ranks
