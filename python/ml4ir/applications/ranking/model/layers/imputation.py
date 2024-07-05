import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class QueryMinImputation(layers.Layer):
    """
    Impute missing feature values with min from query
    """

    def __init__(self,
                 name="query_min_imputation",
                 missing_value: float = 0.,
                 **kwargs):
        """
        Parameters
        ----------
        name: str
            Layer name
        missing_value: float
            Value to be used to identify missing features
            0 value is used to identify missing/null features by default
        kwargs:
            Additional key-value args that will be used for configuring the layer
        """
        self.inf = tf.constant(np.inf)
        self.missing_value = tf.constant(missing_value, tf.float32)
        super().__init__(name=name, **kwargs)

    def call(self, inputs, training=None):
        """
        Invoke the query min imputation layer for the input feature tensor

        Parameters
        ----------
        inputs: Tensor object
            Input ranking feature tensor
            Shape: [batch_size, sequence_len, num_features]
        training: bool
            If the layer should be run as training or not

        Returns
        -------
        Tensor object
            Imputed input feature tensor
            Shape: [batch_size, sequence_len, encoding_size]
        """
        # Set missing values to infinity before min
        masked_inputs = tf.where(tf.equal(inputs, self.missing_value), self.inf, inputs)

        # Find min across feature dimension
        mins = tf.reduce_min(masked_inputs, axis=1)

        # Replace infinities in the mins - could arise when all values are missing
        mins = tf.where(tf.math.is_inf(mins), 0., mins)

        # Set each feature to min
        imputed_inputs = tf.where(tf.equal(inputs, self.missing_value), mins[:, tf.newaxis, :], inputs)

        return imputed_inputs