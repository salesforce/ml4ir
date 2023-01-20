import tensorflow as tf
from tensorflow.keras import layers
from ml4ir.applications.ranking.config.keys import PositionalBiasHandler
from tensorflow.keras import regularizers


class FixedAdditivePositionalBias(layers.Layer):
    """
    This class implements a solution to handle positional bias in logged data.
    During training, a bias term is learned for each rank and is added to the relevance score of each query-document
    pair depending on its rank.
    The process of adding the positional bias is only applied during training because this is a way to de-bias data
    from logs (typically clicked/no clicked pairs). During inference we do not know the actual ranks, so the addition
    is turned off.
    This is an approach we have found useful for logged data as these are presented to the user with some order and
    there is a presentation bias (that this technique tries to model).
    When the data is labeled with actual graded relevance annotations then this technique is not recommended.

    To trigger this technique add this section to the model_config.yaml file:

    positional_bias_handler:
        key: fixed_additive_positional_bias
        max_ranks_count: x

    Where x is the maximum number of documents per query.
    """
    def __init__(self, max_ranks, kernel_initializer='Zeros', l1_coeff=0, l2_coeff=0):
        super(FixedAdditivePositionalBias, self).__init__()
        self.dense = layers.Dense(1,
                                  name=PositionalBiasHandler.FIXED_ADDITIVE_POSITIONAL_BIAS,
                                  kernel_initializer=kernel_initializer,
                                  kernel_regularizer=regularizers.l1_l2(l1=l1_coeff, l2=l2_coeff),
                                  activation=None,
                                  use_bias=False)
        self.max_ranks = max_ranks

    def call(self, inputs, training=False):
        """
        Invoke the positional bias handling

        Parameters
        ----------
        input : Tensor object
            rank index tensor
        training : bool
            If the layer should be run as training or not

        Returns
        -------
        Tensor object
            positional biases resulting from a feedforwrd of the converted one hot tensor through a dense layer.
        """
        features = tf.one_hot(tf.cast(tf.subtract(inputs, 1), dtype=tf.int64), depth=self.max_ranks,
                              dtype=tf.dtypes.float32)
        if not training:
            features = tf.multiply(features, 0.0)
        return self.dense(features)