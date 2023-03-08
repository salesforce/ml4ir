"""
Source : https://github.com/tensorflow/models/blob/master/official/nlp/modeling/models/seq2seq_transformer.py#L360
Relevant Github Issue : https://github.com/tensorflow/models/issues/10927
"""

import math
import tensorflow as tf

from ml4ir.base.model.layers.transformer_encoder_block import TransformerEncoderBlock

EOS_ID = 1


def attention_initializer(hidden_size):
    """Initializer for attention layers in Seq2SeqTransformer."""
    hidden_size = int(hidden_size)
    limit = math.sqrt(6.0 / (hidden_size + hidden_size))
    return tf.keras.initializers.RandomUniform(minval=-limit, maxval=limit)


class TransformerEncoder(tf.keras.layers.Layer):
    """Transformer encoder.
    Transformer encoder is made up of N identical layers. Each layer is composed
    of the sublayers:
      1. Self-attention layer
      2. Feedforward network (which is 2 fully-connected layers)
    """

    def __init__(self,
                 num_layers=6,
                 num_attention_heads=8,
                 intermediate_size=2048,
                 activation="relu",
                 dropout_rate=0.0,
                 attention_dropout_rate=0.0,
                 use_bias=False,
                 norm_first=True,
                 norm_epsilon=1e-6,
                 intermediate_dropout=0.0,
                 **kwargs):
        """Initialize a Transformer encoder.
        Args:
          num_layers: Number of layers.
          num_attention_heads: Number of attention heads.
          intermediate_size: Size of the intermediate (Feedforward) layer.
          activation: Activation for the intermediate layer.
          dropout_rate: Dropout probability.
          attention_dropout_rate: Dropout probability for attention layers.
          use_bias: Whether to enable use_bias in attention layer. If set False,
            use_bias in attention layer is disabled.
          norm_first: Whether to normalize inputs to attention and intermediate
            dense layers. If set False, output of attention and intermediate dense
            layers is normalized.
          norm_epsilon: Epsilon value to initialize normalization layers.
          intermediate_dropout: Dropout probability for intermediate_dropout_layer.
          **kwargs: key word arguemnts passed to tf.keras.layers.Layer.
        """

        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self._intermediate_size = intermediate_size
        self._activation = activation
        self._dropout_rate = dropout_rate
        self._attention_dropout_rate = attention_dropout_rate
        self._use_bias = use_bias
        self._norm_first = norm_first
        self._norm_epsilon = norm_epsilon
        self._intermediate_dropout = intermediate_dropout

    def build(self, input_shape):
        """Implements build() for the layer."""
        self.encoder_layers = []
        for i in range(self.num_layers):
            self.encoder_layers.append(
                TransformerEncoderBlock(
                    num_attention_heads=self.num_attention_heads,
                    inner_dim=self._intermediate_size,
                    inner_activation=self._activation,
                    output_dropout=self._dropout_rate,
                    attention_dropout=self._attention_dropout_rate,
                    use_bias=self._use_bias,
                    norm_first=self._norm_first,
                    norm_epsilon=self._norm_epsilon,
                    inner_dropout=self._intermediate_dropout,
                    attention_initializer=attention_initializer(input_shape[2]),
                    name=("layer_%d" % i)))
        self.output_normalization = tf.keras.layers.LayerNormalization(
            epsilon=self._norm_epsilon, dtype="float32")
        super(TransformerEncoder, self).build(input_shape)

    def get_config(self):
        config = {
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self._intermediate_size,
            "activation": self._activation,
            "dropout_rate": self._dropout_rate,
            "attention_dropout_rate": self._attention_dropout_rate,
            "use_bias": self._use_bias,
            "norm_first": self._norm_first,
            "norm_epsilon": self._norm_epsilon,
            "intermediate_dropout": self._intermediate_dropout
        }
        base_config = super(TransformerEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, encoder_inputs, attention_mask=None):
        """Return the output of the encoder.
        Args:
          encoder_inputs: A tensor with shape `(batch_size, input_length,
            hidden_size)`.
          attention_mask: A mask for the encoder self-attention layer with shape
            `(batch_size, input_length, input_length)`.
        Returns:
          Output of encoder which is a `float32` tensor with shape
            `(batch_size, input_length, hidden_size)`.
        """
        for layer_idx in range(self.num_layers):
            encoder_inputs = self.encoder_layers[layer_idx](
                [encoder_inputs, attention_mask])

        output_tensor = encoder_inputs
        output_tensor = self.output_normalization(output_tensor)

        return output_tensor
