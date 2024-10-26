import tensorflow as tf
from tensorflow.keras import layers

from ml4ir.base.model.layers.transformer_encoder import TransformerEncoder


class SetRankEncoder(layers.Layer):
    """
    SetRank architecture layer that maps features for a document -> encoding
    using a permutation invariant multi-head self attention technique attending to all documents in the query.

    Inspired from the transformer architecture as described in the following paper
    * Liang Pang, Jun Xu, Qingyao Ai, Yanyan Lan, Xueqi Cheng, Jirong Wen. 2020.
    SetRank: Learning a Permutation-Invariant Ranking Model for Information Retrieval. In Proceedings of SIGIR '20

    Reference -> https://arxiv.org/pdf/1912.05891.pdf
    """

    def __init__(self,
                 encoding_size: int,
                 requires_mask: bool = True,
                 projection_dropout_rate: float = 0.0,
                 **kwargs):
        """
        Parameters
        ----------
        encoding_size: int
            Size of the projection which will serve as both the input and output size to the encoder
        requires_mask: bool
            Indicates if the layer requires a mask to be passed to it during forward pass
        projection_dropout_rate: float
            Dropout rate to be applied after the input projection layer
        kwargs:
            Additional key-value args that will be used for configuring the TransformerEncoder

        Notes
        -----
            For a full list of args that can be passed to configure this layer, please check the below official layer doc
            https://www.tensorflow.org/api_docs/python/tfm/nlp/models/TransformerEncoder
        """
        super(SetRankEncoder, self).__init__()

        self.requires_mask = requires_mask
        self.encoding_size = encoding_size
        self.projection_dropout_rate = projection_dropout_rate

        self.input_projection_op = layers.Dense(units=self.encoding_size)
        self.projection_dropout_op = layers.Dropout(rate=self.projection_dropout_rate)
        self.transformer_encoder = TransformerEncoder(**kwargs)

    def call(self, inputs, mask=None, training=None):
        # Project input from shape
        # [batch_size, sequence_len, num_features] -> [batch_size, sequence_len, encoding_size]
        encoder_inputs = self.input_projection_op(inputs, training=training)
        encoder_inputs = self.projection_dropout_op(encoder_inputs, training=training)

        # Compute attention mask if mask is present
        if self.requires_mask and mask is not None:
            # Mask encoder inputs after projection
            encoder_inputs = tf.transpose(
                tf.multiply(
                    tf.transpose(encoder_inputs),
                    tf.transpose(tf.cast(mask, encoder_inputs.dtype))
                )
            )

            # Convert 2D mask to 3D mask to be used for attention
            attention_mask = tf.matmul(mask[:, :, tf.newaxis], mask[:, tf.newaxis, :])
        else:
            # Create default attention mask (all ones)
            batch_size = tf.shape(encoder_inputs)[0]
            sequence_len = tf.shape(encoder_inputs)[1]
            attention_mask = tf.ones((batch_size, sequence_len, sequence_len), dtype=encoder_inputs.dtype)

        encoder_output = self.transformer_encoder(encoder_inputs=encoder_inputs,
                                                  attention_mask=attention_mask,
                                                  training=training)

        return encoder_output

    def get_config(self):
        config = self.transformer_encoder.get_config()
        config.update({
            "encoding_size": self.encoding_size,
            "projection_dropout_rate": self.projection_dropout_rate
        })

        return config
